"""Main LLM job for one (dataset, LLM): computes everything the routing analysis needs.

targets  = test_eval  ∪  val_gain  (∪ annotation set for RQ6 distillation)
1. f0 (text-only) distributions for targets, their neighbour candidates, and neighbours of training nodes
2. label-compatibility matrix Pi (zero-shot: from pseudo-labels; supervised: training labels where available)
3. SIG features (A, D, NB) per target, with estimated Pi and with identity Pi (homophily assumption)
4. neighbour selection: random / similar (SNS-style) / CNS (ours) ; graph-augmented f1 for each, plus CNS with
   pseudo-label hints, a keyword-compression and an LLM neighbour-summary baseline. CNS is run on all targets (val ->
   learned router, annotation set -> distillation); all other strategies only on test_eval (where they are evaluated)
--k K        : k-sensitivity run, re-uses f0 / Pi / SIG of raw.pkl -> raw_kK.pkl
--strategies : subset of strategies (e.g. the 3B / 32B scaling runs)
--graphicl   : also run the GraphICL-style baseline (labelled demonstrations + neighbours; test nodes only)
-> results/llm/<llm>/<dataset>/raw.pkl (+ token costs)
"""
import argparse
import pickle
import time

import numpy as np

from . import prompts as PR
from .data import build, candidates
from .llm import LLM
from .sig import estimate_compat, message, select_cns, select_random, select_similar, sig_features
from .utils import DATA, ROOT, get_logger, load_config, slug


def embeddings(g, cfg):
    p = DATA / "processed" / f"{g['name']}_emb.npy"
    if p.exists():
        return np.load(p)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(cfg["models"]["embedder"], device="cuda")
    e = st.encode([t[:2000] for t in g["texts"]], batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    np.save(p, e.astype(np.float32))
    return e


def keywords(g, nodes, top=10):
    """top TF-IDF terms of each node text (fitted on the texts of the graph)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = np.random.RandomState(0)
    fit_idx = rng.choice(len(g["texts"]), min(len(g["texts"]), 20000), replace=False)
    vec = TfidfVectorizer(stop_words="english", max_features=50000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    vec.fit([g["texts"][i] for i in fit_idx])
    vocab = np.array(vec.get_feature_names_out())
    X = vec.transform([g["texts"][u] for u in nodes])
    out = {}
    for row, u in enumerate(nodes):
        r = X.getrow(row)
        idx = r.indices[np.argsort(-r.data)[:top]]
        out[u] = "keywords: " + ", ".join(vocab[idx]) if len(idx) else "(no text)"
    return out


DEFAULT_STRATEGIES = ["random", "similar", "cns", "cns_id", "cns_nored", "cns_hints", "keywords", "summary"]
ALL_STRATEGIES = DEFAULT_STRATEGIES + ["graphicl"]
N_DEMOS = 3  # GraphICL-style baseline: number of labelled demonstrations (most similar training nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--annotate", type=int, default=0, help="extra non-test annotation nodes (RQ6 distillation)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="k-sensitivity run: re-uses f0/SIG of raw.pkl, only random/similar/CNS with k neighbours")
    ap.add_argument("--strategies", default=None, help="comma-separated subset of " + ",".join(ALL_STRATEGIES))
    ap.add_argument("--graphicl", action="store_true", help="add the GraphICL-style baseline to the strategies")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pc = cfg["prompt"]
    sens = a.k is not None and a.k != pc["k_neighbors"]
    k = a.k if sens else pc["k_neighbors"]
    out = ROOT / "results" / "llm" / slug(a.llm) / a.dataset
    log = get_logger(f"{slug(a.llm)}/{a.dataset}" + (f"/k{k}" if sens else ""),
                     out / (f"run_k{k}.log" if sens else "run.log"))
    strategies = (a.strategies.split(",") if a.strategies else
                  (["random", "similar", "cns"] if sens else list(DEFAULT_STRATEGIES)))
    if a.graphicl and not sens and "graphicl" not in strategies:
        strategies.append("graphicl")
    bad = [s for s in strategies if s not in ALL_STRATEGIES]
    assert not bad, f"unknown strategies {bad}"
    if "cns" not in strategies:
        strategies.append("cns")  # the learned router (trained on val) and the distillation need CNS
    g = build(a.dataset, cfg)
    emb = embeddings(g, cfg)
    C = len(g["label_names"])
    assert C <= len(PR.LETTERS), f"{C} classes: more than {len(PR.LETTERS)} single-character option letters"
    letters = PR.LETTERS[:C]  # A-Z, then a-z (arXiv: 40 classes); each is a single token -> first-token scoring
    rng = np.random.RandomState(cfg["seed"])
    targets = [int(v) for v in g["test_eval"]] + [int(v) for v in g["val_gain"]]
    nt = len(g["test_eval"])
    test_t = targets[:nt]
    annot = []
    if a.annotate and not sens:
        # annotation pool: every node except validation nodes and the fixed LLM evaluation subset of the test set
        # (the distilled GNN is evaluated on that never-annotated subset)
        excl = np.zeros(len(g["y"]), bool)
        excl[g["val"]] = True
        excl[g["test_eval"]] = True
        pool = np.where(~excl)[0]
        annot = [int(x) for x in rng.choice(pool, min(a.annotate, len(pool)), replace=False)]
        tset = set(targets)
        targets += [x for x in annot if x not in tset]
    cand = {v: [int(u) for u in candidates(g, v, pc["max_candidates"], np.random.RandomState(cfg["seed"] + int(v)))]
            for v in targets}
    one_hop = {v: set(map(int, g["adj"][v])) for v in targets}
    # neighbours used by the SIG estimators: the 1-hop part of the (seeded) candidate pool -> deterministic per node
    nbr1 = {v: [u for u in cand[v] if u in one_hop[v]] for v in targets}
    tp = 2 if ("32B" in a.llm or "70B" in a.llm) else 1  # build_vllm clamps tp to the number of visible GPUs
    t0 = time.time()
    base = None
    if sens:  # re-use f0 distributions, Pi and SIG features of the main run (same nodes, same candidate pools)
        base = pickle.load(open(out / "raw.pkl", "rb"))
        dist = {int(v): np.asarray(p) for v, p in base["dist_all"].items()}
        missing = [u for v in targets for u in [v] + cand[v] if u not in dist]
        assert not missing, f"{len(missing)} nodes without f0 distribution in raw.pkl"
        Pi_sup = base["Pi"]["est"]
        log.info(f"k={k}: {len(targets)} targets ({nt} test), strategies={strategies}")
    else:
        train_idx = np.where(g["train"])[0]
        train_nbrs = {int(u) for v in train_idx for u in g["adj"][v][:10]}
        need = sorted(set(targets) | {u for v in targets for u in cand[v]} | train_nbrs | set(map(int, train_idx)))
        log.info(f"{len(targets)} targets ({nt} test), f0 needed for {len(need)} nodes, C={C}, "
                 f"homophily={g['homophily']:.3f}, strategies={strategies}")
    llm = LLM(a.llm, max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"], tp=tp,
              cache_name=a.dataset)
    log.info(f"engine: {llm.engine_info}")
    tau = cfg["routing"]["nb_temperature"]
    Pi_id = np.eye(C) * 0.9 + 0.1 / C                                            # homophily assumption
    if not sens:
        p0_prompts = [PR.f0(g, v, cfg) for v in need]
        P0 = llm.option_probs(p0_prompts, letters, tag="f0")
        tok0 = dict(zip(need, llm.n_tokens(p0_prompts)))
        dist = {v: P0[i] for i, v in enumerate(need)}
        log.info(f"f0 done in {time.time() - t0:.0f}s; llm stats {llm.stats}")
        # contextual calibration prior (Zhao et al., 2021): option distribution for a content-free input
        cf_prior = llm.option_probs([PR.f0(g, int(need[0]), cfg, text=x) for x in ("N/A", "[MASK]", "")], letters,
                                    tag="cf").mean(0)
        # LLM self-assessment router baseline: P("the text alone is NOT sufficient")  (test nodes only)
        self_need = llm.option_probs([PR.self_need(g, v, cfg) for v in test_t], ["A", "B"], tag="self_need")[:, 1]
        # ---- compatibility matrices
        ei = g["edge_index"]
        known = np.array([int(v) in dist and int(u) in dist for v, u in ei.T])
        E = ei[:, known]
        Pi_zero = estimate_compat(E, dist)                                       # zero-shot (pseudo labels only)
        Pi_sup = estimate_compat(E, dist, labels=g["y"], labeled=g["train"])     # + training labels
        ts = time.perf_counter()
        feats = {name: [sig_features(dist[v], [message(dist[u], Pi) for u in nbr1[v]], tau) for v in targets]
                 for name, Pi in [("est", Pi_sup), ("zero", Pi_zero), ("id", Pi_id)]}
        t_sig = (time.perf_counter() - ts) / 3 / max(len(targets), 1)  # CPU seconds per node for one SIG variant
    # ---- neighbour selection: CNS for every target (val -> learned router, annotation -> distillation),
    #      the other strategies only for the test nodes (they are only evaluated there)
    sel = {s: {} for s in ("random", "similar", "cns", "cns_id", "cns_nored") if s in strategies}
    test_set = set(test_t)
    t_cns = 0.0
    for v in targets:
        c = cand[v]
        ec = emb[c] if c else np.zeros((0, emb.shape[1]))
        ts = time.perf_counter()
        msgs = [message(dist[u], Pi_sup, 1 if u in one_hop[v] else 2) for u in c]
        sel["cns"][v] = select_cns(dist[v], c, msgs, emb[v], ec, k, tau)
        t_cns += time.perf_counter() - ts
        if v not in test_set:
            continue
        if "random" in sel:
            sel["random"][v] = [int(x) for x in select_random(g, v, c, k, np.random.RandomState(int(v)))]
        if "similar" in sel:
            sel["similar"][v] = [int(x) for x in select_similar(c, emb[v], ec, k)]
        if "cns_id" in sel:
            sel["cns_id"][v] = select_cns(dist[v], c, [message(dist[u], Pi_id, 1) for u in c], emb[v], ec, k, tau)
        if "cns_nored" in sel:
            sel["cns_nored"][v] = select_cns(dist[v], c, msgs, emb[v], ec, k, tau, lam_red=0.0)
    # ---- graph-augmented f1 (arrays aligned with `targets` for cns, with the test prefix for all others)
    P1, tok1 = {}, {}
    for s, nb in sel.items():
        nodes = targets if s == "cns" else test_t
        pr = [PR.f1(g, v, nb[v], cfg) for v in nodes]
        P1[s] = llm.option_probs(pr, letters, tag=f"f1_{s}")
        tok1[s] = llm.n_tokens(pr)
        log.info(f"f1[{s}] on {len(nodes)} nodes done ({time.time() - t0:.0f}s)")
    if "cns_hints" in strategies:
        pr = [PR.f1(g, v, sel["cns"][v], cfg, hints=[int(np.argmax(dist[u])) for u in sel["cns"][v]]) for v in test_t]
        P1["cns_hints"] = llm.option_probs(pr, letters, tag="f1_cns_hints")
        tok1["cns_hints"] = llm.n_tokens(pr)
    if "keywords" in strategies:
        # GraphiT-style token-efficient baseline: random neighbours compressed to TF-IDF keywords
        rnd = sel.get("random") or {v: [int(x) for x in select_random(g, v, cand[v], k, np.random.RandomState(int(v)))]
                                    for v in test_t}
        kw = keywords(g, sorted({u for v in test_t for u in rnd[v]}))
        pr = [PR.f1(g, v, rnd[v], cfg, nbr_texts=[kw[u] for u in rnd[v]]) for v in test_t]
        P1["keywords"] = llm.option_probs(pr, letters, tag="f1_keywords")
        tok1["keywords"] = llm.n_tokens(pr)
    if "summary" in strategies:
        # LLM neighbour-summary baseline (two calls per node: summarise up to 10 neighbours, then classify);
        # cost = summarisation prompt + generated summary + classification prompt
        nb10 = {v: [int(u) for u in g["adj"][v][:10]] for v in test_t}
        has = [v for v in test_t if nb10[v]]
        sp = [PR.summarize(g, nb10[v], cfg) for v in has]
        summ = dict(zip(has, [s_.strip() for s_ in llm.chat(sp, max_tokens=120, tag="summary")]))
        stok = dict(zip(has, llm.n_tokens(sp)))
        gtok = {v: len(llm.tok(summ[v], add_special_tokens=False)["input_ids"]) for v in has}
        pr = [PR.f1(g, v, [], cfg, summary=summ[v]) if v in summ else PR.f0(g, v, cfg) for v in test_t]
        P1["summary"] = llm.option_probs(pr, letters, tag="f1_summary")
        tok1["summary"] = [t + stok.get(v, 0) + gtok.get(v, 0) for v, t in zip(test_t, llm.n_tokens(pr))]
    if "graphicl" in strategies:
        # GraphICL-style baseline (Sun et al., 2025; training-free): the N_DEMOS most similar *training* nodes as
        # labelled demonstrations + k similarity-selected neighbours whose training labels are shown when known
        tr = np.where(g["train"])[0]
        sim = emb[test_t] @ emb[tr].T
        demos = {v: [int(tr[j]) for j in np.argsort(-sim[i], kind="stable")[:N_DEMOS]] for i, v in enumerate(test_t)}
        nbs = sel.get("similar") or {
            v: [int(x) for x in select_similar(cand[v], emb[v], emb[cand[v]] if cand[v] else
                                               np.zeros((0, emb.shape[1])), k)] for v in test_t}
        pr = [PR.graphicl(g, v, nbs[v], demos[v], cfg) for v in test_t]
        P1["graphicl"] = llm.option_probs(pr, letters, tag="f1_graphicl")
        tok1["graphicl"] = llm.n_tokens(pr)
        log.info(f"f1[graphicl] on {len(test_t)} nodes done ({time.time() - t0:.0f}s)")
    common = dict(dataset=a.dataset, llm=a.llm, targets=np.array(targets), n_test=nt, k=k,
                  P1=P1, tok1={k_: np.array(v_) for k_, v_ in tok1.items()}, sel=sel, time_sec=time.time() - t0,
                  llm_stats=dict(llm.stats), engine=dict(llm.engine_info), strategies=strategies,
                  overhead={"cns_sec_per_node": t_cns / max(len(targets), 1),
                            **({} if sens else {"sig_sec_per_node": t_sig})})
    if sens:
        res = dict(base, **common)  # f0 / SIG of the main run + the k-specific selections and f1
        idx = {int(v): i for i, v in enumerate(base["targets"])}
        ii = [idx[v] for v in targets]
        for key in ("y", "P0", "tok0", "deg", "text_len", "tok_nbr_f0"):
            res[key] = np.asarray(base[key])[ii]
        res["feats"] = {n_: [base["feats"][n_][i] for i in ii] for n_ in base["feats"]}
        res["nbr1"] = [base["nbr1"][i] for i in ii]
        res["n_val"] = len(targets) - nt
        res["annot"] = np.zeros(0, np.int64)
        res.pop("dist_all", None)
        fn = f"raw_k{k}.pkl"
    else:
        # cost view (ii): tokens of the neighbours' text-only calls needed by SIG / CNS for this node alone
        tok_nbr = np.array([sum(tok0.get(u, 0) for u in cand[v]) for v in targets])
        res = dict(common, n_val=len(g["val_gain"]), annot=np.array(annot, dtype=np.int64), y=g["y"][targets],
                   P0=np.stack([dist[v] for v in targets]), tok0=np.array([tok0[v] for v in targets]), feats=feats,
                   Pi={"est": Pi_sup, "zero": Pi_zero}, homophily=g["homophily"], C=C,
                   deg=np.array([len(g["adj"][v]) for v in targets]), nbr1=[nbr1[v] for v in targets],
                   text_len=np.array([len(g["texts"][v].split()) for v in targets]),
                   dist_all={int(v): dist[v] for v in dist}, tok_nbr_f0=tok_nbr, cf_prior=cf_prior,
                   self_need=self_need)
        fn = "raw.pkl"
    tmp = out / (fn + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=4)
    tmp.replace(out / fn)  # atomic: a crash never leaves a truncated done-file behind
    yt = res["y"][:nt]
    log.info(f"done in {time.time() - t0:.0f}s; llm stats {llm.stats}; acc f0={np.mean(res['P0'][:nt].argmax(1) == yt):.4f} "
             + " ".join(f"f1[{s}]={np.mean(P[:nt].argmax(1) == yt):.4f}" for s, P in P1.items()))


if __name__ == "__main__":
    main()
