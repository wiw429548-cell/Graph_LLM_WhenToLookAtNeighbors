"""Offline routing analysis for one (LLM, dataset) from raw.pkl (no LLM calls).

RQ1 heterogeneity  : accuracy of f0 vs f1 per ambiguity quintile (cross-over curve) + by SIG tercile
RQ2 routing        : accuracy-vs-budget and accuracy-vs-tokens for every router x neighbour strategy
RQ3 SIG estimators : Spearman(score, true gain) ; ablations (A only, D only, NB with/without compatibility)
RQ4 selection      : random vs similar vs CNS (+ hints, keywords, LLM summary) at full and partial budgets
RQ5 heterophily    : NB / CNS with estimated vs identity compatibility; accuracy by *local* (per-node) homophily
                     (true labels, analysis only) and the dataset's adjusted homophily
Gain breakdown     : fraction of test nodes helped / hurt by each neighbour strategy, and which of them NeedNbr routes
Baselines w/o routing: GraphICL-style prompting (if run), f0 + GCN probability ensemble and NeedNbr -> GCN cascade
                     (GCN probabilities from results/gnn/probs_<dataset>.npy, if present; no LLM call)

Leakage control: every quantity reported is computed on the fixed test subset (indices 0..n_test-1 of raw targets).
Anything that is *fitted* (learned router g_phi, the NeedNbr-auto routing threshold) is fitted on the val_gain nodes
only; the compatibility matrix uses training labels only; budgets are fixed a priori; the 'oracle' router uses test
labels and is reported only as an upper bound.
--ksens: k-sensitivity table from raw.pkl (k=5) and raw_k*.pkl -> results/analysis/<llm>/ksens_<dataset>.json
"""
import argparse
import pickle

import numpy as np
from scipy.stats import spearmanr

from .data import PROC
from .sig import message, nb_posterior
from .utils import ROOT, entropy, load_config, save_json, slug

STRATS = ["cns", "similar", "random", "cns_id", "cns_nored", "cns_hints", "keywords", "summary", "graphicl"]
HBINS = [0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9]

TF = "NB (Pi=est.) [NeedNbr-TF]"
LR = "learned g_phi [NeedNbr-L]"
N_RANDOM = 200  # the random router is averaged over this many permutations


def budget_curve(order, base_correct, routed_correct, tok0, tok1, budgets):
    """order: nodes sorted by decreasing router score; routes the top-b fraction to f1."""
    n = len(order)
    out = []
    for b in budgets:
        m = np.zeros(n, bool)
        m[order[:int(round(b * n))]] = True
        acc = np.where(m, routed_correct, base_correct).mean()
        toks = (tok0.sum() + tok1[m].sum()) / n
        out.append({"budget": b, "acc": float(acc), "tokens_per_node": float(toks)})
    return out


def random_curve(base_correct, routed_correct, tok0, tok1, budgets, seed):
    rng = np.random.RandomState(seed)
    curves = [budget_curve(rng.permutation(len(base_correct)), base_correct, routed_correct, tok0, tok1, budgets)
              for _ in range(N_RANDOM)]
    return [{"budget": b, "acc": float(np.mean([c[i]["acc"] for c in curves])),
             "acc_std": float(np.std([c[i]["acc"] for c in curves])),
             "tokens_per_node": float(np.mean([c[i]["tokens_per_node"] for c in curves]))}
            for i, b in enumerate(budgets)]


def area(curve):
    xs = [c["budget"] for c in curve]
    ys = [c["acc"] for c in curve]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else float(ys[0])


def rank_bins(score, n_bins):
    """equal-size bins by rank (robust to ties, never empty if len(score) >= n_bins)"""
    return [b for b in np.array_split(np.argsort(score, kind="stable"), n_bins) if len(b)]


def learned_router(F_val, gain_val, F_test, seed):
    from sklearn.ensemble import GradientBoostingRegressor
    if len(F_val) < 20 or np.all(gain_val == gain_val[0]):
        return F_test[:, 2], False  # fall back to NB
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    m.fit(F_val, gain_val)
    return m.predict(F_test), True


def auto_threshold(s_val, c0_val, c1_val):
    """threshold on a router score chosen on the validation nodes: maximise val accuracy of 'route if score >= th';
    ties -> route fewer nodes. Returns +inf (= never route) if no threshold beats text-only on val."""
    best_th, best_acc = np.inf, float(c0_val.mean())
    for th in np.unique(s_val)[::-1]:
        acc = float(np.where(s_val >= th, c1_val, c0_val).mean())
        if acc > best_acc + 1e-12:
            best_th, best_acc = float(th), acc
    return best_th, best_acc


def neighbours(raw, g):
    """SIG neighbour set per target (stored by run_llm; older raw files: 1-hop neighbours with a known f0)"""
    if "nbr1" in raw:
        return [list(map(int, x)) for x in raw["nbr1"]]
    d = raw["dist_all"]
    return [[int(u) for u in g["adj"][v] if int(u) in d][:30] for v in raw["targets"]]


def mlp_proxy_scores(raw, cfg, g, nbr):
    """SIG estimated with neighbour pseudo-labels from a cheap embedding classifier (trained on the training labels
    only) instead of LLM calls on the neighbours (cost variant)."""
    from sklearn.linear_model import LogisticRegression
    emb = np.load(PROC / f"{raw['dataset']}_emb.npy")
    tr = np.where(g["train"])[0]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb[tr], g["y"][tr])
    C = raw["C"]
    need = sorted({u for nb in nbr for u in nb})
    full = {}
    if need:
        pr = clf.predict_proba(emb[need])
        for i, u in enumerate(need):
            p = np.zeros(C)
            p[clf.classes_] = pr[i]
            full[u] = p + 1e-6
    Pi = raw["Pi"]["est"]
    tau = cfg["routing"]["nb_temperature"]
    s = []
    for i in range(len(raw["targets"])):
        msgs = [message(full[u], Pi) for u in nbr[i]]
        p = raw["P0"][i]
        s.append(float(entropy(p) - entropy(nb_posterior(p, msgs, tau))) if msgs else 0.0)
    return np.array(s)


def calibrate(raw):
    """contextual calibration (Zhao et al., 2021): p_cal(y|x) ∝ p(y|x) / p(y|content-free)"""
    w = 1.0 / np.clip(np.asarray(raw["cf_prior"], dtype=float), 1e-6, None)
    out = {}
    for v, p in raw["dist_all"].items():
        q = np.asarray(p) * w
        out[int(v)] = q / q.sum()
    return out


def nb_scores(raw, nbr, tau, dist=None):
    calibrated = dist is not None
    dist = dist if calibrated else {int(k): v for k, v in raw["dist_all"].items()}
    Pi = raw["Pi"]["est"]
    out = []
    for i, v in enumerate(raw["targets"]):
        msgs = [message(dist[u], Pi) for u in nbr[i]]
        p = dist[int(v)] if calibrated else raw["P0"][i]
        out.append(float(entropy(p) - entropy(nb_posterior(p, msgs, tau))) if msgs else 0.0)
    return np.array(out)


def mcnemar_boot(a, b, n_boot=2000, seed=0):
    from scipy.stats import binomtest
    a, b = a.astype(float), b.astype(float)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    rng = np.random.RandomState(seed)
    d = a - b
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"delta_acc": float(d.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "p_mcnemar": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0, "n": int(len(d))}


def routed_at(score, b, n):
    order = np.argsort(-score, kind="stable")
    m = np.zeros(n, bool)
    m[order[:int(round(b * n))]] = True
    return m


def analyze(raw, cfg, g):
    nt, nv = raw["n_test"], raw["n_val"]
    te, va = np.arange(nt), np.arange(nt, nt + nv)
    y = raw["y"]
    c0 = raw["P0"].argmax(1) == y
    c1 = {s: np.asarray(P).argmax(1) == y[:len(P)] for s, P in raw["P1"].items()}   # cns: all targets, others: test
    gain = {s: c.astype(float) - c0[:len(c)].astype(float) for s, c in c1.items()}
    budgets = cfg["routing"]["budgets"]
    tok0 = raw["tok0"][te].astype(float)
    res = {"dataset": raw["dataset"], "llm": raw["llm"], "homophily": raw["homophily"], "C": raw["C"],
           "n_test": int(nt), "n_val": int(nv), "k": raw.get("k"),
           "acc_f0": float(c0[te].mean()), "acc_f1_full": {s: float(c[te].mean()) for s, c in c1.items()},
           "tokens_f0": float(tok0.mean()),
           "tokens_f1_only": {s: float(np.asarray(raw["tok1"][s])[te].mean()) for s in raw["tok1"]},
           "engine": raw.get("engine"), "llm_stats": raw.get("llm_stats")}
    F = {k: np.array([[r["A"], r["D"], r["NB"], r["deg"], r["maxp"]] for r in rows]) for k, rows in raw["feats"].items()}
    nb_pred = np.array([r.get("nb_pred", -1) for r in raw["feats"]["est"]])
    res["acc_nb_posterior"] = float(np.where(nb_pred[te] >= 0, nb_pred[te] == y[te], c0[te]).mean())
    res["frac_no_neighbours"] = float((F["est"][te, 3] == 0).mean())
    # ---- static (non-routing) combination baselines that call f1 for every node
    if "cns" in raw["P1"]:
        P0t, P1t = raw["P0"][te], np.asarray(raw["P1"]["cns"])[te]
        both = float((tok0 + np.asarray(raw["tok1"]["cns"])[te]).mean())
        ens = (P0t + P1t).argmax(1) == y[te]
        mc = np.where(P0t.max(1) >= P1t.max(1), P0t.argmax(1), P1t.argmax(1)) == y[te]
        res["static_baselines"] = {"ensemble f0+f1(CNS)": {"acc": float(ens.mean()), "tokens_per_node": both},
                                   "max-confidence f0 vs f1(CNS)": {"acc": float(mc.mean()), "tokens_per_node": both}}
    # ---- RQ1: cross-over by ambiguity quintile (rank-based bins)
    A = F["est"][:, 0]
    res["crossover_by_ambiguity"] = []
    for b, idx in enumerate(rank_bins(A[te], 5)):
        res["crossover_by_ambiguity"].append(
            {"bin": b, "A_range": [float(A[te][idx].min()), float(A[te][idx].max())], "n": int(len(idx)),
             "acc_f0": float(c0[te][idx].mean()), **{f"acc_f1_{s}": float(c[te][idx].mean()) for s, c in c1.items()}})
    # ---- routers (scores on the test nodes; the learned router is fitted on val_gain only)
    nbr = neighbours(raw, g)
    feat_val = np.column_stack([F["est"][va], F["id"][va][:, 2], np.log1p(raw["text_len"][va])])
    feat_te = np.column_stack([F["est"][te], F["id"][te][:, 2], np.log1p(raw["text_len"][te])])
    lr_score, lr_ok = learned_router(feat_val, gain["cns"][va], feat_te, cfg["seed"])
    res["learned_router_fitted"] = bool(lr_ok)
    routers = {
        "degree": raw["deg"][te].astype(float),
        "A_only (confidence)": F["est"][te, 0],
        "D_only": F["est"][te, 1],
        "AxD": F["est"][te, 0] * F["est"][te, 1],
        "NB (Pi=identity)": F["id"][te, 2],
        "NB (Pi=zero-shot est.)": F["zero"][te, 2],
        TF: F["est"][te, 2],
        LR: lr_score,
    }
    try:
        routers["NB with MLP pseudo-labels (no neighbour LLM calls)"] = mlp_proxy_scores(raw, cfg, g, nbr)[te]
    except Exception as e:  # noqa: BLE001
        res["mlp_proxy_error"] = repr(e)
    if raw.get("self_need") is not None:
        routers["LLM self-assessment"] = np.asarray(raw["self_need"])[te]
    if raw.get("cf_prior") is not None:  # contextual calibration of the option distributions
        try:
            cal = calibrate(raw)
            P0c = np.stack([cal[int(v)] for v in raw["targets"]])
            res["acc_f0_calibrated"] = float((P0c[te].argmax(1) == y[te]).mean())
            routers["A_only (calibrated)"] = -(P0c[te] * np.log(np.clip(P0c[te], 1e-12, 1))).sum(1)
            routers["NB (Pi=est., calibrated)"] = nb_scores(raw, nbr, cfg["routing"]["nb_temperature"], dist=cal)[te]
        except Exception as e:  # noqa: BLE001
            res["calibration_error"] = repr(e)
    for t in (0.25, 1.0):  # sensitivity to the naive-Bayes temperature tau (routing only, no new LLM calls)
        routers[f"NB (Pi=est., tau={t})"] = nb_scores(raw, nbr, t)[te]
    rng = np.random.RandomState(cfg["seed"])
    curves, spear = {}, {}
    for strat in STRATS:
        if strat not in c1:
            continue
        t1 = np.asarray(raw["tok1"][strat])[te].astype(float)
        cur = random_curve(c0[te], c1[strat][te], tok0, t1, budgets, cfg["seed"])
        curves[f"{strat}|random"] = {"curve": cur, "area": area(cur)}
        orc = gain[strat][te] + 1e-3 * rng.rand(nt)  # oracle (uses test labels): upper bound only
        for rname, score in dict(routers, oracle=orc).items():
            order = np.argsort(-score, kind="stable")
            cur = budget_curve(order, c0[te], c1[strat][te], tok0, t1, budgets)
            curves[f"{strat}|{rname}"] = {"curve": cur, "area": area(cur)}
            if strat == "cns" and "tok_nbr_f0" in raw:  # cost view (ii): + neighbours' text-only calls
                cur2 = budget_curve(order, c0[te], c1[strat][te], tok0 + raw["tok_nbr_f0"][te], t1, budgets)
                curves[f"cns_costview2|{rname}"] = {"curve": cur2, "area": area(cur2)}
        for rname, score in routers.items():
            r = spearmanr(score, gain[strat][te]).correlation if np.std(score) > 0 and np.std(gain[strat][te]) > 0 else None
            spear[f"{strat}|{rname}"] = None if r is None or r != r else float(r)
    res["curves"], res["spearman_vs_true_gain"] = curves, spear
    # ---- NeedNbr-auto: routing threshold fitted on the val nodes (no budget given), applied to the test nodes
    res["auto_threshold"] = {}
    t1c = np.asarray(raw["tok1"]["cns"])[te].astype(float)
    for name, col in (("NeedNbr-auto (NB, val-tuned threshold)", 2), ("A_only-auto (val-tuned threshold)", 0)):
        th, acc_val = auto_threshold(F["est"][va, col], c0[va], c1["cns"][va])
        m = F["est"][te, col] >= th
        res["auto_threshold"][name] = {"threshold": th, "val_acc": acc_val, "routed_frac": float(m.mean()),
                                       "acc": float(np.where(m, c1["cns"][te], c0[te]).mean()),
                                       "tokens_per_node": float((tok0.sum() + t1c[m].sum()) / nt)}
    # ---- by-SIG terciles: 'structure does not help' on average but not per group (rank-based)
    sc = F["est"][te, 2]
    res["by_sig_tercile"] = []
    for b, idx in enumerate(rank_bins(sc, 3)):
        row = {"tercile": b, "n": int(len(idx)), "acc_f0": float(c0[te][idx].mean())}
        for s_ in ("random", "similar", "cns"):
            if s_ in c1:
                row[f"acc_f1_{s_}"] = float(c1[s_][te][idx].mean())
        res["by_sig_tercile"].append(row)
    # ---- significance: NeedNbr-TF @20% (CNS) vs alternatives (exact McNemar) + bootstrap CI of the accuracy delta
    routed = routed_at(routers[TF], 0.2, nt)
    ours = np.where(routed, c1["cns"][te], c0[te])
    comps = {"f0 (text only)": c0[te], **{f"f1 {s_} (all nodes)": c1[s_][te]
                                           for s_ in ("random", "similar", "cns", "summary", "graphicl") if s_ in c1}}
    for rname in ("A_only (confidence)", "LLM self-assessment", "NB (Pi=identity)", "degree", LR):
        if rname in routers:
            comps[f"{rname} @20%"] = np.where(routed_at(routers[rname], 0.2, nt), c1["cns"][te], c0[te])
    comps["random router @20% (seeded draw)"] = np.where(routed_at(np.random.RandomState(cfg["seed"]).rand(nt), 0.2, nt),
                                                         c1["cns"][te], c0[te])
    if "random" in c1:  # NeedNbr with CNS vs the same router with random neighbours
        comps["NeedNbr-TF @20% w/ random neighbours"] = np.where(routed, c1["random"][te], c0[te])
    lg = llm_gnn(raw, te, y, c0, routed)
    if lg:
        res["llm_gnn"] = {k_: v_ for k_, v_ in lg.items() if k_ != "_correct"}
        comps.update(lg["_correct"])
    res["significance_needNbr20"] = {name: mcnemar_boot(ours, other) for name, other in comps.items()}
    res["gain_breakdown"] = gain_breakdown(c0[te], {s_: c[te] for s_, c in c1.items()}, routed)
    res["adj_homophily"] = adjusted_homophily(g)
    res["by_local_homophily"] = by_local_homophily(raw, g, te, c0, c1, routed)
    res["overhead"] = raw.get("overhead")
    # ---- who gets routed? (profile of the top-20% nodes)
    prof = {"degree": raw["deg"][te], "text_len": raw["text_len"][te], "ambiguity_A": F["est"][te, 0],
            "f0_correct": c0[te].astype(float)}
    res["routed_profile"] = {k: {"routed": float(v[routed].mean()), "not_routed": float(v[~routed].mean())}
                             for k, v in prof.items()}
    res["cases"] = cases(raw, g, te, c0, c1)
    return res


def llm_gnn(raw, te, y, c0, routed):
    """training-free LLM+GNN combinations (no extra LLM call): average of f0 and GCN probabilities, and a cascade that
    answers the NeedNbr-routed nodes with the GCN instead of a graph-augmented prompt. The GCN uses the 20 training
    labels per class (like Pi), i.e. this is a stronger-supervision baseline."""
    p = ROOT / "results" / "gnn" / f"probs_{raw['dataset']}.npy"
    if not p.exists():
        return None
    try:
        Pg = np.load(p)[np.asarray(raw["targets"])[te]]
    except (OSError, ValueError, IndexError):
        return None
    P0 = raw["P0"][te]
    if Pg.shape != P0.shape:
        return None
    cg = Pg.argmax(1) == y[te]
    ens = (P0 + Pg).argmax(1) == y[te]
    casc = np.where(routed, cg, c0[te])
    return {"acc_gcn": float(cg.mean()), "acc_f0_gcn_ensemble": float(ens.mean()),
            "acc_needNbr20_to_gcn": float(casc.mean()),
            "_correct": {"GCN (20 labels/class)": cg, "f0 + GCN ensemble": ens, "NeedNbr-TF@20% -> GCN": casc}}


def gain_breakdown(c0, c1, routed):
    """per strategy: share of test nodes helped (f0 wrong, f1 right) / hurt (f0 right, f1 wrong); for every strategy
    also how many of its helped / hurt nodes fall into the top-20% NeedNbr-TF set"""
    out = {}
    for s_, c in c1.items():
        helped, hurt = (~c0) & c, c0 & (~c)
        out[s_] = {"helped": float(helped.mean()), "hurt": float(hurt.mean()), "net": float(helped.mean() - hurt.mean()),
                   "helped_in_routed20": float(helped[routed].sum() / max(helped.sum(), 1)),
                   "hurt_in_routed20": float(hurt[routed].sum() / max(hurt.sum(), 1)),
                   "net_routed20": float((helped & routed).mean() - (hurt & routed).mean())}
    anyc = np.zeros_like(c0)
    for c in c1.values():
        anyc |= c
    out["_any_strategy_fixes"] = float(((~c0) & anyc).mean())
    return out


def adjusted_homophily(g):
    """adjusted edge homophily (Platonov et al., 2022): edge homophily corrected for the class-degree distribution"""
    ei, y = np.asarray(g["edge_index"]), np.asarray(g["y"])
    if ei.shape[1] == 0:
        return None
    h = float((y[ei[0]] == y[ei[1]]).mean())
    pk = np.bincount(y[np.concatenate([ei[0], ei[1]])], minlength=int(y.max()) + 1) / (2.0 * ei.shape[1])
    s2 = float((pk ** 2).sum())
    return float((h - s2) / (1 - s2)) if s2 < 1 else None


def by_local_homophily(raw, g, te, c0, c1, routed):
    """accuracy of f0 / f1 / NeedNbr@20% by the test node's local homophily (share of 1-hop neighbours with the same
    true label -- uses test labels, analysis only, never used for routing)"""
    y = np.asarray(g["y"])
    tg = np.asarray(raw["targets"])[te]
    lh = np.array([np.mean(y[np.asarray(g["adj"][v], dtype=np.int64)] == y[v]) if len(g["adj"][v]) else np.nan
                   for v in tg])
    groups = [("isolated", np.isnan(lh))]
    for lo, hi in zip(HBINS[:-1], HBINS[1:]):
        groups.append((f"[{lo:.2f},{min(hi, 1.0):.2f}{']' if hi > 1 else ')'}", (lh >= lo) & (lh < hi)))
    ours = np.where(routed, c1["cns"][te], c0[te])
    out = []
    for name, m in groups:
        if not m.any():
            continue
        row = {"bin": name, "n": int(m.sum()), "acc_f0": float(c0[te][m].mean()),
               "acc_needNbr20": float(ours[m].mean()), "routed_frac": float(routed[m].mean())}
        for s_ in ("random", "similar", "cns", "graphicl"):
            if s_ in c1:
                row[f"acc_f1_{s_}"] = float(c1[s_][te][m].mean())
        out.append(row)
    return out


def cases(raw, g, te, c0, c1, n=3):
    """nodes where CNS neighbours fix the text-only error but similar (SNS-style) neighbours do not"""
    if "similar" not in c1:
        return []
    names = g["label_names"]
    out = []
    idx = [i for i in te if c1["cns"][i] and not c0[i] and not c1["similar"][i]][:n]
    for i in idx:
        v = int(raw["targets"][i])
        out.append({"node": v, "true": names[g["y"][v]], "f0_pred": names[int(raw["P0"][i].argmax())],
                    "text": g["texts"][v][:200],
                    "similar_neighbours_true_labels": [names[g["y"][u]] for u in raw["sel"]["similar"][v]],
                    "cns_neighbours_true_labels": [names[g["y"][u]] for u in raw["sel"]["cns"][v]],
                    "f1_similar_pred": names[int(raw["P1"]["similar"][i].argmax())],
                    "f1_cns_pred": names[int(raw["P1"]["cns"][i].argmax())]})
    return out


def k_sensitivity(llm_dir, dataset, cfg):
    """accuracy (all test nodes, and NeedNbr-TF @20%) and f1 tokens for every available k"""
    out = {}
    for p in sorted((llm_dir / dataset).glob("raw*.pkl")):
        raw = pickle.load(open(p, "rb"))
        nt = raw["n_test"]
        te = np.arange(nt)
        y = raw["y"][:nt]
        c0 = raw["P0"][:nt].argmax(1) == y
        sc = np.array([r["NB"] for r in raw["feats"]["est"]])[te]
        m = routed_at(sc, 0.2, nt)
        row = {}
        for s_ in ("random", "similar", "cns"):
            if s_ in raw["P1"]:
                c1 = np.asarray(raw["P1"][s_])[:nt].argmax(1) == y
                row[s_] = {"all": float(c1.mean()), "needNbr@20%": float(np.where(m, c1, c0).mean()),
                           "tokens_f1": float(np.asarray(raw["tok1"][s_])[:nt].mean())}
        out[int(raw.get("k", cfg["prompt"]["k_neighbors"]))] = {"acc_f0": float(c0.mean()), **row}
    return {f"k={k}": out[k] for k in sorted(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ksens", action="store_true", help="only the k-sensitivity table")
    a = ap.parse_args()
    cfg = load_config(a.config)
    llm_dir = ROOT / "results" / "llm" / slug(a.llm)
    out_dir = ROOT / "results" / "analysis" / slug(a.llm)
    if a.ksens:
        save_json({"dataset": a.dataset, "llm": a.llm, "k_sensitivity": k_sensitivity(llm_dir, a.dataset, cfg)},
                  out_dir / f"ksens_{a.dataset}.json")
        return
    raw = pickle.load(open(llm_dir / a.dataset / "raw.pkl", "rb"))
    g = pickle.load(open(PROC / f"{a.dataset}.pkl", "rb"))
    save_json(analyze(raw, cfg, g), out_dir / f"{a.dataset}.json")


if __name__ == "__main__":
    main()
