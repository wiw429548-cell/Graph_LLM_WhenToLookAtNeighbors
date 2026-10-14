"""Unified text-attributed graph loading + a uniform low-label split for every dataset.

Graph fields: texts (list[str]), y (int64[N]), label_names (list[str]), adj (list of int64 arrays, undirected, no
self loops), edge_index (int64[2, E] symmetric), split masks: train / val / test, test_eval (LLM evaluation subset),
val_gain (subset of val used to train the learned router).
"""
import re
import pickle

import numpy as np

from .utils import DATA

RAW = DATA / "raw"
PROC = DATA / "processed"


def _load_raw(meta):
    if meta["src"] == "llmnodebed":
        import torch
        d = torch.load(RAW / "llmnodebed" / meta["file"], weights_only=False, map_location="cpu")
        texts = [str(t) if t is not None else "" for t in d.raw_texts]
        y = d.y.numpy().astype(np.int64).reshape(-1)
        names = list(d.label_name) if getattr(d, "label_name", None) is not None else [f"class {i}" for i in range(y.max() + 1)]
        ei = d.edge_index.numpy().astype(np.int64)
        return texts, y, names, ei
    d = np.load(RAW / "hetgb" / f"{meta['dir']}.npz", allow_pickle=True)
    return [str(t) for t in d["node_texts"]], d["node_labels"].astype(np.int64), [str(t) for t in d["label_texts"]], \
        d["edges"].astype(np.int64)


ARXIV_CS = {"ai": "Artificial Intelligence", "ar": "Hardware Architecture", "cc": "Computational Complexity",
            "ce": "Computational Engineering, Finance, and Science", "cg": "Computational Geometry",
            "cl": "Computation and Language", "cr": "Cryptography and Security",
            "cv": "Computer Vision and Pattern Recognition", "cy": "Computers and Society", "db": "Databases",
            "dc": "Distributed, Parallel, and Cluster Computing", "dl": "Digital Libraries",
            "dm": "Discrete Mathematics", "ds": "Data Structures and Algorithms", "et": "Emerging Technologies",
            "fl": "Formal Languages and Automata Theory", "gl": "General Literature", "gr": "Graphics",
            "gt": "Computer Science and Game Theory", "hc": "Human-Computer Interaction", "ir": "Information Retrieval",
            "it": "Information Theory", "lg": "Machine Learning", "lo": "Logic in Computer Science",
            "ma": "Multiagent Systems", "mm": "Multimedia", "ms": "Mathematical Software", "na": "Numerical Analysis",
            "ne": "Neural and Evolutionary Computing", "ni": "Networking and Internet Architecture",
            "oh": "Other Computer Science", "os": "Operating Systems", "pf": "Performance",
            "pl": "Programming Languages", "ro": "Robotics", "sc": "Symbolic Computation", "sd": "Sound",
            "se": "Software Engineering", "si": "Social and Information Networks", "sy": "Systems and Control"}


def display_label(n):
    """human-readable class names for the prompts ('arxiv cs lg' -> 'cs.LG (Machine Learning)', 'Rule_Learning' ->
    'Rule Learning'); the label ids are unchanged"""
    m = re.fullmatch(r"arxiv cs ([a-z]{2})", str(n).strip())
    if m and m.group(1) in ARXIV_CS:
        return f"cs.{m.group(1).upper()} ({ARXIV_CS[m.group(1)]})"
    return str(n).replace("_", " ")


def build(name, cfg):
    out = PROC / f"{name}.pkl"
    if out.exists():
        with open(out, "rb") as f:
            g = pickle.load(f)
        # prompt wording (node / nodes / edge / domain / task) always comes from the current config, not from the
        # config at preprocessing time (the cached graph froze it before)
        g["meta"] = {**g.get("meta", {}), **cfg["datasets"][name]}
        g["label_names"] = [display_label(n) for n in g["label_names"]]
        return g
    meta = cfg["datasets"][name]
    texts, y, names, ei = _load_raw(meta)
    N = len(texts)
    ei = ei[:, ei[0] != ei[1]]
    ei = np.unique(np.concatenate([ei, ei[::-1]], 1), axis=1)
    order = np.argsort(ei[0], kind="stable")
    src, dst = ei[0][order], ei[1][order]
    ptr = np.searchsorted(src, np.arange(N + 1))
    adj = [dst[ptr[i]:ptr[i + 1]] for i in range(N)]
    rng = np.random.RandomState(cfg["seed"])
    sc = cfg["split"]
    perm = rng.permutation(N)
    train = np.zeros(N, bool)
    per = {c: 0 for c in range(len(names))}
    for i in perm:
        if per.get(y[i], 0) < sc["train_per_class"]:
            train[i] = True
            per[y[i]] = per.get(y[i], 0) + 1
    rest = perm[~train[perm]]
    n_val = min(sc["val_max"], int(sc["val_frac"] * N))
    val = np.zeros(N, bool)
    val[rest[:n_val]] = True
    test = np.zeros(N, bool)
    test[rest[n_val:]] = True
    test_idx = rest[n_val:]
    test_eval = np.sort(test_idx[:sc["n_test_eval"]])
    val_gain = np.sort(rest[:n_val][:sc["n_val_gain"]])
    g = dict(name=name, texts=texts, y=y, label_names=names, adj=adj, edge_index=np.stack([src, dst]), train=train,
             val=val, test=test, test_eval=test_eval, val_gain=val_gain, meta=meta,
             homophily=float((y[src] == y[dst]).mean()) if len(src) else 0.0)
    PROC.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(g, f, protocol=4)
    return dict(g, label_names=[display_label(n) for n in names])


def candidates(g, v, max_c, rng):
    """1-hop neighbours first, then 2-hop, capped at max_c (random subset if larger)"""
    one = g["adj"][v]
    if len(one) >= max_c:
        return rng.choice(one, max_c, replace=False)
    two = np.unique(np.concatenate([g["adj"][u] for u in one])) if len(one) else np.zeros(0, np.int64)
    two = np.setdiff1d(two, np.append(one, v))
    if len(two) > max_c - len(one):
        two = rng.choice(two, max_c - len(one), replace=False)
    return np.concatenate([one, two]).astype(np.int64)
