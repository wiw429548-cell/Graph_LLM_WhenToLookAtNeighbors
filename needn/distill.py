"""RQ6: Web-scale deployment by distillation. The LLM labels an annotation subset (nodes outside the validation set and
the fixed LLM test subset); a GCN is trained ONLY on those LLM pseudo-labels (no ground-truth label is used to train or
to select the GCN) and classifies the whole graph; it is evaluated on the fixed test subset, which is never annotated.
Note: the NB routing score and CNS use the label-compatibility matrix estimated with the 20 training labels per class
(as in the main experiments). Annotation strategies are compared at their LLM prompt-token cost:
f0 only / NeedNbr routing (20% budget, CNS) / graph-augmented for all / NB posterior (no extra calls)."""
import argparse
import pickle

import numpy as np
import torch

from .data import PROC, build
from .gnn import train_model
from .utils import ROOT, load_config, save_json, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw = pickle.load(open(ROOT / "results" / "llm" / slug(a.llm) / a.dataset / "raw.pkl", "rb"))
    g = build(a.dataset, cfg)
    emb = np.load(PROC / f"{a.dataset}_emb.npy")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(emb, device=dev)
    C = raw["C"]
    pos = {int(v): i for i, v in enumerate(raw["targets"])}
    ann = [int(v) for v in raw["annot"]]
    if not ann:
        raise SystemExit(f"{a.dataset}: raw.pkl has no annotation subset (the dataset was not in distill_datasets when "
                         f"run_llm ran) -> delete results/llm/{slug(a.llm)}/{a.dataset}/raw.pkl and rerun")
    ii = np.array([pos[v] for v in ann], dtype=int)
    P0, P1 = raw["P0"][ii], raw["P1"]["cns"][ii]
    nb = np.array([raw["feats"]["est"][i]["NB"] for i in ii])
    nbp = np.array([raw["feats"]["est"][i].get("nb_pred", -1) for i in ii])
    route = np.zeros(len(ann), bool)
    route[np.argsort(-nb)[:int(a.budget * len(ann))]] = True
    t0, t1 = raw["tok0"][ii], raw["tok1"]["cns"][ii]
    strategies = {
        "f0 only": (P0, t0.sum()),
        f"NeedNbr routing ({int(a.budget * 100)}%)": (np.where(route[:, None], P1, P0), t0.sum() + t1[route].sum()),
        "graph-augmented for all": (P1, t0.sum() + t1.sum()),
        "NB posterior (no extra calls)": (np.where((nbp >= 0)[:, None], np.eye(C)[np.maximum(nbp, 0)], P0), t0.sum()),
    }
    rng = np.random.RandomState(cfg["seed"])
    perm = rng.permutation(len(ann))
    n_va = max(1, len(ann) // 5)
    res = {"n_annotated": len(ann)}
    for name, (soft, tok) in strategies.items():
        soft_full = np.zeros((len(emb), C))
        soft_full[ann] = soft
        tr, va = np.array(ann)[perm[n_va:]], np.array(ann)[perm[:n_va]]
        accs = []
        for seed in range(3):
            pred = train_model("gcn", x, g["edge_index"], g["y"], tr, va, C, seed, dev, soft=soft_full)
            ev = g["test_eval"]  # never annotated, never used for training or validation
            accs.append(float((pred[ev] == g["y"][ev]).mean()))
        res[name] = {"pseudo_label_acc": float((soft.argmax(1) == g["y"][ann]).mean()),
                     "gcn_test_acc": float(np.mean(accs)), "gcn_test_std": float(np.std(accs)),
                     "llm_prompt_tokens": float(tok)}
        print(name, res[name], flush=True)
    save_json(res, ROOT / "results" / "distill" / slug(a.llm) / f"{a.dataset}.json")


if __name__ == "__main__":
    main()
