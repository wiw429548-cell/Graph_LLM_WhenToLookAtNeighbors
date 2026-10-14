"""Classic baselines on frozen text embeddings (same split): MLP, GCN, GAT, GraphSAGE, GPR-GNN, H2GCN(-lite).
Also used by distill.py to train a GNN on LLM pseudo-labels.
--probs: only the GCN class probabilities (mean over 5 seeds) -> results/gnn/probs_<dataset>.npy, used by analyze.py
for the LLM+GNN ensemble / cascade baselines (no LLM call)."""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PROC, build
from .utils import ROOT, load_config, save_json


def norm_adj(ei, N, dev, self_loops=True):
    ei = torch.tensor(ei, device=dev)
    if self_loops:
        loop = torch.arange(N, device=dev)
        ei = torch.cat([ei, torch.stack([loop, loop])], 1)
    deg = torch.zeros(N, device=dev).index_add_(0, ei[0], torch.ones(ei.shape[1], device=dev))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei, w, (N, N)).coalesce()


class MLP(nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), p

    def forward(self, x, A=None):
        return self.l2(F.dropout(F.relu(self.l1(F.dropout(x, self.p, self.training))), self.p, self.training))


class GCN(nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), p

    def forward(self, x, A):
        x = F.relu(torch.sparse.mm(A, self.l1(F.dropout(x, self.p, self.training))))
        return torch.sparse.mm(A, self.l2(F.dropout(x, self.p, self.training)))


class SAGE(nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__()
        self.s1, self.n1, self.s2, self.n2, self.p = nn.Linear(i, h), nn.Linear(i, h), nn.Linear(h, o), nn.Linear(h, o), p

    def forward(self, x, A):  # A: row-normalised mean aggregator without self loops
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.s1(x) + self.n1(torch.sparse.mm(A, x)))
        x = F.dropout(x, self.p, self.training)
        return self.s2(x) + self.n2(torch.sparse.mm(A, x))


class GAT(nn.Module):
    def __init__(self, i, h, o, p, heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.c1 = GATConv(i, h // heads, heads=heads, dropout=p)
        self.c2 = GATConv(h, o, heads=1, dropout=p)
        self.p = p

    def forward(self, x, ei):
        x = F.elu(self.c1(F.dropout(x, self.p, self.training), ei))
        return self.c2(F.dropout(x, self.p, self.training), ei)


class GPRGNN(nn.Module):
    def __init__(self, i, h, o, p, K=10, alpha=0.1):
        super().__init__()
        self.mlp = MLP(i, h, o, p)
        g = alpha * (1 - alpha) ** np.arange(K + 1)
        g[-1] = (1 - alpha) ** K
        self.gamma = nn.Parameter(torch.tensor(g, dtype=torch.float32))
        self.K = K

    def forward(self, x, A):
        z = self.mlp(x)
        out = self.gamma[0] * z
        for k in range(1, self.K + 1):
            z = torch.sparse.mm(A, z)
            out = out + self.gamma[k] * z
        return out


class H2GCN(nn.Module):
    """ego / neighbour separation + 1-hop and 2-hop aggregation (without self loops) + combination of all rounds."""

    def __init__(self, i, h, o, p):
        super().__init__()
        self.emb, self.out, self.p = nn.Linear(i, h), nn.Linear(h * 7, o), p

    def forward(self, x, As):
        A1, A2 = As
        r0 = F.relu(self.emb(F.dropout(x, self.p, self.training)))
        r1 = torch.cat([torch.sparse.mm(A1, r0), torch.sparse.mm(A2, r0)], 1)
        r2 = torch.cat([torch.sparse.mm(A1, r1), torch.sparse.mm(A2, r1)], 1)
        return self.out(F.dropout(torch.cat([r0, r1, r2], 1), self.p, self.training))


def two_hop(ei, N):
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)).tocsr()
    A.data[:] = 1
    A2 = (A @ A).tocoo()
    m = (A2.row != A2.col)
    A2 = sp.coo_matrix((np.ones(m.sum()), (A2.row[m], A2.col[m])), shape=(N, N)).tocsr()
    A2 = A2 - A2.multiply(A)  # strictly 2-hop
    A2.eliminate_zeros()
    A2 = A2.tocoo()
    return np.stack([A2.row, A2.col]).astype(np.int64)


def row_norm(ei, N, dev):
    ei_t = torch.tensor(ei, device=dev)
    deg = torch.zeros(N, device=dev).index_add_(0, ei_t[0], torch.ones(ei_t.shape[1], device=dev)).clamp(min=1)
    return torch.sparse_coo_tensor(ei_t, 1.0 / deg[ei_t[0]], (N, N)).coalesce()


def train_model(name, x, ei, y, train_idx, val_idx, C, seed, dev, soft=None, epochs=300, patience=50, proba=False):
    torch.manual_seed(seed)
    N = x.shape[0]
    h, p = 128, 0.5
    if name == "mlp":
        model, A = MLP(x.shape[1], h, C, p), None
    elif name == "gcn":
        model, A = GCN(x.shape[1], h, C, p), norm_adj(ei, N, dev)
    elif name == "sage":
        model, A = SAGE(x.shape[1], h, C, p), row_norm(ei, N, dev)
    elif name == "gat":
        model, A = GAT(x.shape[1], h, C, p), torch.tensor(ei, device=dev)
    elif name == "gprgnn":
        model, A = GPRGNN(x.shape[1], h, C, p), norm_adj(ei, N, dev)
    elif name == "h2gcn":
        model, A = H2GCN(x.shape[1], 64, C, p), (row_norm(ei, N, dev), row_norm(two_hop(ei, N), N, dev))
    else:
        raise ValueError(name)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yt = torch.tensor(y, device=dev)
    tr = torch.tensor(train_idx, device=dev)
    va = torch.tensor(val_idx, device=dev)
    target = torch.tensor(soft, device=dev, dtype=torch.float32) if soft is not None else None
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        out = model(x, A)
        loss = (-(target[tr] * out[tr].log_softmax(-1)).sum(-1).mean() if target is not None
                else F.cross_entropy(out[tr], yt[tr]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, A)
            if target is not None:  # no true labels in the zero-shot distillation setting: agree with pseudo-labels
                acc = (out[va].argmax(1) == target[va].argmax(1)).float().mean().item()
            else:
                acc = (out[va].argmax(1) == yt[va]).float().mean().item()
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, A)
        return out.softmax(1).cpu().numpy() if proba else out.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--probs", action="store_true", help="only save GCN class probabilities (mean of 5 seeds)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = build(a.dataset, cfg)
    emb = np.load(PROC / f"{a.dataset}_emb.npy")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(emb, device=dev)
    C = len(g["label_names"])
    tr, va = np.where(g["train"])[0], np.where(g["val"])[0]
    if a.probs:
        P = np.mean([train_model("gcn", x, g["edge_index"], g["y"], tr, va, C, seed, dev, proba=True)
                     for seed in range(5)], 0).astype(np.float32)
        print(a.dataset, "gcn probs: test_eval acc", float((P[g["test_eval"]].argmax(1) == g["y"][g["test_eval"]]).mean()),
              flush=True)
        out = ROOT / "results" / "gnn" / f"probs_{a.dataset}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, P)
        tmp.replace(out)
        return
    res = {}
    for name in ["mlp", "gcn", "sage", "gat", "gprgnn", "h2gcn"]:
        if name == "h2gcn" and (len(g["y"]) > 60000 or g["edge_index"].shape[1] > 300000):
            res[name] = {"skipped": "2-hop adjacency too large for full-batch H2GCN"}
            continue
        accs, accs_eval = [], []
        for seed in range(5):
            pred = train_model(name, x, g["edge_index"], g["y"], tr, va, C, seed, dev)
            accs.append(float((pred[g["test"]] == g["y"][g["test"]]).mean()))
            accs_eval.append(float((pred[g["test_eval"]] == g["y"][g["test_eval"]]).mean()))
        res[name] = {"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
                     "test_eval_acc_mean": float(np.mean(accs_eval)), "test_eval_acc_std": float(np.std(accs_eval))}
        print(a.dataset, name, res[name], flush=True)
    # label propagation (structure only, no text) and LP over text-embedding predictions (C&S-style smoothing)
    from torch_geometric.nn import LabelPropagation
    ei_t = torch.tensor(g["edge_index"], device=dev)
    yt = torch.tensor(g["y"], device=dev)
    trm = torch.zeros(len(g["y"]), dtype=torch.bool, device=dev)
    trm[torch.tensor(tr, device=dev)] = True
    lp = LabelPropagation(num_layers=50, alpha=0.9)
    pred = lp(yt, ei_t, mask=trm).argmax(1).cpu().numpy()
    res["label_propagation"] = {"test_acc_mean": float((pred[g["test"]] == g["y"][g["test"]]).mean()), "test_acc_std": 0.0,
                                "test_eval_acc_mean": float((pred[g["test_eval"]] == g["y"][g["test_eval"]]).mean()),
                                "test_eval_acc_std": 0.0}
    save_json(res, ROOT / "results" / "gnn" / f"{a.dataset}.json")


if __name__ == "__main__":
    main()
