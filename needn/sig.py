"""Node-level Structural Information Gain (SIG) estimators and Complementary Neighbor Selection (CNS).

SIG(v) = I(Y_v ; X_N(v) | X_v) = H(Y_v | X_v) - H(Y_v | X_v, X_N(v))
Estimators (all from the text-only LLM distribution p0 of v and of its neighbours; no extra LLM calls):
  A(v)   : textual ambiguity  H(p0_v) / log C                                (upper-bounds SIG)
  D(v)   : neighbourhood discriminativeness  mean_u  I(m_u) * m_u(TopK(p0_v))
  NB(v)  : H(p0_v) - H(q_v),  q_v ∝ p0_v * prod_u m_u^tau   (naive-Bayes posterior with neighbour messages)
  with messages m_u = Pi @ p0_u, Pi[a,b] = P(Y_v = a | Y_u = b)  (label-compatibility matrix; identity = homophily
  assumption, estimated Pi handles *regular* heterophily)
"""
import numpy as np

from .utils import entropy


def estimate_compat(edges, dist, labels=None, labeled=None, smooth=0.1):
    """edges [2,E] among nodes whose distribution is known; dist: dict node -> p0 (np [C]).
    labels/labeled: optionally replace p0 by one-hot ground truth on labeled (training) nodes."""
    C = len(next(iter(dist.values())))
    J = np.full((C, C), smooth)
    for v, u in edges.T:
        if v in dist and u in dist:
            pv = np.eye(C)[labels[v]] if labeled is not None and labeled[v] else dist[v]
            pu = np.eye(C)[labels[u]] if labeled is not None and labeled[u] else dist[u]
            J += np.outer(pv, pu)
    J = (J + J.T) / 2
    return J / J.sum(0, keepdims=True)  # column b = P(Y_v = . | Y_u = b)


def message(p_u, Pi, hops=1):
    m = p_u
    for _ in range(hops):
        m = Pi @ m
    return m / m.sum()


def nb_posterior(p_v, msgs, tau):
    logq = np.log(np.clip(p_v, 1e-12, 1))
    for m in msgs:
        logq = logq + tau * np.log(np.clip(m, 1e-12, 1))
    q = np.exp(logq - logq.max())
    return q / q.sum()


def sig_features(p_v, nbr_msgs, tau, topk=3):
    C = len(p_v)
    A = float(entropy(p_v) / np.log(C))
    if len(nbr_msgs) == 0:
        return {"A": A, "D": 0.0, "NB": 0.0, "deg": 0, "maxp": float(p_v.max())}
    top = np.argsort(-p_v)[:topk]
    D = float(np.mean([(1 - entropy(m) / np.log(C)) * m[top].sum() for m in nbr_msgs]))
    q = nb_posterior(p_v, nbr_msgs, tau)
    return {"A": A, "D": D, "NB": float(entropy(p_v) - entropy(q)), "deg": len(nbr_msgs), "maxp": float(p_v.max()),
            "nb_pred": int(q.argmax())}


def select_cns(p_v, cand, msgs, emb_v, emb_c, k, tau, lam_red=0.5):
    """greedy complementary neighbour selection: maximise the expected entropy drop of the NB posterior,
    penalising redundancy (cosine to already selected neighbours). cand / msgs / emb_c aligned lists."""
    chosen = []
    q_ent = entropy(p_v)
    cur = []
    for _ in range(min(k, len(cand))):
        best, best_s = None, -1e9
        for i in range(len(cand)):
            if i in chosen:
                continue
            q = nb_posterior(p_v, cur + [msgs[i]], tau)
            gain = q_ent - entropy(q)
            red = max([float(emb_c[i] @ emb_c[j]) for j in chosen], default=0.0)
            s = gain - lam_red * max(0.0, red - 0.5) + 1e-3 * float(emb_c[i] @ emb_v)
            if s > best_s:
                best, best_s = i, s
        chosen.append(best)
        cur.append(msgs[best])
        q_ent = entropy(nb_posterior(p_v, cur, tau))
    return [cand[i] for i in chosen]


def select_similar(cand, emb_v, emb_c, k):
    if len(cand) == 0:
        return []
    order = np.argsort(-(emb_c @ emb_v))[:k]
    return [cand[i] for i in order]


def select_random(g, v, cand, k, rng):
    one = list(g["adj"][v])
    pool = one if len(one) >= k else one + [c for c in cand if c not in set(one)]
    if len(pool) <= k:
        return list(pool)
    return list(rng.choice(pool, k, replace=False))
