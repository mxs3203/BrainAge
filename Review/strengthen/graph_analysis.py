"""One 14-node graph per subject, graph measures against age.

Two sources (raw log-covariance, the model's G_log) x two notions of an edge:
marginal correlation, and partial correlation off the precision matrix.
Marginal is dominated by volume conduction and the shared reference so almost
every pair looks conected; the partial one is what you can actually call an
inferred connection. Both are reported so the model and the raw signal get
judged the same way.

Measures run on |w| -- the weights are signed and the usual graph measures
aren't defined for negative ones. Edge-level tests keep the sign.

USAGE: python graph_analysis.py [--cohort ours|all] [--nperm 10000]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REV)
RES = os.path.join(REV, "results")

from pasa_age_groups import IX, perm_p, spearman  # noqa: E402
from pasa_index import _pearson, holm  # noqa: E402

CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
      "P8", "T8", "FC6", "F4", "F8", "AF4"]
ANT, POS = IX["ANT"], IX["POS"]
OFF = ~np.eye(14, dtype=bool)
IU = np.triu_indices(14, 1)


def to_corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def to_partial(C):
    # partial correlation off the precision matrix. pinv not inv: a few
    # subjects come out borderline singular
    T = np.linalg.pinv(C + 1e-8 * np.trace(C) / 14 * np.eye(14))
    d = np.sqrt(np.clip(np.diag(T), 1e-12, None))
    P = -T / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P


def metrics(W):
    """Graph measures, weights |w|."""
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    A = np.clip(A, 0, None)
    g = nx.from_numpy_array(A)
    out = {}
    out["strength"] = A.sum() / 2.0
    out["strength_ANT"] = A[ANT].sum() / len(ANT)
    out["strength_POS"] = A[POS].sum() / len(POS)
    out["strength_ANT_minus_POS"] = out["strength_ANT"] - out["strength_POS"]
    out["clustering"] = float(np.mean(list(
        nx.clustering(g, weight="weight").values())))
    # distance = 1/weight so strong edges are short (recpirocal, not negated)
    inv = np.zeros_like(A)
    nz = A > 0
    inv[nz] = 1.0 / A[nz]
    gi = nx.from_numpy_array(inv)
    sp = dict(nx.all_pairs_dijkstra_path_length(gi, weight="weight"))
    D = np.array([[sp[i].get(j, np.inf) for j in range(14)] for i in range(14)])
    out["path_length"] = float(D[OFF].mean())
    out["efficiency"] = float((1.0 / D[OFF]).mean())
    try:
        comm = nx.community.greedy_modularity_communities(g, weight="weight")
        out["modularity"] = float(nx.community.modularity(g, comm,
                                                          weight="weight"))
        out["n_modules"] = float(len(comm))
    except Exception:
        out["modularity"] = np.nan; out["n_modules"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="ours", choices=["ours", "all"])
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()

    c = np.load(os.path.join(RES, "pasa_cohorts_features.npz"), allow_pickle=True)
    m = (np.ones(len(c["age"]), bool) if a.cohort == "all"
         else c["dataset"].astype(str) == "ours")
    age = c["age"][m]
    subs = c["subject"][m].astype(str)
    print(f"cohort {a.cohort}: {len(age)} subjects")

    SRC = {"raw log-covariance": c["logcov"][m], "model G_log": c["glog"][m]}
    rows, edge_rows = [], []
    for sname, M in SRC.items():
        Cs = np.stack([expm(G) for G in M])
        for kind, fn in (("marginal", to_corr), ("conditional", to_partial)):
            Ws = np.stack([fn(C) for C in Cs])
            feats = pd.DataFrame([metrics(W) for W in Ws])
            for col in feats.columns:
                v = feats[col].to_numpy()
                if not np.isfinite(v).all() or v.std() < 1e-12:
                    continue
                rows.append(dict(source=sname, graph=kind, metric=col,
                                 n=len(age), r=_pearson(v, age),
                                 rho=spearman(v, age),
                                 p_perm=perm_p(v, age,
                                               np.random.default_rng(3),
                                               a.nperm)))
            # edge level, signs kept here
            E = np.stack([W[IU] for W in Ws])
            pe = np.array([perm_p(E[:, k], age, np.random.default_rng(3), 2000)
                           for k in range(E.shape[1])])
            re = np.array([_pearson(E[:, k], age) for k in range(E.shape[1])])
            order = np.argsort(pe)
            q = pe * len(pe) / (np.arange(len(pe)) + 1)[np.argsort(order)]
            for k in range(len(pe)):
                edge_rows.append(dict(source=sname, graph=kind,
                                      edge=f"{CH[IU[0][k]]}-{CH[IU[1][k]]}",
                                      r=round(re[k], 4),
                                      p_perm=round(pe[k], 4),
                                      p_fdr=round(min(q[k], 1.0), 4)))
            print(f"  {sname:<20} {kind:<12} edges with p<0.05: "
                  f"{int((pe < 0.05).sum())} of {len(pe)} "
                  f"({0.05*len(pe):.1f} expected); survive FDR: "
                  f"{int((q < 0.05).sum())}")

    df = pd.DataFrame(rows)
    df["p_holm"] = np.concatenate([
        holm(g.p_perm.to_numpy()) for _, g in df.groupby(["source", "graph"],
                                                         sort=False)])
    df.round(4).to_csv(os.path.join(RES,
                       f"Table_graph_metrics_{a.cohort}.csv"), index=False)
    pd.DataFrame(edge_rows).to_csv(os.path.join(RES,
                       f"Table_graph_edges_{a.cohort}.csv"), index=False)

    print(f"\ngraph metrics against age, {a.cohort} (n={len(age)})\n")
    for (s, k), g in df.groupby(["source", "graph"], sort=False):
        print(f"{s}  |  {k}")
        for _, r in g.sort_values("p_perm").iterrows():
            star = "*" if r.p_perm < 0.05 else " "
            print(f"   {star} {r.metric:<24} r={r.r:+.3f}  rho={r.rho:+.3f}  "
                  f"p={r.p_perm:.4f}  (Holm {r.p_holm:.3f})")
        print()
    print("wrote", os.path.join(RES, f"Table_graph_metrics_{a.cohort}.csv"))


if __name__ == "__main__":
    main()
