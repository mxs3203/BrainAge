"""Anterior/posterior index per cohort, three ways of reading it.

Builds the per-subject feature cache the graph analysis and Figures 6 and S12
all read, then reports the index itself: raw log-covariance diagonal, the
model's G_log diagonal, and the FFT band power.

The cache is the point -- pushing 799 participants through the model is the
expensive bit, everything downstream reads the npz.

USAGE: python pasa_cohorts.py              # build cache if absent, then report
       python pasa_cohorts.py --rebuild    # force the feature pass
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REV)
RES = os.path.join(REV, "results")

from pasa_age_groups import IX, perm_p, spearman, paper_bands, band_power  # noqa: E402
from pasa_index import raw_features, _pearson, holm  # noqa: E402
from stimulus_groups import group as stim_group  # noqa: E402

ANT, POS = IX["ANT"], IX["POS"]
CACHE = os.path.join(RES, "pasa_cohorts_features.npz")

# stride per dataset so everyone contributes a few hundred windows whatever
# the recording lenght
DATASETS = [("ours", 128), ("ds003775", 512), ("ds005385", 1024)]
if os.environ.get("PASA_ONLY"):
    DATASETS = [d for d in DATASETS if d[0] in os.environ["PASA_ONLY"].split(",")]
CHUNK = 40


def ec_stimuli(name):
    recs = pd.read_parquet(os.path.join(REV, "harmonized", name,
                                        "recordings.parquet"))
    return sorted({s for s in recs.stimulus.unique()
                   if stim_group(name, s) == "rest_ec"})


def build_cache(ckpt):
    import torch
    from harmonized_io import build_windows
    from channel_interaction import subject_glog

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(ckpt, map_location=device, weights_only=False)
    model.eval()
    print(f"loaded {ckpt} on {device}")
    BANDS, BS_BANDS, _ = paper_bands()

    sid, dset, age_l, Lc, Gl, Bp = [], [], [], [], [], []
    for name, stride in DATASETS:
        stim = ec_stimuli(name)
        recs = pd.read_parquet(os.path.join(REV, "harmonized", name,
                                            "recordings.parquet"))
        subs = sorted(recs.subject.astype(str).unique())
        print(f"\n=== {name}: {len(subs)} subjects, stride {stride}, "
              f"stimuli {stim}")
        for i in range(0, len(subs), CHUNK):
            part = subs[i:i + CHUNK]
            w = build_windows(name, recipe="prefiltered_uv", win=512,
                              stride=stride, subjects=part, stimuli=stim,
                              verbose=False)
            if not w or len(w["X"]) == 0:
                print(f"  [{i + len(part)}/{len(subs)}] no windows"); continue
            X = w["X"].astype(np.float32)
            sj = np.asarray(w["subject"]).astype(str)
            ag = np.asarray(w["age"], float)

            uniq, L, _ = raw_features(X, sj)
            Pw = band_power(X, sj, uniq, BANDS)
            bs = np.mean([Pw["abs_" + b] for b in BS_BANDS
                          if "abs_" + b in Pw], axis=0)
            per = subject_glog(model, (X, ag, sj), device)

            for k, s in enumerate(uniq):
                if s not in per:
                    continue
                sid.append(s); dset.append(name)
                age_l.append(float(ag[sj == s][0]))
                Lc.append(L[s]); Gl.append(per[s][0]); Bp.append(bs[k])
            print(f"  [{min(i + CHUNK, len(subs))}/{len(subs)}] "
                  f"{len(X):,} windows -> {len(uniq)} subjects")
            del X, w

    np.savez_compressed(CACHE, subject=np.array(sid), dataset=np.array(dset),
                        age=np.array(age_l), logcov=np.stack(Lc),
                        glog=np.stack(Gl), bandpower=np.stack(Bp))
    print(f"\nwrote {CACHE}: {len(sid)} subjects")


def index(V, w):
    """Anterior minus posterior, weighted means, log space.

    Weights are normalised inside each group, so they change the weighting
    within anterior and within posterior but not the contrast between them.
    """
    wa = w[ANT] / w[ANT].sum()
    wp = w[POS] / w[POS].sum()
    return np.asarray([v[ANT] @ wa - v[POS] @ wp for v in V], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(RES,
                    "model_age_phase4_t25_ep10.pt"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()

    if a.rebuild or not os.path.exists(CACHE):
        build_cache(a.ckpt)
    z = np.load(CACHE, allow_pickle=True)
    dset, age = z["dataset"].astype(str), z["age"]
    Dg = np.stack([np.diag(G) for G in z["glog"]])
    Dc = np.stack([np.diag(G) for G in z["logcov"]])
    Df = z["bandpower"]

    w_flat = np.ones(14)
    ARMS = [("raw log-covariance", Dc, w_flat),
            ("model G_log", Dg, w_flat),
            ("FFT band power", Df, w_flat)]
    COHORTS = [("all three corpora", np.ones(len(age), bool)),
               ("in-house (ours)", dset == "ours"),
               ("ds003775", dset == "ds003775"),
               ("ds005385", dset == "ds005385")]

    COHORTS = [(c, m) for c, m in COHORTS if m.sum() >= 10]
    rows = []
    for cname, m in COHORTS:
        for aname, V, w in ARMS:
            v = index(V[m], w)
            rows.append(dict(cohort=cname, arm=aname, n=int(m.sum()),
                             r=_pearson(v, age[m]), rho=spearman(v, age[m]),
                             p_perm=perm_p(v, age[m],
                                           np.random.default_rng(5), a.nperm)))

    def centred(v):
        """z-score inside each dataset before pooling.

        Three corpora on different amplitude scales can manufacture (or mask)
        a correlation through the between-dataset contrast alone. This leaves
        only the within-dataset relationshp.
        """
        o = np.empty_like(v)
        for d in np.unique(dset):
            k = dset == d
            o[k] = (v[k] - v[k].mean()) / (v[k].std(ddof=1) + 1e-12)
        return o

    if len(np.unique(dset)) > 1:
        for aname, V, w in ARMS:
            v = centred(index(V, w))
            rows.append(dict(cohort="all three, dataset-centred", arm=aname,
                             n=len(age), r=_pearson(v, age),
                             rho=spearman(v, age),
                             p_perm=perm_p(v, age, np.random.default_rng(5),
                                           a.nperm)))
        COHORTS.append(("all three, dataset-centred",
                        np.ones(len(age), bool)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_perm.to_numpy())
    df.round(4).to_csv(os.path.join(RES, "Table_pasa_cohorts.csv"), index=False)

    print(f"\nANT/POS against age, {len(age)} eyes-closed participants\n")
    for cname, m in COHORTS:
        print(f"{cname}  (n = {int(m.sum())}, age "
              f"{age[m].min():.0f}-{age[m].max():.0f})")
        for _, r in df[df.cohort == cname].iterrows():
            star = "*" if r.p_perm < 0.05 else " "
            print(f"   {star} {r.arm:<26} r={r.r:+.3f}  rho={r.rho:+.3f}  "
                  f"p={r.p_perm:.4f}  (Holm {r.p_holm:.3f})")
        print()
    print("wrote", os.path.join(RES, "Table_pasa_cohorts.csv"))


if __name__ == "__main__":
    main()
