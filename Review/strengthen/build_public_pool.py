"""Pool all PUBLIC data into one window cache for teh per-channel encoder search."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harmonized_io import load_dataset, DATASETS  # noqa: E402

WIN, STRIDE = 512, 256
PER_SUBJECT = int(os.environ.get("POOL_PER_SUBJECT", 250))   # windows kept per subject
RECIPE = os.environ.get("POOL_RECIPE", "filtered")
# POOL_EXCLUDE: comma-separated datasets to leave out entirely. ds003478 (age
# 18-24, SD 1.2) has a NEGATIVE within-dataset age correlation (-0.094 mean over
EXCLUDE = {d for d in os.environ.get("POOL_EXCLUDE", "").split(",") if d}
# POOL_ONLY restricts to specific datasets; POOL_GROUP to a condition group.
# Default here: the two cohorts whose eyes-closed rest transfers bewteen them
ONLY = [d for d in os.environ.get("POOL_ONLY", "").split(",") if d]
GROUP = os.environ.get("POOL_GROUP", "")


def _in_group(ds, stim):
    """POOL_GROUP accepts a single group name ("rest_ec") or a SETS key
    ("no_inter", "rest_any"), which is a set of group names. Previously only an
    exact group name worked, so a set key silently matched nothing and the build
    died with "need at least one array to concatenate"."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stimulus_groups import group as _g, SETS
    g = _g(ds, stim)
    if GROUP in SETS:
        allowed = SETS[GROUP]
        return True if allowed is None else g in allowed
    return g == GROUP
# POOL_MAX_SUBJECTS: cap subjects per dataset so one cohort cannot dominate.
# ds005385 is 608/995 = 61% of the corpus AND the most spectrally dissimilar to
MAX_SUBJ = int(os.environ.get("POOL_MAX_SUBJECTS", 0)) or None
# POOL_MATCH_AGE=<dataset>: subsample each public dataset so its age histogram
# matches that dataset's. ds005385 is 608 subjets at mean 44.1 +- 14.5 while the
MATCH_AGE = os.environ.get("POOL_MATCH_AGE", "")
HERE = os.path.dirname(os.path.abspath(__file__))
# POOL_TAG lets a second pool live beside the first instead of overwriting it, so
# the 2-dataset pool phase 1 used stays intact while a 5-dataset multi-domain pool
TAG = os.environ.get("POOL_TAG", "")
X_PATH = os.path.join(HERE, f"public_pool{TAG}_X.npy")
META_PATH = os.path.join(HERE, f"public_pool{TAG}_meta.pkl")


def main():
    from harmonized_io import build_windows
    rng = np.random.default_rng(0)
    public = [d for d in DATASETS if d != "ours" and d not in EXCLUDE]
    if ONLY:
        public = [d for d in public if d in ONLY]
        print(f"datasets: {public}", flush=True)
    if EXCLUDE:
        print(f"excluding: {sorted(EXCLUDE)}", flush=True)
    if MAX_SUBJ:
        print(f"capping each dataset at {MAX_SUBJ} subjects", flush=True)
    tgt_age = None
    if MATCH_AGE:
        from harmonized_io import load_dataset as _ld
        _, _tr = _ld(MATCH_AGE)
        if GROUP:
            import sys as _s
            _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from stimulus_groups import group as _g
            _tr = _tr[[_in_group(MATCH_AGE, x) for x in _tr.stimulus]]
        tgt_age = np.array([_tr[_tr.subject == x].age.iloc[0]
                            for x in sorted(_tr.subject.unique())], float)
        print(f"matching age distribution to {MATCH_AGE}: n={len(tgt_age)} "
              f"mean {tgt_age.mean():.1f} sd {tgt_age.std():.1f}", flush=True)
    Xs, subj, ds_name, ages = [], [], [], []

    for ds in public:
        X, recs = load_dataset(ds)
        subs = sorted(recs.subject.unique())
        if tgt_age is not None:
            # 6 bins over the TARGET range; any source subject outside it is
            # dropped, and within each bin teh source is thinned to the target
            edges = np.quantile(tgt_age, np.linspace(0, 1, 7))
            edges[0], edges[-1] = tgt_age.min(), tgt_age.max()
            frac = np.histogram(tgt_age, bins=edges)[0] / len(tgt_age)
            sa = np.array([recs[recs.subject == x].age.iloc[0] for x in subs], float)
            inr = (sa >= edges[0]) & (sa <= edges[-1])
            b = np.clip(np.digitize(sa, edges[1:-1]), 0, len(frac) - 1)
            avail = np.array([np.sum(inr & (b == i)) for i in range(len(frac))])
            cap = min((avail[i] / frac[i]) for i in range(len(frac)) if frac[i] > 0)
            keep = []
            for i in range(len(frac)):
                want = int(round(frac[i] * cap))
                pool_i = np.flatnonzero(inr & (b == i))
                if want and len(pool_i):
                    keep.append(rng.choice(pool_i, min(want, len(pool_i)), replace=False))
            if keep:
                sel = np.sort(np.concatenate(keep))
                kept_age = sa[sel]
                print(f"  {ds}: {len(subs)} -> {len(sel)} subjects, "
                      f"mean {sa.mean():.1f} -> {kept_age.mean():.1f}, "
                      f"sd {sa.std():.1f} -> {kept_age.std():.1f}", flush=True)
                subs = [subs[i] for i in sel]
        if MAX_SUBJ and len(subs) > MAX_SUBJ:
            # deterministic subsample, stratified by age so capping does not
            # distort teh age distribution of the dataset being capped
            a = np.array([recs[recs.subject == s].age.iloc[0] for s in subs], float)
            order = np.argsort(a)
            keep = order[np.linspace(0, len(order) - 1, MAX_SUBJ).round().astype(int)]
            subs = [subs[i] for i in sorted(set(keep))]
        print(f"{ds}: {len(subs)} subjects", flush=True)
        for i, sid in enumerate(subs, 1):
            stims = None
            if GROUP:
                import sys as _s
                _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from stimulus_groups import group as _g
                stims = [x for x in recs[recs.subject == sid].stimulus.unique()
                         if _in_group(ds, x)]
                if not stims:
                    continue
            w = build_windows(ds, recipe=RECIPE, subjects=[sid], stimuli=stims,
                              verbose=False)
            if w is None or len(w["X"]) == 0:
                continue
            n = len(w["X"])
            take = rng.choice(n, size=min(PER_SUBJECT, n), replace=False)
            take.sort()
            Xs.append(w["X"][take])
            subj.append(np.full(len(take), f"{ds}:{sid}"))
            ds_name.append(np.full(len(take), ds))
            ages.append(np.full(len(take), w["age"][0]))
            if i % 100 == 0:
                print(f"   [{i}/{len(subs)}] pooled "
                      f"{sum(len(a) for a in Xs):,} windows", flush=True)

    X = np.concatenate(Xs); del Xs
    meta = {"subject": np.concatenate(subj), "dataset": np.concatenate(ds_name),
            "age": np.concatenate(ages).astype(float),
            "win": WIN, "stride": STRIDE, "recipe": RECIPE,
            "per_subject": PER_SUBJECT,
            "excluded": sorted(EXCLUDE), "max_subjects": MAX_SUBJ,
            "only": ONLY, "group": GROUP, "match_age": MATCH_AGE}
    np.save(X_PATH, X)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"\npooled {X.shape} ({X.nbytes/1e9:.2f} GB) | "
          f"{len(set(meta['subject']))} subjects | "
          f"age {meta['age'].min():.0f}-{meta['age'].max():.0f}")


def load_pool():
    X = np.load(X_PATH, mmap_mode="r")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return X, meta


if __name__ == "__main__":
    main()
