"""Evaluate one trained age model on each in-house eyes-closed STIMULUS.
USAGE: python per_stimulus_eval.py [--config age_phase4_t25] [--split locked]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")
os.environ.setdefault("AGE_GAP_ON", "val")

import optuna
import age_model_search as A

optuna.logging.set_verbosity(optuna.logging.WARNING)
DB = "sqlite:///age_model_optuna.db"
NAVY, RED, GRAY = "#1F4E79", "#C00000", "#808080"
AMBER, PURPLE, GREEN = "#C98B2E", "#6A3D9A", "#2E7D5B"


def get_cfg(name):
    study, n = name.rsplit("_t", 1)
    t = [x for x in optuna.load_study(study_name=study, storage=DB).trials
         if x.number == int(n)][0]
    cfg = dict(t.params)
    cfg.setdefault("train_unit", "bag")
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("pool_mix", "joint")
    return cfg, t


def stats(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    if len(t) < 3 or t.std() == 0:
        return dict(n=len(t), r=np.nan, mae=np.nan, slope=np.nan, chance=np.nan,
                    pct_vs_chance=np.nan)
    ch = float(np.abs(t - np.median(t)).mean())
    mae = float(np.abs(p - t).mean())
    return dict(n=len(t), r=float(np.corrcoef(t, p)[0, 1]), mae=mae,
                slope=float(np.polyfit(t, p - t, 1)[0]), chance=ch,
                pct_vs_chance=100.0 * (1 - mae / ch))



def predict_bag(model, Xi, subj, agew, idx, device, k, seed, min_windows=None):
    """Per-subject prediction over `idx`, using exactly k windows per subject."""
    need = k if min_windows is None else min_windows
    smap = {}
    for i in idx:
        smap.setdefault(subj[i], []).append(i)
    smap = {a_: np.asarray(v) for a_, v in smap.items() if len(v) >= need}
    keys = list(smap)
    if len(keys) < 3:
        return None
    rng = np.random.default_rng(seed)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(keys), 8):
            blk = keys[i:i + 8]
            with A.autocast(device_type="cuda"):
                pr = A.bag_forward_batch(model, Xi, [smap[a_] for a_ in blk],
                                         device, k=k, rng=rng)
            out.append(pr.float().cpu().numpy())
    p = np.concatenate(out).astype(float).ravel()
    t = np.array([agew[smap[a_][0]] for a_ in keys], float)
    return pd.DataFrame({"subject": keys, "t": t, "p": p})


def boot_ci(t, p, n=2000, seed=0):
    """Bootstrap over SUBJECTS (the unit of analysis), percentile CI on r."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        b = rng.integers(0, len(t), len(t))
        if np.std(t[b]) < 1e-9:
            continue
        rs.append(np.corrcoef(t[b], p[b])[0, 1])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--split", default="locked",
                    choices=["locked", "val", "fit"])
    ap.add_argument("--load", action="store_true",
                    help="reuse the cached trained model instead of retraining")
    ap.add_argument("--match-k", type=int, default=24, dest="match_k",
                    help="windows per subject in the matched controls")
    ap.add_argument("--draws", type=int, default=8,
                    help="random window subsamples averaged in the controls")
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = use the epoch the sweep selected")
    # A retrain on CHANGED data must not clobber the run it is being compared
    # against: the checkpoint, the table and the figure all take this suffix,
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint, table and figure paths")
    a = ap.parse_args()

    TAG = ("_" + a.tag) if a.tag else ""
    cfg, trial = get_cfg(a.config)
    sel = int(trial.user_attrs.get("epoch", 12))
    epochs = a.epochs or sel
    print(f"{a.config}: head={cfg['head']} | sweep selected epoch {sel} "
          f"-> training {epochs} epochs so the final model IS that model",
          flush=True)

    POOLS, va, device, X, ages, subjects, datasets = A.prepare()
    # The trained model is cached, so asking a new question about stimulus
    # subsets is an evaluation, not a 30-minute retrain. The cache key includes
    ckpt = f"../results/model_{a.config}_ep{epochs}{TAG}.pt"
    if a.load and os.path.exists(ckpt):
        print(f"  loading cached model {ckpt}", flush=True)
        model = torch.load(ckpt, map_location=device, weights_only=False)
        model.eval()
    else:
        A.EPOCHS = epochs
        A.MODEL_SINK = {}
        m = A.run_fold(cfg, POOLS, va, device, X, ages, subjects, datasets)
        model = A.MODEL_SINK["model"]
        print(f"  retrain reproduces: PUB r {m['r']:+.3f} "
              f"MAE {m['subject_mae']:.2f}", flush=True)
        torch.save(model, ckpt)
        print(f"  cached model -> {ckpt}", flush=True)

    # ---- in-house windows WITH teh stimulus label kept -------------------
    sys.path.insert(0, os.path.dirname(A.HERE))
    from harmonized_io import build_windows, load_dataset
    from stimulus_groups import group as _g
    _, recs = load_dataset("ours")
    ec = [x for x in recs.stimulus.unique() if _g("ours", x) == "rest_ec"]
    w = build_windows("ours", recipe="prefiltered_uv", win=512,
                      stride=A.IH_STRIDE, stimuli=ec, verbose=False)
    Xi = w["X"].astype(np.float32)
    subj = np.asarray(w["subject"]).astype(str)
    stim = np.asarray(w["stimulus"]).astype(str)
    agew = np.asarray(w["age"], float)
    groups = A.inhouse_split(subj, {s: v for s, v in zip(subj, agew)})

    GY = ["gyrus_left_closed_eyes", "gyrus_right_closed_eyes"]
    CONDS = [("eyesclose", ["eyesclose"], "eyes-closed REST"),
             ("gyrus_left_closed_eyes", ["gyrus_left_closed_eyes"], "TASK"),
             ("gyrus_right_closed_eyes", ["gyrus_right_closed_eyes"], "TASK"),
             ("gyrus (L+R pooled)", GY, "TASK, pooled"),
             ("phase_Stimuli_phase_15", ["phase_Stimuli_phase_15"],
              "inter-stimulus phase"),
             ("no gyrus (EC + phase_15)",
              ["eyesclose", "phase_Stimuli_phase_15"],
              "gyrus dropped at EVALUATION only"),
             ("ALL COMBINED", ec, "what the model is normally scored on")]

    def predict(mask):
        """Per-subject prediction over the windows selected by `mask`."""
        idx = np.flatnonzero(mask)
        if not len(idx):
            return None
        smap = {}
        for i in idx:
            smap.setdefault(subj[i], []).append(i)
        smap = {k: np.asarray(v) for k, v in smap.items()}
        keys, p = A.bag_predict(model, Xi, smap, device)
        t = np.array([agew[smap[k][0]] for k in keys], float)
        return pd.DataFrame({"subject": keys, "t": t, "p": np.asarray(p).ravel()})

    rows, frames = [], {}
    for split in ("fit", "val", "locked"):
        keep = np.isin(subj, list(groups[split]))
        for label, stims, kind in CONDS:
            f = predict(keep & np.isin(stim, stims))
            if f is None or len(f) < 3:
                continue
            st = stats(f.t, f.p)
            rows.append(dict(split=split, stimulus=label, kind=kind,
                             windows=int((keep & np.isin(stim, stims)).sum()),
                             **st))
            if split == a.split:
                frames[label] = f
    tab = pd.DataFrame(rows)

    # ---- CONTROL 1: matched windows per subject, matched subject set --------
    # Conditions differ enormously in windows/subject (eyesclose ~90, gyrus ~29,
    K, DRAWS = a.match_k, a.draws
    lockmask = np.isin(subj, list(groups[a.split]))
    MAIN = [c for c in CONDS if c[0] != "phase_Stimuli_phase_15"]
    common = None
    for label, stims, _k in MAIN:
        mk = lockmask & np.isin(stim, stims)
        cnt = pd.Series(subj[mk]).value_counts()
        ok = set(cnt[cnt >= K].index)
        common = ok if common is None else (common & ok)
    print(f"\nCONTROL 1: k={K} windows/subject, {DRAWS} draws, "
          f"{len(common)} subjects common to all conditions", flush=True)
    mrows = []
    for label, stims, kind in MAIN:
        mk = lockmask & np.isin(stim, stims) & np.isin(subj, list(common))
        per = [predict_bag(model, Xi, subj, agew, np.flatnonzero(mk), device,
                           K, 1000 + d) for d in range(DRAWS)]
        per = [f for f in per if f is not None]
        if not per:
            continue
        rs = [stats(f.t, f.p)["r"] for f in per]
        ms = [stats(f.t, f.p)["mae"] for f in per]
        avg = per[0].copy()
        avg["p"] = np.mean([f.sort_values("subject").p.to_numpy()
                            for f in per], axis=0) \
            if all(len(f) == len(per[0]) for f in per) else per[0].p
        st = stats(avg.t, avg.p)
        lo, hi = boot_ci(avg.t, avg.p)
        mrows.append(dict(analysis="matched", stimulus=label, kind=kind,
                          n=st["n"], k=K, draws=len(per),
                          r=float(np.mean(rs)), r_sd=float(np.std(rs)),
                          r_lo=lo, r_hi=hi, mae=float(np.mean(ms)),
                          chance=st["chance"],
                          pct_vs_chance=100 * (1 - np.mean(ms) / st["chance"])))
    mt = pd.DataFrame(mrows)
    print(mt[["stimulus", "n", "r", "r_sd", "r_lo", "r_hi", "mae", "chance",
              "pct_vs_chance"]].round(3).to_string(index=False))

    # ---- CONTROL 2: are the 11 phase_15 subjets just easier? ---------------
    # phase_15 scored best (r +0.594) but on only 11 subjets whose chance MAE
    p15 = lockmask & (stim == "phase_Stimuli_phase_15")
    cnt = pd.Series(subj[p15]).value_counts()
    sub11 = set(cnt[cnt >= K].index)
    print(f"\nCONTROL 2: the same {len(sub11)} phase_15 subjects, every "
          f"condition, k={K}", flush=True)
    crows = []
    for label, stims, kind in CONDS:
        mk = lockmask & np.isin(stim, stims) & np.isin(subj, list(sub11))
        per = [predict_bag(model, Xi, subj, agew, np.flatnonzero(mk), device,
                           K, 2000 + d) for d in range(DRAWS)]
        per = [f for f in per if f is not None]
        if not per:
            continue
        rs = [stats(f.t, f.p)["r"] for f in per]
        ms = [stats(f.t, f.p)["mae"] for f in per]
        st = stats(per[0].t, per[0].p)
        crows.append(dict(analysis="phase15_subjects", stimulus=label,
                          kind=kind, n=st["n"], k=K, draws=len(per),
                          r=float(np.mean(rs)), r_sd=float(np.std(rs)),
                          mae=float(np.mean(ms)), chance=st["chance"],
                          pct_vs_chance=100 * (1 - np.mean(ms) / st["chance"])))
    ct = pd.DataFrame(crows)
    print(ct[["stimulus", "n", "r", "r_sd", "mae", "chance",
              "pct_vs_chance"]].round(3).to_string(index=False))

    tab["analysis"] = "natural_k"
    pd.concat([tab, mt, ct], ignore_index=True).round(4).to_csv(
        f"../results/Table_per_stimulus_{a.config}{TAG}.csv", index=False)
    print()
    print(tab[tab.split == a.split][
        ["stimulus", "kind", "n", "windows", "r", "mae", "chance",
         "pct_vs_chance", "slope"]].round(3).to_string(index=False))

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .25,
                         "axes.axisbelow": True})
    fig, axes = plt.subplots(4, 4, figsize=(21.5, 21.0))
    ax = axes.ravel()
    COL = {"eyesclose": NAVY, "gyrus_left_closed_eyes": GREEN,
           "gyrus_right_closed_eyes": GREEN, "gyrus (L+R pooled)": GREEN,
           "phase_Stimuli_phase_15": AMBER, "ALL COMBINED": PURPLE,
           "no gyrus (EC + phase_15)": "#1F7A8C"}

    for i, (label, _s, kind) in enumerate(CONDS):
        axx = ax[i]
        f = frames.get(label)
        if f is None:
            axx.axis("off"); continue
        st = stats(f.t, f.p)
        c = COL[label]
        axx.scatter(f.t, f.p, s=34, alpha=.8, color=c, edgecolor="w", lw=.5)
        lo = min(f.t.min(), f.p.min()) - 3
        hi = max(f.t.max(), f.p.max()) + 3
        axx.plot([lo, hi], [lo, hi], "--", color=RED, lw=1.5, label="identity")
        b, a0 = np.polyfit(f.t, f.p, 1)
        axx.plot([lo, hi], [a0 + b * lo, a0 + b * hi], "-", color="k", lw=1.8,
                 label=f"fit (slope {b:.2f})")
        axx.axhline(np.median(f.t), ls=":", color=GRAY, lw=1.3,
                    label="median-age guess")
        axx.text(.03, .97,
                 f"n = {st['n']}\nr = {st['r']:+.3f}\nMAE = {st['mae']:.2f}\n"
                 f"chance = {st['chance']:.2f}\nslope = {st['slope']:+.3f}",
                 transform=axx.transAxes, va="top", fontsize=8.5,
                 family="monospace",
                 bbox=dict(fc="w", ec="0.7", alpha=.92))
        mc = (GREEN if st["pct_vs_chance"] >= 10 else
              (AMBER if st["pct_vs_chance"] > 0 else RED))
        axx.text(.03, .03, f"{st['pct_vs_chance']:+.0f}% vs chance",
                 transform=axx.transAxes, fontsize=9, fontweight="bold",
                 color=mc)
        warn = st["n"] < 20
        axx.text(.97, .97, kind + ("\nUNDERPOWERED n<20" if warn else ""),
                 transform=axx.transAxes, ha="right", va="top", fontsize=7.6,
                 fontweight="bold", color=(RED if warn else c),
                 bbox=dict(fc="w", ec=(RED if warn else c), alpha=.85,
                           boxstyle="round,pad=0.28"))
        axx.set_title(f"{i+1}. {label}", fontsize=10.5, fontweight="bold")
        axx.set_xlabel("chronological age (years)")
        axx.set_ylabel("predicted age (years)")
        axx.set_xlim(lo, hi); axx.set_ylim(lo, hi)
        axx.legend(loc="lower right", fontsize=7.5, framealpha=.92)

    NC = len(CONDS)          # scatters occupy 0..NC-1; summaries follow
    for spare in ax[NC + 6:]:
        spare.axis("off")
    cur = tab[tab.split == a.split].set_index("stimulus")
    order = [c[0] for c in CONDS if c[0] in cur.index]
    cols = [COL[o] for o in order]

    axx = ax[NC + 0]
    axx.bar(range(len(order)), [cur.loc[o, "r"] for o in order], color=cols,
            alpha=.85)
    for j, o in enumerate(order):
        axx.text(j, cur.loc[o, "r"] + .01, f"n={int(cur.loc[o,'n'])}",
                 ha="center", fontsize=7.5)
    axx.axhline(0, color="k", lw=1)
    axx.set_xticks(range(len(order)))
    axx.set_xticklabels([o.replace("_closed_eyes", "").replace("_Stimuli", "")
                         for o in order], rotation=35, ha="right", fontsize=8)
    axx.set_ylabel("Pearson r with age")
    axx.set_title(f"{NC+1}. r by stimulus", fontsize=10.5, fontweight="bold")

    axx = ax[NC + 1]
    vals = [cur.loc[o, "pct_vs_chance"] for o in order]
    axx.bar(range(len(order)), vals,
            color=[GREEN if v >= 10 else (AMBER if v > 0 else RED)
                   for v in vals], alpha=.85)
    axx.axhline(0, color="k", lw=1.2)
    axx.set_xticks(range(len(order)))
    axx.set_xticklabels([o.replace("_closed_eyes", "").replace("_Stimuli", "")
                         for o in order], rotation=35, ha="right", fontsize=8)
    axx.set_ylabel("% better than median-age guess")
    axx.set_title(f"{NC+2}. margin over chance by stimulus", fontsize=10.5,
                  fontweight="bold")

    SHORT = {o: o.replace("_closed_eyes", "").replace("_Stimuli", "")
             for o in [c[0] for c in CONDS]}

    # --- 9  matched-k r, the confound-controlled version of panel 7 ---------
    axx = ax[NC + 2]
    if len(mt):
        xs = np.arange(len(mt))
        axx.bar(xs, mt.r, color=[COL[o] for o in mt.stimulus], alpha=.85,
                yerr=mt.r_sd, error_kw=dict(elinewidth=1, ecolor="#333"))
        axx.errorbar(xs, mt.r, yerr=[mt.r - mt.r_lo, mt.r_hi - mt.r],
                     fmt="none", ecolor=RED, elinewidth=1.2, capsize=3)
        axx.set_xticks(xs)
        axx.set_xticklabels([SHORT[o] for o in mt.stimulus], rotation=35,
                            ha="right", fontsize=8)
        axx.axhline(0, color="k", lw=1)
        axx.set_ylabel("Pearson r with age")
        axx.set_title(f"{NC+3}. CONTROL: matched k={K} windows/subject,\n"
                      f"same {int(mt.n.iloc[0])} subjects "
                      f"(red = bootstrap 95% CI)",
                      fontsize=10, fontweight="bold")

    # --- 10  teh confound itself --------------------------------------------
    axx = ax[NC + 3]
    wps = []
    for label, stims, _k in CONDS:
        mk = lockmask & np.isin(stim, stims)
        c_ = pd.Series(subj[mk]).value_counts()
        wps.append(c_.median() if len(c_) else 0)
    axx.bar(range(len(CONDS)), wps, color=[COL[c[0]] for c in CONDS], alpha=.85)
    axx.axhline(A.EVAL_BAG_K, color=RED, ls="--", lw=1.4,
                label=f"EVAL_BAG_K={A.EVAL_BAG_K} (sampling cap)")
    axx.axhline(K, color="k", ls=":", lw=1.4, label=f"matched k={K}")
    axx.set_xticks(range(len(CONDS)))
    axx.set_xticklabels([SHORT[c[0]] for c in CONDS], rotation=35, ha="right",
                        fontsize=8)
    axx.set_ylabel("median windows per subject")
    axx.set_title(f"{NC+4}. WHY RAW AND MATCHED DIFFER:\nwindows per subject by condition",
                  fontsize=10, fontweight="bold")
    axx.legend(fontsize=7.5)

    # --- 11  are the phase_15 subjects simply easier? ------------------------
    axx = ax[NC + 4]
    if len(ct):
        xs = np.arange(len(ct))
        axx.bar(xs, ct.r, color=[COL[o] for o in ct.stimulus], alpha=.85,
                yerr=ct.r_sd, error_kw=dict(elinewidth=1, ecolor="#333"))
        axx.set_xticks(xs)
        axx.set_xticklabels([SHORT[o] for o in ct.stimulus], rotation=35,
                            ha="right", fontsize=8)
        axx.axhline(0, color="k", lw=1)
        axx.set_ylabel("Pearson r with age")
        axx.set_title(f"{NC+5}. CONTROL: the SAME {int(ct.n.max())} phase_15\n"
                      f"subjects, every condition, k={K}",
                      fontsize=10, fontweight="bold")

    # --- 12  matched margin over chance -------------------------------------
    axx = ax[NC + 5]
    if len(mt):
        xs = np.arange(len(mt))
        v = mt.pct_vs_chance.to_numpy()
        axx.bar(xs, v, alpha=.85,
                color=[GREEN if x >= 10 else (AMBER if x > 0 else RED)
                       for x in v])
        axx.axhline(0, color="k", lw=1.2)
        axx.set_xticks(xs)
        axx.set_xticklabels([SHORT[o] for o in mt.stimulus], rotation=35,
                            ha="right", fontsize=8)
        axx.set_ylabel("% better than median-age guess")
        axx.set_title(f"{NC+6}. CONTROL: margin over chance\nat matched k={K}",
                      fontsize=10, fontweight="bold")

    fig.suptitle(f"{a.config} evaluated per in-house stimulus "
                 f"(in-house {a.split.upper()} subjects; one model, "
                 f"scored separately on each recording)",
                 fontsize=13.5, fontweight="bold", y=.999)
    fig.tight_layout(rect=[0, 0, 1, .975])
    for ext in ("pdf", "png"):
        fig.savefig(f"../results/Fig_per_stimulus_{a.config}{TAG}.{ext}", dpi=165,
                    bbox_inches="tight")
    print(f"\nwrote ../results/Fig_per_stimulus_{a.config}{TAG}.pdf/.png")
    print(f"      ../results/Table_per_stimulus_{a.config}{TAG}.csv")


if __name__ == "__main__":
    main()
