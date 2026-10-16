"""PASA as an anterior/posterior RATIO across age GROUPS.
USAGE: python pasa_age_groups.py [--config age_phase4_t25] [--bins decade]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HARMONIZED_F", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pasa_index import (CH, logm_spd, raw_features, _pearson, _rank, FS)

PF  = ["AF3", "AF4"]
FR  = ["F7", "F3", "F4", "F8"]
FC  = ["FC5", "FC6"]
ANT = PF + FR + FC
POS = ["P7", "P8", "O1", "O2"]
IX  = {k: np.array([CH.index(c) for c in v])
       for k, v in (("PF", PF), ("FR", FR), ("FC", FC),
                    ("ANT", ANT), ("POS", POS))}

NAVY, RED, GREY = "#1f3864", "#c00000", "#7f7f7f"
LP_HZ = 45.0            # harmonised low-pass; bands are clipped to this


def paper_bands():
    """fft_bands / brain_score_fft from the project's own config.json."""
    cfgp = os.path.join(HERE, "..", "..", "config.json")
    with open(cfgp) as f:
        c = json.load(f)
    bands, clipped = {}, []
    for k, (lo, hi) in c["fft_bands"].items():
        if hi > LP_HZ:
            clipped.append(f"{k} {hi:g}->{LP_HZ:g}")
        bands[k] = (float(lo), float(min(hi, LP_HZ)))
    return bands, list(c["brain_score_fft"]), clipped


# ------------------------------------------------------------------ scoring
def log_ratio(M_or_P, ia, ip):
    """exp(mean log anterior - mean log posterior), elementwise over subjects."""
    d = np.asarray([v[ia].mean() - v[ip].mean() for v in M_or_P], float)
    return np.exp(d), d


def spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def perm_p(v, age, rng, nperm=10000):
    """Two-sided permutation p on the age labels for Pearson r."""
    r0 = abs(_pearson(v, age))
    a = np.asarray(age, float).copy()
    hits = 0
    for _ in range(nperm):
        rng.shuffle(a)
        if abs(_pearson(v, a)) >= r0 - 1e-15:
            hits += 1
    return (hits + 1) / (nperm + 1)


def group_stats(v, age, edges, labels):
    """Mean +/- SEM of v inside each age bin."""
    gi = np.digitize(age, edges[1:-1])
    out = []
    for g, lab in enumerate(labels):
        m = gi == g
        n = int(m.sum())
        if n == 0:
            out.append(dict(group=lab, n=0, mean=np.nan, sem=np.nan,
                            sd=np.nan, age_mean=np.nan))
            continue
        out.append(dict(group=lab, n=n, mean=float(v[m].mean()),
                        sem=float(v[m].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                        sd=float(v[m].std(ddof=1)) if n > 1 else np.nan,
                        age_mean=float(np.asarray(age)[m].mean())))
    return pd.DataFrame(out), gi


def _letter(ax, ch):
    """Panel letter, placed outside the axes so it never lands on data."""
    ax.text(-0.11, 1.10, ch, transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color="black")


def panel(ax, v, age, edges, labels, title, ylab, colour, note="",
          letter=None, pperm=None):
    """One age-group panel: jittered subjects, group mean +/- SEM, trend line."""
    gs, gi = group_stats(v, age, edges, labels)
    rng = np.random.default_rng(7)
    x = gi + rng.uniform(-0.13, 0.13, len(gi))
    ax.scatter(x, v, s=13, c=colour, alpha=0.35, linewidths=0, zorder=2)
    ok = gs.n > 0
    ax.errorbar(np.flatnonzero(ok), gs.loc[ok, "mean"], yerr=gs.loc[ok, "sem"],
                fmt="o-", color=colour, lw=2.0, ms=7, capsize=4, zorder=4,
                mfc="white", mew=2.0)
    ax.axhline(1.0 if np.nanmedian(v) > 0 else 0.0, color=GREY, lw=0.8,
               ls=":", zorder=1)
    r = _pearson(v, age)
    rs = spearman(v, age)
    ax.set_title(title, fontsize=9.5, loc="left")
    if letter:
        _letter(ax, letter)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\nn={int(n)}" for l, n in zip(labels, gs.n)],
                       fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    lab_ = f"continuous age: r={r:+.3f}  rho={rs:+.3f}"
    if pperm is not None:
        lab_ += f"  p={pperm:.3f}" + ("*" if pperm < 0.05 else " (n.s.)")
    ax.text(0.015, 0.025, lab_ + (f"\n{note}" if note else ""),
            transform=ax.transAxes, fontsize=7.2, va="bottom", color="#333333")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return gs, r, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--bins", default="decade", choices=["decade", "quartile"])
    a = ap.parse_args()
    rng = np.random.default_rng(20261013)
    BANDS, BS_BANDS, clipped = paper_bands()
    print(f"paper bands from config.json: {len(BANDS)} | brain_score_fft = "
          f"{BS_BANDS}")
    if clipped:
        print(f"  clipped to the {LP_HZ:g} Hz low-pass: {', '.join(clipped)}")

    # ---------------- model-free arm: raw log-cov + per-channel power ------
    cache = "../results/pasa_power_table.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        uniq = list(z["uniq"]); age_r = z["age"]; Lm = z["logcov"]
        Pw = {k: z["P_" + k] for k in z["pkeys"]}
        print(f"loaded cached raw features: {len(uniq)} subjects")
    else:
        import age_model_search as A
        parts = [A.inhouse_data(k) for k in ("fit", "val", "locked")]
        parts = [p for p in parts if p]
        Xall = np.concatenate([p[0] for p in parts], 0)
        agall = np.concatenate([p[1] for p in parts], 0)
        sball = np.concatenate([np.asarray(p[2]).astype(str) for p in parts], 0)
        print(f"raw signal: {Xall.shape[0]} windows, {len(set(sball))} subjects",
              flush=True)
        # raw_features uses pasa_index's own BANDS; recompute power here so the
        # paper's band set (and the 45 Hz clip) is what gets used.
        uniq, L, _ = raw_features(Xall, sball)
        amap = {s: float(agall[sball == s][0]) for s in uniq}
        age_r = np.array([amap[s] for s in uniq])
        Lm = np.stack([L[s] for s in uniq])
        Pw = band_power(Xall, sball, uniq, BANDS)
        np.savez(cache, uniq=np.array(uniq, dtype=object), age=age_r,
                 logcov=Lm, pkeys=np.array(list(Pw.keys()), dtype=object),
                 **{"P_" + k: v for k, v in Pw.items()})
        print(f"cached raw features -> {cache}")

    # ---------------- age groups ------------------------------------------
    if a.bins == "decade":
        edges = [20, 30, 40, 50, 63]
        labels = ["20-29", "30-39", "40-49", "50-62"]
    else:
        q = np.quantile(age_r, [0, .25, .5, .75, 1.0])
        edges = list(q); labels = [f"{q[i]:.0f}-{q[i+1]:.0f}" for i in range(4)]
    print(f"age groups ({a.bins}): " + ", ".join(
        f"{l} n={int(((age_r>=edges[i])&(age_r<edges[i+1] if i<3 else age_r<=edges[4])).sum())}"
        for i, l in enumerate(labels)))

    # ---------------- the scores ------------------------------------------
    diag_cov = [np.diag(G) for G in Lm]
    R_cov, d_cov = log_ratio(diag_cov, IX["ANT"], IX["POS"])
    R_pf,  _ = log_ratio(diag_cov, IX["PF"],  IX["POS"])
    R_fr,  _ = log_ratio(diag_cov, IX["FR"],  IX["POS"])
    R_fc,  _ = log_ratio(diag_cov, IX["FC"],  IX["POS"])

    # FFT: paper's brain_score band set, pooled as the mean of its log powers
    bs = np.mean([Pw["abs_" + b] for b in BS_BANDS if "abs_" + b in Pw], axis=0)
    R_bs, d_bs = log_ratio(bs, IX["ANT"], IX["POS"])
    R_band = {}
    for b in BANDS:
        if "abs_" + b in Pw:
            R_band[b] = log_ratio(Pw["abs_" + b], IX["ANT"], IX["POS"])[0]

    # combined, equal a-priori weights on the z-scores
    def z(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    R_comb = (z(d_cov) + z(d_bs)) / 2.0

    # ---------------- model arm ------------------------------------------
    gp = f"../results/glog_{a.config}_locked.npy"
    have_model = os.path.exists(gp)
    if have_model:
        M = np.load(gp)
        age_m = np.load(f"../results/glog_{a.config}_locked_ages.npy")
        R_glog, d_glog = log_ratio([np.diag(G) for G in M], IX["ANT"], IX["POS"])
        print(f"model arm: Glog {M.shape}, {len(age_m)} locked subjects")
    else:
        print(f"model arm skipped: {gp} not found")


    # ---------------- figures --------------------------------------------
    # TWO figures, not one. The MAIN figure carries the three panels that make
    rows, prows = [], []

    _pcache = {}

    def pperm_of(key, v, agev):
        if key not in _pcache:
            _pcache[key] = perm_p(v, agev, np.random.default_rng(11), a.nperm)
        return _pcache[key]

    def record(nm, arm, v, agev, gs, r, rs, unit):
        p_ = pperm_of(nm, v, agev)
        rows.append(dict(arm=arm, score=nm, unit=unit, n=len(v), r=r,
                         spearman=rs, p_perm=p_))
        for _, g in gs.iterrows():
            prows.append(dict(arm=arm, score=nm, **g.to_dict()))
        print(f"  {arm:8s} {nm:34s} n={len(v):3d} r={r:+.3f} rho={rs:+.3f} "
              f"p_perm={p_:.4f}")

    print("\nscores vs age:")
    # the model arm is scored for the table even though it is not plotted
    if have_model:
        gsM, rM = group_stats(R_glog, age_m, edges, labels)[0], _pearson(R_glog, age_m)
        record("Glog ANT/POS (not plotted)", "model", R_glog, age_m, gsM, rM,
               spearman(R_glog, age_m), "ratio")

    # ======================= MAIN FIGURE: A B C ==========================
    figM, axM = plt.subplots(1, 3, figsize=(15.0, 5.2))

    gs, r, rs = panel(axM[0], R_cov, age_r, edges, labels,
                      "Raw log-covariance, anterior/posterior",
                      "ANT / POS  (ratio)", RED, "model-free; all 81 subjects",
                      pperm=pperm_of("logcov ANT/POS", R_cov, age_r), letter="A")
    record("logcov ANT/POS", "logcov", R_cov, age_r, gs, r, rs, "ratio")

    gs, r, rs = panel(axM[1], R_bs, age_r, edges, labels,
                      "FFT, brain-score bands (beta + gamma), anterior/posterior",
                      "ANT / POS power ratio", RED,
                      "model-free; all 81 subjects",
                      pperm=pperm_of("FFT brain_score ANT/POS", R_bs, age_r),
                      letter="B")
    record("FFT brain_score ANT/POS", "fft", R_bs, age_r, gs, r, rs, "ratio")

    gs, r, rs = panel(axM[2], R_comb, age_r, edges, labels,
                      "Combined index, z(log-covariance) + z(FFT)",
                      "combined z score", "#7030a0",
                      "equal weights, fixed a priori",
                      pperm=pperm_of("combined logcov+FFT", R_comb, age_r),
                      letter="C")
    record("combined logcov+FFT", "combined", R_comb, age_r, gs, r, rs, "z")

    figM.suptitle(
        "Frontal/posterior ratio across age groups, in-house EMOTIV cohort (n=81)\n"
        "ANT = AF3 AF4 F7 F3 F4 F8 FC5 FC6   POS = P7 P8 O1 O2   T7/T8 excluded   "
        "ratio = exp(mean log ANT - mean log POS)",
        fontsize=10.5, y=1.045)
    figM.tight_layout()
    for ext in ("pdf", "png"):
        figM.savefig(f"../results/Fig_PASA_main_{a.config}.{ext}", dpi=165,
                     bbox_inches="tight")

    # The anterior SUBREGION breakdown (prefrontal / frontal / frontocentral
    # over posterior) is scored for the table but not plotted: all three are
    for nm, v in (("prefrontal AF3/AF4", R_pf), ("frontal F7 F3 F4 F8", R_fr),
                  ("frontocentral FC5/FC6", R_fc)):
        gsx, _ = group_stats(v, age_r, edges, labels)
        record(nm + " / POS (not plotted)", "logcov", v, age_r, gsx,
               _pearson(v, age_r), spearman(v, age_r), "ratio")

    # ==================== SUPPLEMENT: A B C ==============================
    figS, axS = plt.subplots(1, 3, figsize=(15.0, 5.2))
    axs = axS.ravel()

    # -- A: per-band FFT ratio --------------------------------------------
    axf = axs[0]
    cmap = plt.get_cmap("viridis")
    keys = [b for b in ["delta", "theta", "alpha", "beta", "gamma"] if b in R_band]
    for j, b in enumerate(keys):
        v = R_band[b]
        gsx, _ = group_stats(v, age_r, edges, labels)
        ok = gsx.n > 0
        axf.errorbar(np.flatnonzero(ok), gsx.loc[ok, "mean"],
                     yerr=gsx.loc[ok, "sem"], fmt="o-", lw=1.6, ms=5, capsize=3,
                     color=cmap(j / max(1, len(keys) - 1)),
                     label=f"{b} r={_pearson(v, age_r):+.3f}")
        record(f"FFT {b} ANT/POS", "fft", v, age_r, gsx, _pearson(v, age_r),
               spearman(v, age_r), "ratio")
    axf.axhline(1.0, color=GREY, lw=0.8, ls=":")
    axf.set_title("FFT anterior/posterior ratio, per band", fontsize=9.5,
                  loc="left")
    _letter(axf, "A")
    axf.set_xticks(range(len(labels)))
    axf.set_xticklabels(labels, fontsize=7.5)
    axf.set_ylabel("ANT / POS power ratio", fontsize=8.5)
    axf.legend(fontsize=6.8, frameon=False, ncol=2)
    axf.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)

    # -- B: the combined index against continuous age ---------------------
    axg = axs[1]
    axg.scatter(age_r, R_comb, s=26, c=RED, alpha=0.6, linewidths=0)
    b1, b0 = np.polyfit(age_r, R_comb, 1)
    xs = np.linspace(age_r.min(), age_r.max(), 50)
    axg.plot(xs, b1 * xs + b0, color=NAVY, lw=2)
    axg.set_title(f"Combined index vs continuous age  "
                  f"(r={_pearson(R_comb, age_r):+.3f})", fontsize=9.5, loc="left")
    _letter(axg, "B")
    axg.set_xlabel("age (years)", fontsize=8.5)
    axg.set_ylabel("combined z score", fontsize=8.5)
    axg.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        axg.spines[sp].set_visible(False)

    # -- C: do teh two model-free arms measure the same thing -------------
    axh = axs[2]
    axh.scatter(z(d_cov), z(d_bs), s=26, c="#7030a0", alpha=0.6, linewidths=0)
    r_arms = _pearson(d_cov, d_bs)
    axh.set_title(f"Agreement between the two model-free arms  r={r_arms:+.3f}",
                  fontsize=9.5, loc="left")
    _letter(axh, "C")
    axh.set_xlabel("z  log-covariance ANT/POS (log ratio)", fontsize=8.5)
    axh.set_ylabel("z  FFT brain-score ANT/POS", fontsize=8.5)
    axh.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)

    figS.suptitle(
        "Supplement: composition and robustness of the frontal/posterior age index\n"
        "in-house EMOTIV cohort (n=81), model-free throughout",
        fontsize=10.5, y=1.045)
    figS.tight_layout()
    for ext in ("pdf", "png"):
        figS.savefig(f"../results/Fig_PASA_supp_{a.config}.{ext}", dpi=165,
                     bbox_inches="tight")

    df = pd.DataFrame(rows)

    # the fitted combination, reported WITH its circularity
    Xd = np.c_[z(d_cov), z(d_bs)]
    w = np.linalg.lstsq(np.c_[Xd, np.ones(len(Xd))], age_r, rcond=None)[0]
    fit = Xd @ w[:2]
    print(f"\nfitted combination (CIRCULAR -- weights fitted on these same "
          f"{len(age_r)} subjects):")
    print(f"  weights logcov {w[0]:+.3f}  FFT {w[1]:+.3f}  -> r={_pearson(fit, age_r):+.3f}")
    print(f"  equal-weight a-priori version        -> r={_pearson(R_comb, age_r):+.3f}")

    df.to_csv(f"../results/Table_PASA_ratio_scores_{a.config}.csv", index=False)
    pd.DataFrame(prows).to_csv(
        f"../results/Table_PASA_age_groups_{a.config}.csv", index=False)
    print(f"\nwrote ../results/Fig_PASA_main_{a.config}.pdf/.png   (B C D -> A B C)")
    print(f"      ../results/Fig_PASA_supp_{a.config}.pdf/.png   (F G H -> A B C)")
    print(f"      ../results/Table_PASA_ratio_scores_{a.config}.csv")
    print(f"      ../results/Table_PASA_age_groups_{a.config}.csv")


def band_power(X, subs, uniq, bands, max_windows=400):
    """Per-channel log10 power in each of `bands`, per subject."""
    subs = np.asarray(subs)
    nt = X.shape[1]
    freqs = np.fft.rfftfreq(nt, 1.0 / FS)
    win = np.hanning(nt)
    out = {f"abs_{b}": np.zeros((len(uniq), 14)) for b in bands}
    out["abs_broadband"] = np.zeros((len(uniq), 14))
    for si, s in enumerate(uniq):
        idx = np.flatnonzero(subs == s)
        if len(idx) > max_windows:
            idx = idx[np.linspace(0, len(idx) - 1, max_windows).astype(int)]
        xb = np.asarray(X[idx], dtype=np.float64).transpose(0, 2, 1)
        xb = xb - xb.mean(-1, keepdims=True)
        S = (np.abs(np.fft.rfft(xb * win, axis=-1)) ** 2).mean(0)   # [14,F]
        tot = S[:, (freqs >= 1) & (freqs <= LP_HZ)].sum(-1)
        out["abs_broadband"][si] = np.log10(tot + 1e-20)
        for b, (lo, hi) in bands.items():
            m = (freqs >= lo) & (freqs < hi)
            out[f"abs_{b}"][si] = np.log10(S[:, m].sum(-1) + 1e-20)
    return out


if __name__ == "__main__":
    main()
