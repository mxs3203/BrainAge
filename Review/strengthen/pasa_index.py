"""PASA (Posterior-Anterior Shift in Aging) index, three ways.
USAGE: python pasa_index.py [--config age_phase4_t25] [--nperm 10000]
"""
import argparse, os
import numpy as np
import pandas as pd

os.environ.setdefault("HARMONIZED_F", "1")

CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
      "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
ANT = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6"]
POS = ["P7", "P8", "O1", "O2"]
TMP = ["T7", "T8"]
IA = np.array([CH.index(c) for c in ANT])
IP = np.array([CH.index(c) for c in POS])
IT = np.array([CH.index(c) for c in TMP])
FS = 256
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}


# ---------------------------------------------------------------- statistics
def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank(v):
    return pd.Series(v).rank().to_numpy()


def test_index(name, vals, age, rng, nperm, arm, n_note=""):
    """Pearson + Spearman + permutation p for one index against age."""
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals) & np.isfinite(age)
    v, a = vals[ok], age[ok]
    r = _pearson(v, a)
    rs = _pearson(_rank(v), _rank(a))
    null = np.array([_pearson(v, rng.permutation(a)) for _ in range(nperm)])
    p = float((np.abs(null) >= abs(r)).mean())
    # Fisher z CI, for teh effect size rather than the verdict
    n = len(v)
    if n > 3:
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        se = 1.0 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = np.nan
    return dict(arm=arm, index=name, n=n, r=r, r_lo=lo, r_hi=hi,
                spearman=rs, p_perm=p, notes=n_note)


def holm(ps):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


# ------------------------------------------------------- block contrasts
def block_contrast(M, ia, ip, diag=False):
    """(anterior - posterior) block mean of a [S,14,14] stack."""
    out = np.zeros(len(M))
    for s, G in enumerate(M):
        if diag:
            a = np.diag(G)[ia].mean()
            p = np.diag(G)[ip].mean()
        else:
            ga = G[np.ix_(ia, ia)]
            gp = G[np.ix_(ip, ip)]
            a = ga[~np.eye(len(ia), dtype=bool)].mean()
            p = gp[~np.eye(len(ip), dtype=bool)].mean()
        out[s] = a - p
    return out


def cross_block(M, ia, ip):
    return np.array([G[np.ix_(ia, ip)].mean() for G in M])


def logm_spd(C, eps=1e-6):
    """Matrix log of a symmetric PD matrix via eigendecomposition."""
    w, Q = np.linalg.eigh((C + C.T) / 2.0)
    w = np.clip(w, eps, None)
    return (Q * np.log(w)) @ Q.T


# --------------------------------------------------------- raw-signal feats
def raw_features(X, subs, max_windows=400):
    """Per-subject log-covariance (logm) and per-channel log band power."""
    subs = np.asarray(subs)
    uniq = sorted(set(subs))
    nt = X.shape[1]                      # time is axis 1, channels axis 2
    freqs = np.fft.rfftfreq(nt, 1.0 / FS)
    win = np.hanning(nt)
    L, P = {}, {}
    for s in uniq:
        idx = np.flatnonzero(subs == s)
        if len(idx) > max_windows:
            idx = idx[np.linspace(0, len(idx) - 1, max_windows).astype(int)]
        xb = np.asarray(X[idx], dtype=np.float64).transpose(0, 2, 1)  # [n,14,T]
        xb = xb - xb.mean(-1, keepdims=True)
        # covariance, averaged over windows
        C = np.einsum("nct,ndt->cd", xb, xb) / (xb.shape[0] * xb.shape[-1])
        C = C + 1e-8 * np.trace(C) / 14 * np.eye(14)
        L[s] = logm_spd(C)
        # PSD (Welch-style single-segment periodogram, Hanning, averaged)
        S = np.abs(np.fft.rfft(xb * win, axis=-1)) ** 2
        S = S.mean(0)                                       # [14,F]
        tot = S[:, (freqs >= 1) & (freqs <= 45)].sum(-1)
        row = {"abs_broadband": np.log10(tot + 1e-20),
               "rel_broadband": np.ones(14)}
        for b, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            bp = S[:, m].sum(-1)
            row[f"abs_{b}"] = np.log10(bp + 1e-20)
            row[f"rel_{b}"] = bp / np.maximum(tot, 1e-20)
        P[s] = row
    return uniq, L, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261009)
    rows = []

    # ---------------- arm 1: model-space Glog, locked 41 ------------------
    gp = f"../results/glog_{a.config}_locked.npy"
    M = np.load(gp)
    age_m = np.load(f"../results/glog_{a.config}_locked_ages.npy")
    print(f"arm 1  model-space Glog {M.shape} from {a.config}, "
          f"ages {age_m.min():.0f}-{age_m.max():.0f}")
    g_diag = block_contrast(M, IA, IP, diag=True)
    g_with = block_contrast(M, IA, IP, diag=False)
    g_cross = cross_block(M, IA, IP)
    rows += [
        test_index("Glog diagonal  A-P", g_diag, age_m, rng, a.nperm, "model"),
        test_index("Glog within-block A-P", g_with, age_m, rng, a.nperm, "model"),
        test_index("Glog A-P cross coupling", g_cross, age_m, rng, a.nperm, "model"),
    ]
    # circularity check against the model's own prediction
    pred_f = f"../results/Table_pred_joint_{a.config}.csv"
    part_note = ""
    if os.path.exists(pred_f):
        pr = pd.read_csv(pred_f)
        pr = pr[pr.split == "ih_locked"] if "split" in pr else pr
        # the Glog stack is in sorted-subject order; align on that
        subs_sorted = sorted(pr.subject.astype(str))
        pmap = dict(zip(pr.subject.astype(str), pr.p))
        phat = np.array([pmap[s] for s in subs_sorted])
        if len(phat) == len(age_m):
            for nm, v in (("Glog diagonal  A-P", g_diag),
                          ("Glog within-block A-P", g_with)):
                r_vp = _pearson(v, phat)
                r_va, r_ap = _pearson(v, age_m), _pearson(phat, age_m)
                den = np.sqrt(max(1e-12, (1 - r_vp ** 2) * (1 - r_ap ** 2)))
                rows.append(dict(arm="model", index=f"{nm} | partial out pred",
                                 n=len(v), r=(r_va - r_vp * r_ap) / den,
                                 r_lo=np.nan, r_hi=np.nan, spearman=np.nan,
                                 p_perm=np.nan,
                                 notes=f"r(index,pred)={r_vp:+.3f}"))
            part_note = "partial reported"

    # ---------------- arms 2-3: model-free, all 81 subjects ---------------
    import age_model_search as A
    parts = [A.inhouse_data(k) for k in ("fit", "val", "locked")]
    parts = [p for p in parts if p]
    Xall = np.concatenate([p[0] for p in parts], 0)
    agall = np.concatenate([p[1] for p in parts], 0)
    sball = np.concatenate([np.asarray(p[2]).astype(str) for p in parts], 0)
    print(f"arms 2-3  raw signal: {Xall.shape[0]} windows, "
          f"{len(set(sball))} subjects", flush=True)
    uniq, L, P = raw_features(Xall, sball)
    amap = {s: float(agall[sball == s][0]) for s in uniq}
    age_r = np.array([amap[s] for s in uniq])
    Lm = np.stack([L[s] for s in uniq])
    np.save("../results/pasa_raw_logcov.npy", Lm)
    np.save("../results/pasa_raw_ages.npy", age_r)

    l_diag = block_contrast(Lm, IA, IP, diag=True)
    l_with = block_contrast(Lm, IA, IP, diag=False)
    l_cross = cross_block(Lm, IA, IP)
    rows += [
        test_index("logcov diagonal  A-P", l_diag, age_r, rng, a.nperm, "raw-cov"),
        test_index("logcov within-block A-P", l_with, age_r, rng, a.nperm, "raw-cov"),
        test_index("logcov A-P cross coupling", l_cross, age_r, rng, a.nperm, "raw-cov"),
    ]
    for b in BANDS:
        for kind, lab in (("abs", "log power"), ("rel", "relative power")):
            v = np.array([P[s][f"{kind}_{b}"][IA].mean()
                          - P[s][f"{kind}_{b}"][IP].mean() for s in uniq])
            rows.append(test_index(f"{b} {lab}  A-P", v, age_r, rng, a.nperm,
                                   "raw-power"))

    # ---- is the band effect band-specific, or broadband amplitude? -------
    # Every ABSOLUTE log-power contrast came out positive and the effect grows
    bb = np.array([P[s]["abs_broadband"][IA].mean()
                   - P[s]["abs_broadband"][IP].mean() for s in uniq])
    rows.append(test_index("BROADBAND log power 1-45 Hz  A-P", bb, age_r, rng,
                           a.nperm, "confound"))
    r_bba = _pearson(bb, age_r)
    for b_ in BANDS:
        v = np.array([P[s][f"abs_{b_}"][IA].mean()
                      - P[s][f"abs_{b_}"][IP].mean() for s in uniq])
        r_va, r_vbb = _pearson(v, age_r), _pearson(v, bb)
        den = np.sqrt(max(1e-12, (1 - r_vbb ** 2) * (1 - r_bba ** 2)))
        rows.append(dict(arm="confound", n=len(v),
                         index=f"{b_} log power A-P | partial out broadband",
                         r=(r_va - r_vbb * r_bba) / den, r_lo=np.nan,
                         r_hi=np.nan, spearman=np.nan, p_perm=np.nan,
                         notes=f"r(band,broadband)={r_vbb:+.3f}"))

    # ---- does this cohort show the textbook posterior alpha decline? -----
    # If it does not, the montage/cohort may simply lack sensitivity to the
    for nm, idxs, key in (("posterior alpha (abs)", IP, "abs_alpha"),
                          ("posterior alpha (rel)", IP, "rel_alpha"),
                          ("anterior alpha (abs)", IA, "abs_alpha"),
                          ("whole-head alpha (abs)", np.arange(14), "abs_alpha"),
                          ("whole-head beta (abs)", np.arange(14), "abs_beta")):
        v = np.array([P[s][key][idxs].mean() for s in uniq])
        rows.append(test_index(nm, v, age_r, rng, a.nperm, "sanity"))

    # per-channel r with age, for the topography panel
    pc = []
    for i, c in enumerate(CH):
        reg = "anterior" if c in ANT else ("posterior" if c in POS else "temporal")
        pc.append(dict(channel=c, region=reg,
                       r_logcov_diag=_pearson(Lm[:, i, i], age_r),
                       r_log_alpha=_pearson([P[s]["abs_alpha"][i] for s in uniq],
                                            age_r),
                       r_rel_alpha=_pearson([P[s]["rel_alpha"][i] for s in uniq],
                                            age_r),
                       r_log_beta=_pearson([P[s]["abs_beta"][i] for s in uniq],
                                           age_r),
                       r_log_broadband=_pearson(
                           [P[s]["abs_broadband"][i] for s in uniq], age_r),
                       r_glog_diag=_pearson(M[:, i, i], age_m)))
    pcd = pd.DataFrame(pc)
    pcd.round(4).to_csv("../results/Table_PASA_per_channel.csv", index=False)

    # ---------------- table, with Holm across the tested family -----------
    df = pd.DataFrame(rows)
    fam = df.p_perm.notna()
    df.loc[fam, "p_holm"] = holm(df.loc[fam, "p_perm"].to_numpy())
    df["direction_ok"] = np.where(df.r > 0, "yes (A-P rises with age)",
                                  "no (falls)")
    df.round(4).to_csv("../results/Table_PASA_index.csv", index=False)
    print("\n" + df[["arm", "index", "n", "r", "spearman", "p_perm",
                     "p_holm", "notes"]].round(4).to_string(index=False))

    # ---------------- figure ---------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(19.5, 10.5))
    gs = fig.add_gridspec(2, 4, hspace=.38, wspace=.30)

    def block_box(ax):
        for idx, c in ((IA, "#c0392b"), (IP, "#2e7d5b")):
            for i in idx:
                ax.add_patch(plt.Rectangle((i - .5, -.5), 1, 14, fill=False,
                                           ec=c, lw=.8, alpha=.45))
        ax.set_xticks(range(14)); ax.set_xticklabels(CH, rotation=90, fontsize=7)
        ax.set_yticks(range(14)); ax.set_yticklabels(CH, fontsize=7)

    def heat(ax, mm, ttl):
        # the diagonal is an order of magnitude larger than the off-diagonal in
        # the raw log-covariance, so it is masked out of BOTH panels and tested
        z = np.array(mm, float).copy()
        np.fill_diagonal(z, np.nan)
        v = np.nanmax(np.abs(z))
        im = ax.imshow(np.ma.masked_invalid(z), cmap="RdBu_r", vmin=-v, vmax=v)
        block_box(ax); fig.colorbar(im, ax=ax, fraction=.046)
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)

    heat(fig.add_subplot(gs[0, 0]), M.mean(0),
         "A  model Glog, mean over locked 41\noff-diagonal; anterior red, "
         "posterior green")
    heat(fig.add_subplot(gs[0, 1]), Lm.mean(0),
         "B  raw log-covariance, mean over 81\noff-diagonal; model-free twin "
         "of A")

    def scat(ax, v, age, ttl, col):
        ax.scatter(age, v, s=34, c=col, alpha=.8, edgecolor="w", lw=.6)
        b = np.polyfit(age, v, 1)
        xs = np.linspace(age.min(), age.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=col, lw=1.8)
        ax.set_title(f"{ttl}\nr = {_pearson(v, age):+.3f}  (n={len(v)})",
                     loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("age (years)"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[0, 2])
    scat(ax, g_diag, age_m, "C  model PASA (Glog diag A-P) -- null", "#6a3d9a")
    ax.set_ylabel("anterior - posterior")

    ax = fig.add_subplot(gs[0, 3])
    vb = np.array([P[s]["abs_beta"][IA].mean() - P[s]["abs_beta"][IP].mean()
                   for s in uniq])
    scat(ax, vb, age_r, "D  beta log power A-P -- PASA-consistent", "#2e7d5b")
    ax.set_ylabel("anterior - posterior log beta")

    # per-channel gradient, ordered anterior -> temporal -> posterior
    ax = fig.add_subplot(gs[1, 0])
    order = ANT + TMP + POS
    rb = [_pearson([P[s]["abs_beta"][CH.index(c)] for s in uniq], age_r)
          for c in order]
    cc = ["#c0392b"] * len(ANT) + ["#8a8a8a"] * len(TMP) + ["#2e7d5b"] * len(POS)
    ax.bar(range(len(order)), rb, color=cc, alpha=.88)
    crit = 1.96 / np.sqrt(len(uniq) - 3)
    for sgn in (-1, 1):
        ax.axhline(sgn * crit, color="#333333", ls="--", lw=1)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_ylabel("r(log beta, age)")
    ax.set_title("E  per-channel beta-age r\n(anterior | temporal | posterior)",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=.25, axis="y")

    # sanity panel: the textbook ageing effects this cohort does/does not show
    ax = fig.add_subplot(gs[1, 1])
    sn = df[df.arm == "sanity"]
    ax.bar(np.arange(len(sn)), sn.r, color="#8a8a8a", alpha=.88)
    ax.set_xticks(np.arange(len(sn)))
    ax.set_xticklabels([x.replace(" (", "\n(") for x in sn["index"]],
                       rotation=90, fontsize=8)
    for sgn in (-1, 1):
        ax.axhline(sgn * crit, color="#333333", ls="--", lw=1)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("r with age")
    ax.set_ylim(min(-0.3, sn.r.min() * 1.3), max(0.3, sn.r.max() * 1.3))
    ax.set_title("F  sanity: textbook ageing effects\n(all null in this cohort)",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=.25, axis="y")

    ax = fig.add_subplot(gs[1, 2:4])
    d = df[df.p_perm.notna() & (df.arm != "sanity")].copy().iloc[::-1]
    cols = {"model": "#6a3d9a", "raw-cov": "#3b6ea5", "raw-power": "#2e7d5b",
            "confound": "#c98b2e"}
    yp = np.arange(len(d))
    ax.barh(yp, d.r, color=[cols.get(x, "#555555") for x in d.arm], alpha=.85)
    ax.errorbar(d.r, yp, xerr=[d.r - d.r_lo, d.r_hi - d.r], fmt="none",
                ecolor="#333333", elinewidth=1, capsize=2)
    for i, (_, rw) in enumerate(d.iterrows()):
        if rw.p_holm < .05:
            ax.text(rw.r + .025, i, "*", va="center", fontsize=14)
    ax.set_yticks(yp); ax.set_yticklabels(d["index"], fontsize=8)
    ax.yaxis.tick_right()                 # keep long names clear of panel F
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Pearson r with age  (95% CI; * Holm-corrected p < 0.05)")
    ax.set_xlim(float(np.nanmin(d.r_lo)) - .08, float(np.nanmax(d.r_hi)) + .08)
    ax.set_title("G  every pre-specified index; PASA predicts r > 0",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=.25, axis="x")

    fig.suptitle("Posterior-Anterior Shift in Aging (PASA): learned channel "
                 "interaction vs model-free signal, in-house eyes-closed EEG",
                 fontsize=13, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"../results/Fig_PASA_index.{ext}", dpi=170,
                    bbox_inches="tight")
    print("\nwrote ../results/Fig_PASA_index.pdf/.png")
    print("      ../results/Table_PASA_index.csv")
    print("      ../results/Table_PASA_per_channel.csv")


if __name__ == "__main__":
    main()
