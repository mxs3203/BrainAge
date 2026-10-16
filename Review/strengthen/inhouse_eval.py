"""Run the 14 trained encoders over the in-house cohort and characterise what they"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# The pool and the encoders are both built on per-subject z-scored data, so
# the in-house windows must come from continuous_X_z.npy too. Set before any
os.environ.setdefault("HARMONIZED_ZF", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402
from harmonized_io import build_windows  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
CKPT = os.path.join(HERE, "final_encoders")
WIN_CACHE = os.path.join(CKPT, "ours_windows.npz")
SUFFIX = os.environ.get("CKPT_SUFFIX", "")   # "_adabn" for BN-recalibrated
EMB_CACHE = os.path.join(CKPT, f"inhouse_embeddings{SUFFIX}.npz")
PUB_CACHE = os.path.join(CKPT, "fig4_embeddings.npz")
N_PERM = 500
N_FOLDS = 5
os.makedirs(RES, exist_ok=True)


def ours_windows(recipe="prefiltered_uv"):
    if os.path.exists(WIN_CACHE):
        z = np.load(WIN_CACHE, allow_pickle=True)
        return z["X"], z["subject"], z["age"], z["stimulus"]
    print("building in-house windows (filters 4,100 recordings) ...", flush=True)
    w = build_windows("ours", recipe=recipe, verbose=False)
    X = w["X"].astype(np.float32)
    subject = np.asarray(w["subject"]).astype(str)
    age = np.asarray(w["age"], dtype=float)
    stim = np.asarray(w["stimulus"]).astype(str)
    np.savez(WIN_CACHE, X=X, subject=subject, age=age, stimulus=stim)
    return X, subject, age, stim


def embed_all(X, device, random_init=False):
    """Subject-mean embedding per channel for the in-house cohort."""
    out = {}
    for i, ch in enumerate(A.CHANNELS):
        ck = torch.load(os.path.join(CKPT, f"{ch}{SUFFIX}.pt"), map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        if random_init:
            torch.manual_seed(12345)           # same seed as Figure 4
        enc = A.ChannelEncoderV2(
            conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
            stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
            gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
            embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
            rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
            **cfg["arch_extra"]).to(device)
        if not random_init:
            enc.load_state_dict(ck["state_dict"])
        enc.eval()
        zs = []
        with torch.no_grad():
            for j in range(0, len(X), 1024):
                xb = torch.as_tensor(X[j:j + 1024, :, i], device=device)
                with torch.amp.autocast(device_type="cuda"):
                    z, _ = enc(xb)
                zs.append(z.float().cpu().numpy())
        out[ch] = np.concatenate(zs)
        del enc
        torch.cuda.empty_cache()
        print(f"  {'random' if random_init else 'trained'} {ch}", flush=True)
    return out


def subj_mean(Z, subjects, ages):
    d = pd.DataFrame(Z); d["s"] = subjects
    M = d.groupby("s").mean()
    a = pd.Series(ages).groupby(pd.Series(subjects)).first().loc[M.index]
    return M.to_numpy(), a.to_numpy(), np.array(M.index)


def fit_apply(Mtr, atr, Mva):
    sc = StandardScaler().fit(Mtr)
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr), atr)
    return mdl.predict(sc.transform(Mva))


def stats(a, p):
    if len(a) < 3 or np.std(p) == 0:
        return {"r": np.nan, "mae": np.nan, "slope": np.nan, "sd_ratio": np.nan}
    return {"r": float(np.corrcoef(a, p)[0, 1]),
            "mae": float(np.abs(p - a).mean()),
            "slope": float(np.polyfit(a, p - a, 1)[0]),
            "sd_ratio": float(p.std() / a.std())}


def cv_predict(M, a, folds):
    p = np.empty(len(a))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(a)), te)
        p[te] = fit_apply(M[tr], a[tr], M[te])
    return p


def main():
    device = torch.device("cuda")
    X, subj_w, age_w, stim_w = ours_windows()
    print(f"in-house: {X.shape[0]:,} windows, {len(set(subj_w))} subjects, "
          f"age {age_w.min():.0f}-{age_w.max():.0f}", flush=True)

    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE, allow_pickle=True)
        Et, Er = z["trained"].item(), z["random"].item()
    else:
        Et = embed_all(X, device, random_init=False)
        Er = embed_all(X, device, random_init=True)
        np.savez_compressed(EMB_CACHE, trained=np.array(Et, dtype=object),
                            random=np.array(Er, dtype=object))

    pub = np.load(PUB_CACHE, allow_pickle=True)["data"].item()
    rng = np.random.default_rng(C.SEED)

    rows, quart = [], []
    for ch in A.CHANNELS:
        Mt, a_in, sid = subj_mean(Et[ch], subj_w, age_w)
        Mr, _, _ = subj_mean(Er[ch], subj_w, age_w)
        order = rng.permutation(len(a_in))
        folds = np.array_split(order, N_FOLDS)
        floor = float(np.abs(a_in - np.median(a_in)).mean())

        # --- T0 pure transfer: probe fitted on PUBLIC subjects only
        pt = pub[ch]["trained"]
        p_t0 = fit_apply(pt["Mtr"], pt["atr"], Mt)
        pr = pub[ch]["random"]
        p_t0r = fit_apply(pr["Mtr"], pr["atr"], Mr)

        # --- T1 in-house 5-fold CV
        p_t1 = cv_predict(Mt, a_in, folds)
        p_t1r = cv_predict(Mr, a_in, folds)

        # --- permutation null on T1
        nr = []
        for _ in range(N_PERM):
            sh = rng.permutation(a_in)
            pp = cv_predict(Mt, sh, folds)
            nr.append(np.corrcoef(a_in, pp)[0, 1])
        nr = np.array(nr)

        s0, s1 = stats(a_in, p_t0), stats(a_in, p_t1)
        s0r, s1r = stats(a_in, p_t0r), stats(a_in, p_t1r)
        pub_val = stats(pt["ava"], fit_apply(pt["Mtr"], pt["atr"], pt["Mva"]))
        rows.append({
            "ch": ch, "n_subjects": len(a_in), "floor": floor,
            "T0_r": s0["r"], "T0_mae": s0["mae"], "T0_slope": s0["slope"],
            "T1_r": s1["r"], "T1_mae": s1["mae"], "T1_slope": s1["slope"],
            "T0_r_randinit": s0r["r"], "T1_r_randinit": s1r["r"],
            "null_r_mean": nr.mean(), "null_r_sd": nr.std(),
            "null_r_p95": np.percentile(nr, 95),
            "p_perm_T1": (1 + (nr >= s1["r"]).sum()) / (1 + N_PERM),
            "public_val_r": pub_val["r"], "public_val_mae": pub_val["mae"],
            "transfer_gap_r": pub_val["r"] - s1["r"],
        })
        q = pd.qcut(a_in, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for lab in ["Q1", "Q2", "Q3", "Q4"]:
            m = q == lab
            quart.append({"ch": ch, "quartile": lab, "n": int(m.sum()),
                          "age_mean": float(a_in[m].mean()),
                          "age_lo": float(a_in[m].min()), "age_hi": float(a_in[m].max()),
                          "pred_mean_T1": float(p_t1[m].mean()),
                          "gap_mean_T1": float((p_t1[m] - a_in[m]).mean()),
                          "gap_sd_T1": float((p_t1[m] - a_in[m]).std(ddof=1)),
                          "gap_ci95_T1": float(1.96 * (p_t1[m] - a_in[m]).std(ddof=1)
                                               / np.sqrt(m.sum()))})
        print(f"  {ch:5s} T0 r {s0['r']:+.3f} MAE {s0['mae']:5.2f} | "
              f"T1 r {s1['r']:+.3f} MAE {s1['mae']:5.2f} | rand T1 r {s1r['r']:+.3f} | "
              f"p_perm {rows[-1]['p_perm_T1']:.3f}", flush=True)

    # ---------------- per-stimulus breakdown -------------------------------
    # Pooling all 47 in-house stimuli averages the 4 that carry age signal into
    # the 43 that do not. Evaluating per stimulus shows where the signal is --
    per_stim = []
    for ch in A.CHANNELS:
        Z = Et[ch]
        for stim in np.unique(stim_w):
            m = stim_w == stim
            if m.sum() < 200 or len(np.unique(subj_w[m])) < 20:
                continue
            Ms, a_s, _ = subj_mean(Z[m], subj_w[m], age_w[m])
            if len(a_s) < 20 or np.std(a_s) < 2:
                continue
            o = rng.permutation(len(a_s))
            fo = np.array_split(o, N_FOLDS)
            p1 = cv_predict(Ms, a_s, fo)
            st_ = stats(a_s, p1)
            nr = np.array([np.corrcoef(a_s, cv_predict(Ms, rng.permutation(a_s), fo))[0, 1]
                           for _ in range(100)])
            per_stim.append({"ch": ch, "stimulus": str(stim), "n_subj": len(a_s),
                             "n_windows": int(m.sum()), "r": st_["r"],
                             "mae": st_["mae"], "slope": st_["slope"],
                             "p_perm": (1 + (nr >= st_["r"]).sum()) / 101})
    if per_stim:
        ps = pd.DataFrame(per_stim)
        # BH-FDR across the whole channel x stimulus family
        pv = np.sort(ps.p_perm.values)
        n = len(pv)
        thr = 0.0
        for i, v in enumerate(pv, 1):
            if v <= 0.05 * i / n:
                thr = v
        ps["fdr_sig"] = ps.p_perm <= thr
        ps.round(4).to_csv(os.path.join(RES, "Table_inhouse_per_stimulus.csv"), index=False)
        g = (ps.groupby("stimulus")
               .agg(n_ch=("r", "size"), mean_r=("r", "mean"), max_r=("r", "max"),
                    n_sig=("fdr_sig", "sum"), n_subj=("n_subj", "first"))
               .sort_values("mean_r", ascending=False))
        print("\nper-stimulus (averaged over 14 channels), top 12:")
        print(g.head(12).round(3).to_string())
        print(f"\nBH-FDR 5% threshold p<={thr:.4f} -> "
              f"{int(ps.fdr_sig.sum())}/{len(ps)} channel x stimulus combinations survive")
        print(f"family size {len(ps)}; at p<0.05 uncorrected you would expect "
              f"{0.05*len(ps):.0f} false positives")

    df = pd.DataFrame(rows).sort_values("T1_r", ascending=False)
    df.round(4).to_csv(os.path.join(RES, "Table_inhouse_transfer.csv"), index=False)
    qdf = pd.DataFrame(quart)
    qdf.round(3).to_csv(os.path.join(RES, "Table_inhouse_quartiles.csv"), index=False)

    print("\n" + df[["ch", "T0_r", "T0_mae", "T1_r", "T1_mae", "T1_slope",
                     "T1_r_randinit", "p_perm_T1", "public_val_r",
                     "transfer_gap_r"]].round(3).to_string(index=False))
    print(f"\nin-house no-information floor: {df.floor.iloc[0]:.2f} yr")
    print(f"T0 pure transfer : mean r {df.T0_r.mean():+.3f} | MAE {df.T0_mae.mean():.2f}")
    print(f"T1 in-house CV   : mean r {df.T1_r.mean():+.3f} | MAE {df.T1_mae.mean():.2f} "
          f"| slope {df.T1_slope.mean():+.3f}")
    print(f"T1 random-init   : mean r {df.T1_r_randinit.mean():+.3f}")
    print(f"public validation: mean r {df.public_val_r.mean():+.3f} | "
          f"MAE {df.public_val_mae.mean():.2f}")
    print(f"transfer gap (public r - in-house T1 r): {df.transfer_gap_r.mean():+.3f}")
    print(f"channels with T1 p_perm < 0.05: {(df.p_perm_T1 < 0.05).sum()}/{len(df)}")
    g = qdf.groupby("quartile").agg(age=("age_mean", "mean"), n=("n", "first"),
                                    gap=("gap_mean_T1", "mean"))
    print("\nage quartiles (averaged over channels):")
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
