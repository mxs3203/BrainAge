"""Figure 4: how much age information the encoders actually extract, against chance."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "strengthen"))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "strengthen", "final_encoders")
CACHE = os.path.join(CKPT, "fig4_embeddings.npz")
os.makedirs(RES, exist_ok=True)
NAVY, RED, GRAY, LGRAY = "#1F4E79", "#C00000", "#808080", "#BFBFBF"
N_PERM = 1000

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def build_encoder(cfg, trained_state=None, device="cuda", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    e = A.ChannelEncoderV2(
        conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
        stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
        gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
        embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
        rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
        **cfg["arch_extra"]).to(device)
    if trained_state is not None:
        e.load_state_dict(trained_state)
    return e


def embed(enc, Xc, idx, device, bs=1024):
    enc.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.as_tensor(np.asarray(Xc[b]), device=device)
            with torch.amp.autocast(device_type="cuda"):
                z, _ = enc(xb)
            zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


def subj_mean(Z, subjects, ages):
    d = pd.DataFrame(Z); d["s"] = subjects
    M = d.groupby("s").mean()
    a = pd.Series(ages).groupby(pd.Series(subjects)).first().loc[M.index]
    return M.to_numpy(), a.to_numpy()


def probe(Mtr, atr, Mva, ava):
    sc = StandardScaler().fit(Mtr)
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr), atr)
    p = mdl.predict(sc.transform(Mva))
    return p


def metrics(ava, p):
    return {"mae": float(np.abs(p - ava).mean()),
            "r": float(np.corrcoef(ava, p)[0, 1]),
            "slope": float(np.polyfit(ava, p - ava, 1)[0])}


def compute():
    """Subject-mean embeddings for trained and random-init encoders."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item()
    device = torch.device("cuda")
    _, meta = S.load_pool()
    ages, subjects = meta["age"], meta["subject"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * S.VAL_FRACTION), replace=False))
    is_va = np.array([s in va_subs for s in subjects])

    data = {}
    for ch in A.CHANNELS:
        ck = torch.load(os.path.join(CKPT, f"{ch}.pt"), map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        m = cfg["win_stride_mult"]
        tr = np.flatnonzero(~is_va)[::m].copy()
        va = np.flatnonzero(is_va)[::m].copy()
        Xc = S.channel_array(A.CHANNELS.index(ch))
        d = {}
        for tag, st_, seed in (("trained", ck["state_dict"], None),
                               ("random", None, 12345)):
            enc = build_encoder(cfg, st_, device, seed)
            Mtr, atr = subj_mean(embed(enc, Xc, tr, device), subjects[tr], ages[tr])
            Mva, ava = subj_mean(embed(enc, Xc, va, device), subjects[va], ages[va])
            d[tag] = {"Mtr": Mtr, "atr": atr, "Mva": Mva, "ava": ava}
            del enc
            torch.cuda.empty_cache()
        data[ch] = d
        print(f"  embedded {ch}", flush=True)
    np.savez_compressed(CACHE, data=np.array(data, dtype=object))
    return data


def main():
    data = compute()
    rows, nulls, preds = [], {}, {}
    rng = np.random.default_rng(0)
    for ch, d in data.items():
        t = d["trained"]
        p = probe(t["Mtr"], t["atr"], t["Mva"], t["ava"])
        mt = metrics(t["ava"], p)
        preds[ch] = (t["ava"], p)
        q = d["random"]
        mr = metrics(q["ava"], probe(q["Mtr"], q["atr"], q["Mva"], q["ava"]))
        # permutation null: real embeddings, shuffled ages, probe refitted
        nr, nm = [], []
        for _ in range(N_PERM):
            sh = rng.permutation(t["atr"])
            pp = probe(t["Mtr"], sh, t["Mva"], t["ava"])
            nr.append(np.corrcoef(t["ava"], pp)[0, 1])
            nm.append(np.abs(pp - t["ava"]).mean())
        nr, nm = np.array(nr), np.array(nm)
        nulls[ch] = {"r": nr, "mae": nm}
        rows.append({
            "ch": ch, "r": mt["r"], "mae": mt["mae"], "slope": mt["slope"],
            "r_random_init": mr["r"], "mae_random_init": mr["mae"],
            "null_r_mean": nr.mean(), "null_r_p95": np.percentile(nr, 95),
            "null_mae_mean": nm.mean(), "null_mae_p05": np.percentile(nm, 5),
            "z_r": (mt["r"] - nr.mean()) / nr.std(),
            "p_perm": (1 + (nr >= mt["r"]).sum()) / (1 + N_PERM),
        })
        print(f"  {ch:5s} r {mt['r']:.3f} | random-init {mr['r']:.3f} | "
              f"null {nr.mean():+.3f}+-{nr.std():.3f} | z {rows[-1]['z_r']:.1f}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)
    df.round(4).to_csv(os.path.join(RES, "Table_information_vs_chance.csv"), index=False)

    # best-trial information score, moved over from Figure 3
    st = ("sqlite:///" + os.path.join(HERE, "strengthen",
                                      "channel_encoder_arch_optuna.db") + "?timeout=120")
    score = {}
    for ch in A.CHANNELS:
        try:
            s = optuna.load_study(study_name=f"ch_{ch}", storage=st)
        except KeyError:
            continue
        c = [t for t in s.trials if t.state.name == "COMPLETE" and t.value is not None]
        if c:
            score[ch] = -min(c, key=lambda t: t.value).value

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.95)
    x = np.arange(len(df))

    # ---- A  age correlation vs both nulls ------------------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    ax.bar(x, df.r, color=NAVY, edgecolor="k", lw=0.7, label="Trained encoder")
    ax.bar(x, df.r_random_init, width=0.45, color=LGRAY, edgecolor="k", lw=0.6,
           label="Random-init, same architecture")
    ax.plot(x, df.null_r_p95, ls="--", lw=1.4, color=RED,
            label="Permutation null, 95th pct")
    ax.set_xticks(x); ax.set_xticklabels(df.ch, rotation=45, ha="right",
                                         style="italic", fontsize=8)
    ax.set_ylabel("Subject-level age correlation  r")
    ax.set_ylim(min(0, df.r_random_init.min() - 0.05), df.r.max() * 1.25)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95, ncol=1)
    ax.set_title("A. Age information vs chance")

    # ---- B  subject MAE vs floor and null ------------------------------------
    ax = fig.add_subplot(gs[0, 2:4])
    ax.bar(x, df.mae, color=NAVY, edgecolor="k", lw=0.7, label="Trained encoder")
    ax.bar(x, df.mae_random_init, width=0.45, color=LGRAY, edgecolor="k", lw=0.6,
           label="Random-init")
    ax.axhline(14.07, color=RED, ls="--", lw=1.5, label="No-information floor")
    ax.plot(x, df.null_mae_p05, ls=":", lw=1.5, color=RED,
            label="Permutation null, 5th pct")
    ax.set_xticks(x); ax.set_xticklabels(df.ch, rotation=45, ha="right",
                                         style="italic", fontsize=8)
    ax.set_ylabel("Subject MAE (years)")
    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.9)
    ax.set_title("B. Prediction error vs chance")

    # ---- C  best information score (moved from Figure 3) ---------------------
    ax = fig.add_subplot(gs[0, 4:6])
    d2 = df[df.ch.isin(score)].copy()
    d2["score"] = [score[c] for c in d2.ch]
    d2 = d2.sort_values("score")
    ax.bar(range(len(d2)), d2.score, color=NAVY, edgecolor="k", lw=0.7)
    ax.set_xticks(range(len(d2)))
    ax.set_xticklabels(d2.ch, rotation=45, ha="right", style="italic", fontsize=8)
    ax.set_ylim(d2.score.min() - 0.04, d2.score.max() + 0.02)
    ax.set_xlabel("Channel"); ax.set_ylabel("Best information score")
    ax.set_title("C. Search score by channel")

    # ---- D  permutation null distribution vs observed ------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    best_ch = df.ch.iloc[0]
    nr = nulls[best_ch]["r"]
    ax.hist(nr, bins=40, color=LGRAY, edgecolor="k", lw=0.5,
            label=f"Permutation null ({N_PERM} shuffles)")
    ax.axvline(df.r.iloc[0], color=NAVY, lw=2.4, label=f"{best_ch} observed")
    ax.axvline(df.r_random_init.iloc[0], color=RED, lw=2.0, ls="--",
               label=f"{best_ch} random-init")
    ax.set_xlabel("Age correlation  r"); ax.set_ylabel("Permutations")
    ax.legend(fontsize=7, loc="upper center", framealpha=0.9)
    ax.set_title(f"D. Null distribution vs observed ({best_ch})")

    # ---- E  effect size per channel ------------------------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    ax.bar(x, df.z_r, color=NAVY, edgecolor="k", lw=0.7)
    ax.axhline(1.96, color=RED, ls="--", lw=1.4)
    ax.text(0.0, 2.05, "p = 0.05", fontsize=7.5, color=RED,
            ha="left", va="bottom", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(df.ch, rotation=45, ha="right",
                                         style="italic", fontsize=8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Effect size  (r - null mean) / null SD")
    ax.set_title("E. Distance from chance, in null SDs")

    # ---- F  predicted vs true age, best channel ------------------------------
    ax = fig.add_subplot(gs[1, 4:6])
    ava, p = preds[best_ch]
    ax.scatter(ava, p, s=22, color=NAVY, alpha=0.6, edgecolor="none")
    lo, hi = min(ava.min(), p.min()) - 2, max(ava.max(), p.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1.5, label="Identity")
    b = np.polyfit(ava, p, 1)
    ax.plot([lo, hi], np.polyval(b, [lo, hi]), color="k", lw=1.5,
            label=f"Fit (slope {b[0]:.2f})")
    ax.axhline(np.median(ava), color=GRAY, ls=":", lw=1.2, label="Median-age guess")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title(f"F. Predicted vs true age ({best_ch}, held-out subjects)")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig4_information.{ext}"), dpi=190,
                    bbox_inches="tight")
    print("\nsaved", os.path.join(RES, "Fig4_information.pdf"))
    print(df[["ch", "r", "r_random_init", "null_r_mean", "z_r", "p_perm",
              "mae", "mae_random_init"]].round(3).to_string(index=False))
    print(f"\nall 14 channels beat the permutation null at p<{df.p_perm.max():.4f}")
    print(f"mean r trained {df.r.mean():.3f} vs random-init {df.r_random_init.mean():.3f} "
          f"-> training contributes {df.r.mean()-df.r_random_init.mean():+.3f}")
    print(f"mean MAE trained {df.mae.mean():.2f} vs random-init "
          f"{df.mae_random_init.mean():.2f} vs floor 14.07")


if __name__ == "__main__":
    main()
