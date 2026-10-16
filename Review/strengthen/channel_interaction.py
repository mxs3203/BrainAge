"""Channel-interaction matrices from teh log-Euclidean head, by age group.
USAGE: python channel_interaction.py [--config age_phase4_t25] [--split locked]
"""
import argparse, os
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")

import optuna
import age_model_search as A

optuna.logging.set_verbosity(optuna.logging.WARNING)
DB = "sqlite:///age_model_optuna.db"
CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
      "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def get_cfg(name):
    study, n = name.rsplit("_t", 1)
    t = [x for x in optuna.load_study(study_name=study, storage=DB).trials
         if x.number == int(n)][0]
    cfg = dict(t.params)
    cfg.setdefault("train_unit", "bag")
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("pool_mix", "joint")
    if cfg.get("head") != "logeuclid":
        raise SystemExit(f"{name} is head={cfg.get('head')}; this needs logeuclid")
    return cfg


def subject_glog(model, d, device, max_windows=64, bs=32):
    """Mean Glog per subject. d = (X, ages, subjects) as inhouse_data returns."""
    X, ages, subs = d[0], np.asarray(d[1]), np.asarray(d[2])
    out = {}
    model.eval()
    for s in sorted(set(subs)):
        idx = np.flatnonzero(subs == s)
        if len(idx) > max_windows:
            idx = idx[np.linspace(0, len(idx) - 1, max_windows).astype(int)]
        mats = []
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.as_tensor(np.asarray(X[b]), device=device)
            A.SPD_SINK = []
            with torch.no_grad():
                model(xb)
            if A.SPD_SINK:
                mats.append(np.concatenate([q["Glog"] for q in A.SPD_SINK], 0))
            A.SPD_SINK = None
        if mats:
            out[s] = (np.concatenate(mats, 0).mean(0), float(ages[idx[0]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--split", default="locked", choices=["locked", "val", "fit"])
    ap.add_argument("--epochs", type=int, default=12)
    # Loading the shipped checkpoint instead of retraining guarantees the figure
    # describes the model in the paper, and skips a 30-minute retrain whose
    ap.add_argument("--ckpt", default="",
                    help="use this trained checkpoint instead of retraining")
    a = ap.parse_args()

    cfg = get_cfg(a.config)
    print(f"config {a.config}: spd_dim={cfg.get('spd_dim')} "
          f"spd_norm={cfg.get('spd_norm')} bn={cfg.get('spd_bn')}")
    if a.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(a.ckpt, map_location=device, weights_only=False)
        model.eval()
        print(f"  loaded {a.ckpt}")
    else:
        POOLS, va, device, X, ages, subjects, datasets = A.prepare()
        A.EPOCHS = a.epochs
        A.MODEL_SINK = {}
        A.run_fold(cfg, POOLS, va, device, X, ages, subjects, datasets)
        model = A.MODEL_SINK["model"]

    d = A.inhouse_data(a.split)
    print(f"extracting Glog for in-house '{a.split}' ...", flush=True)
    per = subject_glog(model, d, device)
    print(f"  {len(per)} subjects")

    subs = sorted(per)
    M = np.stack([per[s][0] for s in subs])          # [S,14,14]
    age = np.array([per[s][1] for s in subs])
    np.save(f"../results/glog_{a.config}_{a.split}.npy", M)
    np.save(f"../results/glog_{a.config}_{a.split}_ages.npy", age)

    q = np.quantile(age, [1/3, 2/3])
    grp = np.digitize(age, q)
    names = [f"young (<{q[0]:.0f}, n={np.sum(grp==0)})",
             f"mid ({q[0]:.0f}-{q[1]:.0f}, n={np.sum(grp==1)})",
             f"old (>{q[1]:.0f}, n={np.sum(grp==2)})"]

    # per-pair corelation with age
    R = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            v = M[:, i, j]
            R[i, j] = np.corrcoef(v, age)[0, 1] if v.std() > 1e-12 else 0.0
    rows = [dict(ch_i=CH[i], ch_j=CH[j], r_with_age=R[i, j],
                 mean_young=M[grp == 0, i, j].mean(),
                 mean_old=M[grp == 2, i, j].mean())
            for i in range(14) for j in range(i, 14)]
    pd.DataFrame(rows).round(4).sort_values("r_with_age", key=abs, ascending=False)\
        .to_csv("../results/Table_channel_interaction_age_r.csv", index=False)

    # ---- anatomical blocks -------------------------------------------
    # Individual cells are noisy at n=41; the block means are the level the
    ANT = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6"]
    POS = ["P7", "P8", "O1", "O2"]
    TMP = ["T7", "T8"]
    BI = {k: np.array([CH.index(c) for c in v])
          for k, v in (("ANT", ANT), ("POS", POS), ("TMP", TMP))}

    def block_mean(G, ia, ib):
        """Mean over off-diagonal entries of the (ia, ib) block."""
        sub = G[np.ix_(ia, ib)]
        if ia is ib or (len(ia) == len(ib) and np.array_equal(ia, ib)):
            return sub[~np.eye(len(ia), dtype=bool)].mean()
        return sub.mean()

    brows = []
    for na, nb in (("ANT", "ANT"), ("POS", "POS"), ("ANT", "POS"),
                   ("TMP", "TMP"), ("ANT", "TMP"), ("POS", "TMP")):
        v = np.array([block_mean(G, BI[na], BI[nb]) for G in M])
        rb = float(np.corrcoef(v, age)[0, 1]) if v.std() > 1e-12 else 0.0
        brows.append(dict(block=f"{na}-{nb}", r_with_age=round(rb, 4),
                          young=round(float(v[grp == 0].mean()), 4),
                          mid=round(float(v[grp == 1].mean()), 4),
                          old=round(float(v[grp == 2].mean()), 4),
                          old_minus_young=round(float(v[grp == 2].mean()
                                                      - v[grp == 0].mean()), 4)))
    bdf = pd.DataFrame(brows)
    bdf.to_csv("../results/Table_channel_interaction_blocks.csv", index=False)
    print("\nanatomical blocks (off-diagonal means):")
    print(bdf.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
    gm = [M[grp == k].mean(0) for k in range(3)]
    vmax = max(np.abs(g).max() for g in gm)
    for k, ax in enumerate(axes[0]):
        im = ax.imshow(gm[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(names[k], fontsize=11, fontweight="bold")
        ax.set_xticks(range(14)); ax.set_xticklabels(CH, rotation=90, fontsize=7)
        ax.set_yticks(range(14)); ax.set_yticklabels(CH, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=.046)
    diff = gm[2] - gm[0]
    dv = np.abs(diff).max()
    im = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-dv, vmax=dv)
    axes[1, 0].set_title("old - young", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 0], fraction=.046)
    rv = np.abs(R).max()
    im = axes[1, 1].imshow(R, cmap="RdBu_r", vmin=-rv, vmax=rv)
    axes[1, 1].set_title(f"Pearson r with age (n={len(subs)})",
                         fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 1], fraction=.046)
    off = ~np.eye(14, dtype=bool)
    axes[1, 2].hist(R[off], bins=30, color="#3b6ea5", alpha=.85)
    axes[1, 2].axvline(0, color="k", lw=1)
    crit = 1.96 / np.sqrt(max(len(subs) - 3, 1))
    for sgn in (-1, 1):
        axes[1, 2].axvline(sgn * crit, color="#c0392b", ls="--", lw=1.4,
                           label="p=0.05 (uncorrected)" if sgn > 0 else None)
    axes[1, 2].set_title("off-diagonal r with age", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Pearson r"); axes[1, 2].legend(fontsize=8)
    for ax in axes[1, :2]:
        ax.set_xticks(range(14)); ax.set_xticklabels(CH, rotation=90, fontsize=7)
        ax.set_yticks(range(14)); ax.set_yticklabels(CH, fontsize=7)
    fig.suptitle(f"Log-Euclidean channel interaction, {a.config}, "
                 f"in-house {a.split} (n={len(subs)})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97])
    for ext in ("pdf", "png"):
        fig.savefig(f"../results/Fig_channel_interaction.{ext}", dpi=170,
                    bbox_inches="tight")
    print("saved ../results/Fig_channel_interaction.pdf")
    nsig = int((np.abs(R[off]) > crit).sum())
    print(f"off-diagonal pairs with |r| > {crit:.3f} (p<0.05 uncorr): "
          f"{nsig} of {int(off.sum())}  "
          f"-- expected by chance: {0.05*off.sum():.0f}")


if __name__ == "__main__":
    main()
