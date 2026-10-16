"""Full training of the 14 per-channel encoders using the hyperparameters selected by
Usage:
python train_final_encoders.py                     # all 14 channels
python train_final_encoders.py --channels P7 P8    # a subset
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.amp import autocast, GradScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import optuna  # noqa: E402

import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402
from cross_subject_loss import cross_subject_supervised_contrastive_loss  # noqa: E402
from utils.GpuAugmentor import EEGAugmentorGPU  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

FLOOR = 14.07
R_W, KNN_W = 1.5, 0.5
CKPT_DIR = os.path.join(HERE, "final_encoders")
os.makedirs(CKPT_DIR, exist_ok=True)


# --------------------------------------------------------------- selected configs
def selected_config(ch, db="channel_encoder_arch_optuna.db"):
    st = "sqlite:///" + os.path.join(HERE, db) + "?timeout=120"
    s = optuna.load_study(study_name=f"ch_{ch}", storage=st)
    done = [t for t in s.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not done:
        raise RuntimeError(f"{ch}: no completed trials")
    t = min(done, key=lambda x: x.value)
    p = dict(t.params)
    cfg = dict(p)
    cfg.update(A.FROZEN[ch]["frozen"])          # frozen scalars win
    cfg["dilation"] = S.band_dilations(A.ARCH_BAND_SETS[p["band_set"]], p["kernel_size"])
    cfg["padding"] = None
    cfg["num_layers_gru"] = 1
    cfg["batch_size"] = 256
    # carry the searched normalisation layer through; phase 2 selects it per
    # channel, and overriding it with a CLI default would discard that choice
    cfg["arch_extra"] = {"conv_layers": p["conv_layers"], "branch_mix": p["branch_mix"],
                         "norm": p.get("enc_norm", "none")}
    return cfg, {"search_score": float(t.value), "search_trial": int(t.number),
                 "search_n_trials": len(done)}


# -------------------------------------------------------------------- the monitor
def info_metrics(enc, Xc, ages, subjects, tr_idx, va_idx, device, bs=1024):
    """Subject-level probe r / MAE / slope plus widnow-level kNN-MAE."""
    enc.eval()
    def emb(idx):
        zs = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                xb = torch.as_tensor(np.asarray(Xc[b]), device=device)
                with autocast(device_type="cuda"):
                    z, _ = enc(xb)
                zs.append(z.float().cpu().numpy())
        return np.concatenate(zs)

    Ztr, Zva = emb(tr_idx), emb(va_idx)
    ytr, yva = ages[tr_idx], ages[va_idx]
    s_tr, s_va = subjects[tr_idx], subjects[va_idx]
    kmae, _ = S.knn_mae(Ztr, ytr, Zva, yva)

    dtr = pd.DataFrame(Ztr); dtr["s"] = s_tr
    dva = pd.DataFrame(Zva); dva["s"] = s_va
    Mtr, Mva = dtr.groupby("s").mean(), dva.groupby("s").mean()
    atr = pd.Series(ytr).groupby(pd.Series(s_tr)).first().loc[Mtr.index].to_numpy()
    ava = pd.Series(yva).groupby(pd.Series(s_va)).first().loc[Mva.index].to_numpy()
    sc = StandardScaler().fit(Mtr.to_numpy())
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr.to_numpy()), atr)
    p = mdl.predict(sc.transform(Mva.to_numpy()))
    r = float(np.corrcoef(ava, p)[0, 1])
    mae = float(np.abs(p - ava).mean())
    slope = float(np.polyfit(ava, p - ava, 1)[0])
    sd = float(np.std(Zva, axis=0).mean())
    return {"r": r, "subject_mae": mae, "slope": slope, "knn_mae": kmae,
            "sd_ratio": float(p.std() / ava.std()), "emb_std": sd,
            "score": -R_W * r + KNN_W * (kmae / FLOOR)}


def _pool_info():
    """What the pool was built from, so a history file identifies its own data."""
    import pickle
    try:
        m = pickle.load(open(os.path.join(HERE, "public_pool_meta.pkl"), "rb"))
        return {k: m.get(k) for k in ("only", "group", "recipe", "win", "stride",
                                      "per_subject", "match_age", "excluded")}
    except Exception:
        return {}


def train_channel(ch_idx, ch, meta, device, args, lr_mult=1.0, tag=""):
    pool_info = _pool_info()
    cfg, prov = selected_config(ch, args.db)
    cfg["lr"] = cfg["lr"] * lr_mult
    # --norm overrides only if explicitly passed; otherwise teh per-channel
    # choice from the search wins
    extra = dict(cfg["arch_extra"], use_aux=bool(args.use_aux))
    if args.norm:
        extra["norm"] = args.norm
    cfg["arch_extra"] = extra
    ages, subjects = meta["age"], meta["subject"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * S.VAL_FRACTION), replace=False))
    is_va = np.array([s in va_subs for s in subjects])
    m = cfg["win_stride_mult"]
    tr_idx = np.flatnonzero(~is_va)[::m].copy()
    va_idx = np.flatnonzero(is_va)[::m].copy()
    Xc = S.channel_array(ch_idx)

    # The monitor runs every epoch, and embedding all ~186k train windows each
    # time costs more than the training step itself over a few hundred epochs.
    samp_w = None
    if os.environ.get("MATCH_AGE", "1") != "0":
        sys.path.insert(0, os.path.dirname(HERE))
        from harmonized_io import load_dataset as _ld
        from stimulus_groups import group as _g
        _, _r = _ld("ours")
        _ec = [x for x in _r.stimulus.unique() if _g("ours", x) == "rest_ec"]
        _sub = _r[_r.stimulus.isin(_ec)]
        tgt = np.array([_sub[_sub.subject == x].age.iloc[0]
                        for x in sorted(_sub.subject.unique())], float)
        edges = np.quantile(tgt, np.linspace(0, 1, 7))
        edges[0], edges[-1] = tgt.min() - 1e-6, tgt.max() + 1e-6
        ft = np.histogram(tgt, bins=edges)[0] / len(tgt)
        a_tr = ages[tr_idx]
        b = np.clip(np.digitize(a_tr, edges[1:-1]), 0, len(ft) - 1)
        fs = np.array([(b == i).mean() for i in range(len(ft))])
        ratio = np.where(fs > 0, ft / np.maximum(fs, 1e-9), 0.0)
        # windows outside the target age range keep a small weight rather than
        # zero, so they still regularise without dominating
        out_of_range = (a_tr < edges[0]) | (a_tr > edges[-1])
        samp_w = ratio[b]
        samp_w[out_of_range] = 0.05
        samp_w = samp_w / samp_w.sum()
        eff = 1.0 / np.sum(samp_w ** 2)
        run_cfg = {"age_matched_sampling": True, "target": "ours/rest_ec",
                   "target_mean": float(tgt.mean()), "target_sd": float(tgt.std()),
                   "pool_mean": float(a_tr.mean()), "pool_sd": float(a_tr.std()),
                   "effective_n": float(eff), "n_windows": int(len(tr_idx)),
                   "out_of_range_weight": 0.05}
        print(f"[{ch}] age-matched sampling: target {tgt.mean():.1f}+-{tgt.std():.1f}, "
              f"pool {a_tr.mean():.1f}+-{a_tr.std():.1f}, effective n "
              f"{eff:,.0f} of {len(tr_idx):,} windows", flush=True)

    if samp_w is None:
        run_cfg = {"age_matched_sampling": False}

    mon_tr = tr_idx
    if len(tr_idx) > args.monitor_train:
        mon_tr = np.sort(np.random.default_rng(C.SEED).choice(
            tr_idx, args.monitor_train, replace=False))

    torch.manual_seed(C.SEED)
    enc = A.ChannelEncoderV2(
        conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
        stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
        gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
        embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
        rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
        **cfg["arch_extra"]).to(device)
    n_par = sum(p.numel() for p in enc.parameters())
    aug = EEGAugmentorGPU(noise_std_factor=cfg["aug_noise"], scale_range=cfg["aug_scale"],
                          max_shift=cfg["aug_shift"], max_drop_len=cfg["aug_drop"]).to(device)
    opt = optim.Adam(enc.parameters(), lr=cfg["lr"])
    # LR drops when the monitored information score first plateaus.
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=1e-4, min_lr=args.min_lr)
    scaler = GradScaler(device="cuda")
    ages_d = torch.as_tensor(ages, dtype=torch.float32, device=device)
    su = {s: i for i, s in enumerate(sorted(set(subjects)))}
    subj_d = torch.as_tensor([su[s] for s in subjects], dtype=torch.long, device=device)
    bs = cfg["batch_size"]
    tr_t = torch.as_tensor(tr_idx)

    print(f"[{ch}] {n_par:,} params | k={cfg['kernel_size']} d={cfg['dilation']} "
          f"conv={cfg['conv_channels']} gru={cfg['gru_hidden']} emb={cfg['embedding_dim']} "
          f"| {len(tr_idx):,} train / {len(va_idx):,} val windows "
          f"(monitor on {len(mon_tr):,} train)", flush=True)

    best, best_ep, bad, history = np.inf, -1, 0, []
    ckpt = os.path.join(CKPT_DIR, f"{ch}{tag}.pt")
    for ep in range(1, args.max_epochs + 1):
        enc.train()
        if samp_w is None:
            perm = tr_t[torch.randperm(len(tr_t))]
        else:
            draw = np.random.choice(len(tr_idx), size=len(tr_idx), replace=True,
                                    p=samp_w)
            perm = torch.as_tensor(tr_idx[draw])
        tl = []
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = torch.as_tensor(np.asarray(Xc[b.numpy()]), device=device)
            opt.zero_grad()
            with autocast(device_type="cuda"):
                z1, _ = enc(aug(xb)); z2, _ = enc(aug(xb))
            loss = cross_subject_supervised_contrastive_loss(
                z1.float(), z2.float(), ages=ages_d[b], subjects=subj_d[b],
                temperature=cfg["temperature"], gaussian_sigma=cfg["sigma"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl.append(float(loss))

        mt = info_metrics(enc, Xc, ages, subjects, mon_tr, va_idx, device)
        mt["train_loss"] = float(np.mean(tl))
        mt["lr"] = opt.param_groups[0]["lr"]
        mt["epoch"] = ep
        history.append(mt)
        sched.step(mt["score"])

        improved = mt["score"] < best - 1e-5
        if improved:
            best, best_ep, bad = mt["score"], ep, 0
            torch.save({"state_dict": enc.state_dict(), "cfg": cfg, "lr_mult": lr_mult,
                        "channel": ch, "epoch": ep, "metrics": mt,
                        "provenance": prov, "n_params": n_par}, ckpt)
        else:
            bad += 1
        print(f"  [{ch}] ep{ep:3d} loss {mt['train_loss']:.4f} | score {mt['score']:+.4f} "
              f"| r {mt['r']:.3f} kNN {mt['knn_mae']:.2f} slope {mt['slope']:+.3f} "
              f"MAE {mt['subject_mae']:.2f} | lr {mt['lr']:.2e}"
              f"{'  *' if improved else f'  ({bad}/{args.patience})'}", flush=True)
        if bad >= args.patience:
            print(f"  [{ch}] early stop at epoch {ep}; best epoch {best_ep}", flush=True)
            break

    with open(os.path.join(CKPT_DIR, f"{ch}{tag}_history.json"), "w") as f:
        json.dump({"channel": ch, "cfg": {k: v for k, v in cfg.items()
                                          if k != "arch_extra"},
                   "arch_extra": cfg["arch_extra"], "n_params": n_par,
                   "provenance": prov, "run_config": run_cfg,
                   "lr_mult": lr_mult, "pool_meta": pool_info,
                   "best_epoch": best_ep,
                   "history": history}, f, indent=2, default=str)
    return best, best_ep, history[best_ep - 1] if best_ep > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="channel_encoder_arch_optuna.db")
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--max-epochs", type=int, default=300,
                    help="ceiling only; early stopping decides when to stop")
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    ap.add_argument("--lr-patience", type=int, default=5, help="epochs before LR drop")
    ap.add_argument("--monitor-train", type=int, default=60000,
                    help="train windows used by the per-epoch monitor")
    ap.add_argument("--use-aux", action="store_true",
                    help="feed the window's log amplitude past the normalisation")
    ap.add_argument("--norm", default=None,
                    choices=["none", "instance", "batch", "group"],
                    help="normalisation layer inside the encoder")
    ap.add_argument("--lr-mults", default="1,3,0.3333",
                    help="train each channel at the frozen lr times each of these "
                         "and keep the best; insurance against the frozen lr being "
                         "wrong for the rescaled inputs")
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    args = ap.parse_args()

    device = torch.device("cuda")
    _, meta = S.load_pool()
    chans = args.channels or A.CHANNELS
    shard = int(os.environ.get("FINAL_SHARD", 0))
    nsh = int(os.environ.get("FINAL_NSHARD", 1))
    mine = [(A.CHANNELS.index(c), c) for i, c in enumerate(chans) if i % nsh == shard]
    print(f"worker {shard}/{nsh} -> {[c for _, c in mine]}", flush=True)

    mults = [float(x) for x in args.lr_mults.split(",")]
    summary = {}
    for idx, ch in mine:
        t0 = time.time()
        runs = []
        for m in mults:
            tag = "" if len(mults) == 1 else f"_lr{m:g}"
            sc, ep, mt = train_channel(idx, ch, meta, device, args, lr_mult=m, tag=tag)
            runs.append((sc, m, ep, mt, tag))
            print(f"  [{ch}] lr x{m:g}: score {sc:+.4f} at epoch {ep}", flush=True)
            torch.cuda.empty_cache()
        best = min(runs, key=lambda r: r[0])
        # promote the winning rate to the canonical {ch}.pt the rest of the
        # pipeline loads, so downstream scripts need no knowledge of the sweep
        if best[4]:
            import shutil
            shutil.copy(os.path.join(CKPT_DIR, f"{ch}{best[4]}.pt"),
                        os.path.join(CKPT_DIR, f"{ch}.pt"))
            shutil.copy(os.path.join(CKPT_DIR, f"{ch}{best[4]}_history.json"),
                        os.path.join(CKPT_DIR, f"{ch}_history.json"))
        summary[ch] = {"best_epoch": best[2], "lr_mult": best[1],
                       "all_rates": {f"x{r[1]:g}": r[0] for r in runs},
                       **(best[3] or {})}
        print(f"=== {ch} done in {(time.time()-t0)/60:.1f} min | "
              f"best lr x{best[1]:g}, epoch {best[2]}, score {best[0]:+.4f} ===\n",
              flush=True)

    out = os.path.join(CKPT_DIR, f"summary_shard{shard}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
