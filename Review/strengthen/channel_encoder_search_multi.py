"""Per-channel encoder search on all public data — the paper's original design.
running statistics, so no train/eval mismatch and no recalibration step)
"""
import json
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config_review as C
from build_public_pool import load_pool
from cross_subject_loss import cross_subject_supervised_contrastive_loss
from utils.GpuAugmentor import EEGAugmentorGPU

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.environ.get("CH_TRIALS", 40))
EPOCHS = int(os.environ.get("CH_EPOCHS", 10))
VAL_FRACTION = 0.25
TARGET_ALIGN, TARGET_UNIF = 0.3, -2.0      # the paper's hinge thresholds
SLOPE_W = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "channel_encoder_multi_optuna.db")
STORAGE = "sqlite:///" + DB_PATH


def get_storage():
    """SQLite tuned for concurrent workers."""
    import optuna.storages
    return optuna.storages.RDBStorage(
        url=STORAGE,
        engine_kwargs={"connect_args": {"timeout": 120, "check_same_thread": False}},
    )


def init_studies(names):
    """Create every study once from the parent, before any worker starts."""
    import sqlite3
    st = get_storage()
    for n in names:
        optuna.create_study(direction="minimize", study_name=f"ch_{n}",
                            storage=st, load_if_exists=True)
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("PRAGMA journal_mode=WAL;")
        con.close()
    except Exception:
        pass
    return st
OUT = os.path.join(HERE, "channel_encoder_multi_results.json")
OUT_DIR = os.path.join(HERE, "channel_results_multi"); os.makedirs(OUT_DIR, exist_ok=True)


class ChannelEncoder(nn.Module):
    """Published Conv1d+GRU encoder, BatchNorm removed, dilation added, bidirectional."""

    def __init__(self, conv_channels=16, kernel_size=7, stride=2, padding=None,
                 dilation=(1,), gru_hidden=16, num_layers_gru=1, embedding_dim=16,
                 cnn_dropout=0.05, rnn_dropout=0.0, activation="ReLU"):
        super().__init__()
        act = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}[activation]
        dils = (dilation,) if isinstance(dilation, int) else tuple(dilation)
        # One Conv1d per dilation, run in PARALLEL on teh same input and
        # concatenated — each branch sees a different temporal scale. padding is
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, conv_channels, kernel_size=kernel_size, stride=stride,
                          padding=d * (kernel_size // 2), dilation=d),
                act(),
                nn.Dropout1d(p=cnn_dropout))
            for d in dils])
        self.dilations = dils
        self.gru = nn.GRU(conv_channels * len(dils), gru_hidden, num_layers=num_layers_gru,
                          batch_first=True, bidirectional=True,
                          dropout=rnn_dropout if num_layers_gru > 1 else 0.0)
        self.out_dim = gru_hidden * 2
        self.projector = nn.Linear(self.out_dim, embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                       # [B,1,L]
        h = torch.cat([b(x) for b in self.branches], dim=1)
        if h.shape[-1] < 2:
            raise ValueError("conv collapsed the time axis")
        out, hn = self.gru(h.permute(0, 2, 1))
        out = torch.cat([hn[-2], hn[-1]], dim=1)     # bidirectional final states
        return F.normalize(self.projector(out), dim=1), out


# ----------------------------------------------------- paper's quality metrics
def alignment(z1, z2, alpha=2):
    return (z1 - z2).norm(dim=1).pow(alpha).mean()


def uniformity(z, t=2):
    sq = torch.pdist(z, p=2).pow(2)
    return sq.mul(-t).exp().mean().log()


def collapse(z):
    sd = z.std(dim=0)
    return float(sd.mean()), float((sd < 1e-3).float().mean())


def knn_mae(tr_z, tr_y, va_z, va_y, k=10, max_ref=20000, chunk=4096):
    """k-NN age regression in embedding space (teh paper's kNN-MAE)."""
    rng = np.random.default_rng(C.SEED)
    tr_z = np.ascontiguousarray(tr_z); tr_y = np.asarray(tr_y)
    if len(tr_z) > max_ref:
        sel = rng.choice(len(tr_z), max_ref, replace=False)
        tr_z, tr_y = tr_z[sel], tr_y[sel]
    tr = F.normalize(torch.as_tensor(tr_z), dim=1)
    tr_y_t = torch.as_tensor(tr_y, dtype=torch.float32)
    preds = []
    for i in range(0, len(va_z), chunk):
        q = F.normalize(torch.as_tensor(np.ascontiguousarray(va_z[i:i + chunk])), dim=1)
        sim = q @ tr.T                                # [chunk, max_ref]
        idx = sim.topk(min(k, tr.shape[0]), dim=1).indices
        preds.append(tr_y_t[idx].mean(1).numpy())
        del sim, q
    pred = np.concatenate(preds)
    return float(np.abs(pred - va_y).mean()), pred


def subject_slope(y_true, y_pred, subjects):
    """Aggregate to subjects, then slope of (pred - true) on true."""
    import pandas as pd
    d = pd.DataFrame({"s": subjects, "t": y_true, "p": y_pred})
    g = d.groupby("s").agg(t=("t", "first"), p=("p", "mean"))
    if g.t.nunique() < 3:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.polyfit(g.t, g.p - g.t, 1)[0])
    mae = float(np.abs(g.p - g.t).mean())
    r = float(np.corrcoef(g.t, g.p)[0, 1])
    return slope, mae, r


# --------------------------------------------------------------------- search
def evaluate_cfg(cfg, Xc, ages, subjects, tr_idx, va_idx, device, return_emb=False,
                 encoder_cls=None, extra_X=None):
    torch.manual_seed(C.SEED)
    # encoder_cls lets stage 2 inject a wider architecture without changing teh
    # class this finished search used. Default path is byte-identicial.
    enc = (encoder_cls or ChannelEncoder)(
        conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
        stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
        gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
        embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
        rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
        **cfg.get("arch_extra", {})).to(device)
    aug = EEGAugmentorGPU(noise_std_factor=cfg["aug_noise"], scale_range=cfg["aug_scale"],
                          max_shift=cfg["aug_shift"], max_drop_len=cfg["aug_drop"]).to(device)
    opt = optim.Adam(enc.parameters(), lr=cfg["lr"])
    scaler = GradScaler(device="cuda")
    ages_d = torch.as_tensor(ages, dtype=torch.float32, device=device)
    su = {s: i for i, s in enumerate(sorted(set(subjects)))}
    subj_d = torch.as_tensor([su[s] for s in subjects], dtype=torch.long, device=device)
    bs = cfg["batch_size"]

    for _ in range(EPOCHS):
        enc.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
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

    # ---------------- validation: contrastive loss + geometry ----------------
    enc.eval()
    vl, al, un = [], [], []
    zs_va, zs_tr = [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), 1024):
            b = va_idx[i:i + 1024]
            xb = torch.as_tensor(np.asarray(Xc[b.numpy()]), device=device)
            with autocast(device_type="cuda"):
                z1, _ = enc(aug(xb)); z2, _ = enc(aug(xb))
                zc, _ = enc(xb)                        # clean embedding for probes
            z1, z2 = z1.float(), z2.float()
            vl.append(float(cross_subject_supervised_contrastive_loss(
                z1, z2, ages=ages_d[b], subjects=subj_d[b],
                temperature=cfg["temperature"], gaussian_sigma=cfg["sigma"])))
            al.append(float(alignment(z1, z2)))
            un.append(float(uniformity(torch.cat([z1, z2])[:512])))
            zs_va.append(zc.float().cpu().numpy())
        for i in range(0, len(tr_idx), 1024):
            b = tr_idx[i:i + 1024]
            xb = torch.as_tensor(np.asarray(Xc[b.numpy()]), device=device)
            with autocast(device_type="cuda"):
                zc, _ = enc(xb)
            zs_tr.append(zc.float().cpu().numpy())
    Zva, Ztr = np.concatenate(zs_va), np.concatenate(zs_tr)
    mean_sd, dead = collapse(torch.as_tensor(Zva))

    ytr, yva = ages[tr_idx.numpy()], ages[va_idx.numpy()]
    sva = subjects[va_idx.numpy()]
    kmae, knn_pred = knn_mae(Ztr, ytr, Zva, yva)
    # Ridge probe -> slope / subject-level MAE
    sc = StandardScaler().fit(Ztr)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Ztr), ytr)
    pred = ridge.predict(sc.transform(Zva))
    slope, smae, r = subject_slope(yva, pred, sva)

    res = {"val_loss": float(np.mean(vl)), "alignment": float(np.mean(al)),
           "uniformity": float(np.mean(un)), "emb_std": mean_sd, "dead_frac": dead,
           "knn_mae": kmae, "slope": slope, "subject_mae": smae, "corr_r": r}
    # paper's hinge penalties + slope term
    res["score"] = (kmae
                    + SLOPE_W * (abs(slope) if np.isfinite(slope) else 1.0)
                    + 0.5 * max(0.0, res["alignment"] - TARGET_ALIGN)
                    + 0.5 * max(0.0, res["uniformity"] - TARGET_UNIF))
    # Explicit collapse flag. The first pass produced alignment ~0 AND uniformity ~0
    # AND emb_std ~0.03 — everything mapped to one point, which makes "good"
    res["collapsed"] = bool(res["uniformity"] > -0.5 or res["emb_std"] < 0.10)
    # Raw embeddings, for probe_fix.py to compare probe variants on identical
    # embeddings. Returned only on request — the search path stores res items as
    if return_emb:
        res["_emb"] = {"Ztr": Ztr, "ytr": ytr, "str": subjects[tr_idx.numpy()],
                       "Zva": Zva, "yva": yva, "sva": sva}
        # Embed any extra cohort (e.g. the in-house recordings) while the encoder
        # is still alive -- it is deleted immediately below.
        if extra_X is not None:
            zs = []
            with torch.no_grad():
                for i in range(0, len(extra_X), 1024):
                    xb = torch.as_tensor(np.asarray(extra_X[i:i + 1024]), device=device)
                    with autocast(device_type="cuda"):
                        ze, _ = enc(xb)
                    zs.append(ze.float().cpu().numpy())
            res["_emb"]["Zextra"] = np.concatenate(zs) if zs else None
    del enc
    torch.cuda.empty_cache()
    return res


FS_HZ = 256
# Canonical EEG bands. A branch with dilation d and kernel k passes roughly
#   [fs/((k-1)d+1), fs/(2d)]  Hz,
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "beta_l": (12, 20), "beta_h": (20, 30), "gamma": (30, 45)}

# Which band combinations to offer. Names say what the branches listen to.
BAND_SETS = {
    "all6":          ["delta", "theta", "alpha", "beta_l", "beta_h", "gamma"],
    "classic5":      ["delta", "theta", "alpha", "beta_l", "gamma"],
    "low3":          ["delta", "theta", "alpha"],
    "mid3":          ["theta", "alpha", "beta_l"],
    "high3":         ["beta_l", "beta_h", "gamma"],
    "delta_alpha":   ["delta", "alpha"],
    "theta_beta":    ["theta", "beta_l"],
    "alpha_only":    ["alpha"],
    "delta_only":    ["delta"],
    "theta_alpha_g": ["theta", "alpha", "gamma"],
    "wide4":         ["delta", "theta", "beta_l", "gamma"],
}


def band_dilations(band_names, kernel_size):
    """Dilations centring a branch on each named band, for this kernel."""
    out = []
    for b in band_names:
        lo, hi = BANDS[b]
        fc = (lo * hi) ** 0.5
        d = max(1, int(round(FS_HZ / (fc * (2 * (kernel_size - 1)) ** 0.5))))
        if d not in out:
            out.append(d)
    return tuple(sorted(out))


def suggest(trial, win=512):
    """Narrowed from the completed 1,120-trial single-dilation search."""
    k = trial.suggest_categorical("kernel_size", [7, 11, 15])
    bands = trial.suggest_categorical("band_set", list(BAND_SETS))
    d = band_dilations(BAND_SETS[bands], k)
    cfg = {
        "kernel_size": k, "dilation": d, "band_set": bands,
        "stride": trial.suggest_categorical("stride", [2, 4]),
        "padding": None,                       # derived per branch as d*(k//2)
        "conv_channels": trial.suggest_categorical("conv_channels", [8, 16, 32]),
        "gru_hidden": trial.suggest_categorical("gru_hidden", [24, 32]),
        "num_layers_gru": 1,                   # fixed: no signal in the first pass
        "embedding_dim": trial.suggest_categorical("embedding_dim", [24, 32]),
        "cnn_dropout": trial.suggest_float("cnn_dropout", 0.0, 0.15),
        "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.15),
        "activation": trial.suggest_categorical("activation", ["ReLU", "SiLU"]),
        "temperature": trial.suggest_float("temperature", 0.5, 0.9),
        "sigma": trial.suggest_float("sigma", 1.0, 3.5),
        "lr": trial.suggest_float("lr", 1e-3, 5e-3, log=True),
        "batch_size": 256,                     # fixed: 88% of top trials
        "win_stride_mult": trial.suggest_categorical("win_stride_mult", [2, 4]),
        "aug_noise": trial.suggest_float("aug_noise", 0.01, 0.08),
        "aug_scale": trial.suggest_float("aug_scale", 0.05, 0.4),
        "aug_shift": trial.suggest_int("aug_shift", 5, 30),
        "aug_drop": trial.suggest_int("aug_drop", 10, 80),
    }
    dmax = max(cfg["dilation"])
    eff = (win + 2 * (dmax * (k // 2)) - dmax * (k - 1) - 1) // cfg["stride"] + 1
    if eff < 4:
        raise optuna.TrialPruned()
    if cfg["conv_channels"] * len(cfg["dilation"]) > 128:
        raise optuna.TrialPruned()
    return cfg


def channel_array(ch_idx):
    """One channel's contiguous [N, 512] memmap."""
    p = os.path.join(HERE, f"public_pool_ch{ch_idx:02d}.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    raise FileNotFoundError(f"{p} — run split_pool_channels.py first")


def run_channel(ch_idx, ch_name, X, meta, device, n_trials):
    ages = meta["age"]
    subjects = meta["subject"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * VAL_FRACTION), replace=False))
    is_va = np.array([s in va_subs for s in subjects])
    Xc = channel_array(ch_idx)                # contiguous per-channel memmap

    study = optuna.create_study(direction="minimize", study_name=f"ch_{ch_name}",
                                storage=get_storage(), load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=C.SEED,
                                                                   n_startup_trials=8))

    def objective(trial):
        cfg = suggest(trial)
        m = cfg["win_stride_mult"]
        tr = torch.as_tensor(np.flatnonzero(~is_va)[::m].copy())
        va = torch.as_tensor(np.flatnonzero(is_va)[::m].copy())
        try:
            res = evaluate_cfg(cfg, Xc, ages, subjects, tr, va, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); raise optuna.TrialPruned()
        except Exception:
            torch.cuda.empty_cache(); raise optuna.TrialPruned()
        if not np.isfinite(res["score"]):
            raise optuna.TrialPruned()
        for k_, v_ in res.items():
            trial.set_user_attr(k_, v_)
        return res["score"]

    done = len([t for t in study.trials if t.state.is_finished()])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done,
                       catch=(RuntimeError, ValueError))
    best = study.best_trial
    return {"channel": ch_name, "n_trials": len(study.trials),
            "best_params": best.params, "best_score": best.value,
            **{k: best.user_attrs.get(k) for k in
               ("val_loss", "alignment", "uniformity", "emb_std", "dead_frac",
                "knn_mae", "slope", "subject_mae", "corr_r")}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--parallel", type=int, default=0,
                    help="spawn N worker processes, distributing channels between them")
    ap.add_argument("--vram-frac", type=float, default=0.0)
    a = ap.parse_args()

    if a.parallel > 1:
        # Split the 14 channels across N processes. Studies live in shared SQLite,
        # so each channel's search is independent and resumable; workers never
        import subprocess
        with open(C.CONFIG_JSON) as f:
            names = [c.replace("EEG.", "") for c in json.load(f)["EEG_columns"]]
        if a.channels:
            names = [n for n in names if n in a.channels]
        init_studies(names)          # schema must exist before any worker starts
        shards = [names[i::a.parallel] for i in range(a.parallel)]
        procs = []
        for sh in shards:
            if not sh:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--channels", *sh,
                   "--trials", str(a.trials),
                   "--vram-frac", str(round(0.92 / a.parallel, 3))]
            procs.append(subprocess.Popen(cmd, env={**os.environ,
                                                    "CUDA_VISIBLE_DEVICES": "0"}))
            print(f"  worker -> channels {sh}", flush=True)
            __import__("time").sleep(8)   # stagger so workers don't collide on SQLite
        for pr in procs:
            pr.wait()
        print("all workers finished")
        return

    if a.vram_frac:
        try:
            torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
        except Exception:
            pass
    device = torch.device("cuda:0")
    with open(C.CONFIG_JSON) as f:
        names = [c.replace("EEG.", "") for c in json.load(f)["EEG_columns"]]
    import pickle
    with open(os.path.join(HERE, "public_pool_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    X = None                                  # per-channel files are used instead
    _n = len(meta["age"])
    print(f"pool: ({_n}, 512) per channel | {len(set(meta['subject']))} subjects | "
          f"age {meta['age'].min():.0f}-{meta['age'].max():.0f} | recipe {meta['recipe']}",
          flush=True)

    targets = [(i, n) for i, n in enumerate(names) if not a.channels or n in a.channels]
    out = {}
    for fn in os.listdir(OUT_DIR):          # pick up other workers' finished channels
        if fn.endswith(".json"):
            try:
                out[fn[:-5]] = json.load(open(os.path.join(OUT_DIR, fn)))
            except Exception:
                pass
    for i, n in targets:
        print(f"\n=== channel {n} ({i+1}/{len(names)}) ===", flush=True)
        r = run_channel(i, n, X, meta, device, a.trials)
        out[n] = r
        with open(os.path.join(OUT_DIR, f"{n}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"  best score {r['best_score']:.3f} | kNN-MAE {r['knn_mae']:.2f} | "
              f"slope {r['slope']:+.3f} | align {r['alignment']:.3f} | "
              f"unif {r['uniformity']:.2f} | r {r['corr_r']:+.3f}", flush=True)
    for fn in os.listdir(OUT_DIR):
        if fn.endswith(".json"):
            try:
                out.setdefault(fn[:-5], json.load(open(os.path.join(OUT_DIR, fn))))
            except Exception:
                pass
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'ch':5s}{'score':>8}{'kNN-MAE':>9}{'slope':>8}{'subjMAE':>9}"
          f"{'r':>7}{'align':>8}{'unif':>8}{'val_loss':>10}{'dead':>7}")
    for n, r in sorted(out.items(), key=lambda kv: kv[1]["best_score"]):
        print(f"{n:5s}{r['best_score']:>8.3f}{r['knn_mae']:>9.2f}{r['slope']:>8.3f}"
              f"{r['subject_mae']:>9.2f}{r['corr_r']:>7.3f}{r['alignment']:>8.3f}"
              f"{r['uniformity']:>8.2f}{r['val_loss']:>10.3f}{r['dead_frac']:>7.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
