"""Stage 2: CNN / GRU architecture search per channel, everything else frozen."""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.environ.get("ARCH_TRIALS", 90))
FLOOR = 14.07                       # no-information subject MAE of the public pool
# Information objective: the encoder carries age information, the downstream
# attention model handles calibration and absolute error.
R_W, KNN_W, SLOPE_W, XFER_W, IH_W, HARD_W = 1.0, 0.5, 1.0, 0.75, 1.5, 5.0
DB_PATH = os.path.join(HERE, "channel_encoder_arch_optuna.db")
STORAGE = "sqlite:///" + DB_PATH + "?timeout=120"
_FZ = os.path.join(HERE, "frozen_hparams.json")
# No frozen_hparams.json -> FULL mode: every scalar is searched too, becuase
# there is no earlier stage on this data to freeze from.
FROZEN = json.load(open(_FZ)) if os.path.exists(_FZ) else None
FULL = FROZEN is None
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


_IH = {}


def inhouse_windows(ch_idx):
    """In-house eyes-closed windows for one channel, loaded once per worker."""
    key = "data" if _IH.get("_mode", "ec") == "ec" else "all"
    if "data" not in _IH:
        os.environ.setdefault("HARMONIZED_F", "1")
        sys.path.insert(0, os.path.dirname(HERE))
        from harmonized_io import build_windows, load_dataset
        from stimulus_groups import group as _g
        _, recs = load_dataset("ours")
        ec = [x for x in recs.stimulus.unique() if _g("ours", x) == "rest_ec"]
        w = build_windows("ours", recipe="prefiltered_uv", win=512, stride=256,
                          stimuli=ec, verbose=False)
        _IH["data"] = None if w is None else (
            w["X"], np.asarray(w["subject"]).astype(str),
            np.asarray(w["age"], float))
        if _IH["data"] is not None:
            print(f"  in-house eyes-closed: {len(w['X'])} windows, "
                  f"{len(set(w['subject']))} subjects", flush=True)
        # ALL in-house stimuli as well. Only eyes-closed enters the objective --
        # it is the condition the encoders train on, so it is the comparable one
        wa = build_windows("ours", recipe="prefiltered_uv", win=512, stride=256,
                           verbose=False)
        _IH["all"] = None if wa is None else (
            wa["X"], np.asarray(wa["subject"]).astype(str),
            np.asarray(wa["age"], float), np.asarray(wa["stimulus"]).astype(str))
        if _IH["all"] is not None:
            print(f"  in-house all stimuli: {len(wa['X'])} windows, "
                  f"{len(set(wa['stimulus']))} stimuli", flush=True)
    d = _IH["data"]
    return None if d is None else (np.ascontiguousarray(d[0][:, :, ch_idx]), d[1], d[2])


def get_storage():
    return optuna.storages.RDBStorage(
        url=STORAGE, engine_kwargs={"connect_args": {"timeout": 120}})


class ChannelEncoderV2(nn.Module):
    """Stage-2 encoder: the published Conv+recurrent block with the architecture"""

    def __init__(self, conv_channels=16, kernel_size=7, stride=2, padding=None,
                 dilation=(1,), gru_hidden=16, num_layers_gru=1, embedding_dim=16,
                 cnn_dropout=0.05, rnn_dropout=0.0, activation="ReLU",
                 conv_layers=1, branch_mix="concat", norm="none", use_aux=False):
        super().__init__()
        act = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}[activation]
        dils = (dilation,) if isinstance(dilation, int) else tuple(dilation)
        self.dilations = dils

        def nrm():
            if norm == "instance":
                return [nn.InstanceNorm1d(conv_channels, affine=True)]
            if norm == "batch":
                return [nn.BatchNorm1d(conv_channels)]
            if norm == "group":
                return [nn.GroupNorm(min(4, conv_channels), conv_channels)]
            return []

        def branch(d):
            layers = [nn.Conv1d(1, conv_channels, kernel_size=kernel_size,
                                stride=stride, padding=d * (kernel_size // 2),
                                dilation=d)] + nrm() + [act(), nn.Dropout1d(p=cnn_dropout)]
            for _ in range(conv_layers - 1):
                layers += [nn.Conv1d(conv_channels, conv_channels,
                                     kernel_size=kernel_size, stride=1,
                                     padding=d * (kernel_size // 2), dilation=d)] \
                          + nrm() + [act(), nn.Dropout1d(p=cnn_dropout)]
            return nn.Sequential(*layers)

        self.branches = nn.ModuleList([branch(d) for d in dils])
        wide = conv_channels * len(dils)
        if branch_mix == "proj" and len(dils) > 1:
            self.mix = nn.Sequential(nn.Conv1d(wide, conv_channels, 1), act())
            rnn_in = conv_channels
        else:
            self.mix, rnn_in = None, wide

        self.gru = nn.GRU(rnn_in, gru_hidden, num_layers=num_layers_gru,
                          batch_first=True, bidirectional=True,
                          dropout=rnn_dropout if num_layers_gru > 1 else 0.0)
        self.use_aux = use_aux
        self.out_dim = gru_hidden * 2
        self.projector = nn.Linear(self.out_dim + (1 if use_aux else 0), embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # measured before the conv stack, so InstanceNorm cannot erase it
        aux = (x.float().std(dim=-1).clamp_min(1e-6).log10()
               if self.use_aux else None)
        h = torch.cat([b(x) for b in self.branches], dim=1)
        if h.shape[-1] < 2:
            raise ValueError("conv collapsed the time axis")
        if self.mix is not None:
            h = self.mix(h)
        _, hn = self.gru(h.permute(0, 2, 1))
        out = torch.cat([hn[-2], hn[-1]], dim=1)   # bidirectional final states
        z_in = torch.cat([out, aux.to(out.dtype)], dim=1) if self.use_aux else out
        return F.normalize(self.projector(z_in), dim=1), out


ARCH_BAND_SETS = dict(S.BAND_SETS)
ARCH_BAND_SETS.update({
    "theta_alpha": ["theta", "alpha"],
    "alpha_beta":  ["alpha", "beta_l"],
    "mid4":        ["theta", "alpha", "beta_l", "beta_h"],
    "beta_only":   ["beta_l"],
    "alpha_gamma": ["alpha", "gamma"],
})
KERNELS = [5, 7, 9, 11, 13, 15, 21]


def suggest_scalars(trial):
    """Scalars that stage 2 normally inherits frozen. Searched here because the"""
    return {
        "stride": trial.suggest_categorical("stride", [2, 4]),
        "activation": trial.suggest_categorical("activation", ["ReLU", "SiLU"]),
        "cnn_dropout": trial.suggest_float("cnn_dropout", 0.0, 0.20),
        "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.20),
        "temperature": trial.suggest_float("temperature", 0.5, 1.2),
        "sigma": trial.suggest_float("sigma", 0.3, 3.5),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "win_stride_mult": trial.suggest_categorical("win_stride_mult", [2, 4]),
        "aug_noise": trial.suggest_float("aug_noise", 0.01, 0.10),
        "aug_scale": trial.suggest_float("aug_scale", 0.05, 0.40),
        "aug_shift": trial.suggest_int("aug_shift", 5, 30),
        "aug_drop": trial.suggest_int("aug_drop", 10, 80),
    }


def suggest_arch(trial, ch, win=512):
    fz = suggest_scalars(trial) if FULL else FROZEN[ch]["frozen"]
    k = trial.suggest_categorical("kernel_size", KERNELS)
    bands = trial.suggest_categorical("band_set", list(ARCH_BAND_SETS))
    d = S.band_dilations(ARCH_BAND_SETS[bands], k)
    mix = trial.suggest_categorical("branch_mix", ["concat", "proj"])
    cfg = {
        "kernel_size": k, "dilation": d, "band_set": bands, "padding": None,
        "conv_channels": trial.suggest_categorical("conv_channels", [8, 16, 24, 32, 48]),
        "gru_hidden": trial.suggest_categorical("gru_hidden", [8, 16, 24, 32, 48, 64]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [8, 16, 24, 32, 48, 64]),
        "num_layers_gru": 1,
        "batch_size": 256,
        "arch_extra": {
            "conv_layers": trial.suggest_categorical("conv_layers", [1, 2]),
            "branch_mix": mix,
            # Normalisation layer inside the encoder. It lives HERE, not in
            # suggest_scalars, because suggest_scalars only runs in FULL mode --
            "norm": trial.suggest_categorical(
                "enc_norm", ["none", "instance", "batch", "group"]),
        },
    }
    cfg.update(fz)                   # frozen scalars win over anything above
    dmax = max(d)
    eff = (win + 2 * (dmax * (k // 2)) - dmax * (k - 1) - 1) // cfg["stride"] + 1
    if eff < 4:
        raise optuna.TrialPruned()
    # the 128 cap applies to what the recurrent layer actually receives, so a 1x1
    # mix lets the wide band sets through instead of being pruned outright
    rnn_in = cfg["conv_channels"] if (mix == "proj" and len(d) > 1) \
        else cfg["conv_channels"] * len(d)
    if rnn_in > 128:
        raise optuna.TrialPruned()
    return cfg


def probe_transfer(Ztr, ytr, s_tr, Zte, yte, s_te):
    """Fit a subject-level probe on one cohort, score it on another."""
    dtr = pd.DataFrame(Ztr); dtr["s"] = s_tr
    dte = pd.DataFrame(Zte); dte["s"] = s_te
    Mtr, Mte = dtr.groupby("s").mean(), dte.groupby("s").mean()
    atr = pd.Series(ytr).groupby(pd.Series(s_tr)).first().loc[Mtr.index].to_numpy()
    ate = pd.Series(yte).groupby(pd.Series(s_te)).first().loc[Mte.index].to_numpy()
    if len(ate) < 10 or np.std(ate) < 2:
        return 0.0
    sc = StandardScaler().fit(Mtr.to_numpy())
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr.to_numpy()), atr)
    p_ = mdl.predict(sc.transform(Mte.to_numpy()))
    return 0.0 if np.std(p_) < 1e-9 else float(np.corrcoef(ate, p_)[0, 1])


def cross_cohort_r(Z, ages, subjects, datasets):
    """Fit on one cohort's subjects, predict the other's. Mean r over both
    directions. Returns 0.0 when only one cohort is present."""
    d = pd.DataFrame(Z)
    d["s"] = subjects
    M = d.groupby("s").mean()
    a_ = pd.Series(ages).groupby(pd.Series(subjects)).first().loc[M.index].to_numpy()
    ds = pd.Series(datasets).groupby(pd.Series(subjects)).first().loc[M.index].to_numpy()
    M = M.to_numpy()
    out = []
    uniq = [u for u in np.unique(ds) if (ds == u).sum() >= 20]
    for a_ds in uniq:
        for b_ds in uniq:
            if a_ds == b_ds:
                continue
            ka, kb = ds == a_ds, ds == b_ds
            if np.std(a_[ka]) < 2 or np.std(a_[kb]) < 2:
                continue
            sc = StandardScaler().fit(M[ka])
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(M[ka]), a_[ka])
            p_ = mdl.predict(sc.transform(M[kb]))
            if np.std(p_) > 1e-9:
                out.append(float(np.corrcoef(a_[kb], p_)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def subject_probe(Ztr, ytr, s_tr, Zva, yva, s_va):
    """Probe C: average embeddings per subject, then fit at subject level."""
    dtr = pd.DataFrame(Ztr); dtr["s"] = s_tr
    dva = pd.DataFrame(Zva); dva["s"] = s_va
    Mtr, Mva = dtr.groupby("s").mean(), dva.groupby("s").mean()
    atr = pd.Series(ytr).groupby(pd.Series(s_tr)).first().loc[Mtr.index].to_numpy()
    ava = pd.Series(yva).groupby(pd.Series(s_va)).first().loc[Mva.index].to_numpy()
    sc = StandardScaler().fit(Mtr.to_numpy())
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr.to_numpy()), atr)
    p = mdl.predict(sc.transform(Mva.to_numpy()))
    return (float(np.polyfit(ava, p - ava, 1)[0]),
            float(np.abs(p - ava).mean()),
            float(np.corrcoef(ava, p)[0, 1]),
            float(p.std() / ava.std()))


def run_channel(ch_idx, ch, meta, device, n_trials):
    ages, subjects = meta["age"], meta["subject"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * S.VAL_FRACTION), replace=False))
    is_va = np.array([s in va_subs for s in subjects])
    Xc = S.channel_array(ch_idx)
    study = optuna.create_study(direction="minimize", study_name=f"ch_{ch}",
                                storage=get_storage(), load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(
                                    seed=C.SEED, n_startup_trials=10))

    def objective(trial):
        cfg = suggest_arch(trial, ch)
        m = cfg["win_stride_mult"]
        tr = torch.as_tensor(np.flatnonzero(~is_va)[::m].copy())
        va = torch.as_tensor(np.flatnonzero(is_va)[::m].copy())
        try:
            ihw = inhouse_windows(ch_idx)
            iha = _IH.get("all")
            ihaX = (np.ascontiguousarray(iha[0][:, :, ch_idx])
                    if iha is not None else None)
            # one embedding pass over both: eyes-closed first, then all stimuli
            ex = (np.concatenate([ihw[0], ihaX]) if (ihw and ihaX is not None)
                  else (ihw[0] if ihw else None))
            res = S.evaluate_cfg(cfg, Xc, ages, subjects, tr, va, device,
                                 return_emb=True, encoder_cls=ChannelEncoderV2,
                                 extra_X=ex)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); raise optuna.TrialPruned()
        except Exception:
            torch.cuda.empty_cache(); raise optuna.TrialPruned()
        E = res.pop("_emb")
        slope, mae, r, sdr = subject_probe(E["Ztr"], E["ytr"], E["str"],
                                           E["Zva"], E["yva"], E["sva"])
        kmae = res["knn_mae"]
        # transferability, measured on the validation cohorts only
        xfer = cross_cohort_r(E["Zva"], E["yva"], E["sva"],
                              meta["dataset"][va.numpy()])
        trial.set_user_attr("cross_cohort_r", xfer)
        # transfer to teh in-house cohort: probe fitted on the public TRAIN
        # subjects, applied to the in-house participants
        ih_r = 0.0
        Zall = E.get("Zextra")
        if ihw is not None and Zall is not None:
            n_ec = len(ihw[0])
            Zih = Zall[:n_ec]
            ih_r = probe_transfer(E["Ztr"], E["ytr"], E["str"],
                                  Zih, ihw[2], ihw[1])
            # per-stimulus, over every in-house condition. Logged only -- teh
            # objective still sees eyes-closed alone.
            if iha is not None and len(Zall) > n_ec:
                Za = Zall[n_ec:]
                per = {}
                for st_ in np.unique(iha[3]):
                    m = iha[3] == st_
                    if m.sum() < 100 or len(np.unique(iha[1][m])) < 20:
                        continue
                    rr = probe_transfer(E["Ztr"], E["ytr"], E["str"],
                                        Za[m], iha[2][m], iha[1][m])
                    per[str(st_)] = round(float(rr), 4)
                if per:
                    trial.set_user_attr("inhouse_per_stimulus", per)
                    trial.set_user_attr("inhouse_all_r",
                                        probe_transfer(E["Ztr"], E["ytr"], E["str"],
                                                       Za, iha[2], iha[1]))
                    trial.set_user_attr("inhouse_best_stim", max(per, key=per.get))
                    trial.set_user_attr("inhouse_best_stim_r", max(per.values()))

        if not np.isfinite(slope) or not np.isfinite(mae) or not np.isfinite(kmae):
            raise optuna.TrialPruned()
        # INFORMATION objective. The encoder's job is to carry age information;
        # calibration and absolute error are the downstream attention model's job,
        norm = mae / FLOOR
        score = (-R_W * r + KNN_W * (kmae / FLOOR)
                 + SLOPE_W * abs(slope)
                 - XFER_W * xfer
                 + HARD_W * max(0.0, norm - 1.0)
                 + 0.5 * max(0.0, res["alignment"] - S.TARGET_ALIGN)
                 + 0.5 * max(0.0, res["uniformity"] - S.TARGET_UNIF)
                 + 5.0 * float(res["collapsed"]))
        for k_, v_ in res.items():
            trial.set_user_attr(k_, v_)
        # subject-level probe metrics, kept under distinct names so the old
        # window-level numbers stay comparable in the dashboard
        trial.set_user_attr("c_slope", slope)
        trial.set_user_attr("c_subject_mae", mae)
        trial.set_user_attr("c_r", r)
        trial.set_user_attr("c_sd_ratio", sdr)
        return float(score)

    # n_trials is a TARGET for the study, not a number to add. Without this a
    # resumed run would push allready-finished channels from 100 to 200 trials
    done = len([t for t in study.trials if t.state.name != "RUNNING"])
    remaining = max(0, n_trials - done)
    if remaining == 0:
        print(f"[{ch}] already at {done}/{n_trials} trials, skipping", flush=True)
        return
    if done:
        print(f"[{ch}] resuming: {done}/{n_trials} done, running {remaining} more",
              flush=True)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    b = study.best_trial
    print(f"[{ch}] best score {b.value:.3f} | slope {b.user_attrs['c_slope']:+.3f} "
          f"| MAE {b.user_attrs['c_subject_mae']:.2f} | r {b.user_attrs['c_r']:.3f} "
          f"| emb {b.params['embedding_dim']} gru {b.params['gru_hidden']} "
          f"conv {b.params['conv_channels']} k {b.params['kernel_size']} "
          f"{b.params['band_set']} "
          f"d{S.band_dilations(ARCH_BAND_SETS[b.params['band_set']], b.params['kernel_size'])}"
          f" cv{b.params['conv_layers']} {b.params['branch_mix']}", flush=True)


def init_studies():
    """Pre-create every study in the parent. Workers racing to create the schema
    is what produced 'table studies already exists' and wiped 21 trials before."""
    st = get_storage()
    for ch in CHANNELS:
        optuna.create_study(direction="minimize", study_name=f"ch_{ch}",
                            storage=st, load_if_exists=True)


def main():
    shard = int(os.environ.get("ARCH_SHARD", 0))
    n_shard = int(os.environ.get("ARCH_NSHARD", 1))
    if os.environ.get("ARCH_INIT"):
        init_studies()
        print(f"initialised {len(CHANNELS)} studies")
        return
    device = torch.device("cuda")
    _, meta = S.load_pool()
    mine = [(i, c) for i, c in enumerate(CHANNELS) if i % n_shard == shard]
    print(f"worker {shard}/{n_shard} -> {[c for _, c in mine]}", flush=True)
    for i, ch in mine:
        t0 = time.time()
        print(f"=== {ch} ===", flush=True)
        run_channel(i, ch, meta, device, N_TRIALS)
        print(f"=== {ch} done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
