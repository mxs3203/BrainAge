"""Frozen vs fine-tuned encoders, with a concatenation head."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast, GradScaler

os.environ.setdefault("HARMONIZED_F", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402
from harmonized_io import build_windows  # noqa: E402
from stimulus_groups import group as sgroup  # noqa: E402

CKPT = os.path.join(HERE, "final_encoders")
RES = os.path.join(os.path.dirname(HERE), "results")
FLOOR = 14.07
VAL_STIM = ["__eyes_closed__", "phase_Stimuli_phase_15"]


class ConcatAgeModel(nn.Module):
    """14 channel encoders -> concatenated embeddings -> MLP -> age."""

    def __init__(self, encs, dims, hidden=128, dropout=0.2, freeze=True):
        super().__init__()
        self.encs = nn.ModuleList(encs)
        self.freeze = freeze
        if freeze:
            for e in self.encs:
                for p in e.parameters():
                    p.requires_grad_(False)
        d = sum(dims)
        self.head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))
                     if hidden else nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)))

    def forward(self, x):                       # x [B, 512, 14]
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            zs = [e(x[:, :, i].contiguous())[0] for i, e in enumerate(self.encs)]
        return self.head(torch.cat(zs, 1)).squeeze(-1)


def load_encoders(device):
    encs, dims = [], []
    for ch in A.CHANNELS:
        ck = torch.load(os.path.join(CKPT, f"{ch}.pt"), map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        e = A.ChannelEncoderV2(
            conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
            stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
            gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
            embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
            rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
            **cfg["arch_extra"])
        e.load_state_dict(ck["state_dict"])
        encs.append(e.to(device))
        dims.append(cfg["embedding_dim"])
    return encs, dims


def subj(y, p, s):
    d = pd.DataFrame({"s": s, "t": y, "p": p})
    return d.groupby("s").agg(t=("t", "first"), p=("p", "mean"))


def metrics(y, p, s):
    g = subj(y, p, s)
    if len(g) < 5 or g.p.std() < 1e-9:
        return {"r": 0.0, "mae": float(np.abs(g.p - g.t).mean()), "slope": -1.0}
    return {"r": float(np.corrcoef(g.t, g.p)[0, 1]),
            "mae": float(np.abs(g.p - g.t).mean()),
            "slope": float(np.polyfit(g.t, g.p - g.t, 1)[0])}


def cohort_transfer(y, p, s, ds):
    d = pd.DataFrame({"s": s, "t": y, "p": p, "d": ds})
    g = d.groupby("s").agg(t=("t", "first"), p=("p", "mean"), d=("d", "first"))
    out = []
    uniq = [u for u in g.d.unique() if (g.d == u).sum() >= 20]
    for a in uniq:
        for b in uniq:
            if a == b:
                continue
            ga, gb = g[g.d == a], g[g.d == b]
            if ga.t.std() < 2 or gb.t.std() < 2 or ga.p.std() < 1e-9:
                continue
            k = np.polyfit(ga.p, ga.t, 1)
            pb = np.polyval(k, gb.p)
            if np.std(pb) > 1e-9:
                out.append(float(np.corrcoef(gb.t, pb)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def predict(model, X, device, bs=256):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs]), device=device)
            with autocast(device_type="cuda"):
                ps.append(model(xb).float().cpu().numpy())
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda")

    X, meta = S.load_pool()
    ages, subjects, dsets = meta["age"], meta["subject"], meta["dataset"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * 0.25), replace=False))
    is_va = np.array([s in va_subs for s in subjects])
    tr, va = np.flatnonzero(~is_va), np.flatnonzero(is_va)
    print(f"public: {len(tr):,} train / {len(va):,} val windows, "
          f"{len(subs)} subjects", flush=True)

    w = build_windows("ours", recipe="prefiltered_uv", win=512, stride=256,
                      verbose=False)
    Xi = w["X"].astype(np.float32)
    yi = np.asarray(w["age"], float)
    si = np.asarray(w["subject"]).astype(str)
    sti = np.asarray(w["stimulus"]).astype(str)
    ec = np.array([sgroup("ours", x) == "rest_ec" for x in sti])
    p15 = sti == "phase_Stimuli_phase_15"
    chosen = ec | p15
    print(f"in-house: {len(Xi):,} windows | eyes-closed {ec.sum():,} "
          f"({len(set(si[ec]))} subj) | phase_15 {p15.sum():,} "
          f"({len(set(si[p15]))} subj) | chosen {chosen.sum():,}", flush=True)

    rows = []
    for freeze in (True, False):
        torch.manual_seed(C.SEED)
        encs, dims = load_encoders(device)
        model = ConcatAgeModel(encs, dims, hidden=args.hidden, freeze=freeze).to(device)
        groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
        if not freeze:
            groups.append({"params": model.encs.parameters(), "lr": args.lr_enc})
        opt = optim.AdamW(groups, weight_decay=1e-4)
        scaler = GradScaler(device="cuda")
        crit = nn.SmoothL1Loss()
        y_t = torch.as_tensor(ages, dtype=torch.float32)
        best = None
        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), args.batch):
                b = perm[i:i + args.batch]
                xb = torch.as_tensor(np.asarray(X[b]), device=device)
                opt.zero_grad()
                with autocast(device_type="cuda"):
                    pr = model(xb)
                loss = crit(pr.float(), y_t[b].to(device))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pv = predict(model, X[va], device)
            m = metrics(ages[va], pv, subjects[va])
            m["cross_cohort_r"] = cohort_transfer(ages[va], pv, subjects[va], dsets[va])
            pi = predict(model, Xi, device)
            for nm, msk in (("ec", ec), ("p15", p15), ("chosen", chosen),
                            ("all", np.ones(len(Xi), bool))):
                mi = metrics(yi[msk], pi[msk], si[msk])
                m[f"ih_{nm}_r"], m[f"ih_{nm}_mae"] = mi["r"], mi["mae"]
            # select on the chosen in-house conditions, as agreed
            key = m["ih_chosen_r"]
            if best is None or key > best["ih_chosen_r"]:
                best = dict(m, epoch=ep + 1)
            print(f"  [{'frozen' if freeze else 'finetune'}] ep{ep+1:3d} "
                  f"r {m['r']:.3f} MAE {m['mae']:5.2f} xcoh {m['cross_cohort_r']:+.3f} "
                  f"| IH ec {m['ih_ec_r']:+.3f} p15 {m['ih_p15_r']:+.3f} "
                  f"chosen {m['ih_chosen_r']:+.3f}", flush=True)
        best["mode"] = "frozen" if freeze else "finetuned"
        rows.append(best)
        del model, encs
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    cols = ["mode", "epoch", "r", "mae", "slope", "cross_cohort_r",
            "ih_ec_r", "ih_ec_mae", "ih_p15_r", "ih_p15_mae",
            "ih_chosen_r", "ih_chosen_mae", "ih_all_r"]
    os.makedirs(RES, exist_ok=True)
    df[cols].round(4).to_csv(os.path.join(RES, "Table_frozen_vs_finetune.csv"),
                             index=False)
    print("\n" + df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
