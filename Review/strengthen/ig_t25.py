"""Integrated Gradients for the shipped t25 model (log-euclid head, no attention).

Replaces MainFigures/F7/conv_features_ig.py, which attributed through AttAgeModel.

USAGE: python ig_t25.py [--ckpt ../results/model_age_phase4_t25_ep10.pt] [--nper 48]
"""
import argparse, os
import numpy as np
import torch

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")
os.environ.setdefault("AGE_GAP_ON", "val")
import age_model_search as A
from captum.attr import IntegratedGradients

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
BINS = [(20, 29), (30, 39), (40, 49), (50, 62)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../results/model_age_phase4_t25_ep10.pt")
    ap.add_argument("--nper", type=int, default=48, help="windows sampled per age bin")
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()

    # cudnn's fused GRU refuses to backward in eval mode. the non-cudnn path
    # does, and keeps eval semantics (running BN stats, dropout off).
    torch.backends.cudnn.enabled = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.eval()
    # the encoders are wrapped in no_grad when freeze=True, which would zero every
    # attribution. turn it off for the backward pass only.
    was = getattr(model, "freeze", None)
    model.freeze = False
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"loaded {a.ckpt}  (freeze {was} -> False for attribution)")

    d = A.inhouse_data("locked")
    if d is None:
        raise SystemExit("no in-house windows")
    X, ages, subs = d
    print(f"in-house locked: {X.shape[0]} windows, {len(set(subs))} subjects")

    def fwd(x):
        y = model(x)
        return y.squeeze(-1) if y.ndim > 1 else y

    ig = IntegratedGradients(fwd)
    rng = np.random.default_rng(0)
    out, counts = [], []
    per_win, per_age = [], []        # per-window mean|IG| per channel, for CIs
    for lo, hi in BINS:
        idx = np.where((ages >= lo) & (ages <= hi))[0]
        if len(idx) == 0:
            out.append(np.zeros((512, 14))); counts.append(0); continue
        take = rng.choice(idx, size=min(a.nper, len(idx)), replace=False)
        acc = []
        for s in range(0, len(take), 8):
            xb = torch.tensor(X[take[s:s + 8]], dtype=torch.float32, device=dev)
            xb.requires_grad_(True)
            at = ig.attribute(xb, baselines=torch.zeros_like(xb),
                              n_steps=a.steps, internal_batch_size=8)
            acc.append(at.detach().cpu().numpy())
        A_ = np.concatenate(acc, 0)              # [n, 512, 14]
        per_win.append(np.abs(A_).mean(1))       # [n, 14]
        per_age.append(ages[take])
        out.append(A_.mean(0)); counts.append(len(take))
        print(f"  {lo}-{hi}: {len(take)} windows | mean|IG| {np.abs(A_).mean():.3e}")
    M = np.stack(out)                            # [bins, 512, 14]
    np.savez(os.path.join(RES, "ig_t25.npz"), M=M, counts=np.array(counts),
             bins=np.array([f"{l}-{h}" for l, h in BINS]), ch=np.array(CH),
             W=np.concatenate(per_win, 0), Wage=np.concatenate(per_age, 0))
    print("wrote ../results/ig_t25.npz", M.shape)
    if was is not None:
        model.freeze = was


if __name__ == "__main__":
    main()
