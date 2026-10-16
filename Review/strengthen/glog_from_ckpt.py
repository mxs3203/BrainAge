"""Extract per-subject Glog from an ALREADY-TRAINED checkpoint.
USAGE: python glog_from_ckpt.py --ckpt ../results/model_age_phase4_t25_ep10_newdata.pt \
                                --out  age_phase4_t25_newdata --split locked
"""
import argparse, os
import numpy as np
import torch

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")

import age_model_search as A
from channel_interaction import subject_glog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True,
                    help="name used in glog_<out>_<split>.npy")
    ap.add_argument("--split", default="locked", choices=["locked", "val", "fit"])
    a = ap.parse_args()
    print(f"pulling glog out of {a.ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.eval()
    print(f"loaded {a.ckpt}")

    d = A.inhouse_data(a.split)
    print(f"extracting Glog for in-house '{a.split}' ...", flush=True)
    per = subject_glog(model, d, device)
    subs = sorted(per)
    M = np.stack([per[s][0] for s in subs])
    age = np.array([per[s][1] for s in subs])
    np.save(f"../results/glog_{a.out}_{a.split}.npy", M)
    np.save(f"../results/glog_{a.out}_{a.split}_ages.npy", age)
    print(f"  {len(subs)} subjects -> ../results/glog_{a.out}_{a.split}.npy "
          f"{M.shape}")


if __name__ == "__main__":
    main()
