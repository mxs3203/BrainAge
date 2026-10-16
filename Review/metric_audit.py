"""Metric reconciliation / audit (reviewer response W2; R1 comments 2 & 7).
Usage:
python metric_audit.py ../Modeling_ContrastiveChannelWise/results_all.csv
python metric_audit.py <preds.csv> --beta 1.0
"""
import argparse

import numpy as np
import pandas as pd


def smooth_l1(abs_err, beta=1.0):
    e = np.asarray(abs_err, dtype=float)
    return float(np.mean(np.where(e < beta, 0.5 * e ** 2 / beta, e - 0.5 * beta)))


def metrics(err, beta=1.0):
    err = np.asarray(err, dtype=float)
    ae = np.abs(err)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        f"SmoothL1(beta={beta})": smooth_l1(ae, beta),
        "n": int(len(err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_csv")
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)
    win_err = (df["predicted_age"] - df["true_age"]).values

    pp = df.groupby("ID").agg(true=("true_age", "first"),
                              pred=("predicted_age", "mean")).reset_index()
    subj_err = (pp["pred"] - pp["true"]).values

    win = metrics(win_err, args.beta)
    subj = metrics(subj_err, args.beta)

    print(f"File: {args.preds_csv}")
    print(f"  windows: {len(df):,}   participants: {df['ID'].nunique()}\n")
    hdr = f"{'level':<14}{'MAE':>8}{'RMSE':>9}{'SmoothL1':>11}"
    print(hdr); print("-" * len(hdr))
    for name, m in [("per-window", win), ("per-subject", subj)]:
        print(f"{name:<14}{m['MAE']:>8.3f}{m['RMSE']:>9.3f}"
              f"{m[f'SmoothL1(beta={args.beta})']:>11.3f}")

    # sanity check the reviewers' point
    ok_w = win["MAE"] >= win[f"SmoothL1(beta={args.beta})"]
    ok_s = subj["MAE"] >= subj[f"SmoothL1(beta={args.beta})"]
    print(f"\nSanity: MAE >= SmoothL1 at same level?  per-window {ok_w}   per-subject {ok_s}")
    print("(This must always be True for beta=1. The manuscript's 2.6>1.8 came from "
          "mixing per-window SmoothL1 with per-subject MAE.)")


if __name__ == "__main__":
    main()
