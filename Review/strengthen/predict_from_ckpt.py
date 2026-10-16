"""Per-subject predictions from an ALREADY-TRAINED checkpoint.
USAGE: python predict_from_ckpt.py \
         --ckpt ../results/model_age_phase4_t25_ep10.pt \
         --config age_phase4_t25 --tag oldmodel_newdata --train-preds
"""
import argparse, os
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")
os.environ.setdefault("AGE_GAP_ON", "val")

import age_model_search as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="age_phase4_t25",
                    help="only used to name the output file")
    ap.add_argument("--tag", default="oldmodel_newdata")
    ap.add_argument("--train-preds", action="store_true", dest="train_preds",
                    help="also predict the 539-subject public TRAINING pool "
                         "(in-sample; panel A of the pred figures needs it)")
    a = ap.parse_args()
    print(f"scoring {a.ckpt} on the current arrays (no retrain)")

    POOLS, va, device, X_eval, ages_eval, subj_eval, ds_eval = A.prepare()
    P = POOLS["2ds"]
    X, ages, subjects, datasets, tr = (P["X"], P["ages"], P["subjects"],
                                       P["datasets"], P["tr"])
    tr_map = A.subject_index(tr.numpy(), subjects)
    va_map = A.subject_index(va.numpy(), subj_eval)

    model = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.eval()
    print(f"loaded {a.ckpt}")

    rows = []

    # ---- public validation, the shared 179 subjects ----------------------
    sv, p = A.bag_predict(model, X_eval, va_map, device)
    yv = np.array([ages_eval[va_map[k][0]] for k in sv], dtype=float)
    dsv = np.array([str(ds_eval[va_map[k][0]]) for k in sv])
    d = pd.DataFrame({"subject": sv, "t": yv, "p": np.asarray(p).ravel(),
                      "cohort": "public: " + pd.Series(dsv).astype(str),
                      "split": "public_val"})
    rows.append(d)

    # ---- public TRAINING pool: in-sample, shown only as the memorisation
    #      panel, never as a result
    if a.train_preds:
        ts, tp = A.bag_predict(model, X, tr_map, device)
        rows.append(pd.DataFrame({
            "subject": ts,
            "t": [float(ages[tr_map[k][0]]) for k in ts],
            "p": np.asarray(tp).ravel(),
            "cohort": ["public: " + str(datasets[tr_map[k][0]]) for k in ts],
            "split": "public_train"}))

    # ---- in-house, one frame per split -----------------------------------
    for key, lbl, split in (("fit", "in-house: FIT (in training)", "ih_fit"),
                            ("val", "in-house: VAL (selection)", "ih_val"),
                            ("locked", "in-house: LOCKED (held out)", "ih_locked")):
        ihd = A.inhouse_data(key)
        if not ihd:
            continue
        m, g = A.inhouse_metrics(model, ihd[0], ihd[1], ihd[2], device,
                                 bag=True, return_frame=True)
        if g is None or not len(g):
            continue
        gg = g.reset_index().rename(columns={"s": "subject", "index": "subject"})
        gg["cohort"], gg["split"] = lbl, split
        rows.append(gg[["subject", "t", "p", "cohort", "split"]])

    df = pd.concat(rows, ignore_index=True)
    out = f"../results/Table_pred_joint_{a.config}_{a.tag}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'split':12s} {'n':>4} {'r':>7} {'MAE':>7} {'slope':>7}")
    for sp, g in df.groupby("split"):
        if len(g) < 3:
            continue
        r = float(np.corrcoef(g.t, g.p)[0, 1])
        mae = float(np.abs(g.p - g.t).mean())
        sl = float(np.polyfit(g.t, g.p - g.t, 1)[0])
        print(f"{sp:12s} {len(g):4d} {r:+7.3f} {mae:7.2f} {sl:+7.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
