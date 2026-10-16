"""Train the sweep's best JOINT config and dump per-subject predictions.
USAGE: python predict_best.py [--config age_phase6_t4] [--seeds 1]
"""
import argparse, json, os
import numpy as np
import pandas as pd

os.environ.setdefault("HARMONIZED_F", "1")
os.environ.setdefault("AGE_IH_W_LOCK", "0")
os.environ.setdefault("AGE_IH_W_ALL", "0")
os.environ.setdefault("AGE_GAP_ON", "val")

import optuna
import age_model_search as A

optuna.logging.set_verbosity(optuna.logging.WARNING)
DB = "sqlite:///age_model_optuna.db"


def get_cfg(name):
    """name like 'age_phase6_t4' -> that trial's params, verbatim."""
    study, n = name.rsplit("_t", 1)
    t = [x for x in optuna.load_study(study_name=study, storage=DB).trials
         if x.number == int(n)][0]
    cfg = dict(t.params)
    # suggest() sets these two as CONSTANTS, so they never reach trial.params and
    # a config rebuilt from params alone is incomplete. Verified by inspection:
    cfg.setdefault("train_unit", "bag")
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("pool_mix", "joint")
    return cfg, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase6_t4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, so a variant run does not "
                         "overwrite the baseline table")
    ap.add_argument("--at-epoch", type=int, default=0, dest="at_epoch",
                    help="capture predictions at exactly this epoch instead of "
                         "the objective's best epoch (which for t25 is epoch 2, "
                         "a different model from the one the paper reports)")
    ap.add_argument("--train-preds", action="store_true",
                    help="also predict the 539-subject public TRAINING pool "
                         "(in-sample; for the training panel of the 6-panel figure)")
    a = ap.parse_args()

    cfg, t = get_cfg(a.config)
    print(f"config {a.config}: head={cfg['head']} pool={cfg['pool_mix']} "
          f"ovs={cfg.get('ih_oversample')} bag_k={cfg.get('bag_k')}")
    print(f"  as recorded in the sweep: IHtest r {t.user_attrs.get('test_r'):+.3f} "
          f"MAE {t.user_attrs.get('test_mae'):.2f}")

    POOLS, va, device, X, ages, subjects, datasets = A.prepare()
    A.EPOCHS = a.epochs
    A.PRED_SINK = {}
    A.PRED_TRAIN = bool(a.train_preds)
    A.PRED_AT_EPOCH = a.at_epoch or None
    m = A.run_fold(cfg, POOLS, va, device, X, ages, subjects, datasets)
    sink = A.PRED_SINK

    rows = []
    pub = sink["public"].rename(columns={"s": "subject"})
    pub["cohort"] = "public: " + pub["dataset"].astype(str)
    pub["split"] = "public_val"
    rows.append(pub[["subject", "t", "p", "cohort", "split"]])
    if "public_train" in sink:
        tp = sink["public_train"].rename(columns={"s": "subject"})
        tp["cohort"] = "public: " + tp["dataset"].astype(str)
        tp["split"] = "public_train"      # IN the training loss, in-sample
        rows.append(tp[["subject", "t", "p", "cohort", "split"]])
    for key, lbl, split in (("ih_fit", "in-house: FIT (in training)", "ih_fit"),
                            ("ih_val", "in-house: VAL (selection)", "ih_val"),
                            ("ih_locked", "in-house: LOCKED (held out)", "ih_locked")):
        f = sink.get(key)
        if f is None or not len(f):
            continue
        # _subject_frame() returns the subject as the INDEX, not a column
        d = f.reset_index().rename(columns={"s": "subject", "index": "subject"})
        d["cohort"], d["split"] = lbl, split
        rows.append(d[["subject", "t", "p", "cohort", "split"]])
    df = pd.concat(rows, ignore_index=True)
    out = (f"../results/Table_pred_joint_{a.config}"
           f"{('_' + a.tag) if a.tag else ''}.csv")
    df.to_csv(out, index=False)

    print(f"\nepoch used: {sink['epoch']}")
    print(f"{'split':12s} {'n':>4} {'r':>7} {'MAE':>7} {'slope':>7}")
    for sp, g in df.groupby("split"):
        if len(g) < 3:
            continue
        r = np.corrcoef(g.t, g.p)[0, 1]
        sl = np.polyfit(g.t, g.p - g.t, 1)[0]
        print(f"{sp:12s} {len(g):4d} {r:+7.3f} {np.abs(g.p-g.t).mean():7.2f} {sl:+7.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
