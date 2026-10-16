"""Freeze teh settled hyperparameters per channel, ready for the stage-2"""
import collections
import json
import os

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.environ.get("FREEZE_DB",
                                       "channel_encoder_arch_optuna.db"))
OUT = os.path.join(HERE, "frozen_hparams.json")

FREEZE = ["temperature", "sigma", "lr", "stride", "activation",
          "cnn_dropout", "rnn_dropout", "aug_noise", "aug_scale",
          "aug_shift", "aug_drop", "win_stride_mult"]
INT_PARAMS = {"stride", "aug_shift", "aug_drop", "win_stride_mult"}
TOP_FRAC = 0.30


def robust(values, name):
    if isinstance(values[0], str):
        return collections.Counter(values).most_common(1)[0][0]
    m = float(np.median(values))
    return int(round(m)) if name in INT_PARAMS else m


def main():
    st = "sqlite:///" + DB + "?timeout=120"
    out, report = {}, []
    for study in sorted(optuna.study.get_all_study_names(st)):
        ch = study[3:]
        s = optuna.load_study(study_name=study, storage=st)
        done = [t for t in s.trials if t.state.name == "COMPLETE" and t.value is not None]
        if len(done) < 5:
            print(f"  !! {ch}: only {len(done)} trials -- SKIPPED, no reliable freeze")
            continue
        done.sort(key=lambda t: t.value)
        k = max(3, int(round(TOP_FRAC * len(done))))
        top = done[:k]
        frozen, spread = {}, {}
        for p in FREEZE:
            v = [t.params[p] for t in top if p in t.params]
            if not v:
                continue
            frozen[p] = robust(v, p)
            if not isinstance(v[0], str):
                lo, hi = np.percentile(v, [25, 75])
                spread[p] = float(hi - lo)
        out[ch] = {"frozen": frozen, "n_trials": len(done), "n_top": k,
                   "best_score": float(done[0].value),
                   "iqr_within_top30": spread}
        report.append((ch, len(done), k, done[0].value, frozen))

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"\nfroze {len(FREEZE)} params for {len(out)} channels -> {OUT}\n")
    cols = ["temperature", "sigma", "lr", "stride", "activation",
            "cnn_dropout", "rnn_dropout", "aug_noise", "aug_scale",
            "aug_shift", "aug_drop", "win_stride_mult"]
    hdr = {"temperature": "temp", "cnn_dropout": "cnnDO", "rnn_dropout": "rnnDO",
           "aug_noise": "nois", "aug_scale": "scal", "aug_shift": "shft",
           "aug_drop": "drop", "activation": "act", "win_stride_mult": "wsm",
           "stride": "str"}
    print(f"{'ch':5s}{'n':>4s}{'top':>4s} " +
          " ".join(f"{hdr.get(c, c):>7s}" for c in cols))
    for ch, n, k, _, fr in sorted(report):
        cells = []
        for c in cols:
            v = fr.get(c)
            if v is None:
                cells.append(f"{'-':>7s}")
            elif isinstance(v, str):
                cells.append(f"{v:>7s}")
            elif isinstance(v, int):
                cells.append(f"{v:>7d}")
            else:
                cells.append(f"{v:7.4g}")
        print(f"{ch:5s}{n:>4d}{k:>4d} " + " ".join(cells))

    # how much disagreement did we avoid by not using the single best trial?
    print("\nbest-trial vs top-30% disagreement avoided (max over channels):")
    for p in ["aug_scale", "lr", "temperature", "aug_noise", "cnn_dropout"]:
        worst, wch = 0.0, ""
        for study in optuna.study.get_all_study_names(st):
            s = optuna.load_study(study_name=study, storage=st)
            d = sorted([t for t in s.trials
                        if t.state.name == "COMPLETE" and t.value is not None],
                       key=lambda t: t.value)
            if len(d) < 5 or p not in d[0].params:
                continue
            k = max(3, int(round(TOP_FRAC * len(d))))
            med = float(np.median([t.params[p] for t in d[:k]]))
            rel = abs(d[0].params[p] - med) / (abs(med) or 1)
            if rel > worst:
                worst, wch = rel, study[3:]
        print(f"  {p:14s} up to {100*worst:5.0f}%  ({wch})")


if __name__ == "__main__":
    main()
