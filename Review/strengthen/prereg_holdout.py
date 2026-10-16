"""Pre-registered single-configuration hold-out estimate.
USAGE: python prereg_holdout.py [--seeds 5] [--config phase4_t5]
"""
import argparse, json, os, subprocess, sys

# --- teh pre-registered configurations. Frozen: do not re-pick after seeing
# --- the numbers this script produces. That would reintroduce the exact
CONFIGS = {
    # the replicated anchor (age_phase4 t5 == age_joint4 t22 on nine terms)
    "phase4_t5": dict(
        head="logeuclid", pool_mix="joint", train_unit="bag",
        ih_oversample=8, age_strat=True, aux="coral", aux_weight=0.0017318,
        freeze_encoders=False, dropout=0.3013, head_hidden=256,
        lr_head=0.0007435, lr_encoder=4.5819e-05, weight_decay=0.0001095,
        loss="smoothl1", bag_passes=8, bag_k=8, bag_subjects=32,
        agg="mean", enc_adapt="full", domains="2ds",
        spd_dim=16, spd_norm="trace", spd_bn=True, spd_recenter="none",
        spd_eps=0.00078866, spd_with_mean=False,
    ),
    # BEST cohort parity (0.307) and best hold-out r (+0.430); public MAE 8.09
    "age_phase4_t25": dict(
        age_strat=True,
        agg='mean',
        aux='coral',
        aux_weight=0.0017318,
        bag_k=32,
        bag_passes=8,
        bag_subjects=32,
        domains='2ds',
        dropout=0.3013,
        enc_adapt='full',
        freeze_encoders=False,
        head='logeuclid',
        head_hidden=256,
        ih_oversample=8,
        loss='smoothl1',
        lr_encoder=4.5819e-05,
        lr_head=0.0007435,
        pool_mix='joint',
        spd_bn=True,
        spd_dim=16,
        spd_eps=0.00078866,
        spd_norm='trace',
        spd_recenter='none',
        spd_with_mean=False,
        train_unit='bag',
        weight_decay=0.0001095,
    ),
    # best FEASIBLE under the original 0.75/8.0 caps: +0.403 / public 7.83
    "age_phase4_t26": dict(
        age_strat=True,
        agg='mean',
        aux='coral',
        aux_weight=0.0017318,
        bag_k=8,
        bag_passes=16,
        bag_subjects=32,
        domains='2ds',
        dropout=0.3013,
        enc_adapt='full',
        freeze_encoders=False,
        head='logeuclid',
        head_hidden=256,
        ih_oversample=8,
        loss='smoothl1',
        lr_encoder=4.5819e-05,
        lr_head=0.0007435,
        pool_mix='joint',
        spd_bn=True,
        spd_dim=16,
        spd_eps=0.00078866,
        spd_norm='trace',
        spd_recenter='none',
        spd_with_mean=False,
        train_unit='bag',
        weight_decay=0.0001095,
    ),
    # the phase-1 half of the cross-objective replication
    "age_joint4_t22": dict(
        age_strat=True,
        aux='coral',
        aux_weight=0.023669336786305804,
        bag_k=32,
        bag_passes=4,
        bag_subjects=32,
        dropout=0.3056023874726369,
        freeze_encoders=False,
        head='logeuclid',
        head_hidden=256,
        ih_oversample=8,
        loss='huber',
        lr_encoder=2.3314110555418265e-05,
        lr_head=0.0004988436534066726,
        pool_mix='joint',
        spd_dim=64,
        spd_eps=0.0014040489178414083,
        spd_norm='trace',
        spd_with_mean=False,
        train_unit='bag',
        weight_decay=0.00012140797099822535,
    ),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phase4_t5", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    print(f"pre-registered config: {a.config}")
    print(json.dumps(cfg, indent=1, sort_keys=True))
    print(f"\nseeds: {a.seeds}")
    print("objective: AGE_IH_W_LOCK=0 AGE_IH_W_ALL=0  "
          "(locked 41 subjects excluded from training AND selection)")
    print("reports:  per-seed hold-out r/MAE/slope + subject-level bootstrap CI")
    if a.dry_run:
        print("\n[dry run] nothing executed")
        sys.exit(0)
    print("\nNOT YET WIRED to age_model_search.run_trial -- see TODO below.")
    sys.exit(1)

# TODO to make this runnable:
#   1. age_model_search.py needs run_trial() importable with an explicit cfg
#   3. bootstrap: resample the 41 locked SUBJECTS with replacement (not
