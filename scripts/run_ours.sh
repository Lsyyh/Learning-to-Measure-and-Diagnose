#!/usr/bin/env bash
# Train the proposed method (Ours = family-balanced Stage 1 + joint-refined
# dictionary-concept Stage 2) for one cohort, seeds 42/43/44.
#
#   bash scripts/run_ours.sh --dataset thyroidxl --device cuda:0
#
# Prerequisite: bash scripts/prepare_data.sh --dataset <cohort> has completed.
# Cost on one RTX 4090: ThyroidXL ~1-2 GPU-days, TN3K ~0.5 GPU-day.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=""; DEVICE="cuda:0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$DATASET" == thyroidxl || "$DATASET" == tn3k ]] || { echo "usage: $0 --dataset {thyroidxl,tn3k} [--device cuda:N]" >&2; exit 2; }

CFG=configs/$DATASET/ours_oof.yaml

# 1) Stage-1 OOF cross-fitting (family-balanced arm; 3 seeds x 3 folds of
#    80-epoch warm-up + 20-epoch continuation chains). ThyroidXL freezes the
#    fold assignment once first.
if [[ "$DATASET" == thyroidxl && ! -f experiments/ours/thyroidxl/oof/oof_assignment.csv ]]; then
  python scripts/training/train_ours_oof.py --config $CFG --arm family_balanced --seed 42 --fold 0 --freeze-assignment --device $DEVICE
fi
for SEED in 42 43 44; do
  for FOLD in 0 1 2; do
    python scripts/training/train_ours_oof.py --config $CFG --arm family_balanced --seed $SEED --fold $FOLD --device $DEVICE --execute
  done
done

# 2) Freeze the OOF-80 selection cells (student table, member rule,
#    anchor weights, Stage-2 head initialization).
python scripts/training/build_ours.py cells --config $CFG --arms family_balanced

# 3) Full-data 80-epoch family-balanced warm-ups (the deployed Stage 1).
for SEED in 42 43 44; do
  python scripts/training/train_stage1_final.py --config configs/$DATASET/stage1_final.yaml \
    --output experiments/ours/$DATASET/warmup/family_balanced/seed$SEED --arm W --seed $SEED --epochs 80 --device $DEVICE
done

# 4) Stage-2 anchored joint refinement (20 epochs; reads the warm-ups from
#    step 3 and the student tables from step 2 via configs/<c>/stage2_joint_final.yaml).
for SEED in 42 43 44; do
  python scripts/training/train_stage2_joint.py --config configs/$DATASET/stage2_joint_final.yaml \
    --output experiments/ours/$DATASET/stage2_joint/seed$SEED --seed $SEED --device $DEVICE
done

echo "== Ours ($DATASET) complete: experiments/ours/$DATASET/stage2_joint/seed{42,43,44} =="
