#!/usr/bin/env bash
# One-click data preparation per cohort (run from the repository root).
#
#   bash scripts/prepare_data.sh --dataset thyroidxl
#   bash scripts/prepare_data.sh --dataset tn3k
#
# Prerequisites: the raw cohorts are downloaded and placed under
#   data/ThyroidXL   (HuggingFace snapshot, see README "Data")
#   data/TN3K        (TN3K repository clone, incl. metadata/TU1.0_S1.xlsx)
# Override with CBM_DATA_ROOT / CBM_TU10_WORKBOOK if they live elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$DATASET" == thyroidxl || "$DATASET" == tn3k ]] || { echo "usage: $0 --dataset {thyroidxl,tn3k}" >&2; exit 2; }

if [[ "$DATASET" == thyroidxl ]]; then
  python scripts/preparation/thyroidxl/audit_raw.py
  python scripts/preparation/thyroidxl/preprocess.py
  python scripts/preparation/thyroidxl/extract_radiomics.py
  python scripts/preparation/thyroidxl/freeze_development_split.py
else
  python scripts/preparation/tn3k/prepare_manifest.py
  python scripts/preparation/tn3k/extract_radiomics.py
  python scripts/preparation/tn3k/freeze_final_split.py
  python scripts/preparation/tn3k/prepare_train_oof_inputs.py
fi

# Shared label-blind dictionary audit (consumed by Stage-2 of both cohorts).
python scripts/evaluation/audit_tu10_mapping.py

echo "== $DATASET data contracts ready under experiments/data/$DATASET =="
