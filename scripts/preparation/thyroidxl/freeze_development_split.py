#!/usr/bin/env python3
"""Freeze the patient-level ThyroidXL development split before model training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cbm.data.thyroidxl import assign_development_partitions
from cbm.utils import atomic_csv

DEFAULT_EXPERIMENT = ROOT / "experiments" / "data" / "thyroidxl"




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.experiment_root / "preprocessing_v1" / "manifest_preprocessed_fullframe_v1.csv"
    frame = pd.read_csv(source, dtype={"patient_id": str})
    output = assign_development_partitions(
        frame, validation_fraction=args.validation_fraction, seed=args.seed
    )
    destination = args.experiment_root / "development_split_v1" / "manifest_development_split_v1.csv"
    atomic_csv(destination, output)
    included = output.loc[output["status"].eq("included")]
    summary = {
        "source_manifest": str(source),
        "output_manifest": str(destination),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "official_test_status": "locked_not_used_for_model_selection",
        "partitions": {
            name: {
                "images": int(len(group)),
                "patients": int(group["patient_id"].nunique()),
                "positive_patients": int(
                    group[["patient_id", "label"]].drop_duplicates()["label"].sum()
                ),
            }
            for name, group in included.groupby("development_partition", sort=True)
        },
    }
    summary_path = args.experiment_root / "development_split_v1" / "development_split_v1_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
