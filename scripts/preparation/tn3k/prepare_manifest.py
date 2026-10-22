#!/usr/bin/env python3
"""Build the TN3K training manifest for within-cohort Stage-1 OOF experiments.

Writes the schema-compatible development split manifest (every frame is its
own patient — TN3K has no patient identifiers) plus the 102-feature
inventory regenerated from the TU1.0 workbook (the parser enforces the
SHA-pinned workbook contract), so the standard OOF trainer can run against
``--experiment-root experiments/data/tn3k`` with the TN3K preprocessing version
and radiomics tables. No ThyroidXL artifact is read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cbm.data.tn3k import TN3K_PREPROCESSING_VERSION, build_manifest
from cbm.utils import atomic_csv

DEFAULT_WORKBOOK = Path(
    os.environ.get("CBM_TU10_WORKBOOK", "data/TN3K/metadata/TU1.0_S1.xlsx")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "experiments" / "data" / "tn3k"
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    args = parser.parse_args()

    manifest = build_manifest()
    rows = manifest.assign(
        source="tn3k",
        split="development",
        fold=0,
        status="included",
        exclusion_reason="",
        preprocessing_status="complete",
        preprocessing_version=TN3K_PREPROCESSING_VERSION,
        development_partition="train",
        official_split="train",
    )
    rows = rows[
        [
            "sample_id",
            "patient_id",
            "source",
            "split",
            "fold",
            "label",
            "image_path",
            "mask_path",
            "network_input_path",
            "preprocessing_version",
            "preprocessing_status",
            "status",
            "exclusion_reason",
            "development_partition",
            "official_split",
            "source_split",
        ]
    ]
    rows["radiomics_path"] = ""
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_root / "train_manifest_v1" / "manifest_train_v1.csv", rows)
    from cbm.radiomics.extraction import parse_tu10_inventory

    inventory = parse_tu10_inventory(args.workbook)
    if len(inventory) != 102 or inventory["pyradiomics_name"].duplicated().any():
        raise ValueError("TU1.0 inventory parse did not yield the frozen 102 unique features")
    inventory.to_csv(
        args.output_root / "train_manifest_v1" / "tu10_inventory_102.csv", index=False
    )
    print(f"wrote {len(rows)} rows to {args.output_root / 'train_manifest_v1' / 'manifest_train_v1.csv'}")


if __name__ == "__main__":
    main()
