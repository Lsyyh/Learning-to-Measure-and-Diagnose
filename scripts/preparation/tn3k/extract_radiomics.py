#!/usr/bin/env python3
"""Extract the frozen 102-feature oracle radiomics for the TN3K external cohort.

Runs the exact ThyroidXL extraction protocol (same inventory, native-frame
foreground z-score, bin width 0.075, pixel spacing) over every TN3K raw frame
with its JPEG mask thresholded at the fixed >127 rule. No ThyroidXL patient,
label, or model artifact is read. Output is the external-cohort analogue of
``radiomics_features_native_v1.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "experiments" / "data" / "tn3k"
DEFAULT_WORKBOOK = Path(
    os.environ.get("CBM_TU10_WORKBOOK", "data/TN3K/metadata/TU1.0_S1.xlsx")
)

_INVENTORY: pd.DataFrame | None = None


def _initialize(workbook: str) -> None:
    global _INVENTORY
    from cbm.radiomics.extraction import parse_tu10_inventory

    _INVENTORY = parse_tu10_inventory(workbook)


def _one(task: dict) -> dict:
    from cbm.data.thyroidxl import grayscale_array
    from cbm.data.tn3k import tn3k_mask_array
    from cbm.radiomics.extraction import extract_arrays

    started = time.monotonic()
    values, audit = extract_arrays(
        grayscale_array(task["image_path"]),
        tn3k_mask_array(task["mask_path"]),
        _INVENTORY,
        bin_width=float(task["bin_width"]),
    )
    return {
        "sample_id": task["sample_id"],
        "label": int(task["label"]),
        "mask_voxels": audit["mask_voxels"],
        "seconds": time.monotonic() - started,
        **values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--bin-width", type=float, default=0.075)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    from cbm.data.tn3k import (
        TN3K_MASK_THRESHOLD,
        TN3K_RADIOMICS_VERSION,
        TN3K_TOTAL,
        build_manifest,
    )

    manifest = build_manifest()
    if len(manifest) != TN3K_TOTAL:
        raise ValueError("TN3K manifest size mismatch")
    output = args.output / "radiomics_native_v1" / "radiomics_features_native_v1.csv"
    if output.exists():
        raise FileExistsError(f"TN3K radiomics already exist: {output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = manifest.to_dict("records")
    for task in tasks:
        task["bin_width"] = args.bin_width
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialize, initargs=(str(args.workbook),)
    ) as pool:
        rows = list(pool.map(_one, tasks, chunksize=8))
    frame = pd.DataFrame(rows)
    frame.insert(1, "patient_id", frame["sample_id"])
    frame.insert(2, "split", "external")
    frame.insert(3, "radiomics_version", TN3K_RADIOMICS_VERSION)
    from cbm.utils import atomic_csv, atomic_json

    atomic_csv(output, frame)
    atomic_json(
        args.output / "radiomics_native_v1" / "radiomics_provenance.json",
        {
            "cohort": "tn3k_tncd",
            "n_images": len(frame),
            "radiomics_version": TN3K_RADIOMICS_VERSION,
            "protocol_source": "cbm.radiomics.extraction.extract_arrays (frozen ThyroidXL inventory/normalization/bin width)",
            "mask_threshold": TN3K_MASK_THRESHOLD,
            "mask_rule": "jpeg_midband_ring_excluded_gt_127",
            "bin_width": float(args.bin_width),
            "workbook": str(args.workbook),
            "elapsed_seconds": time.monotonic() - started,
            "thyroidxl_patients_or_labels_used": False,
        },
    )
    print(json.dumps({"output": str(output), "n_images": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
