#!/usr/bin/env python3
"""Publish full-cohort ThyroidXL image-only inputs after staging validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cbm.data.thyroidxl import (
    THYROIDXL_PREPROCESSING_VERSION,
    build_preprocessed_manifest,
    publish_staging,
    run_preprocessing_case,
    validate_staging,
)
from cbm.utils import atomic_csv, atomic_json, start_estimated_process, update_estimated_process

DEFAULT_DATA_ROOT = Path(
    os.environ.get("CBM_DATA_ROOT", "data/ThyroidXL")
)
DEFAULT_EXPERIMENT = ROOT / "experiments" / "data" / "thyroidxl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--estimated-minutes", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Workers must be in [1, 8]")
    manifest = pd.read_csv(args.experiment_root / "raw_audit_v1" / "manifest_raw.csv", dtype={"patient_id": str})
    included = manifest.loc[manifest["status"].eq("included")].copy()
    if len(manifest) != 11635 or len(included) != 11554:
        raise ValueError("ThyroidXL raw manifest does not match the frozen audited cohort")

    final_root = args.data_root / "preprocessed_fullframe_v1"
    stage_root = args.data_root / "preprocessed_fullframe_v1_staging"
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite published preprocessing: {final_root}")
    stage_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "sample_id": str(row.sample_id),
            "patient_id": str(row.patient_id),
            "source_image": str(row.image_path),
            "source_mask": str(row.mask_path),
            "stage_root": str(stage_root),
        }
        for row in included.itertuples(index=False)
    ]
    start_estimated_process(args.estimated_minutes)
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_preprocessing_case, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 250 == 0 or completed == len(tasks):
                remaining = (time.monotonic() - started) / completed * (len(tasks) - completed)
                update_estimated_process(remaining)
                print(f"[{completed}/{len(tasks)}] preprocessing eta={remaining:.1f}s", flush=True)
    update_estimated_process(0)
    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise RuntimeError(f"ThyroidXL preprocessing failures: {failures[:5]}")
    audits = validate_staging(stage_root, included)
    output = build_preprocessed_manifest(manifest, final_root)
    publish_staging(stage_root, final_root)
    atomic_csv(args.experiment_root / "preprocessing_v1" / "manifest_preprocessed_fullframe_v1.csv", output)
    atomic_csv(
        args.experiment_root / "preprocessing_v1" / "preprocessing_full_audit.csv", pd.DataFrame(audits)
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "preprocessing_version": THYROIDXL_PREPROCESSING_VERSION,
        "included_images": len(included),
        "included_patients": int(included["patient_id"].nunique()),
        "development_images": int(included["split"].eq("development").sum()),
        "official_test_images": int(included["split"].eq("test").sum()),
        "target_fov_retention_minimum": float(
            pd.DataFrame(audits)["target_fov_retention_fraction_qc_only"].min()
        ),
        "mask_used_for_scale_or_center": False,
        "official_test_used_for_model_selection": False,
    }
    atomic_json(args.experiment_root / "preprocessing_v1" / "preprocessing_full_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
