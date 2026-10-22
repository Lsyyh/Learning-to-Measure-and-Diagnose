#!/usr/bin/env python3
"""Publish full-cohort ThyroidXL native-frame radiomics after staging validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cbm.radiomics.extraction import (
    EXPECTED_INCLUDED_ROWS,
    THYROIDXL_RADIOMICS_FEATURES,
    THYROIDXL_RADIOMICS_VERSION,
    case_is_resumable,
    configuration_hash,
    initialize_radiomics_worker,
    load_case,
    parse_tu10_inventory,
    run_radiomics_case,
    validate_input_manifest,
)
from cbm.utils import atomic_csv, atomic_json, drain_futures, start_estimated_process

DEFAULT_DATA_ROOT = Path(
    os.environ.get("CBM_DATA_ROOT", "data/ThyroidXL")
)
DEFAULT_EXPERIMENT = ROOT / "experiments" / "data" / "thyroidxl"
DEFAULT_WORKBOOK = Path(
    os.environ.get("CBM_TU10_WORKBOOK", "data/TN3K/metadata/TU1.0_S1.xlsx")
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--bin-width", type=float, default=0.075)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--estimated-minutes", type=float, default=120.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("ThyroidXL radiomics workers must be in [1, 8]")
    if args.bin_width <= 0:
        raise ValueError("ThyroidXL radiomics bin width must be positive")

    manifest_path = args.experiment_root / "preprocessing_v1" / "manifest_preprocessed_fullframe_v1.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    included = validate_input_manifest(manifest)
    inventory = parse_tu10_inventory(args.workbook)
    feature_names = inventory["pyradiomics_name"].astype(str).tolist()
    if len(feature_names) != THYROIDXL_RADIOMICS_FEATURES:
        raise ValueError("ThyroidXL production inventory must contain exactly 102 features")
    hash_value = configuration_hash(feature_names, args.bin_width)

    final_root = args.experiment_root / "radiomics_native_v1"
    stage_root = args.experiment_root / "radiomics_native_v1_staging"
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite published ThyroidXL radiomics: {final_root}")
    stage_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "sample_id": str(row.sample_id),
            "patient_id": str(row.patient_id),
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path),
            "official_split": str(row.official_split),
            "preprocessing_version": str(row.preprocessing_version),
            "stage_root": str(stage_root),
            "configuration_hash": hash_value,
            "bin_width": float(args.bin_width),
        }
        for row in included.itertuples(index=False)
    ]

    start_estimated_process(args.estimated_minutes)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_radiomics_worker,
        initargs=(str(args.workbook),),
    ) as pool:
        futures = [pool.submit(run_radiomics_case, task) for task in tasks]
        results = drain_futures(futures, label="radiomics")
    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"ThyroidXL radiomics failed for {len(failures)} cases; staging retained: "
            f"{failures[:5]}"
        )

    cases = []
    audits = []
    for sample_id in included["sample_id"].astype(str):
        case_path = stage_root / "cases" / f"{sample_id}.csv"
        audit_path = stage_root / "audit" / f"{sample_id}.json"
        cases.append(load_case(case_path, feature_names, sample_id))
        if not case_is_resumable(
            case_path,
            audit_path,
            sample_id=sample_id,
            feature_names=feature_names,
            expected_hash=hash_value,
        ):
            raise ValueError(f"Invalid ThyroidXL radiomics staging pair: {sample_id}")
        audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
    features_only = pd.concat(cases, ignore_index=True)
    if features_only.shape != (EXPECTED_INCLUDED_ROWS, THYROIDXL_RADIOMICS_FEATURES + 1):
        raise ValueError("ThyroidXL radiomics aggregate violates the 11,554 x 102 contract")
    if features_only["sample_id"].duplicated().any():
        raise ValueError("ThyroidXL radiomics aggregate sample IDs are not unique")

    os.replace(stage_root, final_root)
    output_manifest = manifest.copy()
    eligible = output_manifest["status"].eq("included")
    output_manifest["radiomics_version"] = THYROIDXL_RADIOMICS_VERSION
    output_manifest["radiomics_status"] = np.where(eligible, "complete", "not_applicable")
    output_manifest["radiomics_qc_status"] = np.where(eligible, "pass", "not_applicable")
    output_manifest["radiomics_exclusion_reason"] = np.where(
        eligible, "", "preprocessing_not_included"
    )
    output_manifest["radiomics_configuration_hash"] = hash_value
    output_manifest["radiomics_path"] = ""
    output_manifest["radiomics_audit_path"] = ""
    output_manifest.loc[eligible, "radiomics_path"] = output_manifest.loc[
        eligible, "sample_id"
    ].map(lambda value: str(final_root / "cases" / f"{value}.csv"))
    output_manifest.loc[eligible, "radiomics_audit_path"] = output_manifest.loc[
        eligible, "sample_id"
    ].map(lambda value: str(final_root / "audit" / f"{value}.json"))
    output_manifest["official_test_used_for_model_selection"] = False

    metadata_columns = [
        "sample_id",
        "patient_id",
        "source",
        "split",
        "official_split",
        "fold",
        "label",
        "radiomics_version",
    ]
    aggregate = output_manifest.loc[eligible, metadata_columns].merge(
        features_only, on="sample_id", validate="one_to_one"
    )
    aggregate = aggregate.loc[:, [*metadata_columns, *feature_names]]
    values = aggregate[feature_names].to_numpy(dtype=float)
    if aggregate.shape != (EXPECTED_INCLUDED_ROWS, len(metadata_columns) + len(feature_names)):
        raise ValueError("ThyroidXL final radiomics table has an invalid shape")
    if not np.isfinite(values).all():
        raise ValueError("ThyroidXL final radiomics table contains non-finite values")

    audit_frame = pd.DataFrame(audits).sort_values("sample_id")
    atomic_csv(args.experiment_root / "radiomics_native_v1" / "manifest_radiomics_native_v1.csv", output_manifest)
    atomic_csv(args.experiment_root / "radiomics_native_v1" / "radiomics_features_native_v1.csv", aggregate)
    atomic_csv(args.experiment_root / "radiomics_native_v1" / "radiomics_native_v1_audit.csv", audit_frame)
    atomic_csv(args.experiment_root / "raw_audit_v1" / "tu10_inventory_102.csv", inventory)
    variance = aggregate[feature_names].var(axis=0, ddof=0)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "full_audited_cohort_including_locked_official_test",
        "radiomics_version": THYROIDXL_RADIOMICS_VERSION,
        "configuration_hash": hash_value,
        "cases": int(len(aggregate)),
        "patients": int(aggregate["patient_id"].nunique()),
        "features": len(feature_names),
        "all_features_finite": bool(np.isfinite(values).all()),
        "zero_variance_features_descriptive_qc_only": variance[variance == 0].index.tolist(),
        "development_cases": int(aggregate["official_split"].eq("train").sum()),
        "official_test_cases": int(aggregate["official_split"].eq("test").sum()),
        "official_test_extracted_for_locked_evaluation": True,
        "official_test_used_for_model_selection": False,
        "normalization": "native_full_frame_foreground_zscore",
        "bin_width": float(args.bin_width),
        "workers": int(args.workers),
    }
    atomic_json(args.experiment_root / "radiomics_native_v1" / "radiomics_native_v1_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
