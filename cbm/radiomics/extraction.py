"""Frozen TU1.0-compatible radiomics extraction and full-cohort staging.

The 102-feature inventory, the native and fixed-grid extractors, and the
resumable staging pipeline with its strict configuration hashing. The
label-blind relationship audit lives in ``mapping.py``; the workbook
parsers live in ``tu10.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbm.utils import atomic_csv, atomic_json, sha256_file

from .tu10 import (
    TU10_COMMIT,
    TU10_SHAPE2D_EXTENSION_FEATURES,
    TU10_WORKBOOK_SHA256,
    execute_inventory_features,
    occupied_bin_stats,
    parse_tu10_shape2d_extension_inventory,
    sitk_2d_pair,
)

__all__ = [
    "EXPECTED_INCLUDED_ROWS",
    "EXPECTED_RAW_ROWS",
    "THYROIDXL_RADIOMICS_FEATURES",
    "THYROIDXL_RADIOMICS_VERSION",
    "case_is_resumable",
    "configuration_hash",
    "extract",
    "extract_arrays",
    "extract_preprocessed_arrays",
    "initialize_radiomics_worker",
    "load_case",
    "parse_tu10_inventory",
    "run_radiomics_case",
    "validate_input_manifest",
]

# ------------------------------------------------------------- frozen extract

THYROIDXL_RADIOMICS_VERSION = (
    "thyroidxl_tu10_actual93_plus_shape2d9_native_zscore_bin0075_pixelspace_v1"
)
THYROIDXL_RADIOMICS_FEATURES = 102


def parse_tu10_inventory(workbook_path: Path | str) -> pd.DataFrame:
    """Return the frozen 93-feature TU1.0 core plus the audited Shape2D extension.

    Parser chain, widest last: ``parse_tu10_actual_inventory`` (the workbook's
    93-feature core) -> ``parse_tu10_shape2d_extension_inventory`` (+9 Shape2D
    features) -> this function (adds the ThyroidXL role column over the full
    102-feature inventory).
    """
    inventory = parse_tu10_shape2d_extension_inventory(workbook_path).copy()
    inventory["thyroidxl_role"] = np.where(
        inventory["feature_class"].eq("shape2D"),
        "pixel_space_shape_sensitivity",
        "tu10_cross_dataset_common",
    )
    return inventory


def extract_arrays(
    image_array: np.ndarray,
    mask_array_value: np.ndarray,
    inventory: pd.DataFrame,
    *,
    bin_width: float = 0.075,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract the frozen ThyroidXL 102-feature inventory from one native frame."""
    from cbm.data.thyroidxl import normalize_full_frame

    image = np.asarray(image_array, dtype=np.float32)
    mask = np.asarray(mask_array_value, dtype=bool)
    if image.ndim != 2 or image.shape != mask.shape or not np.isfinite(image).all():
        raise ValueError("ThyroidXL radiomics requires one finite size-matched 2D image")
    if not mask.any():
        raise ValueError("ThyroidXL radiomics mask is empty")
    normalized, normalization_audit = normalize_full_frame(image)
    return _extract_normalized_arrays(
        normalized,
        mask_array_value,
        inventory,
        bin_width=bin_width,
        normalization_audit=normalization_audit,
        representation="native_frame_then_full_frame_foreground_zscore",
    )


def extract_preprocessed_arrays(
    image_array: np.ndarray,
    mask_array_value: np.ndarray,
    inventory: pd.DataFrame,
    *,
    bin_width: float = 0.075,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract from the already normalized fixed grid for representation QC."""
    image = np.asarray(image_array, dtype=np.float32).squeeze()
    mask = np.asarray(mask_array_value, dtype=bool)
    if image.ndim != 2 or image.shape != mask.shape or not np.isfinite(image).all():
        raise ValueError("Fixed-grid radiomics requires one finite size-matched 2D image")
    if not mask.any():
        raise ValueError("Fixed-grid radiomics mask is empty")
    sentinel = float(image.min())
    if np.any(image[mask] == sentinel):
        raise ValueError("Fixed-grid radiomics mask overlaps physical padding")
    return _extract_normalized_arrays(
        image,
        mask,
        inventory,
        bin_width=bin_width,
        normalization_audit={"input_sentinel": sentinel},
        representation="preprocessed_384_grid_no_additional_normalization",
    )


def _extract_normalized_arrays(
    normalized: np.ndarray,
    mask_array_value: np.ndarray,
    inventory: pd.DataFrame,
    *,
    bin_width: float,
    normalization_audit: dict[str, Any],
    representation: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract the ordered inventory from one normalized 2D representation."""
    image = np.asarray(normalized, dtype=np.float32)
    mask = np.asarray(mask_array_value, dtype=bool)
    if image.ndim != 2 or image.shape != mask.shape or not np.isfinite(image).all():
        raise ValueError("ThyroidXL radiomics requires one finite size-matched 2D image")
    if not mask.any():
        raise ValueError("ThyroidXL radiomics mask is empty")

    sitk_image, sitk_mask = sitk_2d_pair(image, mask)
    roi_values = image[mask]
    requested = inventory["pyradiomics_name"].astype(str).tolist()
    if len(requested) != THYROIDXL_RADIOMICS_FEATURES or len(set(requested)) != len(requested):
        raise ValueError("ThyroidXL radiomics requires 102 unique ordered features")
    bins = occupied_bin_stats(roi_values, bin_width)
    values, missing, nonfinite = execute_inventory_features(
        sitk_image,
        sitk_mask,
        requested,
        bin_width=bin_width,
    )
    audit: dict[str, Any] = {
        "radiomics_version": THYROIDXL_RADIOMICS_VERSION,
        "tu10_commit": TU10_COMMIT,
        "workbook_sha256": TU10_WORKBOOK_SHA256,
        "configuration_status": "independently_frozen_tu10_compatible_not_author_equivalent",
        "image_type": "Original",
        "representation": representation,
        "normalization": (
            "none_already_preprocessed"
            if representation.startswith("preprocessed_")
            else "full_frame_foreground_zscore"
        ),
        "normalize_inside_pyradiomics": False,
        "resampled_pixel_spacing": None,
        "bin_width": float(bin_width),
        "spacing_xy": [1.0, 1.0],
        "spacing_interpretation": "pixel_units_not_physical_millimeters",
        "source_size_yx": [int(value) for value in image.shape],
        "mask_voxels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "roi_min": float(roi_values.min()),
        "roi_max": float(roi_values.max()),
        "roi_mean": float(roi_values.mean()),
        "roi_std": float(roi_values.std()),
        **bins,
        "requested_features": len(requested),
        "extracted_features": len(values),
        "missing_features": missing,
        "nonfinite_features": nonfinite,
        "shape2d_features": len(TU10_SHAPE2D_EXTENSION_FEATURES),
        "shape2d_units": "pixel_space_not_physical_millimeters",
        **normalization_audit,
    }
    if missing or nonfinite:
        raise ValueError(
            f"ThyroidXL radiomics incomplete: missing={missing}, nonfinite={nonfinite}"
        )
    return values, audit


def extract(
    image_path: Path | str,
    mask_path: Path | str,
    inventory: pd.DataFrame,
    *,
    bin_width: float = 0.075,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract one source pair and record source checksums."""
    from cbm.data.thyroidxl import grayscale_array, mask_array

    values, audit = extract_arrays(
        grayscale_array(image_path),
        mask_array(mask_path),
        inventory,
        bin_width=bin_width,
    )
    audit.update(
        {
            "image_sha256": sha256_file(image_path),
            "mask_sha256": sha256_file(mask_path),
        }
    )
    return values, audit


# -------------------------------------------- full-cohort staging pipeline

EXPECTED_RAW_ROWS = 11635
EXPECTED_INCLUDED_ROWS = 11554

_INVENTORY: pd.DataFrame | None = None
_FEATURE_NAMES: list[str] | None = None


def configuration_hash(feature_names: list[str], bin_width: float) -> str:
    """Hash the frozen extraction configuration used for resume validation."""
    payload = {
        "radiomics_version": THYROIDXL_RADIOMICS_VERSION,
        "feature_names": feature_names,
        "workbook_sha256": TU10_WORKBOOK_SHA256,
        "bin_width": float(bin_width),
        "image_type": "Original",
        "normalization": "native_full_frame_foreground_zscore",
        "resampled_pixel_spacing": None,
        "spacing_interpretation": "pixel_units_not_physical_millimeters",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_case(path: Path, feature_names: list[str], sample_id: str | None = None) -> pd.DataFrame:
    """Load and strictly validate one staged radiomics case row."""
    frame = pd.read_csv(path)
    expected = ["sample_id", *feature_names]
    if frame.shape != (1, len(expected)) or frame.columns.tolist() != expected:
        raise ValueError(f"Invalid ThyroidXL radiomics case schema: {path}")
    if sample_id is not None and str(frame.at[0, "sample_id"]) != sample_id:
        raise ValueError(f"ThyroidXL radiomics case ID mismatch: {path}")
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite ThyroidXL radiomics values: {path}")
    return frame


def case_is_resumable(
    case_path: Path,
    audit_path: Path,
    *,
    sample_id: str,
    feature_names: list[str],
    expected_hash: str,
) -> bool:
    """Accept a staged pair only under the exact frozen configuration."""
    if not case_path.is_file() or not audit_path.is_file():
        return False
    try:
        load_case(case_path, feature_names, sample_id)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    return bool(
        audit.get("sample_id") == sample_id
        and audit.get("status") == "complete"
        and audit.get("radiomics_version") == THYROIDXL_RADIOMICS_VERSION
        and audit.get("configuration_hash") == expected_hash
        and audit.get("workbook_sha256") == TU10_WORKBOOK_SHA256
        and audit.get("requested_features") == THYROIDXL_RADIOMICS_FEATURES
        and audit.get("extracted_features") == THYROIDXL_RADIOMICS_FEATURES
    )


def initialize_radiomics_worker(workbook: str) -> None:
    """Pool initializer: single-threaded backends and the frozen inventory."""
    global _FEATURE_NAMES, _INVENTORY
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)
    _INVENTORY = parse_tu10_inventory(workbook)
    _FEATURE_NAMES = _INVENTORY["pyradiomics_name"].astype(str).tolist()


def run_radiomics_case(task: dict[str, Any]) -> dict[str, str]:
    """Extract one case into staging with resume and audited failures."""
    if _INVENTORY is None or _FEATURE_NAMES is None:
        raise RuntimeError("ThyroidXL radiomics worker inventory was not initialized")
    sample_id = str(task["sample_id"])
    stage_root = Path(task["stage_root"])
    case_path = stage_root / "cases" / f"{sample_id}.csv"
    audit_path = stage_root / "audit" / f"{sample_id}.json"
    if case_is_resumable(
        case_path,
        audit_path,
        sample_id=sample_id,
        feature_names=_FEATURE_NAMES,
        expected_hash=str(task["configuration_hash"]),
    ):
        return {"sample_id": sample_id, "status": "resumed"}

    started = time.monotonic()
    try:
        values, audit = extract(
            task["image_path"],
            task["mask_path"],
            _INVENTORY,
            bin_width=float(task["bin_width"]),
        )
        atomic_csv(case_path, pd.DataFrame([{"sample_id": sample_id, **values}]))
        audit.update(
            {
                "sample_id": sample_id,
                "patient_id": str(task["patient_id"]),
                "status": "complete",
                "error": "",
                "configuration_hash": str(task["configuration_hash"]),
                "preprocessing_version": str(task["preprocessing_version"]),
                "radiomics_version": THYROIDXL_RADIOMICS_VERSION,
                "radiomics_scope": "full_audited_cohort_frozen_before_modeling",
                "official_split": str(task["official_split"]),
                "official_test_used_for_model_selection": False,
                "elapsed_seconds": time.monotonic() - started,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_json(audit_path, audit)
        return {"sample_id": sample_id, "status": "complete"}
    except Exception as error:  # noqa: BLE001 - preserve failures without stopping the cohort
        failure = {
            "sample_id": sample_id,
            "patient_id": str(task["patient_id"]),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "configuration_hash": str(task["configuration_hash"]),
            "preprocessing_version": str(task["preprocessing_version"]),
            "radiomics_version": THYROIDXL_RADIOMICS_VERSION,
            "official_split": str(task["official_split"]),
            "official_test_used_for_model_selection": False,
            "elapsed_seconds": time.monotonic() - started,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(audit_path, failure)
        return {"sample_id": sample_id, "status": "failed", "error": failure["error"]}


def validate_input_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Require the frozen preprocessed cohort before full-cohort extraction."""
    required = {
        "sample_id",
        "patient_id",
        "source",
        "split",
        "official_split",
        "fold",
        "label",
        "image_path",
        "mask_path",
        "status",
        "preprocessing_status",
        "preprocessing_version",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"ThyroidXL preprocessing manifest is missing columns: {missing}")
    included = manifest.loc[manifest["status"].eq("included")].copy()
    if len(manifest) != EXPECTED_RAW_ROWS or len(included) != EXPECTED_INCLUDED_ROWS:
        raise ValueError("ThyroidXL manifest does not match the frozen 11,554-case cohort")
    if not included["preprocessing_status"].eq("complete").all():
        raise ValueError("ThyroidXL included cases do not all have complete preprocessing")
    if included["sample_id"].duplicated().any():
        raise ValueError("ThyroidXL included sample IDs are not unique")
    source_exists = included["image_path"].map(lambda value: Path(str(value)).is_file())
    mask_exists = included["mask_path"].map(lambda value: Path(str(value)).is_file())
    if not source_exists.all() or not mask_exists.all():
        raise FileNotFoundError("ThyroidXL native radiomics source pairs are incomplete")
    return included
