#!/usr/bin/env python3
"""Audit the released ThyroidXL files and freeze a patient-level smoke cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from cbm.data.thyroidxl import (
    THYROIDXL_SOURCE,
    apply_exact_duplicate_exclusions,
    build_raw_manifest,
    select_smoke_patients,
)
from cbm.radiomics.tu10 import TU10_WORKBOOK_SHA256
from cbm.utils import (
    atomic_csv,
    atomic_json,
    sha256_file,
    start_estimated_process,
    update_estimated_process,
)

DEFAULT_DATA_ROOT = Path(
    os.environ.get("CBM_DATA_ROOT", "data/ThyroidXL")
)
DEFAULT_OUTPUT = ROOT / "experiments" / "data" / "thyroidxl"
DEFAULT_TU10_WORKBOOK = Path(
    os.environ.get("CBM_TU10_WORKBOOK", "data/TN3K/metadata/TU1.0_S1.xlsx")
)






def _dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    output = 0
    for bit in bits:
        output = (output << 1) | int(bit)
    return output


def _polygon_mask(size_xy: tuple[int, int], segments: list[list[float]]) -> np.ndarray:
    image = Image.new("1", size_xy, 0)
    draw = ImageDraw.Draw(image)
    for segment in segments:
        if len(segment) < 6 or len(segment) % 2:
            raise ValueError("Invalid COCO polygon")
        points = list(zip(segment[0::2], segment[1::2], strict=True))
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=bool)


class _BKTree:
    """Small integer Hamming-distance tree for cross-split dHash screening."""

    def __init__(self) -> None:
        self.root: tuple[int, list[str], dict[int, Any]] | None = None

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, sample_id: str) -> None:
        if self.root is None:
            self.root = (value, [sample_id], {})
            return
        node = self.root
        while True:
            distance = self._distance(value, node[0])
            if distance == 0:
                node[1].append(sample_id)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [sample_id], {})
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        matches: list[tuple[int, str]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self._distance(value, node[0])
            if distance <= maximum_distance:
                matches.extend((distance, sample_id) for sample_id in node[1])
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(child for edge, child in node[2].items() if lower <= edge <= upper)
        return matches


def _annotation_indexes(
    data_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    images: dict[str, dict[str, Any]] = {}
    annotations: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        payload = json.loads(
            (data_root / split / f"{split}_annotations.json").read_text(encoding="utf-8")
        )
        image_by_id = {int(item["id"]): item for item in payload["images"]}
        for annotation in payload["annotations"]:
            item = image_by_id[int(annotation["image_id"])]
            sample_id = f"thyroidxl_{Path(item['file_name']).stem}"
            images[sample_id] = item
            annotations[sample_id] = annotation
    return images, annotations


def _source_qc(
    manifest: pd.DataFrame,
    *,
    data_root: Path,
    estimated_minutes: float,
) -> pd.DataFrame:
    image_metadata, annotations = _annotation_indexes(data_root)
    start_estimated_process(estimated_minutes)
    started = time.monotonic()
    rows = []
    for completed, row in enumerate(manifest.itertuples(index=False), start=1):
        audit: dict[str, Any] = {
            "sample_id": row.sample_id,
            "source_read_status": "complete",
            "source_read_error": "",
        }
        try:
            with Image.open(row.image_path) as image_handle:
                image = image_handle.convert("RGB")
                image_array = np.asarray(image)
                image_sha256 = sha256_file(row.image_path)
                dhash = _dhash64(image)
            mask_raw = np.asarray(Image.open(row.mask_path).convert("L"), dtype=np.uint8)
            mask = mask_raw > 0
            if image_array.shape[:2] != mask.shape:
                raise ValueError("image_mask_size_mismatch")
            if not mask.any():
                raise ValueError("empty_mask")

            labels, component_count = ndimage.label(mask)
            component_sizes = np.bincount(labels.ravel())[1:]
            substantial = component_sizes[
                (component_sizes >= 25) & (component_sizes / component_sizes.sum() >= 0.01)
            ]
            item = image_metadata[row.sample_id]
            annotation = annotations[row.sample_id]
            polygon = _polygon_mask(
                (int(item["width"]), int(item["height"])), annotation["segmentation"]
            )
            intersection = int(np.logical_and(mask, polygon).sum())
            union = int(np.logical_or(mask, polygon).sum())
            polygon_dice = float(2 * intersection / (mask.sum() + polygon.sum()))
            polygon_iou = float(intersection / union)
            bbox = annotation["bbox"]
            bbox_area = float(bbox[2] * bbox[3])
            rgb_range = image_array.max(axis=2) - image_array.min(axis=2)

            audit.update(
                {
                    "source_height": int(mask.shape[0]),
                    "source_width": int(mask.shape[1]),
                    "image_mode": "RGB",
                    "image_sha256": image_sha256,
                    "mask_sha256": sha256_file(row.mask_path),
                    "image_dhash64": f"{dhash:016x}",
                    "rgb_grayscale_pixel_fraction": float((rgb_range == 0).mean()),
                    "mask_voxels": int(mask.sum()),
                    "mask_fraction": float(mask.mean()),
                    "mask_touches_border": bool(
                        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                    ),
                    "component_count_raw": int(component_count),
                    "component_count_substantial": int(len(substantial)),
                    "mask_unique_values": int(np.unique(mask_raw).size),
                    "polygon_voxels": int(polygon.sum()),
                    "polygon_mask_dice": polygon_dice,
                    "polygon_mask_iou": polygon_iou,
                    "coco_area_field": float(annotation["area"]),
                    "coco_bbox_area": bbox_area,
                    "coco_area_equals_bbox_area": bool(
                        np.isclose(float(annotation["area"]), bbox_area)
                    ),
                    "coco_area_to_mask_ratio": float(float(annotation["area"]) / mask.sum()),
                }
            )
        except Exception as error:  # noqa: BLE001 - every source failure belongs in the audit
            audit["source_read_status"] = "failed"
            audit["source_read_error"] = f"{type(error).__name__}: {error}"
        rows.append(audit)
        if completed % 500 == 0 or completed == len(manifest):
            elapsed = time.monotonic() - started
            remaining = elapsed / completed * (len(manifest) - completed)
            update_estimated_process(remaining)
            print(
                f"[{completed}/{len(manifest)}] ThyroidXL raw QC eta={remaining:.1f}s", flush=True
            )
    update_estimated_process(0)
    return pd.DataFrame(rows)


def _duplicate_audit(
    source_qc: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = manifest.set_index("sample_id")[["patient_id", "official_split"]].to_dict("index")
    exact_rows = []
    group_index = 0
    for digest, group in source_qc.groupby("image_sha256", dropna=True, sort=True):
        if len(group) < 2:
            continue
        group_index += 1
        sample_ids = group["sample_id"].astype(str).tolist()
        patients = {str(metadata[value]["patient_id"]) for value in sample_ids}
        splits = {str(metadata[value]["official_split"]) for value in sample_ids}
        for sample_id in sample_ids:
            exact_rows.append(
                {
                    "duplicate_group": group_index,
                    "image_sha256": digest,
                    "sample_id": sample_id,
                    "patient_id": metadata[sample_id]["patient_id"],
                    "official_split": metadata[sample_id]["official_split"],
                    "different_patient_duplicate": len(patients) > 1,
                    "cross_split_duplicate": len(splits) > 1,
                }
            )

    tree = _BKTree()
    development = source_qc.merge(
        manifest[["sample_id", "patient_id", "official_split"]],
        on="sample_id",
        validate="one_to_one",
    )
    for row in development.loc[development["official_split"].eq("train")].itertuples():
        tree.add(int(row.image_dhash64, 16), str(row.sample_id))
    near_rows = []
    seen: set[tuple[str, str]] = set()
    for row in development.loc[development["official_split"].eq("test")].itertuples():
        for distance, train_sample in tree.query(int(row.image_dhash64, 16), 4):
            pair = (train_sample, str(row.sample_id))
            if pair in seen:
                continue
            seen.add(pair)
            train_meta = metadata[train_sample]
            near_rows.append(
                {
                    "train_sample_id": train_sample,
                    "test_sample_id": row.sample_id,
                    "hamming_distance": int(distance),
                    "train_patient_id": train_meta["patient_id"],
                    "test_patient_id": row.patient_id,
                    "same_patient": str(train_meta["patient_id"]) == str(row.patient_id),
                }
            )
    return pd.DataFrame(exact_rows), pd.DataFrame(near_rows)


def _patient_qc(manifest: pd.DataFrame, source_qc: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(source_qc, on="sample_id", validate="one_to_one")
    rows = []
    for patient_id, group in merged.groupby("patient_id", sort=True):
        included_rows = int(group["status"].eq("included").sum())
        patient_status = (
            "excluded"
            if included_rows == 0
            else ("included" if included_rows == len(group) else "partial")
        )
        rows.append(
            {
                "patient_id": patient_id,
                "official_split": group["official_split"].iloc[0],
                "label": int(group["label"].iloc[0]),
                "endpoint_code": int(group["endpoint_code"].iloc[0]),
                "endpoint_provenance": group["endpoint_provenance"].iloc[0],
                "endpoint_qc_warning": group["endpoint_qc_warning"].iloc[0],
                "tirads": int(group["tirads"].iloc[0]),
                "image_count": int(len(group)),
                "included_image_count": included_rows,
                "minimum_mask_fraction": float(group["mask_fraction"].min()),
                "median_mask_fraction": float(group["mask_fraction"].median()),
                "maximum_mask_fraction": float(group["mask_fraction"].max()),
                "minimum_polygon_dice": float(group["polygon_mask_dice"].min()),
                "status": patient_status,
                "exclusion_reason": ";".join(
                    sorted(
                        {
                            reason
                            for value in group["exclusion_reason"].fillna("")
                            for reason in str(value).split(";")
                            if reason
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _huggingface_source_revision(data_root: Path) -> tuple[str, int]:
    metadata_files = list((data_root / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    revisions = {
        path.read_text(encoding="utf-8").splitlines()[0].strip() for path in metadata_files
    }
    if len(revisions) != 1:
        raise ValueError(f"ThyroidXL download contains inconsistent revisions: {sorted(revisions)}")
    return revisions.pop(), len(metadata_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tu10-workbook", type=Path, default=DEFAULT_TU10_WORKBOOK)
    parser.add_argument("--smoke-patients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimated-minutes", type=float, default=3.0)
    parser.add_argument("--reuse-source-qc", action="store_true")
    args = parser.parse_args()

    manifest = build_raw_manifest(args.data_root)
    source_qc_path = args.output / "raw_audit_v1" / "source_qc.csv"
    if args.reuse_source_qc and source_qc_path.is_file():
        source_qc = pd.read_csv(source_qc_path)
        if set(source_qc["sample_id"]) != set(manifest["sample_id"]) or len(source_qc) != len(
            manifest
        ):
            raise ValueError("Existing ThyroidXL source QC does not match the raw manifest")
        print(f"Reusing complete source QC: {source_qc_path}", flush=True)
    else:
        source_qc = _source_qc(
            manifest,
            data_root=args.data_root,
            estimated_minutes=args.estimated_minutes,
        )
    failures = source_qc["source_read_status"].ne("complete")
    if failures.any():
        failed_ids = set(source_qc.loc[failures, "sample_id"])
        selected = manifest["sample_id"].isin(failed_ids)
        manifest.loc[selected, "status"] = "excluded"
        manifest.loc[selected, "exclusion_reason"] = manifest.loc[selected, "exclusion_reason"].map(
            lambda value: ";".join(filter(None, [str(value), "source_read_error"]))
        )

    exact_duplicates, near_duplicates = _duplicate_audit(source_qc, manifest)
    manifest = apply_exact_duplicate_exclusions(manifest, source_qc, exact_duplicates)
    patient_qc = _patient_qc(manifest, source_qc)
    smoke_patients, smoke_manifest = select_smoke_patients(
        manifest,
        source_qc,
        size=args.smoke_patients,
        seed=args.seed,
    )

    workbook_digest = sha256_file(args.tu10_workbook)
    if workbook_digest != TU10_WORKBOOK_SHA256:
        raise ValueError(f"Unexpected TU1.0 workbook SHA256: {workbook_digest}")
    source_revision, source_metadata_files = _huggingface_source_revision(args.data_root)

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "raw_audit_v1" / "manifest_raw.csv", manifest)
    atomic_csv(args.output / "raw_audit_v1" / "source_qc.csv", source_qc)
    atomic_csv(args.output / "raw_audit_v1" / "patient_qc.csv", patient_qc)
    atomic_csv(args.output / "raw_audit_v1" / "exact_duplicate_groups.csv", exact_duplicates)
    atomic_csv(args.output / "raw_audit_v1" / "cross_split_near_duplicate_pairs.csv", near_duplicates)
    atomic_csv(
        args.output / "raw_audit_v1" / "endpoint_conflicts.csv",
        manifest.loc[manifest["exclusion_reason"].str.contains("endpoint_conflict", na=False)],
    )
    smoke_dir = args.output / "smoke_v1"
    smoke_dir.mkdir(exist_ok=True)
    atomic_csv(smoke_dir / "smoke_patients_50.csv", smoke_patients)
    atomic_csv(smoke_dir / "smoke_manifest.csv", smoke_manifest)
    atomic_json(
        args.output / "raw_audit_v1" / "tu10_provenance.json",
        {
            "workbook_path": str(args.tu10_workbook),
            "workbook_sha256": workbook_digest,
            "expected_sha256": TU10_WORKBOOK_SHA256,
            "status": "verified",
        },
    )

    included = manifest["status"].eq("included")
    included_patients = manifest.loc[included].drop_duplicates("patient_id")
    qc_complete = source_qc.loc[source_qc["source_read_status"].eq("complete")]
    fractions = qc_complete["mask_fraction"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": THYROIDXL_SOURCE,
        "huggingface_revision": source_revision,
        "huggingface_metadata_files": source_metadata_files,
        "data_root": str(args.data_root),
        "rows": int(len(manifest)),
        "patients": int(manifest["patient_id"].nunique()),
        "included_rows": int(included.sum()),
        "included_patients": int(manifest.loc[included, "patient_id"].nunique()),
        "official_split_rows": {
            str(key): int(value) for key, value in manifest["official_split"].value_counts().items()
        },
        "official_split_patients": {
            str(key): int(value)
            for key, value in patient_qc["official_split"].value_counts().items()
        },
        "patient_split_overlap": int(
            len(
                set(manifest.loc[manifest["official_split"].eq("train"), "patient_id"])
                & set(manifest.loc[manifest["official_split"].eq("test"), "patient_id"])
            )
        ),
        "all_patient_label_counts": {
            str(key): int(value) for key, value in patient_qc["label"].value_counts().items()
        },
        "included_patient_label_counts": {
            str(key): int(value) for key, value in included_patients["label"].value_counts().items()
        },
        "endpoint_provenance_counts": {
            str(key): int(value)
            for key, value in patient_qc["endpoint_provenance"].value_counts().items()
        },
        "excluded_row_reason_counts": dict(
            Counter(
                reason
                for value in manifest.loc[~included, "exclusion_reason"]
                for reason in str(value).split(";")
                if reason
            )
        ),
        "fully_excluded_patients": int(patient_qc["status"].eq("excluded").sum()),
        "partially_excluded_patients": int(patient_qc["status"].eq("partial").sum()),
        "source_read_failures": int(failures.sum()),
        "empty_masks": int(qc_complete["mask_voxels"].eq(0).sum()),
        "mask_fraction": {
            "minimum": float(fractions.min()),
            "p01": float(fractions.quantile(0.01)),
            "p25": float(fractions.quantile(0.25)),
            "median": float(fractions.median()),
            "p75": float(fractions.quantile(0.75)),
            "maximum": float(fractions.max()),
        },
        "polygon_mask_dice": {
            "minimum": float(qc_complete["polygon_mask_dice"].min()),
            "p01": float(qc_complete["polygon_mask_dice"].quantile(0.01)),
            "median": float(qc_complete["polygon_mask_dice"].median()),
        },
        "coco_area_equals_bbox_area": int(qc_complete["coco_area_equals_bbox_area"].sum()),
        "exact_duplicate_rows": int(len(exact_duplicates)),
        "exact_duplicate_groups": int(
            exact_duplicates.get("duplicate_group", pd.Series(dtype=int)).nunique()
        ),
        "exact_cross_split_duplicates": int(
            exact_duplicates.get("cross_split_duplicate", pd.Series(dtype=bool)).sum()
        ),
        "cross_split_dhash_le4_pairs": int(len(near_duplicates)),
        "smoke_patients": int(len(smoke_patients)),
        "smoke_images": int(len(smoke_manifest)),
        "tu10_workbook_sha256": workbook_digest,
    }
    atomic_json(args.output / "raw_audit_v1" / "raw_audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
