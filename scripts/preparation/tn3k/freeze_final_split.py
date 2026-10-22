#!/usr/bin/env python3
"""Freeze the source-respecting TN3K train/validation/test split."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from cbm.data.tn3k import assign_final_partitions
from cbm.utils import atomic_csv, atomic_json, canonical_config_hash, sha256_file

DEFAULT_CONFIG = ROOT / "configs" / "tn3k" / "final_split.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "data" / "tn3k"


def _network_input_hashes(frame: pd.DataFrame, *, path_column: str, workers: int) -> pd.DataFrame:
    paths = [Path(path) for path in frame[path_column]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TN3K network inputs are missing: {missing[:3]}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(sha256_file, paths))
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "patient_id": frame["patient_id"].astype(str).to_numpy(),
            "source_split": frame["source_split"].astype(str).to_numpy(),
            "final_partition": frame["final_partition"].astype(str).to_numpy(),
            "network_input_sha256": hashes,
        }
    ).sort_values("sample_id", kind="stable").reset_index(drop=True)


def _class_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in ("train", "validation", "test"):
        subset = frame.loc[frame["final_partition"].eq(partition)]
        row: dict[str, object] = {
            "final_partition": partition,
            "images": len(subset),
            "labels_used_for_split": partition != "test",
        }
        if partition == "test":
            row.update({"negative": None, "positive": None, "positive_fraction": None})
        else:
            positives = int(subset["label"].sum())
            row.update(
                {
                    "negative": int(len(subset) - positives),
                    "positive": positives,
                    "positive_fraction": positives / len(subset),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _assert_zero_overlap(frame: pd.DataFrame, hashes: pd.DataFrame) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_rows = frame.loc[frame["final_partition"].eq(left)]
        right_rows = frame.loc[frame["final_partition"].eq(right)]
        sample_overlap = set(left_rows["sample_id"]) & set(right_rows["sample_id"])
        patient_overlap = set(left_rows["patient_id"]) & set(right_rows["patient_id"])
        left_hashes = set(hashes.loc[hashes["final_partition"].eq(left), "network_input_sha256"])
        right_hashes = set(hashes.loc[hashes["final_partition"].eq(right), "network_input_sha256"])
        image_hash_overlap = left_hashes & right_hashes
        key = f"{left}_vs_{right}"
        audit[key] = {
            "sample_id_overlap": len(sample_overlap),
            "patient_id_overlap": len(patient_overlap),
            "network_input_hash_overlap": len(image_hash_overlap),
        }
        if sample_overlap or patient_overlap or image_hash_overlap:
            raise ValueError(f"TN3K final split overlap detected: {key}={audit[key]}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from setproctitle import setproctitle

    setproctitle("cbm_r0_tn3k_final_split")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root
    outputs = {name: output_root / relative for name, relative in config["outputs"].items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"final-split outputs already exist: {existing}")

    source = output_root / config["input_manifest"]
    frame = pd.read_csv(source, dtype={"patient_id": str})
    expected = config["source_splits"]
    observed = frame["source_split"].value_counts().to_dict()
    if observed != {"trainval": int(expected["trainval_images"]), "test": int(expected["test_images"])}:
        raise ValueError(f"TN3K source-split counts differ from frozen protocol: {observed}")

    splitter = config["splitter"]
    output = assign_final_partitions(
        frame,
        validation_fraction=float(splitter["validation_fraction"]),
        seed=int(splitter["seed"]),
    )
    output["final_split_version"] = str(config["protocol"])
    hashes = _network_input_hashes(
        output,
        path_column=str(config["hashing"]["input_column"]),
        workers=int(config["hashing"]["workers"]),
    )
    overlap = _assert_zero_overlap(output, hashes)
    counts = _class_counts(output)

    atomic_csv(outputs["manifest"], output.sort_values("sample_id", kind="stable").reset_index(drop=True))
    atomic_csv(outputs["class_counts"], counts)
    atomic_csv(outputs["image_hashes"], hashes)
    snapshot = {
        "status": "complete",
        "protocol": config["protocol"],
        "config_sha256": canonical_config_hash(config),
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "output_manifest": str(outputs["manifest"]),
        "output_manifest_sha256": sha256_file(outputs["manifest"]),
        "image_hashes": str(outputs["image_hashes"]),
        "image_hashes_sha256": sha256_file(outputs["image_hashes"]),
        "class_counts": str(outputs["class_counts"]),
        "source_test": {
            "images": int(expected["test_images"]),
            "label_used_for_split": False,
            "role": "held_unchanged_for_final_evaluation_only",
        },
        "zero_overlap": overlap,
    }
    atomic_json(outputs["audit"], snapshot)
    atomic_json(outputs["config_snapshot"], config)
    print(snapshot)


if __name__ == "__main__":
    main()
