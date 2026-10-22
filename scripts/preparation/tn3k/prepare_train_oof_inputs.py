#!/usr/bin/env python3
"""Create the split-selected train-only OOF3 assignment and train-only radiomics table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from cbm.eval.stage1 import build_stage1_oof_assignment
from cbm.utils import atomic_csv, atomic_json, canonical_config_hash, sha256_file

DEFAULT_CONFIG = ROOT / "configs" / "tn3k" / "train_oof_inputs.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "data" / "tn3k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from setproctitle import setproctitle

    setproctitle("cbm_tn3k_train_oof_inputs")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root
    outputs = {name: output_root / relative for name, relative in config["outputs"].items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"train-only OOF input outputs already exist: {existing}")

    source = output_root / config["input_manifest"]
    final = pd.read_csv(source, dtype={"patient_id": str})
    selection = config["selection"]
    radiomics_selection = config["radiomics"]
    for key in ("split_column", "split", "source_split", "oof"):
        if radiomics_selection[key] != selection[key]:
            raise ValueError(f"TN3K train-only OOF radiomics {key} must equal the image selection")
    split_column = str(selection["split_column"])
    split = str(selection["split"])
    required_columns = {"source_split": str(selection["source_split"])}
    required = {
        "sample_id", "patient_id", "label", "status", "preprocessing_status", split_column,
        *required_columns,
    }
    missing = sorted(required - set(final.columns))
    if missing:
        raise ValueError(f"TN3K train-only OOF final manifest is missing columns: {missing}")
    manifest = final.loc[
        final["status"].eq("included")
        & final["preprocessing_status"].eq("complete")
        & final[split_column].eq(split)
    ].copy()
    expected = config["expected"]
    if len(manifest) != int(expected["final_train_images"]):
        raise ValueError(f"TN3K train-only OOF expected {expected['final_train_images']} train rows, got {len(manifest)}")
    if not manifest["source_split"].eq(required_columns["source_split"]).all():
        raise ValueError("TN3K train-only OOF manifest contains a non-trainval source row")
    if manifest["sample_id"].duplicated().any() or manifest["patient_id"].duplicated().any():
        raise ValueError("TN3K train-only OOF OOF input requires one unique image per patient_id")

    crossfit = config["crossfit"]
    assignment = build_stage1_oof_assignment(
        manifest,
        n_splits=int(crossfit["folds"]),
        seed=int(crossfit["seed"]),
        split_column=split_column,
        split=split,
        required_columns=required_columns,
    )
    if set(assignment["sample_id"]) != set(manifest["sample_id"]):
        raise AssertionError("TN3K train-only OOF assignment does not cover exactly final-train rows")
    radiomics_source = output_root / config["input_radiomics"]
    radiomics_all = pd.read_csv(radiomics_source, dtype={"patient_id": str})
    if radiomics_all["sample_id"].duplicated().any():
        raise ValueError("TN3K train-only OOF radiomics source has duplicate sample IDs")
    radiomics = radiomics_all.set_index("sample_id").loc[manifest["sample_id"]].reset_index()
    radiomics = radiomics.merge(
        manifest[["sample_id", "final_partition", "source_split"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if (
        radiomics["patient_id"].astype(str).tolist() != manifest["patient_id"].astype(str).tolist()
        or radiomics["label"].astype(int).tolist() != manifest["label"].astype(int).tolist()
    ):
        raise ValueError("TN3K train-only OOF radiomics metadata disagrees with final-train manifest")
    fold_counts = assignment.groupby("oof_fold", sort=True).agg(
        images=("sample_id", "size"), positives=("label", "sum")
    )

    atomic_csv(outputs["radiomics"], radiomics)
    atomic_csv(outputs["assignment"], assignment)
    audit = {
        "status": "complete",
        "protocol": config["protocol"],
        "config_sha256": canonical_config_hash(config),
        "final_manifest": str(source),
        "final_manifest_sha256": sha256_file(source),
        "selection": {
            "split_column": split_column,
            "split": split,
            "required_columns": required_columns,
        },
        "radiomics_selection": dict(radiomics_selection),
        "radiomics_source": str(radiomics_source),
        "radiomics_source_sha256": sha256_file(radiomics_source),
        "train_radiomics": str(outputs["radiomics"]),
        "train_radiomics_sha256": sha256_file(outputs["radiomics"]),
        "assignment": str(outputs["assignment"]),
        "assignment_sha256": sha256_file(outputs["assignment"]),
        "train_images": len(manifest),
        "excluded_final_validation_images": int(final["final_partition"].eq("validation").sum()),
        "excluded_source_test_images": int(final["final_partition"].eq("test").sum()),
        "source_split": required_columns["source_split"],
        "folds": {
            str(fold): {key: int(value) for key, value in row.items()}
            for fold, row in fold_counts.to_dict(orient="index").items()
        },
        "validation_model_or_target_accessed": False,
        "source_test_model_or_target_accessed": False,
        "scope_metadata_read_for_filtering": True,
    }
    atomic_json(outputs["audit"], audit)
    atomic_json(outputs["config_snapshot"], config)
    print(audit)


if __name__ == "__main__":
    main()
