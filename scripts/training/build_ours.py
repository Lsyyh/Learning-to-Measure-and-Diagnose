#!/usr/bin/env python3
"""Freeze the Ours OOF-derived artifacts for one cohort.

Two subcommands over the campaign protocol yaml
(``experiments/ours/protocol/<cohort>.yaml``):

* ``cells``: for every (arm, seed), pool the three fold ``oof80.csv``
  tables into the patient-level student table (the v1 member-rule input),
  recompute the unchanged label-free v1 rule -- reliability, redundancy,
  dictionary members, anchor weights -- fit the frozen endpoint pipeline
  (the C/D head initialization), and write the frozen cell directory
  with SHA provenance. ``train_stage2_joint.py`` consumes
  ``student.csv`` and re-derives the same selection deterministically;
  the saved artifacts are the audit anchors the preregistration's
  member-identity checks point at.
* ``endpoints``: for every (arm, seed) of the A/B ablation arms, pool
  the fold ``oof100.csv`` tables into the patient-level table, keep the
  members frozen by the OOF-80 rule (``cells/{arm}/seed{N}/cell.json``),
  and fit the unchanged frozen endpoint pipeline (median-impute ->
  z-score -> balanced logistic, C=1, seed 42) on the pooled OOF-100
  members. The fitted pipeline is the deployed head of ablation arms
  A/B; it never sees any test partition.

Usage: python scripts/training/build_ours.py {cells,endpoints}
--config experiments/ours/protocol/<cohort>.yaml

2026-09-24 consolidation (PI-directed): merges the retired
``build_ours_cells.py`` / ``build_ours_ablation_endpoints.py``
unchanged in behavior; the historical tool SHAs recorded in
``experiments/ours/protocol/manifest.json`` remain the provenance
anchor for the completed campaign (pre-consolidation tree recoverable
from git history).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cbm.eval.concept_groups import select_dictionary_members  # noqa: E402
from cbm.eval.panel import (  # noqa: E402
    frozen_panel_endpoint_pipeline,
    relationship_hypergraph,
    reliability_scores,
)
from cbm.eval.stage1 import PREDICTION_PREFIX  # noqa: E402
from cbm.models.zoo import radiomics_feature_family  # noqa: E402
from cbm.train.stage2_joint import anchor_weights, pipeline_affine  # noqa: E402
from cbm.utils import sha256_file  # noqa: E402

EXPECTED_FEATURES = 102
ARM_DIRS = {"conventional": "a_baseline", "family_balanced": "b_fb"}


def sha256_relative(path: Path) -> str:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def build_student(fold_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Pool the fold OOF frames into the patient-level student table."""

    pooled = pd.concat(fold_tables, ignore_index=True)
    folds_per_patient = pooled.groupby("patient_id")["oof_fold"].nunique()
    if int(folds_per_patient.max()) != 1:
        leaking = folds_per_patient[folds_per_patient > 1].index.tolist()
        raise ValueError(f"patients span several OOF folds: {leaking[:5]}")
    features = sorted(c for c in pooled.columns if c.startswith("original_"))
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"expected {EXPECTED_FEATURES} features, got {len(features)}")
    student = pooled.groupby("patient_id", sort=True)[features].mean().reset_index()
    labels = pooled.groupby("patient_id", sort=True)["label"].first().reset_index()
    student = labels.merge(student, on="patient_id")
    if (
        student["patient_id"].duplicated().any()
        or not np.isfinite(student[features].to_numpy(float)).all()
    ):
        raise ValueError("student table is not clean patient-level finite data")
    return student


def _cells(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Ours OOF-80 selection cells for one cohort."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="campaign protocol yaml, e.g. experiments/ours/protocol/thyroidxl.yaml",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="subset of arms to build (default: all in the config)",
    )
    args = parser.parse_args(argv)

    campaign = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = str(campaign["dataset"])
    oof = campaign["oof"]
    # The v1 rule's oracle/audit contract lives in the frozen method-of-record
    # config; the campaign yaml owns the arms/seeds/OOF paths.
    stage2_config = yaml.safe_load(
        (ROOT / f"configs/{dataset}/stage2_joint_final.yaml").read_text(encoding="utf-8")
    )
    stage2 = stage2_config["stage2"]
    oracle = pd.read_csv(ROOT / str(stage2["oracle_table"]), dtype={"patient_id": str})
    split_filter = stage2.get("oracle_split")
    if split_filter:
        oracle = oracle.loc[oracle[split_filter["column"]].eq(str(split_filter["value"]))].copy()
    audit = pd.read_csv(ROOT / str(stage2["dictionary_audit"]))

    families_per_group = {
        group: set(families) for group, families in campaign["model"]["groups"].items()
    }
    groups: dict[str, list[int]] = {group: [] for group in families_per_group}

    arms = list(args.arms or oof["arms"])
    for arm in arms:
        for seed in [int(seed) for seed in oof["seeds"]]:
            run_root = ROOT / str(oof["output_root"]) / f"{arm}/seed{seed}"
            fold_tables = []
            sources = []
            for fold in range(int(oof["assignment"]["folds"])):
                table = pd.read_csv(run_root / f"fold{fold}/oof80.csv", dtype={"patient_id": str})
                fold_tables.append(table)
                sources.append(sha256_relative(run_root / f"fold{fold}/oof80.csv"))
            student = build_student(fold_tables)

            features = [
                c
                for c in student.columns
                if c.startswith(PREDICTION_PREFIX) or c.startswith("original_")
            ]
            features = sorted(
                set(features) & set(c for c in oracle.columns if c.startswith("original_"))
            )
            if len(features) != EXPECTED_FEATURES:
                raise ValueError(f"{dataset}: expected 102 common features, got {len(features)}")
            pooled = oracle.groupby("patient_id", sort=True)[features].mean().reset_index()
            pooled = pooled.set_index("patient_id").reindex(student["patient_id"]).reset_index()
            if pooled["patient_id"].tolist() != student["patient_id"].tolist():
                raise ValueError(f"{dataset}/{arm}/seed{seed}: oracle misses student patients")
            pooled["label"] = student["label"].to_numpy()

            if not groups[next(iter(groups))]:
                for index, name in enumerate(features):
                    family = radiomics_feature_family(name)
                    members = [g for g, fs in families_per_group.items() if family in fs]
                    if len(members) != 1:
                        raise ValueError(f"feature {name!r} matches groups {members}")
                    groups[members[0]].append(index)

            support, categories, _ = relationship_hypergraph(audit, features)
            reliability, redundancy = reliability_scores(student, pooled, features)
            correlation = student[features].corr(method="spearman").abs()
            members, _ = select_dictionary_members(
                support, categories, features, reliability, correlation
            )
            dictionary_features = sorted(
                features[index] for index in np.flatnonzero(np.asarray(support).sum(axis=0) > 0)
            )
            weights = anchor_weights(
                features,
                groups,
                reliability,
                frozenset(dictionary_features),
                dict_multiplier=float(stage2_config["joint"]["dict_anchor_multiplier"]),
            )
            pipeline = frozen_panel_endpoint_pipeline()
            pipeline.fit(student[features].loc[:, members], student["label"].astype(int))

            cell = run_root.parent.parent.parent / "cells" / f"{arm}/seed{seed}"
            cell.mkdir(parents=True, exist_ok=True)
            student.to_csv(cell / "student.csv", index=False)
            reliability.rename("reliability").to_csv(cell / "reliability.csv")
            pd.Series(weights, index=features, name="anchor_weight").to_csv(
                cell / "anchor_weights.csv"
            )
            joblib.dump(pipeline, cell / "head_init.joblib")
            correlation.to_csv(cell / "redundancy.csv")
            affine = pipeline_affine(pipeline)
            (cell / "head_init.json").write_text(
                json.dumps(
                    {
                        "members": members,
                        "affine": {
                            key: (value.tolist() if isinstance(value, np.ndarray) else value)
                            for key, value in affine.items()
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            record = {
                "status": "complete",
                "cohort": dataset,
                "arm": arm,
                "seed": seed,
                "rule": "v1 label-free member rule on OOF-80 patient-pooled predictions",
                "n_patients": int(len(student)),
                "n_members": len(members),
                "n_dictionary_features": len(dictionary_features),
                "members": members,
                "dictionary_features": dictionary_features,
                "source_oof80": sources,
                "source_oracle": sha256_relative(ROOT / str(stage2["oracle_table"])),
                "source_audit": sha256_relative(ROOT / str(stage2["dictionary_audit"])),
            }
            (cell / "cell.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(
                f"[cell] {dataset}/{arm}/seed{seed}: {len(student)} patients, "
                f"{len(members)} members, dict support {len(dictionary_features)}"
            )


def _endpoints(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the A/B ablation OOF-100 endpoints for one cohort."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="campaign protocol yaml, e.g. experiments/ours/protocol/thyroidxl.yaml",
    )
    args = parser.parse_args(argv)

    campaign = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = str(campaign["dataset"])
    oof = campaign["oof"]
    for arm, arm_dir in ARM_DIRS.items():
        for seed in [int(seed) for seed in oof["seeds"]]:
            run_root = ROOT / str(oof["output_root"]) / f"{arm}/seed{seed}"
            fold_tables, sources = [], []
            for fold in range(int(oof["assignment"]["folds"])):
                table = run_root / f"fold{fold}/oof100.csv"
                fold_tables.append(pd.read_csv(table, dtype={"patient_id": str}))
                sources.append({"path": str(table.relative_to(ROOT)), "sha256": sha256_file(table)})
            student = build_student(fold_tables)
            cell_record = json.loads(
                (ROOT / f"experiments/ours/{dataset}/cells/{arm}/seed{seed}/cell.json").read_text(
                    encoding="utf-8"
                )
            )
            members = list(cell_record["members"])
            pipeline = frozen_panel_endpoint_pipeline()
            pipeline.fit(student[members], student["label"].astype(int))
            affine = pipeline_affine(pipeline)

            out = ROOT / f"experiments/ours/{dataset}/ablation/{arm_dir}/seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, out / "endpoint_oof100.joblib")
            (out / "endpoint.json").write_text(
                json.dumps(
                    {
                        "arm": arm_dir,
                        "members": members,
                        "affine": {
                            key: (value.tolist() if hasattr(value, "tolist") else value)
                            for key, value in affine.items()
                        },
                        "source_oof100": sources,
                        "source_cell": f"experiments/ours/{dataset}/cells/{arm}/seed{seed}/cell.json",
                        "fit": "frozen endpoint pipeline on patient-pooled OOF-100 members",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[endpoint] {dataset}/{arm_dir}/seed{seed}: {len(student)} patients, "
                f"{len(members)} members"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("cells", description="freeze the OOF-80 selection cells")
    subparsers.add_parser("endpoints", description="freeze the A/B OOF-100 endpoints")
    args, rest = parser.parse_known_args(argv)
    {"cells": _cells, "endpoints": _endpoints}[args.command](rest)


if __name__ == "__main__":
    main()
