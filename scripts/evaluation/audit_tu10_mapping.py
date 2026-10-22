#!/usr/bin/env python3
"""Audit TU1.0 relationship coverage without endpoint-derived mappings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from cbm.radiomics.mapping import audit_tu10_relationships

DEFAULT_CONFIG = ROOT / "configs" / "tu10_mapping_audit.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "data" / "thyroidxl" / "tu10_mapping_audit_v2"


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    workbook_path = str(config["workbook"]["path"])
    if workbook_path.startswith("${CBM_DATA_ROOT}"):
        data_root = os.environ.get("CBM_DATA_ROOT")
        if data_root is None:
            raise ValueError("CBM_DATA_ROOT is required for the TU1.0 workbook path")
        workbook_path = workbook_path.replace("${CBM_DATA_ROOT}", data_root, 1)
    relationship, inventory, technical, summary = audit_tu10_relationships(workbook_path)
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    relationship.to_csv(args.output / "relationship_audit_301.csv.tmp", index=False)
    os.replace(
        args.output / "relationship_audit_301.csv.tmp",
        args.output / "relationship_audit_301.csv",
    )
    inventory.to_csv(args.output / "inventory_coverage_102.csv.tmp", index=False)
    os.replace(
        args.output / "inventory_coverage_102.csv.tmp",
        args.output / "inventory_coverage_102.csv",
    )
    technical.to_csv(args.output / "technical_axes_mapping_35.csv.tmp", index=False)
    os.replace(
        args.output / "technical_axes_mapping_35.csv.tmp",
        args.output / "technical_axes_mapping_35.csv",
    )
    explicit_panel = inventory.loc[
        inventory["author_explicit_relationship_rows"].gt(0),
        [
            "feature_index",
            "pyradiomics_name",
            "feature_class",
            "inventory_status",
            "author_explicit_relationship_rows",
            "implementation_warning_relationship_rows",
        ],
    ]
    explicit_panel.to_csv(args.output / "author_explicit_feature_panel.csv.tmp", index=False)
    os.replace(
        args.output / "author_explicit_feature_panel.csv.tmp",
        args.output / "author_explicit_feature_panel.csv",
    )
    _atomic_text(args.output / "summary.json", json.dumps(summary, indent=2) + "\n")
    report = f"""# ThyroidXL TU1.0 mapping audit

The checksum-locked workbook contains {summary["formal_relationship_rows"]} formal relationship
rows but no machine-readable mapping from those natural-language descriptors to the frozen
PyRadiomics inventory and no normalized signed weight. All 301 rows and all 102 inventory
features are retained in the audit outputs.

The existing four-axis projection has {summary["technical_mapping_rows"]} rows over
{summary["technical_unique_inventory_features"]}/102 unique features. It is a label-blind
technical heuristic, not an author-published TU1.0 concept map. The separate Shape2D features
are pixel-space extensions and are not part of the 93-feature cross-dataset common inventory.

An exact, conservative crosswalk identifies
{summary["author_explicit_unique_inventory_features"]} unique inventory features mentioned by
unambiguous author descriptors. These rows remain non-executable because the workbook provides
relationship prose rather than a normalized direction. Semantic umbrella terms and unsupported
descriptors are reported separately. Fuzzy matching, endpoint labels, and invented directions
are prohibited, so executable coverage is not increased in this audit.

The conservative alternative is a {summary["author_explicit_unique_inventory_features"]}-feature
author-linked panel that retains raw standardized feature values and author-defined membership.
It may be evaluated as a fixed sensitivity representation, but it is not a signed concept
bottleneck and downstream endpoint coefficients cannot be presented as dictionary directions.
Exact-name linkage does not establish implementation equivalence: the audit explicitly flags
PyRadiomics definition mismatches for Elongation and run-emphasis descriptors.

Official-test images and labels used: 0.
"""
    _atomic_text(args.output / "README.md", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
