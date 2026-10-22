"""Shared TU1.0 workbook contract and PyRadiomics inventory execution.

The TU1.0 dictionary (commit ``17e21abb``) is the frozen dictionary used by
the ThyroidXL ultrasound mainline. This module owns the frozen workbook
constants and parsers plus the low-level PyRadiomics execution helpers.
Per-cohort extractors keep their own frozen audit schemas and call these
helpers instead of re-implementing them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import sha256_file

__all__ = [
    "POLARITY_VERSION",
    "TU10_ACTUAL_CLASS_COUNTS",
    "TU10_COMMIT",
    "TU10_SHAPE2D_EXTENSION_FEATURES",
    "TU10_WORKBOOK_SHA256",
    "build_tu10_polarity_mapping",
    "execute_inventory_features",
    "occupied_bin_stats",
    "parse_tu10_actual_inventory",
    "parse_tu10_shape2d_extension_inventory",
    "sha256_file",
    "sitk_2d_pair",
]

TU10_COMMIT = "17e21abbe781090a8d704fc2736d49503763d181"
TU10_WORKBOOK_SHA256 = "95481193f5828b32995cda149d123990a1b584fefcfeaac1fb18b5e952a5f76f"
TU10_ACTUAL_CLASS_COUNTS = {
    "firstorder": 18,
    "glcm": 24,
    "gldm": 14,
    "glrlm": 16,
    "glszm": 16,
    "ngtdm": 5,
}
TU10_SHAPE2D_EXTENSION_FEATURES = (
    "Elongation",
    "MajorAxisLength",
    "MaximumDiameter",
    "MeshSurface",
    "MinorAxisLength",
    "Perimeter",
    "PerimeterSurfaceRatio",
    "PixelSurface",
    "Sphericity",
)


def parse_tu10_actual_inventory(workbook_path: Path | str) -> pd.DataFrame:
    """Parse the 93 features actually present for all three TU1.0 source datasets."""
    import openpyxl

    path = Path(workbook_path)
    digest = sha256_file(path)
    if digest != TU10_WORKBOOK_SHA256:
        raise ValueError(f"Unexpected TU1.0 workbook SHA256: {digest}")
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book["Complete extracted RFs_Sheet 1"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    source_sets = []
    for column in range(3):
        source_sets.append(
            {
                str(row[column]).strip()
                for row in rows
                if row[column] is not None and str(row[column]).startswith("original_")
            }
        )
    actual = source_sets[0].intersection(*source_sets[1:])
    included_order = [
        str(row[5]).strip()
        for row in rows
        if row[5] is not None and str(row[5]).startswith("original_")
    ]
    if set(included_order) != actual or len(included_order) != 93:
        raise ValueError("TU1.0 actual common inventory is not the expected 93-feature set")
    counts = Counter(name.split("_", maxsplit=2)[1] for name in included_order)
    if dict(counts) != TU10_ACTUAL_CLASS_COUNTS:
        raise ValueError(f"Unexpected TU1.0 actual class counts: {dict(counts)}")
    return pd.DataFrame(
        [
            {
                "feature_index": index,
                "pyradiomics_name": name,
                "feature_class": name.split("_", maxsplit=2)[1],
                "tu10_commit": TU10_COMMIT,
                "workbook_sha256": digest,
                "inventory_status": "actual_cross_dataset_feature",
                "paper_107_discrepancy": "paper_reports_107_but_workbook_outputs_93_nonshape",
            }
            for index, name in enumerate(included_order)
        ]
    )


def parse_tu10_shape2d_extension_inventory(workbook_path: Path | str) -> pd.DataFrame:
    """Append nine supported pixel-space Shape2D features to the frozen 93 features."""
    from radiomics import getFeatureClasses

    inventory = parse_tu10_actual_inventory(workbook_path)
    available = getFeatureClasses()["shape2D"].getFeatureNames()
    unsupported = [name for name in TU10_SHAPE2D_EXTENSION_FEATURES if name not in available]
    deprecated = [name for name in TU10_SHAPE2D_EXTENSION_FEATURES if available.get(name)]
    if unsupported or deprecated:
        raise ValueError(
            f"Unexpected PyRadiomics Shape2D inventory: unsupported={unsupported}, "
            f"deprecated={deprecated}"
        )
    extension = pd.DataFrame(
        [
            {
                "feature_index": len(inventory) + index,
                "pyradiomics_name": f"original_shape2D_{name}",
                "feature_class": "shape2D",
                "tu10_commit": TU10_COMMIT,
                "workbook_sha256": TU10_WORKBOOK_SHA256,
                "inventory_status": "tncd_shape2d_extension_not_tu10_cross_dataset_common",
                "paper_107_discrepancy": "not_applicable_extension_feature",
                "unit_interpretation": (
                    "pixel_squared"
                    if name in {"MeshSurface", "PixelSurface"}
                    else (
                        "pixel_inverse"
                        if name == "PerimeterSurfaceRatio"
                        else (
                            "pixel"
                            if name
                            in {
                                "MajorAxisLength",
                                "MaximumDiameter",
                                "MinorAxisLength",
                                "Perimeter",
                            }
                            else "dimensionless"
                        )
                    )
                ),
                "deprecated": False,
            }
            for index, name in enumerate(TU10_SHAPE2D_EXTENSION_FEATURES)
        ]
    )
    inventory = inventory.copy()
    inventory["unit_interpretation"] = "intensity_or_dimensionless_texture"
    inventory["deprecated"] = False
    return pd.concat([inventory, extension], ignore_index=True)


def sitk_2d_pair(image: np.ndarray, mask: np.ndarray) -> tuple[Any, Any]:
    """Wrap finite size-matched 2D arrays as unit-spaced SimpleITK objects."""
    import SimpleITK as sitk

    sitk_image = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
    sitk_mask = sitk.GetImageFromArray(np.asarray(mask, dtype=bool).astype(np.uint8))
    sitk_image.SetSpacing((1.0, 1.0))
    sitk_mask.CopyInformation(sitk_image)
    return sitk_image, sitk_mask


def occupied_bin_stats(roi_values: np.ndarray, bin_width: float) -> dict[str, Any]:
    """Return fixed-grid bin occupancy diagnostics for one ROI."""
    bin_indices = np.floor(np.asarray(roi_values) / float(bin_width)).astype(np.int64)
    occupied = int(np.unique(bin_indices).size)
    span = int(bin_indices.max() - bin_indices.min() + 1)
    return {
        "occupied_bin_count": occupied,
        "bin_index_span": span,
        "empty_bin_fraction": float(1.0 - occupied / span),
    }


def execute_inventory_features(
    sitk_image: Any,
    sitk_mask: Any,
    requested: list[str],
    *,
    bin_width: float,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Run one explicit PyRadiomics inventory and report missing/nonfinite features."""
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=float(bin_width),
        normalize=False,
        resampledPixelSpacing=None,
        label=1,
        minimumROIDimensions=2,
        correctMask=False,
    )
    extractor.disableAllFeatures()
    grouped: dict[str, list[str]] = {}
    for name in requested:
        _, feature_class, feature_name = name.split("_", maxsplit=2)
        grouped.setdefault(feature_class, []).append(feature_name)
    extractor.enableFeaturesByName(**grouped)
    raw = extractor.execute(sitk_image, sitk_mask)
    values = {
        name: float(np.asarray(raw[name]).reshape(-1)[0]) for name in requested if name in raw
    }
    missing = [name for name in requested if name not in values]
    nonfinite = [name for name, value in values.items() if not np.isfinite(value)]
    return values, missing, nonfinite


# ---------------------------------------------------------------------------
# TU1.0-informed fixed four-axis projection (label-blind heuristic).
#
# Single home of the audited axis construction. POLARITY_VERSION keeps its historical
# string value because it is a frozen data identifier written into existing
# artifacts (technical_axes_mapping_35.csv, locked-test metadata).
# ---------------------------------------------------------------------------
POLARITY_VERSION = "tncd_tu10_fixed_polarity_axes4_v1"

# Each sign orients a standardized feature toward the more suspicious TI-RADS pole.
# The mapping is label-blind and restricted to directions stated in workbook Sheet 4.
_AXES: dict[str, tuple[tuple[str, int, str, str], ...]] = {
    "composition_heterogeneity": (
        ("original_firstorder_Entropy", 1, "FO_Entropy", "higher complexity"),
        ("original_firstorder_InterquartileRange", 1, "FO_IQR", "higher dispersion"),
        ("original_firstorder_Range", 1, "FO_Range", "higher dispersion"),
        ("original_firstorder_MeanAbsoluteDeviation", 1, "FO_MAD", "higher dispersion"),
        (
            "original_firstorder_RobustMeanAbsoluteDeviation",
            1,
            "FO_rMAD",
            "higher core dispersion",
        ),
        ("original_firstorder_Variance", 1, "FO_Variance", "higher dispersion"),
        ("original_firstorder_Energy", -1, "FO_E", "lower uniformity"),
        ("original_firstorder_Uniformity", -1, "FO_Uniformity", "lower uniformity"),
        ("original_glcm_Contrast", 1, "GLCM_Contrast", "higher local heterogeneity"),
        ("original_glcm_JointEntropy", 1, "GLCM_JointEntropy", "higher heterogeneity"),
        ("original_glcm_JointEnergy", -1, "GLCM_JointEnergy", "lower homogeneity"),
        ("original_glcm_Idm", -1, "GLCM_Idm", "lower homogeneity"),
        (
            "original_glcm_MaximumProbability",
            -1,
            "GLCM_MaximumProbability",
            "lower homogeneity",
        ),
        (
            "original_glrlm_GrayLevelNonUniformity",
            1,
            "GLRLM_GLN",
            "higher gray-level non-uniformity",
        ),
        ("original_glrlm_RunEntropy", 1, "GLRLM_RE", "higher run complexity"),
        (
            "original_glszm_GrayLevelNonUniformity",
            1,
            "GLSZM_GLN",
            "higher gray-level non-uniformity",
        ),
        ("original_glszm_ZoneEntropy", 1, "GLSZM_ZE", "higher zone complexity"),
        (
            "original_gldm_DependenceVariance",
            1,
            "GLDM_DV",
            "higher dependence heterogeneity",
        ),
        (
            "original_gldm_DependenceEntropy",
            1,
            "GLDM_DE",
            "higher dependence complexity",
        ),
    ),
    "hypoechogenicity": (
        ("original_firstorder_Mean", -1, "FO_Mean", "lower relative intensity"),
        ("original_firstorder_Median", -1, "FO_Median", "lower relative intensity"),
        (
            "original_firstorder_10Percentile",
            -1,
            "FO_10P",
            "lower dark-tail intensity",
        ),
        ("original_firstorder_RootMeanSquared", -1, "FO_RMS", "lower signal magnitude"),
        (
            "original_gldm_LowGrayLevelEmphasis",
            1,
            "GLDM_LGLE",
            "greater low-gray dependence",
        ),
        (
            "original_glrlm_LowGrayLevelRunEmphasis",
            1,
            "GLRLM_LGLRE",
            "greater low-gray runs",
        ),
        (
            "original_glszm_LowGrayLevelZoneEmphasis",
            1,
            "GLSZM_LGLZE",
            "greater low-gray zones",
        ),
    ),
    "margin_irregularity": (
        (
            "original_shape2D_PerimeterSurfaceRatio",
            1,
            "SF_PerimeterToSurfaceRatio",
            "higher boundary complexity",
        ),
        ("original_shape2D_Sphericity", -1, "SF_Sphericity", "lower roundness"),
    ),
    "echogenic_foci": (
        (
            "original_firstorder_90Percentile",
            1,
            "FO_90P",
            "higher bright-tail intensity",
        ),
        ("original_firstorder_Maximum", 1, "FO_Maximum", "higher focal intensity"),
        ("original_firstorder_Range", 1, "FO_Range", "higher bright-dark contrast"),
        ("original_firstorder_Skewness", 1, "FO_Skewness", "positive bright-tail skew"),
        (
            "original_gldm_SmallDependenceHighGrayLevelEmphasis",
            1,
            "GLDM_SDHGE",
            "small high-gray dependencies",
        ),
        (
            "original_glrlm_ShortRunHighGrayLevelEmphasis",
            1,
            "GLRLM_SRHGLE",
            "short high-gray runs",
        ),
        (
            "original_glszm_SmallAreaHighGrayLevelEmphasis",
            1,
            "GLSZM_SAHGE",
            "small high-gray zones",
        ),
    ),
}


def build_tu10_polarity_mapping(workbook_path: Path | str) -> pd.DataFrame:
    """Return the audited label-blind four-axis mapping."""
    workbook = Path(workbook_path)
    inventory = parse_tu10_shape2d_extension_inventory(workbook)
    available = set(inventory["pyradiomics_name"])
    definitions = pd.read_excel(workbook, sheet_name="Complete RFs list_Sheet 4")
    abbreviations = set(definitions["Abbreviation"].dropna().astype(str).str.strip())
    rows = []
    for axis, entries in _AXES.items():
        for feature, direction, abbreviation, evidence in entries:
            if feature not in available:
                raise ValueError(f"TU1.0 polarity feature is outside the 102 inventory: {feature}")
            if abbreviation not in abbreviations:
                raise ValueError(f"TU1.0 polarity evidence abbreviation is absent: {abbreviation}")
            rows.append(
                {
                    "polarity_version": POLARITY_VERSION,
                    "axis": axis,
                    "pole_definition": "more_suspicious_minus_less_suspicious",
                    "pyradiomics_name": feature,
                    "direction": direction,
                    "source_sheet": "Complete RFs list_Sheet 4",
                    "source_abbreviation": abbreviation,
                    "direction_evidence": evidence,
                    "workbook_sha256": TU10_WORKBOOK_SHA256,
                    "mapping_status": "tu10_informed_fixed_technical_not_author_published_axis",
                }
            )
    return pd.DataFrame(rows)

# -------------------------------------- TU1.0 relationship parsing


def parse_relationship_inventory(workbook_path: str) -> pd.DataFrame:
    """Parse the formal TU1.0 semantic relationship sheet without endpoint labels."""
    relationships = pd.read_excel(workbook_path, sheet_name="Relationships_Sheet 5")
    required = {
        "TI-RADS score",
        "Semantic_Based on TIRADS",
        "Radiomics",
        "Meaning of Radiomics",
        "Relationship",
    }
    if not required.issubset(relationships):
        raise ValueError("TU1.0 relationship sheet schema is incomplete")
    relationships = relationships.dropna(subset=["Semantic_Based on TIRADS", "Radiomics"]).copy()
    relationships["TI-RADS score"] = relationships["TI-RADS score"].ffill()
    relationships["concept_category"] = (
        relationships["Semantic_Based on TIRADS"]
        .astype(str)
        .str.extract(r"^\s*([^\(]+)", expand=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    relationships.insert(0, "workbook_row", relationships.index + 2)
    relationships = relationships.reset_index(drop=True)
    expected_categories = {
        "composition",
        "echogenicity",
        "shape",
        "margin",
        "echogenic foci",
    }
    if len(relationships) != 301:
        raise ValueError(f"Unexpected TU1.0 relationship count: {len(relationships)}")
    if set(relationships["concept_category"]) != expected_categories:
        raise ValueError("Unexpected TU1.0 semantic categories")
    return relationships
