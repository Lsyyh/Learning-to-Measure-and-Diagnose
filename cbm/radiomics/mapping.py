"""Label-blind TU1.0 relationship audit.

Resolves author-written relationship descriptors to inventory features
without endpoint labels, signs, or invented links; the audit never
executes a mapping. Extraction machinery lives in ``extraction.py`` and
the workbook parsers in ``tu10.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from cbm.utils import sha256_file

from .tu10 import (
    TU10_COMMIT,
    TU10_WORKBOOK_SHA256,
    build_tu10_polarity_mapping,
    parse_relationship_inventory,
    parse_tu10_shape2d_extension_inventory,
)

__all__ = [
    "THYROIDXL_TU10_MAPPING_AUDIT_VERSION",
    "audit_tu10_relationships",
]


# ------------------------------------------------- TU1.0 relationship audit

THYROIDXL_TU10_MAPPING_AUDIT_VERSION = "thyroidxl_tu10_relationship_audit_v2"


def _key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


# These entries resolve an author-written relationship descriptor to exactly one
# feature in the frozen inventory. They do not assign a numeric direction. In
# particular, the relationship prose is retained for human review rather than
# converted into a sign by endpoint-dependent or keyword heuristics.
_AUTHOR_EXPLICIT_FEATURES: dict[str, str] = {
    "correlation": "original_glcm_Correlation",
    "elongation": "original_shape2D_Elongation",
    "entropy": "original_firstorder_Entropy",
    "gray-level co-occurrence matrix (glcm) features (contrast)": "original_glcm_Contrast",
    "gray-level co-occurrence matrix (glcm) features (correlation )": ("original_glcm_Correlation"),
    "gray-level co-occurrence matrix (glcm) features (energy )": ("original_glcm_JointEnergy"),
    "gray-level co-occurrence matrix (glcm) features (energy)": ("original_glcm_JointEnergy"),
    "gray-level run length matrix (glrlm) features (gray-level non-uniformity (gln))": (
        "original_glrlm_GrayLevelNonUniformity"
    ),
    "gray-level run length matrix (glrlm) features (long run emphasis (lre))": (
        "original_glrlm_LongRunEmphasis"
    ),
    "high gray-level run emphasis (hgre)": "original_glrlm_HighGrayLevelRunEmphasis",
    "kurtosis": "original_firstorder_Kurtosis",
    "low gray-level run emphasis (lgre)": "original_glrlm_LowGrayLevelRunEmphasis",
    "maximum intensity": "original_firstorder_Maximum",
    "mean intensity": "original_firstorder_Mean",
    "skewness": "original_firstorder_Skewness",
    "small area emphasis (sae)": "original_glszm_SmallAreaEmphasis",
    "small area high gray-level emphasis (sahge)": (
        "original_glszm_SmallAreaHighGrayLevelEmphasis"
    ),
    "variance": "original_firstorder_Variance",
}

# These phrases have a plausible relationship to one or more inventory columns,
# but the workbook does not give a unique PyRadiomics key and normalized sign.
# They remain non-executable by design.
_SEMANTIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "area": ("original_shape2D_MeshSurface", "original_shape2D_PixelSurface"),
    "circularity / roundness": ("original_shape2D_Sphericity",),
    "compactness": ("original_shape2D_Sphericity",),
    "contrast": ("original_glcm_Contrast", "original_ngtdm_Contrast"),
    "contrast / dissimilarity": (
        "original_glcm_Contrast",
        "original_glcm_DifferenceAverage",
    ),
    "energy / uniformity": (
        "original_firstorder_Energy",
        "original_firstorder_Uniformity",
        "original_glcm_JointEnergy",
    ),
    "equivalent diameter": ("original_shape2D_MaximumDiameter",),
    "gray-level co-occurrence matrix (glcm) features (homogeneity )": (
        "original_glcm_Id",
        "original_glcm_Idm",
        "original_glcm_Idn",
        "original_glcm_Idmn",
        "original_glcm_InverseVariance",
    ),
    "gray-level non-uniformity (gln)": (
        "original_gldm_GrayLevelNonUniformity",
        "original_glrlm_GrayLevelNonUniformity",
        "original_glszm_GrayLevelNonUniformity",
    ),
    "homogeneity": (
        "original_glcm_Id",
        "original_glcm_Idm",
        "original_glcm_Idn",
        "original_glcm_Idmn",
    ),
    "homogeneity / similarity": (
        "original_glcm_Id",
        "original_glcm_Idm",
        "original_glcm_Idn",
        "original_glcm_Idmn",
    ),
    "irregularity / roughness": (
        "original_shape2D_PerimeterSurfaceRatio",
        "original_shape2D_Sphericity",
    ),
    "major axis length to minor axis length ratio": (
        "original_shape2D_Elongation",
        "original_shape2D_MajorAxisLength",
        "original_shape2D_MinorAxisLength",
    ),
    "mean / median / minimum intensity": (
        "original_firstorder_Mean",
        "original_firstorder_Median",
        "original_firstorder_Minimum",
    ),
    "mean / median intensity": (
        "original_firstorder_Mean",
        "original_firstorder_Median",
    ),
    "perimeter-to-area ratio": ("original_shape2D_PerimeterSurfaceRatio",),
    "variance / standard deviation": ("original_firstorder_Variance",),
}

_IMPLEMENTATION_WARNINGS: dict[str, str] = {
    "elongation": (
        "workbook_states_higher_is_more_elongated_but_pyradiomics_shape2d_elongation_"
        "approaches_one_for_circle_like_shapes"
    ),
    "high gray-level run emphasis (hgre)": (
        "workbook_prose_adds_long_contiguous_run_interpretation_not_present_in_"
        "pyradiomics_high_gray_level_run_emphasis"
    ),
    "low gray-level run emphasis (lgre)": (
        "some_workbook_prose_adds_short_run_interpretation_not_present_in_"
        "pyradiomics_low_gray_level_run_emphasis"
    ),
}


def audit_tu10_relationships(
    workbook_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit all formal relationships without learning or inventing a concept map."""
    workbook = Path(workbook_path)
    digest = sha256_file(workbook)
    if digest != TU10_WORKBOOK_SHA256:
        raise ValueError(f"Unexpected TU1.0 workbook SHA256: {digest}")
    inventory = parse_tu10_shape2d_extension_inventory(workbook).copy()
    relationships = parse_relationship_inventory(str(workbook)).copy()
    technical = build_tu10_polarity_mapping(workbook).copy()
    available = set(inventory["pyradiomics_name"].astype(str))
    if set(_AUTHOR_EXPLICIT_FEATURES.values()) - available:
        raise ValueError("TU1.0 explicit crosswalk contains unavailable inventory features")
    for candidates in _SEMANTIC_CANDIDATES.values():
        if set(candidates) - available:
            raise ValueError("TU1.0 semantic candidate crosswalk contains unavailable features")

    audited_rows: list[dict[str, Any]] = []
    for row in relationships.to_dict(orient="records"):
        descriptor = _key(row["Radiomics"])
        explicit = _AUTHOR_EXPLICIT_FEATURES.get(descriptor)
        semantic = _SEMANTIC_CANDIDATES.get(descriptor, ())
        if explicit:
            evidence_class = "author_explicit"
            candidates = (explicit,)
            limitation = "relationship_prose_has_no_machine_normalized_direction"
        elif semantic:
            evidence_class = "semantic_but_no_direction"
            candidates = semantic
            limitation = "descriptor_does_not_uniquely_define_feature_and_signed_weight"
        else:
            evidence_class = "unsupported"
            candidates = ()
            limitation = "descriptor_not_supported_by_frozen_102_feature_inventory"
        audited_rows.append(
            {
                **row,
                "relationship_evidence_class": evidence_class,
                "candidate_inventory_features": "|".join(candidates),
                "candidate_feature_count": len(candidates),
                "normalized_direction": pd.NA,
                "executable_mapping": False,
                "implementation_semantics_status": (
                    "review_required" if descriptor in _IMPLEMENTATION_WARNINGS else "not_verified"
                ),
                "implementation_semantics_warning": _IMPLEMENTATION_WARNINGS.get(descriptor, ""),
                "limitation": limitation,
                "audit_version": THYROIDXL_TU10_MAPPING_AUDIT_VERSION,
                "tu10_commit": TU10_COMMIT,
                "workbook_sha256": digest,
            }
        )
    relationship_audit = pd.DataFrame(audited_rows)
    if len(relationship_audit) != 301:
        raise ValueError("TU1.0 relationship audit did not retain all 301 rows")

    technical_features = set(technical["pyradiomics_name"].astype(str))
    explicit_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for row in relationship_audit.itertuples(index=False):
        candidates = str(row.candidate_inventory_features).split("|")
        candidates = [name for name in candidates if name]
        target = explicit_counts if row.relationship_evidence_class == "author_explicit" else None
        if row.relationship_evidence_class == "semantic_but_no_direction":
            target = semantic_counts
        if target is not None:
            for name in candidates:
                target[name] = target.get(name, 0) + 1
                if row.implementation_semantics_warning:
                    warning_counts[name] = warning_counts.get(name, 0) + 1
    inventory["technical_axis_mapping_rows"] = (
        inventory["pyradiomics_name"]
        .map(technical["pyradiomics_name"].value_counts())
        .fillna(0)
        .astype(int)
    )
    inventory["author_explicit_relationship_rows"] = (
        inventory["pyradiomics_name"].map(explicit_counts).fillna(0).astype(int)
    )
    inventory["semantic_candidate_relationship_rows"] = (
        inventory["pyradiomics_name"].map(semantic_counts).fillna(0).astype(int)
    )
    inventory["implementation_warning_relationship_rows"] = (
        inventory["pyradiomics_name"].map(warning_counts).fillna(0).astype(int)
    )
    inventory["technical_axis_mapped"] = inventory["pyradiomics_name"].isin(technical_features)
    inventory["formal_executable_mapping"] = False
    inventory["mapping_interpretation"] = inventory.apply(
        lambda row: (
            "technical_heuristic_only"
            if row["technical_axis_mapped"]
            else (
                "author_relationship_requires_direction_review"
                if row["author_explicit_relationship_rows"] > 0
                else (
                    "semantic_candidate_not_executable"
                    if row["semantic_candidate_relationship_rows"] > 0
                    else "not_linked_by_supported_formal_relationship"
                )
            )
        ),
        axis=1,
    )
    inventory["audit_version"] = THYROIDXL_TU10_MAPPING_AUDIT_VERSION

    classes = relationship_audit["relationship_evidence_class"].value_counts().to_dict()
    explicit_features = set(
        relationship_audit.loc[
            relationship_audit["relationship_evidence_class"].eq("author_explicit"),
            "candidate_inventory_features",
        ]
    )
    explicit_features.discard("")
    evidence_by_category = (
        relationship_audit.groupby(["concept_category", "relationship_evidence_class"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    summary: dict[str, Any] = {
        "audit_version": THYROIDXL_TU10_MAPPING_AUDIT_VERSION,
        "tu10_commit": TU10_COMMIT,
        "workbook_sha256": digest,
        "formal_relationship_rows": len(relationship_audit),
        "formal_relationship_descriptors": int(relationship_audit["Radiomics"].nunique()),
        "relationship_evidence_counts": {key: int(value) for key, value in classes.items()},
        "relationship_evidence_by_semantic_category": {
            str(category): {str(key): int(value) for key, value in row.items()}
            for category, row in evidence_by_category.to_dict(orient="index").items()
        },
        "inventory_features": len(inventory),
        "author_explicit_unique_inventory_features": len(explicit_features),
        "author_explicit_feature_fraction": len(explicit_features) / len(inventory),
        "technical_mapping_rows": len(technical),
        "technical_unique_inventory_features": len(technical_features),
        "technical_feature_fraction": len(technical_features) / len(inventory),
        "formal_executable_mapping_rows": 0,
        "formal_executable_mapping_status": "not_released_by_authors",
        "coverage_improvement_status": "not_defensible_without_inventing_links_or_directions",
        "implementation_semantics_review_rows": int(
            relationship_audit["implementation_semantics_status"].eq("review_required").sum()
        ),
        "implementation_semantics_warnings": _IMPLEMENTATION_WARNINGS,
        "label_blind_alternative_representation": {
            "name": "tu10_author_explicit_linked_feature_panel",
            "features": len(explicit_features),
            "interpretation": (
                "dictionary-linked raw-feature sensitivity panel; not a signed concept "
                "bottleneck or complete TU1.0 representation"
            ),
            "membership_source": "author relationship descriptors only",
            "direction_source": "none",
        },
        "shape2d_status": "nine_feature_pixel_space_extension_not_tu10_common_inventory",
        "endpoint_labels_used": False,
        "official_test_used": False,
    }
    return relationship_audit, inventory, technical, summary
