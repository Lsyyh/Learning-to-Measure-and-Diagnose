"""CBM: ThyroidXL mask-free Stage-1 radiomics prediction and TU1.0 panel Stage-2.

The active task is ThyroidXL thyroid-ultrasound malignancy prediction. Masks
create offline radiomics targets only; inference uses full frames. The package
exposes shared components for:

- audited manifests and the 2D ultrasound preprocessing pipeline;
- TU1.0 dictionary parsing and PyRadiomics extraction (102 frozen features);
- the pooled and per-cell ConvNeXt Stage-1 regressors, the end-to-end CBM
  comparators, and the dictionary-panel additive endpoint with its exact
  per-feature attribution tables.
"""

__version__ = "0.1.0"
