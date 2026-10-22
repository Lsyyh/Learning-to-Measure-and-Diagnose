"""Lazy imports for the model zoo.

Every model family lives in the single self-contained module
:mod:`cbm.models.zoo`. These classes require ``torch`` and therefore
live behind a lazy import: ``import cbm.models`` does NOT pull torch.
Callers that need a model import the symbols directly, e.g.

    from cbm.models.zoo import ThyroidXLConvNeXtRadiomicsRegressor

The registry maps the config-facing ``model.name`` values used by the
protocol configs. The comparator trainers import their classes directly,
but every config key resolves here; note ``ThyroidXLConvNeXtUNet`` maps to
the ``build_thyroidxl_convnext_unet`` factory (the network class is nested
inside it to close over the pretrained flag).
"""

__all__ = [
    "ThyroidXLConvNeXtRadiomicsRegressor",
    "ThyroidXLConvNeXtClassifier",
    "ThyroidXLConvNeXtUNet",
    "radiomics_feature_family",
    "build_thyroidxl_stage1_model",
]

# PEP 562: defer heavy imports until a symbol is actually accessed.
# name -> (module, attribute).
_REGISTRY: dict[str, tuple[str, str]] = {
    "ThyroidXLConvNeXtRadiomicsRegressor": (".zoo", "ThyroidXLConvNeXtRadiomicsRegressor"),
    "ThyroidXLConvNeXtClassifier": (".zoo", "ThyroidXLConvNeXtClassifier"),
    "ThyroidXLConvNeXtUNet": (".zoo", "build_thyroidxl_convnext_unet"),
    "radiomics_feature_family": (".zoo", "radiomics_feature_family"),
    "build_thyroidxl_stage1_model": (".zoo", "build_thyroidxl_stage1_model"),
}


def __getattr__(name):
    entry = _REGISTRY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(entry[0], __package__)
    return getattr(module, entry[1])


def __dir__():
    return sorted(__all__)
