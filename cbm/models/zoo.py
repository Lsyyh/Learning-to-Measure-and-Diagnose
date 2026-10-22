"""ThyroidXL model zoo in one self-contained module.

The grayscale ConvNeXt-Tiny Stage-1 radiomics regressor, the direct
classification comparator sharing its backbone, and the segmentation
comparator's ConvNeXt U-Net — the three model families of the paper mainline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _grayscale_convnext_stem(network: nn.Module, *, pretrained: bool) -> None:
    """Replace a torchvision ConvNeXt RGB stem with a 1-channel ultrasound stem.

    The RGB stem's channel-mean initializes the grayscale weights so the
    ImageNet pretraining transfers; ``pretrained=False`` keeps random init.
    """
    rgb_stem = network.features[0][0]
    grayscale_stem = nn.Conv2d(
        1,
        rgb_stem.out_channels,
        kernel_size=rgb_stem.kernel_size,
        stride=rgb_stem.stride,
        padding=rgb_stem.padding,
        bias=rgb_stem.bias is not None,
    )
    with torch.no_grad():
        if pretrained:
            grayscale_stem.weight.copy_(rgb_stem.weight.mean(dim=1, keepdim=True))
        if rgb_stem.bias is not None and grayscale_stem.bias is not None:
            grayscale_stem.bias.copy_(rgb_stem.bias)
    network.features[0][0] = grayscale_stem


# ------------------------------------------------------- shared 2D backbones


class UltrasoundConvNeXtRadiomicsRegressor(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny adapted from RGB to one ultrasound channel.

    The RGB stem's channel-mean initializes the grayscale stem so the
    ImageNet pretraining transfers; state-dict structure matches
    ``network.*`` regardless of ``n_features``.
    """

    def __init__(
        self,
        *,
        n_features: int = 102,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        network = convnext_tiny(weights=weights)
        _grayscale_convnext_stem(network, pretrained=pretrained)
        classifier_norm = network.classifier[0]
        in_features = network.classifier[-1].in_features
        # Standard head: LayerNorm -> Flatten -> Dropout -> Linear.
        network.classifier = nn.Sequential(
            classifier_norm,
            nn.Flatten(1),
            nn.Dropout(float(dropout)),
            nn.Linear(in_features, n_features),
        )
        self.network = network

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)

    def load_incumbent_state(self, state: dict[str, torch.Tensor]) -> dict:
        """Load a same-architecture Stage-1 state dict (warm-start reference).

        Used by continued-training reference arms: every tensor whose name and
        shape matches transfers. Returns the load report for run metadata.
        """
        model_state = self.state_dict()
        keep = {
            key: value
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        result = self.load_state_dict(keep, strict=False)
        return {
            "n_loaded": len(keep),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }


class UltrasoundConvNeXtClassifier(UltrasoundConvNeXtRadiomicsRegressor):
    """Mask-free binary endpoint classifier with the shared ConvNeXt backbone."""

    def __init__(self, *, dropout: float = 0.1, pretrained: bool = True) -> None:
        super().__init__(n_features=1, dropout=dropout, pretrained=pretrained)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).squeeze(-1)


# ------------------------------------------------- ThyroidXL Stage-1 models

# The ThyroidXL* classes below are config-facing aliases of the Ultrasound*
# bases above (identical defaults): configs and trainers name the dataset-
# scoped class, smoke/tests use either.


class ThyroidXLConvNeXtRadiomicsRegressor(UltrasoundConvNeXtRadiomicsRegressor):
    """ConvNeXt-Tiny Stage-1 model for the frozen 102-feature inventory."""

    def __init__(
        self,
        *,
        n_features: int = 102,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__(n_features=n_features, dropout=dropout, pretrained=pretrained)


class ThyroidXLConvNeXtClassifier(UltrasoundConvNeXtClassifier):
    """Direct image-only benign/malignant comparator on the identical full frame."""

    def __init__(self, *, dropout: float = 0.1, pretrained: bool = True) -> None:
        super().__init__(dropout=dropout, pretrained=pretrained)


# ------------------------------------------------------ feature-family split


def radiomics_feature_family(name: str) -> str:
    """Family of a frozen PyRadiomics name: the ``original_<family>_`` class.

    The family is the radiomics computation class (firstorder, shape2D, glcm,
    glrlm, glszm, gldm, ngtdm) — fixed by the feature's definition, not by
    any clinical dictionary. Unknown classes raise rather than guess.
    """
    parts = str(name).split("_")
    if len(parts) < 3 or parts[0] != "original":
        raise ValueError(f"not a canonical PyRadiomics original_* name: {name!r}")
    family = parts[1]
    if family not in {"firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm"}:
        raise ValueError(f"unknown radiomics family {family!r} in {name!r}")
    return family


def build_thyroidxl_stage1_model(model_settings, feature_names: list[str]) -> nn.Module:
    """Construct the Stage-1 regressor named by a protocol config's ``model``.

    Single dispatch shared by the OOF and final-train entry points so both
    accept the same ``model.name`` value with identical construction
    arguments (and therefore identical seeded initialization).
    """
    name = str(model_settings.get("name", "ThyroidXLConvNeXtRadiomicsRegressor"))
    if name == "ThyroidXLConvNeXtRadiomicsRegressor":
        return ThyroidXLConvNeXtRadiomicsRegressor(
            n_features=len(feature_names),
            dropout=float(model_settings["dropout"]),
            pretrained=bool(model_settings["pretrained"]),
        )
    raise ValueError(f"unknown ThyroidXL Stage-1 model {name!r}")


# -------------------------------------------------- segmentation comparator


def build_thyroidxl_convnext_unet(*, pretrained: bool = True):
    """Build a full-frame segmenter with the same ConvNeXt-Tiny encoder as Stage 1."""
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    class DecoderBlock(nn.Sequential):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            )

    class ConvNeXtUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            network = convnext_tiny(weights=weights)
            _grayscale_convnext_stem(network, pretrained=pretrained)
            self.encoder = network.features
            channels = (96, 192, 384, 768)
            self.up32 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
            self.dec16 = DecoderBlock(channels[2] * 2, channels[2])
            self.up16 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
            self.dec8 = DecoderBlock(channels[1] * 2, channels[1])
            self.up8 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
            self.dec4 = DecoderBlock(channels[0] * 2, channels[0])
            self.full_resolution = nn.Sequential(
                nn.ConvTranspose2d(channels[0], 48, 2, stride=2),
                DecoderBlock(48, 48),
                nn.ConvTranspose2d(48, 24, 2, stride=2),
                DecoderBlock(24, 24),
            )
            self.output = nn.Conv2d(24, 1, 1)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            value = image
            skips = []
            for index, layer in enumerate(self.encoder):
                value = layer(value)
                if index in {1, 3, 5, 7}:
                    skips.append(value)
            value = self.dec16(torch.cat([self.up32(skips[3]), skips[2]], dim=1))
            value = self.dec8(torch.cat([self.up16(value), skips[1]], dim=1))
            value = self.dec4(torch.cat([self.up8(value), skips[0]], dim=1))
            return self.output(self.full_resolution(value))

    return ConvNeXtUNet()
