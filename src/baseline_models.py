"""
Baseline segmentation architectures for the model-comparison benchmark.

Provides a small registry/factory mapping architecture names to builders.
Every builder returns a model honoring the common training contract:

    logits = model(images)          # images: (B, 1, H, W)
                                    # logits: (B, num_classes, H, W)

Covered here (conventional CNN baselines):
- "unetpp": U-Net++ (segmentation_models_pytorch, ResNet-34 encoder)
- "attention_unet": Attention U-Net with Oktay-style attention gates
- "deeplabv3plus": DeepLabV3+ (segmentation_models_pytorch, ResNet-50 encoder)
- "cenet": CE-Net with DAC + RMP context blocks (ResNet-34 encoder)

The default "2d_unet" path lives in src.segmentation_model and is untouched.
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional

from .encoder import ConvBlock, SharedEncoder
from .segmentation_model import SegmentationUNet


# ---------------------------------------------------------------------------
# Attention gate (Oktay et al., "Attention U-Net")
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """
    Additive attention gate for a skip connection.

    Args:
        gate_channels: Channels of the gating signal (coarser decoder level)
        skip_channels: Channels of the skip connection
        inter_channels: Intermediate channel width
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int,
    ):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from the coarser level (B, Cg, Hg, Wg)
            skip: Skip-connection features (B, Cs, Hs, Ws)

        Returns:
            Attended skip features, same shape as ``skip``.
        """
        theta = self.theta_x(skip)
        phi = self.phi_g(gate)
        if phi.shape[2:] != theta.shape[2:]:
            phi = F.interpolate(phi, size=theta.shape[2:], mode='bilinear', align_corners=False)
        attention = self.sigmoid(self.psi(self.relu(theta + phi)))
        return skip * attention


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates on every skip connection.

    Randomly initialized (the original Attention U-Net trains from scratch).
    Reuses SharedEncoder so encoder depth/width match the project U-Net.

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Output segmentation classes
        encoder_channels: Channel sizes per encoder level
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        encoder_channels: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        self.num_classes = num_classes

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            dropout=dropout,
            use_residual=True,
        )

        reversed_channels = list(reversed(encoder_channels))
        self.upconvs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.gates.append(
                AttentionGate(
                    gate_channels=out_ch,
                    skip_channels=out_ch,
                    inter_channels=max(1, out_ch // 2),
                )
            )
            self.decoders.append(ConvBlock(out_ch * 2, out_ch, dropout if i < len(reversed_channels) - 2 else 0.0))
        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, 1, H, W). Returns: logits (B, num_classes, H, W)."""
        bottleneck, skips = self.encoder(x)
        skips = list(reversed(skips))
        h = bottleneck
        for upconv, gate, decoder, skip in zip(self.upconvs, self.gates, self.decoders, skips):
            h = upconv(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, gate(h, skip)], dim=1)
            h = decoder(h)
        return self.final_conv(h)


# ---------------------------------------------------------------------------
# CE-Net: DAC (Dense Atrous Convolution) + RMP (Residual Multi-kernel Pooling)
# ---------------------------------------------------------------------------
class _BasicConv(nn.Module):
    """Conv-BN-ReLU block used inside DAC/RMP."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DACBlock(nn.Module):
    """
    Dense Atrous Convolution block (cascaded dilated branches).

    Three cascaded branches with dilation rates 1, 3, 5 enlarge the
    receptive field while preserving resolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dilate1 = _BasicConv(channels, channels, kernel_size=3, dilation=1)
        self.dilate2 = _BasicConv(channels, channels, kernel_size=3, dilation=3)
        self.dilate3 = _BasicConv(channels, channels, kernel_size=3, dilation=5)
        self.conv1x1 = _BasicConv(channels, channels, kernel_size=1, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.dilate1(x)
        out2 = self.dilate2(out1)
        out3 = self.dilate3(out2)
        out = out1 + out2 + out3
        return F.relu(self.conv1x1(out) + x)


class RMPBlock(nn.Module):
    """
    Residual Multi-kernel Pooling block.

    Four pooling branches (2x2, 3x3, 5x5, 6x6) capture multi-scale context,
    compressed with 1x1 convolutions, upsampled and fused residually.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        branch_ch = max(1, in_channels // 4)
        self.branches = nn.ModuleList()
        for kernel_size in (2, 3, 5, 6):
            self.branches.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                _BasicConv(in_channels, branch_ch, kernel_size=1, dilation=1),
            ))
        self.fuse = _BasicConv(in_channels + branch_ch * 4, in_channels, kernel_size=1, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pooled = [branch(x) for branch in self.branches]
        pooled = [F.interpolate(p, size=size, mode='bilinear', align_corners=False) for p in pooled]
        out = torch.cat([x] + pooled, dim=1)
        return F.relu(self.fuse(out) + x)


class CENet(nn.Module):
    """
    CE-Net with a ResNet-34 encoder, DAC + RMP bottleneck context,
    and a lightweight U-Net-style decoder.

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Output segmentation classes
        pretrained: Load ImageNet weights into the ResNet-34 encoder
            (canonical CE-Net initialization); the 1-channel stem is
            adapted by averaging the pretrained RGB kernel.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        import torchvision.models as tv_models
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet34(weights=weights)
        if in_channels != 3:
            stem = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    stem.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = stem
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64 ch, /4
        self.layer2 = backbone.layer2  # 128 ch, /8
        self.layer3 = backbone.layer3  # 256 ch, /16
        self.layer4 = backbone.layer4  # 512 ch, /32

        self.context = nn.Sequential(DACBlock(512), RMPBlock(512))

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _upsample_cat(self, x: torch.Tensor, skip: torch.Tensor, up, dec) -> torch.Tensor:
        x = up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return dec(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, 1, H, W). Returns: logits (B, num_classes, H, W)."""
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bottleneck = self.context(self.layer4(s3))
        h = self._upsample_cat(bottleneck, s3, self.up4, self.dec4)
        h = self._upsample_cat(h, s2, self.up3, self.dec3)
        h = self._upsample_cat(h, s1, self.up2, self.dec2)
        h = self._upsample_cat(h, s0, self.up1, self.dec1)
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.final_conv(h)


# ---------------------------------------------------------------------------
# segmentation_models_pytorch-backed baselines (lazy import)
# ---------------------------------------------------------------------------
def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for this baseline "
            "(pip install segmentation-models-pytorch)."
        ) from exc
    return smp


def build_unetpp(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """U-Net++ with a ResNet-34 encoder (ImageNet weights are canonical)."""
    smp = _require_smp()
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )


def build_deeplabv3plus(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """DeepLabV3+ with a ResNet-50 encoder (ImageNet weights are canonical)."""
    smp = _require_smp()
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )


def build_attention_unet(num_classes: int = 4, in_channels: int = 1, **kwargs) -> AttentionUNet:
    """Original-style Attention U-Net (attention gates; random init is standard)."""
    return AttentionUNet(in_channels=in_channels, num_classes=num_classes)


def build_cenet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> CENet:
    """CE-Net with DAC + RMP context (ImageNet ResNet-34 stem is canonical)."""
    return CENet(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)


def build_transunet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """TransUNet: R50 stem + ImageNet-21k ViT-B/16 (canonical) + CNN decoder."""
    from .baseline_transformer import build_transunet as _build
    return _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


def build_swin_unet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """Swin-UNet: ImageNet-1k Swin-Tiny encoder (canonical) + patch-expanding decoder."""
    from .baseline_transformer import build_swin_unet as _build
    return _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


def build_nn_transunet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """nn-TransUNet (Zhao et al.): paper architecture + planning (reconstruction)."""
    from .nntransunet import build_nn_transunet as _build
    return _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


def build_segformer(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    """SegFormer MiT-B0: ImageNet-pretrained encoder (canonical) + 4-class head."""
    from .baseline_transformer import build_segformer as _build
    return _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


def build_medsam(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True, **kwargs) -> nn.Module:
    """MedSAM zero-shot: official medsam_vit_b.pth + train-fitted prior boxes (no GT at inference)."""
    from .medsam_baseline import build_medsam as _build
    return _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, **kwargs)


BASELINE_BUILDERS = {
    "unetpp": build_unetpp,
    "attention_unet": build_attention_unet,
    "deeplabv3plus": build_deeplabv3plus,
    "cenet": build_cenet,
    "transunet": build_transunet,
    "swin_unet": build_swin_unet,
    "nn_transunet": build_nn_transunet,
    "segformer": build_segformer,
    "medsam": build_medsam,
}


def build_baseline_model(
    architecture: str,
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Build a comparison baseline by registry name.

    Args:
        architecture: Registry key ("unetpp", "attention_unet",
            "deeplabv3plus", "cenet", "transunet", "swin_unet",
            "nn_transunet", "segformer", "medsam").
        num_classes: Output segmentation classes (4 for ACDC).
        in_channels: Input channels (1 for grayscale MRI).
        pretrained: Use canonical ImageNet weights where applicable
            (smp encoders, CE-Net stem). Attention U-Net ignores this flag
            (random init is standard) but accepts it for a uniform call shape.
        **kwargs: Forwarded to the architecture builder (used by baselines
            needing extra configuration, e.g. MedSAM checkpoint/prompts).

    Returns:
        Model mapping (B, 1, H, W) -> (B, num_classes, H, W) logits.
    """
    try:
        builder = BASELINE_BUILDERS[architecture]
    except KeyError:
        raise ValueError(
            f"Unknown baseline architecture: {architecture!r}. "
            f"Available: {sorted(BASELINE_BUILDERS)} (plus '2d_unet' via segmentation_model)."
        )
    if kwargs:
        # Explicit per-builder validation: reject unsupported arguments with
        # a clear error instead of silently discarding them or crashing deep
        # inside a builder. Builders accepting **kwargs take all of them.
        params = inspect.signature(builder).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            unexpected = sorted(set(kwargs) - set(params))
            if unexpected:
                raise TypeError(
                    f"Builder for architecture {architecture!r} does not accept "
                    f"argument(s): {unexpected}. Accepted: {sorted(params)}."
                )
    return builder(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, **kwargs)
