"""
Transformer segmentation baselines for the model-comparison benchmark.

- TransUNet: hybrid ResNet-50 CNN stem -> ViT-B/16 encoder (timm) with
  1x1 patch projection, interpolated positional embeddings and a
  skip-connected CNN decoder (cascaded upsampler style).
- Swin-UNet: hierarchical Swin-Tiny encoder (timm, features-only) with a
  patch-expanding decoder and U-shaped skip connections.

Both honor the common training contract: (B, 1, H, W) -> (B, C, H, W).
Single-channel adaptation stays inside the model; preprocessing untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .encoder import ConvBlock

VIT_MODEL = "vit_base_patch16_224"
VIT_21K_TAG = "vit_base_patch16_224.augreg_in21k"
SWIN_MODEL = "swin_tiny_patch4_window7_224"
SWIN_1K_TAG = "swin_tiny_patch4_window7_224.ms_in1k"


def _adapt_stem_to_1ch(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a pretrained RGB stem for single-channel input by kernel averaging."""
    if in_channels == conv.in_channels:
        return conv
    adapted = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None,
    )
    with torch.no_grad():
        adapted.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """Convert a timm channels-last feature map (B, H, W, C) to NCHW.

    Disambiguated by square spatial maps (d1 == d2 != d3), which holds for
    every Swin-Tiny stage used here.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D feature map, got {tuple(x.shape)}")
    b, d1, d2, d3 = x.shape
    if d1 == d2 and d1 != d3:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


# ---------------------------------------------------------------------------
# TransUNet
# ---------------------------------------------------------------------------
class TransUNet(nn.Module):
    """
    Hybrid CNN-Transformer segmentation network (TransUNet style).

    ResNet-50 stem supplies /2, /4, /8 skips and a /16 feature map that is
    projected to ViT-B/16 tokens (positional embeddings bilinearly
    interpolated to the input size, CLS token dropped). A 12-layer
    transformer encodes global context; a skip-connected CNN decoder
    restores full resolution.

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Output segmentation classes
        vit_pretrained_tag: timm pretrained tag for the ViT-B/16 blocks
            (ImageNet-21k weights are canonical for TransUNet)
        pretrained_backbone: Load ImageNet weights into ResNet-50 stem
        img_size: Native input size assumed for positional interpolation
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        vit_pretrained_tag: str = VIT_21K_TAG,
        pretrained_backbone: bool = True,
        img_size: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.vit_pretrained_tag = vit_pretrained_tag

        import timm
        import torchvision.models as tv_models

        # --- ResNet-50 CNN stem (skips at /2, /4, /8; features at /16) ---
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = tv_models.resnet50(weights=weights)
        backbone.conv1 = _adapt_stem_to_1ch(backbone.conv1, in_channels)
        self.stem_head = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64 ch
        self.pool = backbone.maxpool  # /4
        self.stage1 = backbone.layer1  # /4, 256 ch
        self.stage2 = backbone.layer2  # /8, 512 ch
        self.stage3 = backbone.layer3  # /16, 1024 ch

        # --- ViT-B/16 transformer blocks (patch projection done above) ---
        vit = timm.create_model(VIT_MODEL, pretrained=bool(vit_pretrained_tag), num_classes=0)
        self.patch_project = nn.Conv2d(1024, vit.embed_dim, kernel_size=1)
        self.transformer_blocks = vit.blocks
        self.transformer_norm = vit.norm
        pos_embed = vit.pos_embed.detach()  # (1, 197, 768) incl. CLS
        patch_tokens = pos_embed[:, 1:, :].reshape(1, vit.embed_dim, 14, 14)
        target_grid = img_size // 16
        if target_grid != 14:
            patch_tokens = F.interpolate(patch_tokens, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
        self.register_buffer('pos_embed_grid', patch_tokens.reshape(1, vit.embed_dim, target_grid, target_grid))
        self.embed_dim = vit.embed_dim

        # --- Skip-connected CNN decoder (cascaded upsampler style) ---
        # --- Skip-connected CNN decoder (published R50-ViT-B/16 config) ---
        # decoder_channels = (256, 128, 64, 16), n_skip = 3,
        # skip_channels = [512, 256, 64, 16] (only the first three skips exist
        # in the stem; the final stage upsamples without a skip).
        self.dec_up1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256 + 512, 256)
        self.dec_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 256, 128)
        self.dec_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64)
        self.dec_up4 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(16, 16)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _decode(self, x: torch.Tensor, skip: Optional[torch.Tensor], up, dec) -> torch.Tensor:
        x = up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return dec(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, 1, H, W). Returns: logits (B, num_classes, H, W)."""
        input_size = x.shape[2:]
        s2 = self.stem_head(x)
        s4 = self.stage1(self.pool(s2))
        s8 = self.stage2(s4)
        deep = self.stage3(s8)  # /16

        tokens = self.patch_project(deep)  # (B, 768, H/16, W/16)
        grid = tokens.shape[2:]
        pos = self.pos_embed_grid
        if pos.shape[2:] != grid:
            pos = F.interpolate(pos, size=grid, mode='bicubic', align_corners=False)
        seq = tokens.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)
        for block in self.transformer_blocks:
            seq = block(seq)
        seq = self.transformer_norm(seq)
        encoded = seq.transpose(1, 2).reshape(x.shape[0], self.embed_dim, *grid)

        h = self._decode(encoded, s8, self.dec_up1, self.dec1)
        h = self._decode(h, s4, self.dec_up2, self.dec2)
        h = self._decode(h, s2, self.dec_up3, self.dec3)
        h = self._decode(h, None, self.dec_up4, self.dec4)
        if h.shape[2:] != input_size:
            h = F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)
        return self.final_conv(h)


# ---------------------------------------------------------------------------
# Swin-UNet
# ---------------------------------------------------------------------------
class PatchExpand(nn.Module):
    """2x patch-expanding upsampler: Linear(C, 2C) + rearrange to (2H, 2W, C/2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        x = self.expand(x).reshape(b, h, w, 2, 2, c // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * 2, w * 2, c // 2)
        return self.norm(x)


class SkipFuse(nn.Module):
    """Fuse upsampled decoder features with the encoder skip by concat + projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([x, skip], dim=-1)))


class PatchExpandLast(nn.Module):
    """Final 4x patch-expanding upsampler back to full input resolution."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 16 == 0, f"PatchExpandLast requires dim divisible by 16, got {dim}"
        self.expand = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) -> (B, 4H, 4W, C/16)
        b, h, w, c = x.shape
        x = self.expand(x).reshape(b, h, w, 4, 4, c // 16)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * 4, w * 4, c // 16)
        return x


def _swin_block(dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                shift_size: int = 0) -> nn.Module:
    from timm.models.swin_transformer import SwinTransformerBlock
    return SwinTransformerBlock(
        dim=dim, input_resolution=input_resolution, num_heads=num_heads,
        window_size=window_size, shift_size=shift_size,
        mlp_ratio=4.0, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, drop_path=0.0,
        norm_layer=nn.LayerNorm,
    )


def _swin_stage(dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7) -> nn.Sequential:
    """Two Swin blocks with alternating regular/shifted windows (Swin-UNet style)."""
    return nn.Sequential(
        _swin_block(dim, input_resolution, num_heads, window_size, shift_size=0),
        _swin_block(dim, input_resolution, num_heads, window_size, shift_size=window_size // 2),
    )


class SwinUNet(nn.Module):
    """
    Swin-UNet style pure-transformer U-shaped network.

    Hierarchical Swin-Tiny encoder (timm, ImageNet-1k) with patch-merging
    stages, a Swin bottleneck, and a patch-expanding decoder with
    U-shaped skip connections (Swin-UNet concept).

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Output segmentation classes
        swin_pretrained_tag: timm pretrained tag (ImageNet-1k is canonical)
        window_size: Attention window (7 per the Swin-UNet paper)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        swin_pretrained_tag: str = SWIN_1K_TAG,
        window_size: int = 7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.swin_pretrained_tag = swin_pretrained_tag
        self.window_size = window_size

        import timm
        self.encoder = timm.create_model(
            SWIN_MODEL, pretrained=bool(swin_pretrained_tag),
            features_only=True, out_indices=(0, 1, 2, 3), img_size=256,
        )
        stem = self.encoder.patch_embed.proj
        self.encoder.patch_embed.proj = _adapt_stem_to_1ch(stem, in_channels)

        # Stage dims of Swin-Tiny: 96, 192, 384, 768 with heads 3, 6, 12, 24.
        self.bottleneck = _swin_stage(768, (8, 8), 24, window_size)
        self.expand3 = PatchExpand(768)   # /32 -> /16, 384 ch
        self.fuse3 = SkipFuse(384)
        self.dec3 = _swin_stage(384, (16, 16), 12, window_size)
        self.expand2 = PatchExpand(384)   # /16 -> /8, 192 ch
        self.fuse2 = SkipFuse(192)
        self.dec2 = _swin_stage(192, (32, 32), 6, window_size)
        self.expand1 = PatchExpand(192)   # /8 -> /4, 96 ch
        self.fuse1 = SkipFuse(96)
        self.dec1 = _swin_stage(96, (64, 64), 3, window_size)
        self.expand_last = PatchExpandLast(96)  # /4 -> x1, 6 ch
        self.output_norm = nn.LayerNorm(6)
        self.final_conv = nn.Conv2d(6, num_classes, kernel_size=1)

    def _to_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_channels_first(x)
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, 1, H, W). Returns: logits (B, num_classes, H, W)."""
        input_size = x.shape[2:]
        s4, s8, s16, s32 = [self._to_channels_last(f) for f in self.encoder(x)]
        h = self.bottleneck(s32)
        h = self.dec3(self.fuse3(self.expand3(h), s16))
        h = self.dec2(self.fuse2(self.expand2(h), s8))
        h = self.dec1(self.fuse1(self.expand1(h), s4))
        h = self.output_norm(self.expand_last(h))
        h = h.permute(0, 3, 1, 2).contiguous()
        if h.shape[2:] != input_size:
            h = F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)
        return self.final_conv(h)


def build_transunet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> TransUNet:
    """TransUNet: R50 stem + ImageNet-21k ViT-B/16 (canonical) + CNN decoder."""
    return TransUNet(
        in_channels=in_channels, num_classes=num_classes,
        vit_pretrained_tag=VIT_21K_TAG if pretrained else "",
        pretrained_backbone=pretrained,
    )


def build_swin_unet(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> SwinUNet:
    """Swin-UNet: ImageNet-1k Swin-Tiny encoder (canonical) + patch-expanding decoder."""
    return SwinUNet(
        in_channels=in_channels, num_classes=num_classes,
        swin_pretrained_tag=SWIN_1K_TAG if pretrained else "",
    )


# ---------------------------------------------------------------------------
# SegFormer (MiT-B0)
# ---------------------------------------------------------------------------
SEGFORMER_CHECKPOINT_ID = "nvidia/mit-b0"
SEGFORMER_ID2LABEL = {0: "background", 1: "LV", 2: "myocardium", 3: "RV"}


def _replace_stem_1ch(model: nn.Module, in_channels: int) -> None:
    """Replace the 7x7 RGB patch-embed stem with a 1-channel averaging init, in place."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (7, 7) and module.in_channels == 3:
            node = model
            *path, attr = name.split(".")
            for part in path:
                node = node[int(part)] if part.isdigit() else getattr(node, part)
            setattr(node, attr, _adapt_stem_to_1ch(module, in_channels))
            return
    raise AssertionError("RGB patch-embed stem not found")


class SegFormerBaseline(nn.Module):
    """
    SegFormer with MiT-B0 encoder for 4-class cardiac MRI segmentation.

    Wraps Hugging Face ``SegformerForSemanticSegmentation`` so the model
    honors the common training contract ``(B, 1, H, W) -> (B, 4, H, W)``:
    the 1-channel MRI is adapted inside the model (pretrained RGB patch-
    embed kernel averaged to one channel), and the native H/4 logits are
    bilinearly upsampled to the input size.

    Args:
        num_classes: Output segmentation classes (4 for ACDC).
        in_channels: Input channels (1 for grayscale MRI).
        checkpoint_id: HF Hub ID of the ImageNet-pretrained MiT-B0 encoder.
        pretrained: Load ImageNet weights (canonical); randomly initialize
            the stem adaptation target and decode head otherwise.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        checkpoint_id: str = SEGFORMER_CHECKPOINT_ID,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.checkpoint_id = checkpoint_id if pretrained else ""

        from transformers import SegformerForSemanticSegmentation

        id2label = {i: SEGFORMER_ID2LABEL[i] for i in range(num_classes)}
        label2id = {v: k for k, v in id2label.items()}
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint_id,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import SegformerConfig
            self.model = SegformerForSemanticSegmentation(
                SegformerConfig(
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                )
            )
        if in_channels != 3:
            _replace_stem_1ch(self.model, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 1, H, W).

        Returns:
            Logits (B, num_classes, H, W).
        """
        logits = self.model(pixel_values=x).logits  # (B, C, H/4, W/4)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits


def build_segformer(num_classes: int = 4, in_channels: int = 1, pretrained: bool = True) -> SegFormerBaseline:
    """SegFormer MiT-B0: ImageNet-pretrained encoder (canonical) + 4-class head."""
    return SegFormerBaseline(
        num_classes=num_classes, in_channels=in_channels,
        checkpoint_id=SEGFORMER_CHECKPOINT_ID, pretrained=pretrained,
    )
