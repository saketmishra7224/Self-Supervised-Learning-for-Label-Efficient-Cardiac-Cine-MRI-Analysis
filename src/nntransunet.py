"""
nn-TransUNet baseline (Zhao et al., Life 2022;12(10):1570, DOI 10.3390/life12101570).

Reconstruction from the published paper (no official code was released by the
authors), combining:
  - TransUNet network architecture (Chen et al.): hybrid CNN stem, ViT
    encoder, 3 skip connections, CNN up-convolution decoder;
  - nnU-Net experiment planning (Isensee et al.): rule-based preprocessing /
    training schedule, fixed-epoch budget, best checkpoint by validation loss.

Published architecture implemented here:
  CNN stem (ResNet-50 stages) -> ViT-B encoder (12 transformer layers,
  hidden dim 768, patch 16, 12 heads) -> 3 CNN-to-decoder skip connections
  (/2 @ 64 ch, /4 @ 256 ch, /8 @ 512 ch) -> CNN up-convolution decoder with
  decoder_channels = (256, 128, 64, 16) -> 4-class output at 256x256.

Implementation choices / adaptations for this project:
  - 1-channel MRI: the ImageNet RGB stem conv is adapted by averaging the
    pretrained RGB kernel weights (deterministic, inside the model).
  - 256x256 native input: ViT positional embeddings (learned for 14x14
    tokens) are bicubic-interpolated to the 16x16 token grid; CLS token
    dropped (sequence is purely spatial, as in TransUNet).
  - Pretrained init: ImageNet-21k ViT-B/16 transformer blocks
    (timm ``vit_base_patch16_224.augreg_in21k``) + ImageNet-1k ResNet-50
    stem; decoder randomly initialized (standard for TransUNet-family).
  - Full-resolution planning (§3.5.2 of the paper): native 256x256 inputs,
    no cropping to 224.

Training protocol (paper §3.2.3, §3.5; factories below, not executed here):
  - Loss: Dice + Cross-Entropy (existing DiceCELoss).
  - Optimizer: SGD, Nesterov momentum 0.99.
  - Schedule: polynomial LR decay per epoch.
  - Learning rate locked to 0.01. NOTE on paper discrepancy: the paper text
    mentions 0.1 while Table 1 (the tabulated configuration) specifies
    0.01; the tabulated value is used (it also matches nnU-Net convention).
  - Budget: 500 epochs x 250 iterations/epoch (125k iterations), batch size
    fitted to GPU RAM at launch (paper: 36 on 24 GB; configure down for T4).
  - No early stopping; best checkpoint selected by validation loss
    (Trainer monitor_metric="val_loss"); single train/val split (no CV,
    per paper §3.5.1).
  - Weight decay is unspecified in the paper and defaults to 0.0 here.

Deviations from the paper forced by our benchmark protocol (documented,
not silent): fixed 70/10/20 patient splits instead of the paper's random
split; per-volume z-score 1.5 mm preprocessing instead of re-running the
nnU-Net fingerprint; HD95 + patient-level aggregation via our evaluator
(the paper reports max-HD/JSC; JSC is an exact function of Dice).

Experiment planning (§3.1, §3.2.2): plan_nntransunet_experiment() below
implements the paper's nnU-Net-style rules (MRI per-image z-score,
anisotropic-vs-median target spacing, topology- and VRAM-driven patch and
batch selection) over a dataset fingerprint. Fingerprints can be measured
read-only via fingerprint_dataset(). The locked 256x256 benchmark input is
an explicit benchmark adaptation recorded in the plan output — the project
is not claimed to be an exact reproduction.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .encoder import ConvBlock

# Project label convention (BG=0, LV=1, MYO=2, RV=3); -1 marks unlabeled.
EXPECTED_LABELS = {0, 1, 2, 3}

# Paper architecture constants (R50-ViT-B/16 configuration).
VIT_EMBED_DIM = 768
VIT_DEPTH = 12
VIT_HEADS = 12
VIT_PATCH = 16
N_SKIP = 3
SKIP_CHANNELS = [512, 256, 64]      # /8, /4, /2 stem features
SKIP_STRIDES = [8, 4, 2]
DECODER_CHANNELS = (256, 128, 64, 16)

# Paper training protocol (locked values; batch size fitted at launch).
NNTRANSUNET_TRAINING: Dict[str, object] = {
    "epochs": 500,
    "iters_per_epoch": 250,
    "batch_size": 12,          # conservative locked default for 15 GB T4;
                              # confirm against run_nnunet_reference_planning()
                              # output at launch before raising it
    "lr": 0.01,                # locked: tabulated value (see module docstring)
    "momentum": 0.99,
    "nesterov": True,
    "weight_decay": 0.0,       # unspecified in paper
    "lr_power": 0.9,           # polynomial decay power (nnU-Net convention)
    "monitor_metric": "val_loss",
    "early_stopping": None,    # paper uses none; do not enable
}


class NNTransUNet(nn.Module):
    """
    nn-TransUNet segmentation network (paper reconstruction).

    Distinct from the generic TransUNet baseline: explicit paper constants
    (12 layers / 768 dim / 3 skips / (256,128,64,16) decoder), full-
    resolution planning, and a bundled paper training protocol
    (see NNTRANSUNET_TRAINING and the build_* factories below).

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Output segmentation classes
        vit_pretrained_tag: timm tag for ViT-B/16 blocks (ImageNet-21k)
        pretrained_backbone: Load ImageNet-1k ResNet-50 stem weights
        img_size: Native input size for positional interpolation
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        vit_pretrained_tag: str = "vit_base_patch16_224.augreg_in21k",
        pretrained_backbone: bool = True,
        img_size: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.vit_pretrained_tag = vit_pretrained_tag

        import timm
        import torchvision.models as tv_models

        # --- ResNet-50 CNN stem: skips at /2 (64 ch), /4 (256 ch),
        # --- /8 (512 ch); deep features at /16 (1024 ch) ---
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = tv_models.resnet50(weights=weights)
        if in_channels != 3:
            adapted = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                adapted.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = adapted
        self.stem_head = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3

        # --- ViT-B transformer encoder (12 layers, dim 768, 12 heads) ---
        vit = timm.create_model("vit_base_patch16_224", pretrained=bool(vit_pretrained_tag), num_classes=0)
        assert len(vit.blocks) == VIT_DEPTH, f"Expected {VIT_DEPTH} ViT layers, got {len(vit.blocks)}"
        assert vit.embed_dim == VIT_EMBED_DIM, f"Expected dim {VIT_EMBED_DIM}, got {vit.embed_dim}"
        assert vit.blocks[0].attn.num_heads == VIT_HEADS
        self.patch_project = nn.Conv2d(1024, VIT_EMBED_DIM, kernel_size=1)
        self.transformer_blocks = vit.blocks
        self.transformer_norm = vit.norm
        pos_embed = vit.pos_embed.detach()  # (1, 197, 768) incl. CLS
        grid_tokens = pos_embed[:, 1:, :].reshape(1, VIT_EMBED_DIM, 14, 14)
        target_grid = img_size // VIT_PATCH
        if target_grid != 14:
            grid_tokens = F.interpolate(grid_tokens, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
        self.register_buffer('pos_embed_grid', grid_tokens.reshape(1, VIT_EMBED_DIM, target_grid, target_grid))

        # --- CNN up-convolution decoder: (256, 128, 64, 16), 3 skips ---
        self.dec_up1 = nn.ConvTranspose2d(VIT_EMBED_DIM, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256 + SKIP_CHANNELS[0], 256)
        self.dec_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + SKIP_CHANNELS[1], 128)
        self.dec_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + SKIP_CHANNELS[2], 64)
        self.dec_up4 = nn.ConvTranspose2d(64, DECODER_CHANNELS[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(DECODER_CHANNELS[3], DECODER_CHANNELS[3])
        self.final_conv = nn.Conv2d(DECODER_CHANNELS[3], num_classes, kernel_size=1)

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
        s2 = self.stem_head(x)          # /2
        s4 = self.stage1(self.pool(s2))  # /4
        s8 = self.stage2(s4)            # /8
        deep = self.stage3(s8)          # /16

        tokens = self.patch_project(deep)
        grid = tokens.shape[2:]
        pos = self.pos_embed_grid
        if pos.shape[2:] != grid:
            pos = F.interpolate(pos, size=grid, mode='bicubic', align_corners=False)
        seq = tokens.flatten(2).transpose(1, 2) + pos.flatten(2).transpose(1, 2)
        for block in self.transformer_blocks:
            seq = block(seq)
        seq = self.transformer_norm(seq)
        encoded = seq.transpose(1, 2).reshape(x.shape[0], VIT_EMBED_DIM, *grid)

        h = self._decode(encoded, s8, self.dec_up1, self.dec1)
        h = self._decode(h, s4, self.dec_up2, self.dec2)
        h = self._decode(h, s2, self.dec_up3, self.dec3)
        h = self._decode(h, None, self.dec_up4, self.dec4)
        if h.shape[2:] != input_size:
            h = F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)
        return self.final_conv(h)


def training_budget(epochs: int = 500, iters_per_epoch: int = 250) -> Dict[str, int]:
    """Paper iteration budget helpers for Kaggle planning (no training here)."""
    total_iters = epochs * iters_per_epoch
    return {"epochs": epochs, "iters_per_epoch": iters_per_epoch, "total_iterations": total_iters}


def build_nntransunet_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.99,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """SGD + Nesterov per paper §3.2.3 (lr locked to tabulated 0.01)."""
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        nesterov=True, weight_decay=weight_decay,
    )


def build_nntransunet_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int = 500,
    power: float = 0.9,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Polynomial LR decay stepped per epoch (paper §3.2.3)."""
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_epochs, power=power)


def build_nntransunet_training(
    model: nn.Module,
    overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Assemble the paper training configuration (factories only, no training).

    Returns optimizer, poly scheduler, epoch/iteration budget, batch size,
    val-loss monitoring and disabled early stopping, ready to be consumed
    by the existing Trainer (which accepts arbitrary optimizer/scheduler).
    """
    cfg = dict(NNTRANSUNET_TRAINING)
    cfg.update(overrides or {})
    optimizer = build_nntransunet_optimizer(
        model, lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = build_nntransunet_scheduler(
        optimizer, total_epochs=int(cfg["epochs"]), power=float(cfg["lr_power"]),
    )
    return {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epochs": int(cfg["epochs"]),
        "iters_per_epoch": int(cfg["iters_per_epoch"]),
        "batch_size": int(cfg["batch_size"]),
        "monitor_metric": str(cfg["monitor_metric"]),
        "early_stopping": cfg["early_stopping"],
        "config": cfg,
    }


def build_nn_transunet(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> NNTransUNet:
    """nn-TransUNet: paper architecture (ImageNet-21k ViT + IN1k R50 stem)."""
    return NNTransUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        vit_pretrained_tag="vit_base_patch16_224.augreg_in21k" if pretrained else "",
        pretrained_backbone=pretrained,
    )


# ---------------------------------------------------------------------------
# nnU-Net-style experiment planning (paper §3.1, §3.2.2; dry-run capable)
# ---------------------------------------------------------------------------
# The constants below are PROJECT REFERENCE HEURISTICS for fast,
# dependency-free estimates. They are NOT nnU-Net's planning machinery
# (nnU-Net uses ExperimentPlanner.static_estimate_VRAM_usage plus a
# GPU-targeted batch search); do not describe them as nnU-Net planning.
# Use run_nnunet_reference_planning() for outputs from the real planner.
REF_VRAM_OVERHEAD_MB = 2048.0    # assumed CUDA context / workspace reserve
REF_VRAM_PER_SAMPLE_MB = 600.0   # rough scale: paper batch 36 on 24 GB @ 256x256
REF_PLANNING_SAFETY = 0.85
PATCH_ALIGNMENT = 16         # ViT/16 tokens + 4 downsampling stages
MIN_PATCH = 128


def fingerprint_dataset(
    processed_dir: str,
    max_cases: int = 200,
) -> Dict[str, object]:
    """Measure a project-side dataset fingerprint (read-only, CPU/numpy).

    Reads up to ``max_cases`` .npz files (sorted order, deterministic) and
    records original in-plane spacings and image shapes. This is a project
    measurement helper, not nnU-Net's DatasetFingerprintExtractor (see
    run_nnunet_reference_planning() for the real machinery).
    """
    files = sorted(Path(processed_dir).glob("*.npz"))[:max(1, max_cases)]
    spacings, shapes = [], []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            spacings.append(tuple(float(v) for v in np.atleast_1d(data["original_spacing"])[:2]))
            shapes.append(tuple(int(v) for v in np.asarray(data["image"]).shape))
    return {"n_cases": len(files), "spacings": spacings, "shapes": shapes, "modality": "MRI"}


def plan_nntransunet_experiment(
    fingerprint: Dict[str, object],
    gpu_mem_gb: Optional[float] = None,
) -> Dict[str, object]:
    """Fast reference planning estimate (pure python, no GPU).

    Applies the paper's stated rules (MRI per-image z-score; tenth
    percentile on the lowest-resolution axis when anisotropic, else median
    spacing) and then a PROJECT REFERENCE HEURISTIC for patch/batch sizing
    (REF_* constants above). The heuristic is not nnU-Net planning; for
    authoritative values call run_nnunet_reference_planning().

    Returns {"rules_applied": ..., "reference_estimates": ...,
    "project_adaptations": ...}.
    """
    spacings = np.asarray(fingerprint["spacings"], dtype=np.float64)
    shapes = np.asarray(fingerprint["shapes"], dtype=np.float64)
    sy, sx = float(np.median(spacings[:, 0])), float(np.median(spacings[:, 1]))
    ratio = max(sy, sx) / max(min(sy, sx), 1e-8)
    anisotropic = bool(ratio > 3.0)  # nnU-Net anisotropy convention
    if anisotropic:
        # Tenth percentile on the lowest-resolution axis, median elsewhere.
        ty = float(np.percentile(spacings[:, 0], 10)) if sy >= sx else sy
        tx = float(np.percentile(spacings[:, 1], 10)) if sx > sy else sx
    else:
        ty, tx = sy, sx
    med_h, med_w = float(np.median(shapes[:, 0])), float(np.median(shapes[:, 1]))
    res_h, res_w = med_h * (sy / ty), med_w * (sx / tx)
    if gpu_mem_gb is None:
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            raise ValueError("gpu_mem_gb is required when CUDA is unavailable (dry-run must be explicit).")
    budget_mb = (gpu_mem_gb * 1024 - REF_VRAM_OVERHEAD_MB) * REF_PLANNING_SAFETY
    patch = int(min(res_h, res_w) // PATCH_ALIGNMENT * PATCH_ALIGNMENT)
    batch = 0
    while patch >= MIN_PATCH:
        per_sample = REF_VRAM_PER_SAMPLE_MB * (patch / 256.0) ** 2
        batch = int(budget_mb // max(per_sample, 1e-8))
        if batch >= 2:
            break
        patch -= PATCH_ALIGNMENT
    if batch < 2:
        raise ValueError("GPU budget cannot fit even the minimum patch size.")
    batch = batch // 2 * 2  # even batch size (nnU-Net convention)
    rules_applied = {
        "normalization": "per-image z-score (MRI rule)",
        "spacing_median_yx": [sy, sx],
        "spacing_max_min_ratio": ratio,
        "anisotropic": anisotropic,
        "target_spacing_yx": [ty, tx],
        "resampled_median_shape": [res_h, res_w],
        "gpu_mem_gb": gpu_mem_gb,
    }
    reference_estimates = {
        "estimator": ("project reference heuristic (REF_* constants); NOT "
                      "nnU-Net static_estimate_VRAM_usage — see "
                      "run_nnunet_reference_planning() for real planner output"),
        "planned_patch_size": [patch, patch],
        "planned_batch_size": batch,
    }
    project_adaptations = {
        "input_size": [256, 256],
        "note": ("Locked benchmark adaptation: native 256x256 project inputs "
                 "are used directly instead of re-running fingerprint "
                 "resampling; planned values above are the paper-rule "
                 "reference, not a silent replacement."),
        "splits": "fixed 70/10/20 project patients (paper used a random split)",
        "evaluator": "project HD95 + patient-level aggregation (paper: max-HD/JSC)",
    }
    return {"rules_applied": rules_applied, "reference_estimates": reference_estimates,
            "project_adaptations": project_adaptations}


def run_nnunet_reference_planning(
    processed_dir: str,
    scratch_root: str,
    dataset_id: int = 999,
    dataset_name: str = "ACDCDryRun",
    max_cases: int = 8,
    gpu_memory_target_gb: float = 8.0,
    num_processes: int = 2,
) -> Dict[str, object]:
    """Run nnU-Net v2's ACTUAL fingerprint + experiment planner (dry-run).

    Converts up to ``max_cases`` labeled .npz slices (sorted order,
    deterministic) into a scratch nnU-Net dataset (one NIfTI case per
    slice, shape (H, W, 1); through-plane spacing metadata is dry-run
    scaffolding, not a benchmark value), then invokes the real
    ``DatasetFingerprintExtractor.run()`` and
    ``ExperimentPlanner(gpu_memory_target_in_gb=...).plan_experiment()``.

    All artifacts stay under ``scratch_root`` (e.g. /tmp); the repository,
    splits, and checkpoints are untouched. nnU-Net env vars are pointed at
    the scratch tree for the duration of the call and restored afterwards.
    No training is executed.

    Returns the 2D configuration actually planned by nnU-Net
    (target spacing, median shape, patch size, batch size).
    """
    import os
    import tempfile

    import nibabel as nib
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
        DatasetFingerprintExtractor,
    )
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
        ExperimentPlanner,
    )

    scratch = Path(scratch_root)
    raw_root = scratch / "nnUNet_raw"
    pre_root = scratch / "nnUNet_preprocessed"
    res_root = scratch / "nnUNet_results"
    ds_name = f"Dataset{dataset_id}_{dataset_name}"
    ds_dir = raw_root / ds_name
    (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    picked = []
    for path in sorted(Path(processed_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            mask = np.asarray(data["mask"])
            if (mask == -1).all():
                continue  # dry-run needs labeled slices only
            values = set(np.unique(mask).tolist()) - EXPECTED_LABELS
            if values:
                raise ValueError(f"Unexpected labels {sorted(values)} in {path}")
            picked.append((path.stem, np.asarray(data["image"], dtype=np.float32), mask.astype(np.uint8)))
        if len(picked) >= max(1, max_cases):
            break
    if not picked:
        raise ValueError("No labeled slices found for the planning dry-run.")
    labels_present = set()
    for _, _, mask in picked:
        labels_present.update(np.unique(mask).tolist())
    if not {1, 2, 3} <= labels_present:
        raise ValueError(f"Dry-run subset lacks foreground classes: {sorted(labels_present)}")

    for index, (stem, image, mask) in enumerate(picked):
        case_id = f"ACDC_{index:04d}"
        affine = np.diag([1.5, 1.5, 5.0, 1.0])  # dry-run scaffolding only (see docstring)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(image[:, :, None]), affine),
                 ds_dir / "imagesTr" / f"{case_id}_0000.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(mask[:, :, None]), affine),
                 ds_dir / "labelsTr" / f"{case_id}.nii.gz")
    (ds_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "cineMRI"},
            "labels": {"background": 0, "LV": 1, "myocardium": 2, "RV": 3},
            "numTraining": len(picked),
            "file_ending": ".nii.gz",
        }, indent=2))

    saved_env = {key: os.environ.get(key) for key in
                 ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(pre_root)
    os.environ["nnUNet_results"] = str(res_root)
    try:
        fingerprint = DatasetFingerprintExtractor(ds_name, num_processes=num_processes, verbose=False).run()
        planner = ExperimentPlanner(ds_name, gpu_memory_target_in_gb=gpu_memory_target_gb)
        plans = planner.plan_experiment()
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    config_2d = plans["configurations"].get("2d")
    if config_2d is None:
        raise RuntimeError(f"Real planner produced no 2D configuration: {sorted(plans['configurations'])}")
    return {
        "source": "nnU-Net v2 installed planner (DatasetFingerprintExtractor + ExperimentPlanner)",
        "n_dryrun_cases": len(picked),
        "target_spacing": [float(v) for v in np.atleast_1d(config_2d["spacing"])],
        "median_image_size": [float(v) for v in np.atleast_1d(config_2d["median_image_size_in_voxels"])],
        "patch_size": [int(v) for v in np.atleast_1d(config_2d["patch_size"])],
        "batch_size": int(config_2d["batch_size"]),
        "normalization": list(config_2d.get("normalization_schemes", [])),
    }


# ---------------------------------------------------------------------------
# nnU-Net v2 training augmentation for nn-TransUNet (paper §3.2.3, §3.5)
# ---------------------------------------------------------------------------
# Exact nnU-Net v2 2D pipeline (nnunetv2.training.nnUNetTrainer.nnUNetTrainer.
# get_training_transforms), verified against nnU-Net v2 2.8.1 installed here:
#   - SpatialTransform: rotation ±180° (square 256 patch rule), scaling
#     (0.7, 1.4) p=0.2, no elastic deformation
#   - Gaussian noise, variance (0, 0.1), p=0.1
#   - Gaussian blur, sigma (0.5, 1.0), p=0.2
#   - Multiplicative brightness (0.75, 1.25), p=0.15
#   - Contrast (0.75, 1.25), p=0.15
#   - Low-resolution simulation, scale (0.5, 1), p=0.25
#   - Gamma (0.7, 1.5) with inversion p=0.1, without p=0.3
#   - Mirroring on both spatial axes; RemoveLabel(-1 -> 0) (no-op here:
#     supervised training masks contain no -1)
# Adaptation (documented, minimal): nnU-Net transforms consume per-sample
# {'image', 'segmentation'} torch dicts (batchgeneratorsv2 applies per
# sample; the dataloader collates), so each project sample
# ({'image': (1,H,W), 'mask': (H,W)}) is translated key-for-key with no
# batch dim added. do_dummy_2d_data_aug=False because project data
# is natively 2D. Not applied globally to other baselines.
_NNUNET_ROTATION_2D_SQUARE = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
_NNUNET_MIRROR_AXES_2D = (0, 1)


def build_nnunet_augmentation(patch_size=(256, 256)):
    """Construct nnU-Net v2's actual training augmentation (no training).

    Calls the installed nnU-Net trainer's static pipeline builder, so
    probabilities and ranges are nnU-Net's own, not project inventions.
    """
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    return nnUNetTrainer.get_training_transforms(
        patch_size=tuple(patch_size),
        rotation_for_DA=_NNUNET_ROTATION_2D_SQUARE,
        deep_supervision_scales=None,
        mirror_axes=_NNUNET_MIRROR_AXES_2D,
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=None,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    )


def get_nntransunet_train_transforms(input_size=(256, 256)):
    """nnU-Net v2 augmentation adapted to per-sample project dicts.

    Wraps each {'image': (1,H,W), 'mask': (H,W)} sample as nnU-Net's
    per-sample {'image', 'segmentation'} torch dict (batchgeneratorsv2
    transforms apply per sample; the dataloader collates, so no batch dim
    is added here) for the real nnU-Net pipeline built by
    build_nnunet_augmentation, then unwraps afterwards. Intensity,
    geometric, and mirroring ops all come from nnU-Net; the wrapper only
    translates container formats. Not applied globally to baselines.
    """
    pipeline = build_nnunet_augmentation(patch_size=tuple(input_size))

    def transform(sample):
        image = torch.as_tensor(np.asarray(sample["image"]), dtype=torch.float32)
        batch = {"image": image}
        mask = sample.get("mask", None)
        if mask is not None:
            batch["segmentation"] = torch.as_tensor(
                np.asarray(mask), dtype=torch.int64).unsqueeze(0)
        out = pipeline(**batch)
        sample["image"] = out["image"]
        if mask is not None:
            sample["mask"] = out["segmentation"].squeeze(0)
        return sample

    return transform


def build_nn_transunet(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> NNTransUNet:
    """nn-TransUNet: paper architecture (ImageNet-21k ViT + IN1k R50 stem)."""
    return NNTransUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        vit_pretrained_tag="vit_base_patch16_224.augreg_in21k" if pretrained else "",
        pretrained_backbone=pretrained,
    )
