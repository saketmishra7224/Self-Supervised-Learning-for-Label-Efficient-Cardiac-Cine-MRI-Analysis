"""
Motion estimation and temporal consistency module for cardiac cine MRI.

Implements:
1. SimpleFlowNet: Lightweight, fully differentiable CNN for dense 2D displacement
   field estimation between adjacent cardiac frames (t, t+1).
2. SpatialTransformer: Differentiable bilinear/nearest spatial warping using
   PyTorch's grid_sample operator.
3. Feature & Mask Warping: Rescaling and warping functions for multi-scale
   feature maps and categorical segmentation masks.
4. Consistency Losses: Photometric intensity matching and spatial smoothness
   regularization (Total Variation).
5. MotionEstimator: High-level wrapper providing clean inference, warping, and loss APIs.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Sys.path guard: ensure project root is on sys.path and prevent src/ from
# shadowing standard library modules.
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import ACDCTemporalDataset


# ---------------------------------------------------------------------------
# 1. SimpleFlowNet: Dense Displacement Field Estimator
# ---------------------------------------------------------------------------

class SimpleFlowNet(nn.Module):
    """
    Lightweight CNN for optical flow / displacement field estimation.
    
    Takes a concatenated adjacent frame pair [I_t, I_{t+1}] as input and
    predicts a dense 2D displacement field (dx, dy) in pixel units.
    
    Architecture:
    - 3-level residual convolution encoder (downsampling by 4x)
    - 2-level transposed convolution decoder with skip connections
    - Flow head with zero-weight initialization (initial flow = 0, identity warp)
    
    Args:
        in_channels: Number of concatenated frame channels (2 for two grayscale frames)
        channels: Channel dimensions at each stage (default [16, 32, 64, 32])
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 32]
        
        self.channels = channels
        
        # Encoder stages
        self.enc1 = self._conv_block(in_channels, channels[0])
        self.enc2 = self._conv_block(channels[0], channels[1])
        self.enc3 = self._conv_block(channels[1], channels[2])
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder stages with skip connections
        self.up3 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec3 = self._conv_block(channels[1] * 2, channels[1])
        
        self.up2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec2 = self._conv_block(channels[0] * 2, channels[0])
        
        # Flow prediction head: outputs 2 channels (dx along width, dy along height)
        self.flow_head = nn.Conv2d(channels[0], 2, kernel_size=3, padding=1)
        
        # Initialize flow head with near-zero weights for identity initialization
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
    
    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    
    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """
        Estimate dense displacement field mapping frame_t to frame_{t+1}.
        
        Args:
            frame_t: Source frame tensor (B, 1, H, W)
            frame_t1: Target frame tensor (B, 1, H, W)
            
        Returns:
            flow: Dense displacement field (B, 2, H, W) in pixel units.
                  flow[:, 0, :, :] is horizontal displacement dx (columns).
                  flow[:, 1, :, :] is vertical displacement dy (rows).
        """
        x = torch.cat([frame_t, frame_t1], dim=1)  # (B, 2, H, W)
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder path with skip connections
        d3 = self.up3(e3)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        flow = self.flow_head(d2)  # (B, 2, H, W)
        return flow


# ---------------------------------------------------------------------------
# 2. SpatialTransformer: Differentiable 2D Warping
# ---------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    """
    Differentiable 2D spatial warping module using bilinear interpolation.
    
    Applies a pixel displacement field `flow` to a source tensor `src` using
    PyTorch's `F.grid_sample`.
    
    Coordinate convention:
        Identity grid coordinates span [-1, 1] across height and width.
        Displacement (dx, dy) in pixels is normalized:
            dx_norm = dx * (2.0 / (W - 1))
            dy_norm = dy * (2.0 / (H - 1))
        Warped pixel at (x, y) samples source at (x + dx, y + dy).
    
    Args:
        size: Optional (H, W) spatial dimensions to pre-register buffer grid
        mode: Interpolation mode ('bilinear' for images/features, 'nearest' for masks)
        padding_mode: 'border' (replicates border pixel values) or 'zeros'
        align_corners: Whether grid_sample aligns corner pixels
    """
    
    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'border',
        align_corners: bool = True,
    ):
        super().__init__()
        self.size = size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        
        if size is not None:
            self._register_grid(size)
    
    def _register_grid(self, size: Tuple[int, int]):
        """Pre-compute and register the canonical identity grid [-1, 1]."""
        H, W = size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H),
            torch.linspace(-1.0, 1.0, W),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        self.register_buffer('grid', grid.unsqueeze(0))  # (1, H, W, 2)
    
    def forward(
        self,
        src: torch.Tensor,
        flow: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Warp source tensor using displacement field.
        
        Args:
            src: Source tensor to warp (B, C, H, W)
            flow: Displacement field (B, 2, H, W) in pixel units
            mode: Optional interpolation override ('bilinear' or 'nearest')
            
        Returns:
            warped: Warped tensor (B, C, H, W)
        """
        B, C, H, W = src.shape
        interpolation_mode = mode if mode is not None else self.mode
        
        # Use registered grid if dimensions match, else create on-the-fly
        if hasattr(self, 'grid') and self.grid is not None and self.grid.shape[1:3] == (H, W) and self.grid.device == flow.device:
            grid = self.grid.expand(B, -1, -1, -1)
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=flow.dtype),
                torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=flow.dtype),
                indexing='ij',
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Convert pixel displacement into normalized [-1, 1] coordinate shifts
        # Width factor (x-direction, index 0) and Height factor (y-direction, index 1)
        denom_w = (W - 1) if self.align_corners else W
        denom_h = (H - 1) if self.align_corners else H
        
        norm_factor = torch.tensor(
            [2.0 / max(denom_w, 1), 2.0 / max(denom_h, 1)],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 2, 1, 1)
        
        flow_normalized = flow * norm_factor  # (B, 2, H, W)
        sample_grid = grid + flow_normalized.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        warped = F.grid_sample(
            src,
            sample_grid,
            mode=interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return warped


# ---------------------------------------------------------------------------
# 3. Feature and Mask Warping Helpers
# ---------------------------------------------------------------------------

def warp_features(
    features: torch.Tensor,
    flow: torch.Tensor,
    spatial_transformer: Optional[SpatialTransformer] = None,
) -> torch.Tensor:
    """
    Warp multi-scale encoder features from frame_t space to frame_{t+1} space.
    
    If feature spatial resolution (H_f, W_f) differs from full flow resolution (H, W),
    the flow is bilinearly downsampled and scaled proportionally.
    
    Args:
        features: (B, C, H_f, W_f) encoder feature map
        flow: (B, 2, H, W) displacement field at full resolution
        spatial_transformer: Optional pre-allocated SpatialTransformer instance
        
    Returns:
        warped_features: (B, C, H_f, W_f) warped feature map
    """
    _, _, H_f, W_f = features.shape
    _, _, H, W = flow.shape
    
    if (H_f, W_f) != (H, W):
        scale_h = H_f / float(H)
        scale_w = W_f / float(W)
        flow_scaled = F.interpolate(flow, size=(H_f, W_f), mode='bilinear', align_corners=False)
        flow_scaled = flow_scaled.clone()
        flow_scaled[:, 0] *= scale_w
        flow_scaled[:, 1] *= scale_h
    else:
        flow_scaled = flow
    
    if spatial_transformer is None:
        spatial_transformer = SpatialTransformer(align_corners=True)
    
    return spatial_transformer(features, flow_scaled, mode='bilinear')


def warp_mask(
    mask: torch.Tensor,
    flow: torch.Tensor,
    spatial_transformer: Optional[SpatialTransformer] = None,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Warp segmentation mask or soft predictions using displacement field.
    
    - Hard discrete masks (B, H, W): one-hot encodes, warps bilinearly, then takes argmax.
    - Soft probability maps / logits (B, C, H, W): warps bilinearly across all channels.
    
    Args:
        mask: (B, H, W) integer label mask or (B, C, H, W) probability/logit tensor
        flow: (B, 2, H, W) displacement field
        spatial_transformer: Optional SpatialTransformer
        num_classes: Number of segmentation classes
        
    Returns:
        Warped mask in same format as input
    """
    if spatial_transformer is None:
        spatial_transformer = SpatialTransformer(align_corners=True)
    
    if mask.dim() == 3:
        # Categorical mask: convert to one-hot for continuous bilinear warping
        B, H, W = mask.shape
        one_hot = F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        warped_onehot = spatial_transformer(one_hot, flow, mode='bilinear')
        return torch.argmax(warped_onehot, dim=1)
    elif mask.dim() == 4:
        return spatial_transformer(mask, flow, mode='bilinear')
    else:
        raise ValueError(f"Invalid mask shape: {mask.shape}, expected 3D or 4D tensor")


# ---------------------------------------------------------------------------
# 4. Consistency Losses
# ---------------------------------------------------------------------------

def compute_flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """
    Total Variation (TV) first-order smoothness loss for displacement fields.
    
    Penalizes high-frequency gradients in the displacement field to enforce
    spatially coherent, physically plausible cardiac tissue motion:
        L_smooth = mean(|dx/du| + |dx/dv| + |dy/du| + |dy/dv|)
        
    Args:
        flow: (B, 2, H, W) displacement field
        
    Returns:
        Scalar smoothness loss tensor
    """
    diff_x = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    diff_y = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return diff_x + diff_y


def compute_photometric_loss(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    flow: torch.Tensor,
    spatial_transformer: Optional[SpatialTransformer] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Photometric consistency loss: warped frame_t should match target frame_{t+1}.
    
    Computes L1 intensity error between W(frame_t, flow) and frame_{t+1}.
    
    Args:
        frame_t: Source frame (B, 1, H, W)
        frame_t1: Target frame (B, 1, H, W)
        flow: Estimated displacement field (B, 2, H, W)
        spatial_transformer: Optional SpatialTransformer
        
    Returns:
        photo_loss: Scalar L1 photometric loss
        warped_t: Warped frame_t (B, 1, H, W)
    """
    if spatial_transformer is None:
        spatial_transformer = SpatialTransformer(align_corners=True)
    
    warped_t = spatial_transformer(frame_t, flow, mode='bilinear')
    photo_loss = F.l1_loss(warped_t, frame_t1)
    return photo_loss, warped_t


def compute_temporal_consistency_loss(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    flow: torch.Tensor,
    spatial_transformer: Optional[SpatialTransformer] = None,
    photometric_weight: float = 1.0,
    smoothness_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Combined motion self-supervision loss:
        L_motion = lambda_photo * L_photo + lambda_smooth * L_smooth
        
    Args:
        frame_t: Source frame (B, 1, H, W)
        frame_t1: Target frame (B, 1, H, W)
        flow: Estimated flow (B, 2, H, W)
        spatial_transformer: Optional SpatialTransformer
        photometric_weight: Scalar weight for photometric loss
        smoothness_weight: Scalar weight for smoothness loss
        
    Returns:
        Dict with 'photo_loss', 'smooth_loss', 'total_loss', and 'warped_t'
    """
    photo_loss, warped_t = compute_photometric_loss(
        frame_t, frame_t1, flow, spatial_transformer=spatial_transformer
    )
    smooth_loss = compute_flow_smoothness(flow)
    total_loss = photometric_weight * photo_loss + smoothness_weight * smooth_loss
    
    return {
        'photo_loss': photo_loss,
        'smooth_loss': smooth_loss,
        'total_loss': total_loss,
        'warped_t': warped_t,
    }


# ---------------------------------------------------------------------------
# 5. MotionEstimator: High-Level Motion Wrapper
# ---------------------------------------------------------------------------

class MotionEstimator(nn.Module):
    """
    High-level Motion & Temporal Consistency Module.
    
    Combines SimpleFlowNet and SpatialTransformer into a unified interface
    for motion estimation, feature warping, mask warping, and loss calculation.
    """
    
    def __init__(
        self,
        channels: Optional[List[int]] = None,
        align_corners: bool = True,
        padding_mode: str = 'border',
        photometric_weight: float = 1.0,
        smoothness_weight: float = 0.1,
    ):
        super().__init__()
        self.flownet = SimpleFlowNet(in_channels=2, channels=channels)
        self.transformer = SpatialTransformer(
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        self.photometric_weight = photometric_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self, frame_t: torch.Tensor, frame_t1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate flow, warp frame_t, and compute self-supervised motion losses.
        
        Args:
            frame_t: (B, 1, H, W)
            frame_t1: (B, 1, H, W)
            
        Returns:
            Dict containing flow, warped_t, photo_loss, smooth_loss, and total_loss.
        """
        flow = self.flownet(frame_t, frame_t1)
        loss_dict = compute_temporal_consistency_loss(
            frame_t=frame_t,
            frame_t1=frame_t1,
            flow=flow,
            spatial_transformer=self.transformer,
            photometric_weight=self.photometric_weight,
            smoothness_weight=self.smoothness_weight,
        )
        loss_dict['flow'] = flow
        return loss_dict
    
    def warp_features(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp feature maps with automatic flow scaling."""
        return warp_features(features, flow, self.transformer)
    
    def warp_mask(self, mask: torch.Tensor, flow: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
        """Warp discrete or continuous segmentation mask."""
        return warp_mask(mask, flow, self.transformer, num_classes=num_classes)


# ---------------------------------------------------------------------------
# 6. Diagnostic Visualization Helper
# ---------------------------------------------------------------------------

def create_motion_diagnostic_figure(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    flow: np.ndarray,
    warped_t: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Generate a 5-panel diagnostic figure visualizing motion compensation:
    1. Frame t (Source)
    2. Frame t+1 (Target)
    3. Estimated Flow Magnitude (Spatial displacement in pixels)
    4. Warped Frame t (Motion-compensated toward t+1)
    5. Residual Difference Comparison: Uncompensated |t+1 - t| vs Compensated |t+1 - Warped(t)|
    """
    import matplotlib.pyplot as plt
    
    # Extract components
    img_t = frame_t.squeeze()
    img_t1 = frame_t1.squeeze()
    img_warped = warped_t.squeeze()
    
    # Compute flow magnitude: sqrt(dx^2 + dy^2)
    dx = flow[0]
    dy = flow[1]
    flow_mag = np.sqrt(dx**2 + dy**2)
    
    # Residual error maps
    raw_diff = np.abs(img_t1 - img_t)
    comp_diff = np.abs(img_t1 - img_warped)
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    
    axes[0].imshow(img_t, cmap='gray')
    axes[0].set_title("Source: Frame t", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_t1, cmap='gray')
    axes[1].set_title("Target: Frame t+1", fontsize=12)
    axes[1].axis('off')
    
    im_flow = axes[2].imshow(flow_mag, cmap='viridis')
    axes[2].set_title(f"Flow Magnitude (Max: {flow_mag.max():.2f} px)", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im_flow, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(img_warped, cmap='gray')
    axes[3].set_title("Warped: W(Frame t, flow)", fontsize=12)
    axes[3].axis('off')
    
    im_err = axes[4].imshow(comp_diff, cmap='inferno')
    axes[4].set_title(f"Compensated Residual |t+1 - W(t)|\n(Mean: {comp_diff.mean():.4f} vs Raw: {raw_diff.mean():.4f})", fontsize=11)
    axes[4].axis('off')
    plt.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Motion diagnostic figure saved to: {save_path}")
    
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI & Smoke-Test
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cardiac Cine MRI Motion Estimation & Temporal Consistency Module"
    )
    parser.add_argument(
        "--config", type=str, default="configs/motion.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run lightweight CPU smoke test verifying flow, warping, and loss"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device)
    print(f"Loading Motion Module on device: {device}")
    
    model = MotionEstimator(
        channels=config['motion_model'].get('channels', [16, 32, 64, 32]),
        align_corners=config['warping'].get('align_corners', True),
        padding_mode=config['warping'].get('padding_mode', 'border'),
        photometric_weight=config['loss'].get('photometric_weight', 1.0),
        smoothness_weight=config['loss'].get('smoothness_weight', 0.1),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SimpleFlowNet trainable parameters: {total_params:,}")
    
    if args.smoke_test:
        print("\n--- Running Motion Module Smoke Test ---")
        # Load sample from dataset
        data_cfg = config['data']
        dataset = ACDCTemporalDataset(
            processed_dir=data_cfg['processed_dir'],
            split_file=data_cfg['train_split'],
        )
        sample = dataset[0]
        frame_t = sample['frame_t'].unsqueeze(0).to(device)
        frame_t1 = sample['frame_t1'].unsqueeze(0).to(device)
        
        output = model(frame_t, frame_t1)
        print(f"Flow shape:       {output['flow'].shape}")
        print(f"Warped shape:     {output['warped_t'].shape}")
        print(f"Photometric Loss: {output['photo_loss'].item():.4f}")
        print(f"Smoothness Loss:  {output['smooth_loss'].item():.4f}")
        print(f"Total Loss:       {output['total_loss'].item():.4f}")
        
        # Test backward
        output['total_loss'].backward()
        has_grads = all(p.grad is not None for p in model.flownet.parameters() if p.requires_grad)
        print(f"Gradient flow verified: {has_grads}")
        assert has_grads, "Missing gradients in SimpleFlowNet!"
        print("--- Motion Module Smoke Test PASSED ---")


if __name__ == "__main__":
    main()
