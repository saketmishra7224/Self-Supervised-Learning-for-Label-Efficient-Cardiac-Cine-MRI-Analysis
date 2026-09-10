"""
Lightweight CPU smoke test for Motion Estimation and Temporal Consistency.

Validates:
1. ACDCTemporalDataset loading of adjacent frame pairs (t, t+1).
2. SimpleFlowNet forward pass and displacement field output dimensions (B, 2, H, W).
3. Differentiable SpatialTransformer warping of images (B, 1, H, W).
4. Feature-level warping with spatial scaling on multi-scale feature maps (B, C, H_f, W_f).
5. Segmentation mask warping with discrete classes (B, H, W).
6. Photometric loss, Total Variation flow smoothness loss, and weighted total loss.
7. Backward gradient computation and parameter update step.
8. Diagnostic visualization generation (results/figures/motion_diagnostics.png).
"""

import os
import sys

# Sys.path guard
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.motion import (
    SimpleFlowNet,
    SpatialTransformer,
    MotionEstimator,
    warp_features,
    warp_mask,
    create_motion_diagnostic_figure,
)
from src.dataset import ACDCTemporalDataset


def run_motion_smoke_test(config_path: str = "configs/motion.yaml"):
    print("=" * 70)
    print("STARTING LIGHTWEIGHT MOTION / TEMPORAL CONSISTENCY SMOKE TEST")
    print("=" * 70)
    
    # 1. Load Configuration
    cfg_file = Path(config_path)
    assert cfg_file.exists(), f"Configuration file {config_path} not found!"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    print(f"[1/7] Configuration successfully loaded from: {config_path}")
    print(f"      Motion Method:       {config['motion_model']['method']}")
    print(f"      Channels:            {config['motion_model']['channels']}")
    print(f"      Photometric Weight:  {config['loss']['photometric_weight']}")
    print(f"      Smoothness Weight:   {config['loss']['smoothness_weight']}")
    
    # 2. Dataset & Pair Loading
    data_cfg = config['data']
    dataset = ACDCTemporalDataset(
        processed_dir=data_cfg['processed_dir'],
        split_file=data_cfg['train_split'],
    )
    print(f"[2/7] Temporal dataset loaded: {len(dataset):,} available adjacent pairs.")
    
    # Sample batch of 2 pairs
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    frame_t = batch['frame_t']
    frame_t1 = batch['frame_t1']
    print(f"      Batch shape: frame_t={frame_t.shape}, frame_t1={frame_t1.shape}")
    assert frame_t.shape == (2, 1, 256, 256), f"Unexpected shape {frame_t.shape}"
    
    # 3. Instantiate Motion Estimator
    device = torch.device("cpu")
    motion_module = MotionEstimator(
        channels=config['motion_model']['channels'],
        align_corners=config['warping']['align_corners'],
        padding_mode=config['warping']['padding_mode'],
        photometric_weight=config['loss']['photometric_weight'],
        smoothness_weight=config['loss']['smoothness_weight'],
    ).to(device)
    
    param_count = sum(p.numel() for p in motion_module.parameters() if p.requires_grad)
    print(f"[3/7] MotionEstimator initialized: {param_count:,} trainable parameters.")
    
    # 4. Forward Motion Estimation & Image Warping
    out = motion_module(frame_t, frame_t1)
    flow = out['flow']
    warped_t = out['warped_t']
    photo_loss = out['photo_loss']
    smooth_loss = out['smooth_loss']
    total_loss = out['total_loss']
    
    print(f"[4/7] Forward pass completed:")
    print(f"      Displacement field: {flow.shape} (dx, dy)")
    print(f"      Warped frame:       {warped_t.shape}")
    print(f"      Photometric loss:   {photo_loss.item():.4f}")
    print(f"      Smoothness loss:    {smooth_loss.item():.4f}")
    print(f"      Total motion loss:  {total_loss.item():.4f}")
    
    assert flow.shape == (2, 2, 256, 256), f"Bad flow shape {flow.shape}"
    assert warped_t.shape == (2, 1, 256, 256), f"Bad warped shape {warped_t.shape}"
    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "Loss is NaN/Inf!"
    
    # 5. Feature Warping Test (Downsampled resolution)
    # Simulate encoder feature map at level 1: (2, 64, 128, 128)
    dummy_features = torch.randn(2, 64, 128, 128)
    warped_features = motion_module.warp_features(dummy_features, flow)
    assert warped_features.shape == (2, 64, 128, 128), f"Bad warped features shape {warped_features.shape}"
    print(f"[5/7] Feature warping verified: {dummy_features.shape} -> {warped_features.shape}")
    
    # 6. Mask Warping Test (Categorical 4-class)
    dummy_mask = torch.randint(0, 4, (2, 256, 256))
    warped_mask = motion_module.warp_mask(dummy_mask, flow, num_classes=4)
    assert warped_mask.shape == (2, 256, 256), f"Bad warped mask shape {warped_mask.shape}"
    print(f"      Mask warping verified:    {dummy_mask.shape} -> {warped_mask.shape}")
    
    # 7. Backward Gradient Flow & Parameter Step
    optimizer = torch.optim.AdamW(motion_module.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    
    has_grads = all(p.grad is not None and not torch.isnan(p.grad).any() for p in motion_module.flownet.parameters() if p.requires_grad)
    assert has_grads, "SimpleFlowNet gradients are missing or NaN!"
    optimizer.step()
    print(f"[6/7] Backward gradient flow verified: All SimpleFlowNet layers received valid gradients.")
    print("      Optimizer step executed successfully.")
    
    # 8. Generate Diagnostic Visualization
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    diag_path = fig_dir / "motion_diagnostics.png"
    
    # Use first sample in batch
    sample_t = frame_t[0].detach().numpy()
    sample_t1 = frame_t1[0].detach().numpy()
    sample_flow = flow[0].detach().numpy()
    sample_warped = warped_t[0].detach().numpy()
    
    create_motion_diagnostic_figure(
        frame_t=sample_t,
        frame_t1=sample_t1,
        flow=sample_flow,
        warped_t=sample_warped,
        save_path=str(diag_path),
    )
    assert diag_path.exists(), "Diagnostic figure was not saved!"
    print(f"[7/7] Diagnostic visualization saved to: {diag_path}")
    
    print("=" * 70)
    print("LIGHTWEIGHT MOTION SMOKE TEST RESULT: ALL CHECKS PASSED (OK)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_motion_smoke_test()
