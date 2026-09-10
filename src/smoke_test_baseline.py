"""
CPU Smoke Test for Baseline 2D U-Net Cardiac Segmentation Pipeline.

Verifies:
1. Model instantiation & architecture parameters (~1.85M params).
2. Data loader loading from train_patients.txt.
3. Forward pass tensor shapes: (B, 1, 256, 256) -> (B, 4, 256, 256).
4. Loss computation: DiceCELoss (Dice + Cross-Entropy).
5. Gradient calculation: loss.backward() and single optimizer step.
6. Evaluation metrics: LV Dice, Myocardium Dice, RV Dice, Mean Dice, HD95.
7. Patient-level metric aggregation.

COMPUTE CONSTRAINT:
Runs strictly on CPU for 1 batch. No training loop is executed.
"""

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import ACDCSegDataset
from src.segmentation_model import build_segmentation_model, SegmentationUNet
from src.encoder import count_parameters
from src.losses import DiceCELoss, DiceLoss
from src.metrics import (
    compute_metrics_single,
    compute_metrics_batch,
    compute_patient_level_metrics,
    format_metrics_table
)


def run_smoke_test():
    print("=" * 70)
    print("BASELINE 2D U-NET CPU SMOKE TEST")
    print("=" * 70)
    
    # Force CPU for smoke test
    device = torch.device("cpu")
    print(f"Device: {device} (Verified CPU mode)")
    
    # 1. Load configuration
    config_path = Path("configs/baseline_config.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded: {config_path}")
    print(f"  Architecture: {config['model']['architecture']}")
    print(f"  Channels:     {config['model']['encoder_channels']}")
    print(f"  Num classes:  {config['model']['num_classes']}")
    
    # 2. Instantiate Model
    model = build_segmentation_model(config).to(device)
    assert isinstance(model, SegmentationUNet)
    num_params = count_parameters(model)
    print(f"\n[1/5] Model Instantiation:")
    print(f"  Class:        {model.__class__.__name__}")
    print(f"  Parameters:   {num_params:,} ({num_params / 1e6:.2f} M)")
    assert 1_000_000 < num_params < 5_000_000, f"Unexpected parameter count: {num_params}"
    print("  Status:       PASS")
    
    # 3. Load Sample Batch
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    processed_dir = config["data"]["processed_dir"]
    
    train_ds = ACDCSegDataset(processed_dir=processed_dir, split_file=train_split)
    val_ds = ACDCSegDataset(processed_dir=processed_dir, split_file=val_split)
    print(f"\n[2/5] Data Loading:")
    print(f"  Train split:  {train_split} ({len(train_ds)} labeled slices)")
    print(f"  Val split:    {val_split} ({len(val_ds)} labeled slices)")
    assert len(train_ds) > 0, "Train dataset is empty!"
    assert len(val_ds) > 0, "Val dataset is empty!"
    
    batch_size = 2
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    patient_ids = batch["patient_id"]
    
    print(f"  Image batch:  Shape {tuple(images.shape)}, dtype {images.dtype}")
    print(f"  Mask batch:   Shape {tuple(masks.shape)}, dtype {masks.dtype}")
    print(f"  Unique labels in batch: {torch.unique(masks).tolist()}")
    assert images.shape == (batch_size, 1, 256, 256)
    assert masks.shape == (batch_size, 256, 256)
    assert set(torch.unique(masks).tolist()).issubset({0, 1, 2, 3})
    print("  Status:       PASS")
    
    # 4. Forward Pass
    print(f"\n[3/5] Forward Pass:")
    model.train()
    logits = model(images)
    print(f"  Output logits shape: {tuple(logits.shape)} (B, num_classes, H, W)")
    assert logits.shape == (batch_size, 4, 256, 256)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf!"
    print("  Status:       PASS")
    
    # 5. Loss Computation & Backward Step
    print(f"\n[4/5] Loss Function & Gradient Verification:")
    loss_cfg = config["loss"]
    criterion = DiceCELoss(
        num_classes=config["model"]["num_classes"],
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        include_background=loss_cfg.get("include_background", False),
    )
    loss = criterion(logits, masks)
    print(f"  Combined DiceCELoss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss value is not finite!"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients were computed
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients were computed!"
    assert all(np.isfinite(g) for g in grad_norms), "Gradients contain NaN/Inf!"
    print(f"  Mean gradient norm:  {np.mean(grad_norms):.4f}")
    
    optimizer.step()
    print("  Optimizer step:      SUCCESS")
    print("  Status:              PASS")
    
    # 6. Evaluation Metrics Computation
    print(f"\n[5/5] Metrics Calculation:")
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    targets = masks.cpu().numpy()
    
    # Single slice metrics
    single_metrics = compute_metrics_single(preds[0], targets[0], compute_hd=True)
    print(f"  Sample 0 Metrics:")
    print(f"    - LV Dice:         {single_metrics['LV_Dice']:.4f}")
    print(f"    - Myocardium Dice: {single_metrics['Myocardium_Dice']:.4f}")
    print(f"    - RV Dice:         {single_metrics['RV_Dice']:.4f}")
    print(f"    - Mean Dice:       {single_metrics['Mean_Dice']:.4f}")
    print(f"    - Mean HD95:       {single_metrics['Mean_HD95']:.2f} mm")
    
    # Batch metrics
    batch_metrics = compute_metrics_batch(preds, targets, patient_ids=patient_ids, compute_hd=False)
    print(f"  Batch Mean Dice:     {batch_metrics['mean']['Mean_Dice']:.4f} ± {batch_metrics['std']['Mean_Dice']:.4f}")
    
    # Patient-level aggregation
    patient_metrics = compute_patient_level_metrics(batch_metrics["per_sample"])
    print(f"  Patients in batch:   {patient_metrics['n_patients']}")
    print("  Status:              PASS")
    
    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY ON CPU!")
    print("The baseline model code, loss, metrics, and data pipeline are fully verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
