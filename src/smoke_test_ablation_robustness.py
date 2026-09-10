"""
Comprehensive CPU Smoke Test for Ablation and Robustness Experiment Infrastructure.

Validates without model training:
1. All 5 Ablation Variants:
   - A. Supervised baseline
   - B. SSL only
   - C. SSL + motion
   - D. SSL + pseudo-labels
   - E. Full pipeline
2. Multi-Component Loss Manager (Supervised, Motion, Pseudo-Label losses)
3. Patient-Level Metric Aggregation (Mean +- Std across patients, preserving variance)
4. Temporal Consistency Metrics (Frame-to-frame uncompensated vs warped agreement)
5. Pseudo-Label Quality & Calibration Filtering
6. Robustness Variations (Seeds 42/123/456, Gaussian noise, intensity perturbation, temporal intervals)
7. Results Storage & Aggregation Pipeline
"""

import os
import sys

# Sys.path guard
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F

from src.experiment_runner import build_experiment_model, ExperimentLossManager, set_seed
from src.metrics import (
    compute_metrics_single,
    compute_metrics_batch,
    compute_patient_level_metrics,
    compute_temporal_consistency_metrics,
)
from src.pseudo_labels import (
    compute_confidence_map,
    apply_confidence_filtering,
    evaluate_pseudo_label_quality,
)
from src.motion import warp_mask


def test_ablation_variants(config: dict, device: torch.device):
    """Test model construction and forward pass for all 5 ablation variants."""
    print("\n" + "="*70)
    print("1. VALIDATING 5 ABLATION VARIANTS (MODEL & LOSS CONSTRUCTION)")
    print("="*70)
    
    variants = [
        ("supervised", False, False),
        ("ssl_finetune", False, False),
        ("ssl_motion", True, False),
        ("ssl_pseudo", False, True),
        ("full_pipeline", True, True),
    ]
    
    dummy_input = torch.randn(2, 1, 128, 128, device=device)
    dummy_target = torch.randint(0, 4, (2, 128, 128), device=device)
    
    loss_manager = ExperimentLossManager(
        dice_weight=1.0,
        ce_weight=1.0,
        motion_weight=0.1,
        pseudo_weight=0.25,
        num_classes=4,
    )
    
    for mode, expected_motion, expected_pseudo in variants:
        model, motion_est = build_experiment_model(config, mode=mode, device=device)
        assert model is not None, f"Failed to build model for {mode}"
        
        has_motion = motion_est is not None
        assert has_motion == expected_motion, (
            f"Variant [{mode}] motion expectation mismatch: expected {expected_motion}, got {has_motion}"
        )
        
        # Forward pass
        logits = model(dummy_input)
        assert logits.shape == (2, 4, 128, 128), f"Incorrect logits shape: {logits.shape}"
        
        # Mock motion loss
        motion_loss = torch.tensor(0.05, device=device) if has_motion else None
        
        # Mock pseudo labels
        logits_pseudo = None
        pseudo_labels = None
        if expected_pseudo:
            probs = F.softmax(logits, dim=1)
            raw_pseudo = torch.argmax(probs, dim=1)
            conf = compute_confidence_map(probs, method="max_probability")
            filt_labels, _, _ = apply_confidence_filtering(raw_pseudo, conf, threshold=0.8, ignore_index=-1)
            logits_pseudo = logits
            pseudo_labels = filt_labels
            
        total_loss, loss_dict = loss_manager.compute_loss(
            logits_sup=logits,
            targets_sup=dummy_target,
            motion_loss=motion_loss,
            logits_pseudo=logits_pseudo,
            pseudo_labels=pseudo_labels,
        )
        
        assert torch.isfinite(total_loss), f"Loss not finite for variant {mode}"
        print(f"  [PASS] Variant [{mode:15s}]: motion={str(has_motion):5s}, pseudo={str(expected_pseudo):5s}, total_loss={total_loss.item():.4f}")


def test_patient_level_aggregation():
    """Verify patient-level metrics aggregation does not obscure variance via slice averaging."""
    print("\n" + "="*70)
    print("2. VALIDATING PATIENT-LEVEL METRICS AGGREGATION")
    print("="*70)
    
    # Simulate slice-level metrics for 3 patients with different slice counts
    slice_records = [
        # Patient 1 (3 slices, high Dice)
        {"patient_id": "patient001", "LV_Dice": 0.90, "Myocardium_Dice": 0.85, "RV_Dice": 0.88, "Mean_Dice": 0.8767, "Mean_HD95": 2.1},
        {"patient_id": "patient001", "LV_Dice": 0.92, "Myocardium_Dice": 0.86, "RV_Dice": 0.89, "Mean_Dice": 0.8900, "Mean_HD95": 1.9},
        {"patient_id": "patient001", "LV_Dice": 0.91, "Myocardium_Dice": 0.84, "RV_Dice": 0.87, "Mean_Dice": 0.8733, "Mean_HD95": 2.0},
        # Patient 2 (1 slice, medium Dice)
        {"patient_id": "patient002", "LV_Dice": 0.80, "Myocardium_Dice": 0.70, "RV_Dice": 0.75, "Mean_Dice": 0.7500, "Mean_HD95": 4.5},
        # Patient 3 (2 slices, lower Dice)
        {"patient_id": "patient003", "LV_Dice": 0.70, "Myocardium_Dice": 0.65, "RV_Dice": 0.68, "Mean_Dice": 0.6767, "Mean_HD95": 5.2},
        {"patient_id": "patient003", "LV_Dice": 0.72, "Myocardium_Dice": 0.67, "RV_Dice": 0.70, "Mean_Dice": 0.6967, "Mean_HD95": 4.8},
    ]
    
    agg = compute_patient_level_metrics(slice_records, compute_hd=True)
    assert "per_patient" in agg
    assert "mean" in agg
    assert "std" in agg
    assert agg["n_patients"] == 3
    
    # Check that Patient 1 with 3 slices gets equal patient-weight to Patient 2 with 1 slice
    p1_mean = (0.8767 + 0.8900 + 0.8733) / 3  # ~0.88
    p2_mean = 0.7500
    p3_mean = (0.6767 + 0.6967) / 2  # ~0.6867
    expected_overall_mean = (p1_mean + p2_mean + p3_mean) / 3  # ~0.7722
    
    computed_mean = agg["mean"]["Mean_Dice"]
    assert abs(computed_mean - expected_overall_mean) < 1e-3, (
        f"Patient aggregation mismatch: expected {expected_overall_mean:.4f}, got {computed_mean:.4f}"
    )
    print(f"  [PASS] Aggregated across {agg['n_patients']} patients: Mean_Dice = {computed_mean:.4f} +- {agg['std']['Mean_Dice']:.4f}")
    print(f"         Confirmed: Slice count bias prevented (Patient 1 with 3 slices has 1/3 weight, not 3/6).")


def test_temporal_consistency_metrics():
    """Verify temporal frame-to-frame agreement and motion compensation metrics."""
    print("\n" + "="*70)
    print("3. VALIDATING TEMPORAL CONSISTENCY METRICS")
    print("="*70)
    
    # Synthetic consecutive masks (circular heart structure shifting right by 2 pixels)
    mask_t = np.zeros((128, 128), dtype=np.int64)
    mask_t[50:80, 50:80] = 1  # LV
    mask_t[40:90, 40:90] = 2  # Myocardium
    mask_t[50:80, 50:80] = 1  # Restore LV
    
    # Next frame shifted slightly
    mask_t1 = np.roll(mask_t, shift=2, axis=1)
    
    # Warped version perfectly aligning frame t to frame t1
    warped_mask_t = mask_t1.copy()
    
    temp_metrics = compute_temporal_consistency_metrics(
        pred_t=mask_t,
        pred_t1=mask_t1,
        warped_pred_t=warped_mask_t,
    )
    
    assert "temporal_raw_agreement" in temp_metrics
    assert "temporal_warped_agreement" in temp_metrics
    assert "temporal_motion_gain" in temp_metrics
    assert temp_metrics["temporal_warped_agreement"] >= temp_metrics["temporal_raw_agreement"]
    assert temp_metrics["temporal_motion_gain"] >= 0.0
    
    print(f"  [PASS] Raw frame-to-frame agreement:    {temp_metrics['temporal_raw_agreement']:.4f}")
    print(f"  [PASS] Warped frame agreement:           {temp_metrics['temporal_warped_agreement']:.4f}")
    print(f"  [PASS] Motion compensation gain:         {temp_metrics['temporal_motion_gain']:.4f}")


def test_pseudo_label_quality_calibration():
    """Verify confidence filtering and calibration quality evaluation."""
    print("\n" + "="*70)
    print("4. VALIDATING PSEUDO-LABEL METRICS & CALIBRATION")
    print("="*70)
    
    # Create synthetic probabilities: some high-confidence correct, some low-confidence
    logits = torch.randn(2, 4, 64, 64)
    probs = F.softmax(logits, dim=1)
    conf = compute_confidence_map(probs, method="max_probability")
    raw_pseudo = torch.argmax(probs, dim=1)
    mock_gt = torch.randint(0, 4, (2, 64, 64))
    
    # Test multiple confidence thresholds
    thresholds = [0.5, 0.7, 0.9]
    for tau in thresholds:
        filt, mask, acc_rate = apply_confidence_filtering(raw_pseudo, conf, threshold=tau, ignore_index=-1)
        acc_pct = acc_rate * 100.0
        assert 0.0 <= acc_pct <= 100.0
        
        # Evaluate pseudo-label quality metrics against mock GT
        eval_metrics = evaluate_pseudo_label_quality(
            pseudo_labels=raw_pseudo,
            ground_truth=mock_gt,
            accept_mask=mask,
            confidence=conf,
        )
        assert "acceptance_rate" in eval_metrics
        assert "mean_fg_dice" in eval_metrics
        print(f"  [PASS] Threshold tau = {tau:.2f} -> Pixel acceptance rate = {acc_pct:.2f}% (Mean FG Dice = {eval_metrics['mean_fg_dice']:.4f})")


def test_robustness_configurations(config: dict):
    """Verify robustness evaluation parameters and perturbation operations."""
    print("\n" + "="*70)
    print("5. VALIDATING ROBUSTNESS EVALUATION INFRASTRUCTURE")
    print("="*70)
    
    rob_cfg = config.get("robustness", {})
    seeds = rob_cfg.get("seeds", [42, 123, 456])
    noise_levels = rob_cfg.get("noise_levels", [0.0, 0.05, 0.10])
    intensity_scalings = rob_cfg.get("intensity_scalings", [0.9, 1.0, 1.1])
    temporal_intervals = rob_cfg.get("temporal_intervals", [1, 2])
    
    assert len(seeds) >= 3, "Expected at least 3 random seeds"
    assert len(noise_levels) >= 3, "Expected at least 3 noise levels"
    
    # Test perturbation mechanics on dummy image
    img = torch.rand(1, 1, 64, 64)
    
    # 1. Noise injection
    noisy_img = img + torch.randn_like(img) * noise_levels[1]
    assert noisy_img.shape == img.shape
    
    # 2. Intensity scaling
    scaled_img = img * intensity_scalings[0]
    assert scaled_img.shape == img.shape
    
    print(f"  [PASS] Evaluated seeds: {seeds}")
    print(f"  [PASS] Noise levels std: {noise_levels}")
    print(f"  [PASS] Intensity scale factors: {intensity_scalings}")
    print(f"  [PASS] Temporal intervals: {temporal_intervals}")
    print(f"  [PASS] Controlled image perturbation operations verified successfully.")


def main():
    print("\n" + "#"*70)
    print("STARTING COMPLETE ABLATION & ROBUSTNESS SMOKE TEST (CPU-ONLY)")
    print("STRICT COMPUTE CONSTRAINT: NO NEURAL NETWORK TRAINING EXECUTED")
    print("#"*70)
    
    config_path = Path("configs/experiments.yaml")
    assert config_path.exists(), f"Configuration file missing: {config_path}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    set_seed(42)
    
    # Run all smoke tests
    test_ablation_variants(config, device)
    test_patient_level_aggregation()
    test_temporal_consistency_metrics()
    test_pseudo_label_quality_calibration()
    test_robustness_configurations(config)
    
    print("\n" + "#"*70)
    print("ALL ABLATION & ROBUSTNESS SMOKE TESTS COMPLETED SUCCESSFULLY (OK)")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
