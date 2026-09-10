"""
Standalone lightweight CPU smoke test for Confidence-Filtered Pseudo-Labeling.

Validates:
1. Loading configuration from configs/pseudo_labels.yaml.
2. Loading real labeled cardiac cine MRI slices from ACDCSegDataset.
3. Model inference: logits and probability calculation.
4. Confidence computation:
   - Maximum Softmax Probability (MSP)
   - Normalized Negative Entropy
   - Hybrid confidence
5. Motion-based temporal agreement using MotionEstimator.
6. Confidence threshold filtering and ignore_index assignment.
7. Pseudo-label quality evaluation against ground truth:
   - Acceptance rate (overall and foreground)
   - Accuracy on accepted pixels
   - Per-class pseudo-label Dice (LV, Myocardium, RV)
   - Mean foreground Dice
   - Confidence statistics
8. Metadata export to results/pseudo_labels/pseudo_label_metadata.json.
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
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.segmentation_model import SegmentationUNet
from src.motion import MotionEstimator
from src.dataset import ACDCSegDataset, ACDCTemporalDataset
from src.pseudo_labels import (
    compute_confidence_map,
    compute_temporal_agreement,
    apply_confidence_filtering,
    evaluate_pseudo_label_quality,
    save_pseudo_label_metadata,
)


def run_pseudo_label_smoke_test(config_path: str = "configs/pseudo_labels.yaml"):
    print("=" * 70)
    print("STARTING LIGHTWEIGHT CONFIDENCE PSEUDO-LABELING SMOKE TEST")
    print("=" * 70)
    
    # 1. Load Configuration
    cfg_file = Path(config_path)
    assert cfg_file.exists(), f"Config file not found: {config_path}"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    print(f"[1/7] Configuration loaded from: {config_path}")
    print(f"      Confidence metric:    {config['filtering']['confidence_metric']}")
    print(f"      Threshold:            {config['filtering']['confidence_threshold']}")
    print(f"      Temporal consistency: {config['filtering']['use_temporal_consistency']}")
    
    # 2. Load Real Labeled Samples from ACDCSegDataset
    data_cfg = config['data']
    val_split = data_cfg['val_split']
    processed_dir = data_cfg['processed_dir']
    
    val_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=val_split,
    )
    print(f"[2/7] Validation dataset loaded: {len(val_dataset)} labeled slices available.")
    assert len(val_dataset) > 0, "No labeled slices found!"
    
    loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    images = batch['image']
    gt_masks = batch['mask']
    print(f"      Batch loaded: images={images.shape}, gt_masks={gt_masks.shape}")
    
    # 3. Model Inference (CPU)
    device = torch.device("cpu")
    model = SegmentationUNet(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        encoder_channels=config['model']['encoder_channels'],
    ).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        raw_pseudo = torch.argmax(probs, dim=1)
    
    print(f"[3/7] Model inference completed:")
    print(f"      Logits shape:        {logits.shape}")
    print(f"      Probabilities shape: {probs.shape}")
    # Verify probabilities sum to 1
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities do not sum to 1!"
    print("      Probabilities sum-to-1 verified across all pixels.")
    
    # 4. Confidence Metric Calculation
    conf_msp = compute_confidence_map(probs, method="max_probability")
    conf_ent = compute_confidence_map(probs, method="entropy")
    conf_hyb = compute_confidence_map(probs, method="hybrid")
    
    print(f"[4/7] Confidence maps computed:")
    print(f"      MSP Mean:    {conf_msp.mean().item():.4f} (range: [{conf_msp.min().item():.3f}, {conf_msp.max().item():.3f}])")
    print(f"      Entropy Mean:{conf_ent.mean().item():.4f} (range: [{conf_ent.min().item():.3f}, {conf_ent.max().item():.3f}])")
    print(f"      Hybrid Mean: {conf_hyb.mean().item():.4f} (range: [{conf_hyb.min().item():.3f}, {conf_hyb.max().item():.3f}])")
    
    assert conf_msp.shape == (2, 256, 256), f"Bad conf shape {conf_msp.shape}"
    assert (conf_msp >= 0.25).all() and (conf_msp <= 1.0).all(), "MSP values outside theoretical bounds [0.25, 1.0]!"
    
    # 5. Temporal Consistency Verification
    temporal_dataset = ACDCTemporalDataset(
        processed_dir=processed_dir,
        split_file=val_split,
    )
    temp_sample = temporal_dataset[0]
    frame_t = temp_sample['frame_t'].unsqueeze(0)
    frame_t1 = temp_sample['frame_t1'].unsqueeze(0)
    
    motion_est = MotionEstimator(channels=[16, 32, 64, 32]).to(device).eval()
    with torch.no_grad():
        probs_t = F.softmax(model(frame_t), dim=1)
        probs_t1 = F.softmax(model(frame_t1), dim=1)
        motion_out = motion_est(frame_t, frame_t1)
        flow = motion_out['flow']
        
        agreement = compute_temporal_agreement(probs_t, probs_t1, flow, motion_est.transformer)
    
    print(f"[5/7] Temporal consistency agreement verified:")
    print(f"      Agreement map shape: {agreement.shape}")
    print(f"      Mean agreement:      {agreement.mean().item():.4f}")
    assert agreement.shape == (1, 256, 256), f"Bad agreement shape {agreement.shape}"
    
    # 6. Confidence Filtering
    tau = config['filtering']['confidence_threshold']
    ignore_idx = config['filtering']['ignore_index']
    
    filtered_labels, accept_mask, acc_rate = apply_confidence_filtering(
        pseudo_labels=raw_pseudo,
        confidence=conf_msp,
        threshold=tau,
        ignore_index=ignore_idx,
    )
    
    print(f"[6/7] Confidence filtering applied (threshold={tau}):")
    print(f"      Acceptance rate:      {acc_rate*100:.2f}%")
    print(f"      Accepted pixels:      {accept_mask.sum().item():,}")
    print(f"      Rejected pixels:      {(~accept_mask).sum().item():,} (set to {ignore_idx})")
    assert (filtered_labels[~accept_mask] == ignore_idx).all(), "Rejected pixels not set to ignore_index!"
    
    # 7. Quality Evaluation against Ground Truth & Metadata Export
    metrics = evaluate_pseudo_label_quality(
        pseudo_labels=raw_pseudo,
        ground_truth=gt_masks,
        accept_mask=accept_mask,
        confidence=conf_msp,
        num_classes=config['model']['num_classes'],
        ignore_index=ignore_idx,
    )
    
    print(f"[7/7] Pseudo-label quality metrics against GT:")
    print(f"      Acceptance rate:          {metrics['acceptance_rate']*100:.2f}%")
    print(f"      Foreground acceptance:    {metrics['fg_acceptance_rate']*100:.2f}%")
    print(f"      Accepted pixel accuracy:  {metrics['accuracy_on_accepted']*100:.2f}%")
    print(f"      Mean Foreground Dice:     {metrics['mean_fg_dice']:.4f}")
    print(f"      LV Dice:                  {metrics['dice_lv']:.4f}")
    print(f"      Myocardium Dice:          {metrics['dice_myo']:.4f}")
    print(f"      RV Dice:                  {metrics['dice_rv']:.4f}")
    
    # Export metadata
    out_dir = Path(config['logging']['output_dir'])
    metadata = {
        "experiment": config['experiment_name'],
        "confidence_threshold": tau,
        "confidence_metric": config['filtering']['confidence_metric'],
        "samples_evaluated": 2,
        "metrics": metrics,
    }
    save_pseudo_label_metadata(metadata, out_dir, config['logging']['metadata_file'])
    
    meta_path = out_dir / config['logging']['metadata_file']
    assert meta_path.exists(), "Metadata file was not created!"
    with open(meta_path, "r") as f:
        loaded_meta = json.load(f)
    assert loaded_meta['experiment'] == config['experiment_name'], "Metadata content mismatch!"
    print(f"      Metadata verified on disk at: {meta_path}")
    
    print("=" * 70)
    print("LIGHTWEIGHT PSEUDO-LABEL SMOKE TEST: ALL CHECKS PASSED (OK)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_pseudo_label_smoke_test()
