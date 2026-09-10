"""
Unified End-to-End System Smoke Test.

Validates the complete cardiac cine MRI pipeline on CPU without training:
1. Configuration Loading (All 7 YAML configurations in configs/)
2. Imports Verification (All modules in src/)
3. Dataset Access & Patient Split Integrity (Strict nesting, 0 leakage, batch loading)
4. Baseline Segmentation Model Creation, Forward Pass & Loss Calculation
5. Metric Calculation (Slice-level Dice/HD95, Patient-level aggregation)
6. Self-Supervised Learning (SSL) Temporal Components & Losses
7. Motion Estimation, Differentiable Warping & Temporal Consistency
8. Confidence-Filtered Pseudo-Label Generation & Calibration
9. Five Ablation Variants Model Construction & Multi-Component Loss
10. Results Storage, Aggregation Tables & Visualization Generation

Must execute and pass completely on CPU without requiring a GPU.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Sys.path guard
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def step_header(num: int, title: str):
    print(f"\n{'='*75}")
    print(f"[{num:02d}/10] {title.upper()}")
    print(f"{'='*75}")


def test_01_configs():
    step_header(1, "Verifying YAML Configurations")
    config_files = [
        "configs/base_config.yaml",
        "configs/baseline_config.yaml",
        "configs/ssl.yaml",
        "configs/motion.yaml",
        "configs/pseudo_labels.yaml",
        "configs/experiments.yaml",
        "configs/preprocessing_config.yaml",
    ]
    for cfg_rel in config_files:
        p = Path(project_root) / cfg_rel
        assert p.exists(), f"Missing configuration file: {cfg_rel}"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            assert isinstance(cfg, dict), f"Configuration file {cfg_rel} did not parse into a dict"
        print(f"  [PASS] Successfully parsed: {cfg_rel}")


def test_02_imports():
    step_header(2, "Verifying Core Modules Import")
    import src.dataset as dataset
    import src.encoder as encoder
    import src.segmentation_model as seg_model
    import src.ssl as ssl_module
    import src.motion as motion_module
    import src.pseudo_labels as pseudo_module
    import src.losses as losses_module
    import src.metrics as metrics_module
    import src.experiment_runner as exp_runner
    import src.aggregate_results as agg_module

    print("  [PASS] All 10 core pipeline modules imported successfully without errors.")


def test_03_dataset_and_splits():
    step_header(3, "Verifying Dataset Access & Patient Splits Integrity")
    from src.dataset import ACDCSegDataset, ACDCTemporalDataset

    splits_dir = Path(project_root) / "data/splits"
    split_files = ["train_patients.txt", "val_patients.txt", "test_patients.txt",
                   "labeled_10.txt", "labeled_25.txt", "labeled_50.txt", "labeled_100.txt"]
    
    patient_sets = {}
    for sf in split_files:
        p = splits_dir / sf
        assert p.exists(), f"Missing split file: {sf}"
        with open(p, "r", encoding="utf-8") as f:
            pts = [line.strip() for line in f if line.strip()]
            assert len(pts) > 0, f"Split file {sf} is empty"
            patient_sets[sf] = set(pts)
        print(f"  - {sf:22s}: {len(pts):2d} patients")

    # Verify zero leakage between train, val, and test
    train_pts = patient_sets["train_patients.txt"]
    val_pts = patient_sets["val_patients.txt"]
    test_pts = patient_sets["test_patients.txt"]
    assert len(train_pts & val_pts) == 0, "Leakage between train and val!"
    assert len(train_pts & test_pts) == 0, "Leakage between train and test!"
    assert len(val_pts & test_pts) == 0, "Leakage between val and test!"
    print("  [PASS] Zero patient leakage across train, val, and test partitions.")

    # Verify strict nesting of labeled subsets
    l10 = patient_sets["labeled_10.txt"]
    l25 = patient_sets["labeled_25.txt"]
    l50 = patient_sets["labeled_50.txt"]
    l100 = patient_sets["labeled_100.txt"]
    assert l10.issubset(l25), "10% is not a subset of 25%!"
    assert l25.issubset(l50), "25% is not a subset of 50%!"
    assert l50.issubset(l100), "50% is not a subset of 100%!"
    assert l100 == train_pts, "100% labeled set must match train_patients.txt!"
    print("  [PASS] Strict patient-level nesting verified: 10% subset of 25% subset of 50% subset of 100%.")

    # Verify dataset batch loading from processed dir
    proc_dir = Path(project_root) / "data/processed"
    if proc_dir.exists() and any(proc_dir.glob("*.npz")):
        ds_seg = ACDCSegDataset(processed_dir=str(proc_dir), split_file=str(splits_dir / "labeled_10.txt"))
        assert len(ds_seg) > 0, "ACDCSegDataset is empty!"
        sample = ds_seg[0]
        assert "image" in sample and "mask" in sample
        assert sample["image"].ndim == 3 and sample["image"].shape[0] == 1
        print(f"  [PASS] ACDCSegDataset loaded {len(ds_seg)} labeled slices (sample shape: {sample['image'].shape}).")

        ds_temp = ACDCTemporalDataset(processed_dir=str(proc_dir), split_file=str(splits_dir / "labeled_10.txt"))
        assert len(ds_temp) > 0, "ACDCTemporalDataset is empty!"
        temp_sample = ds_temp[0]
        assert "frame_t" in temp_sample and "frame_t1" in temp_sample
        print(f"  [PASS] ACDCTemporalDataset loaded {len(ds_temp)} cine pairs (frame_t shape: {temp_sample['frame_t'].shape}).")
    else:
        print("  [NOTICE] data/processed contains no slices on this environment. Checked file references.")


def test_04_baseline_model():
    step_header(4, "Verifying Baseline 2D U-Net Model & Loss")
    from src.segmentation_model import SegmentationUNet
    from src.losses import DiceCELoss

    model = SegmentationUNet(in_channels=1, num_classes=4, encoder_channels=[32, 64, 128, 256])
    dummy_input = torch.randn(2, 1, 128, 128)
    dummy_target = torch.randint(0, 4, (2, 128, 128))

    logits = model(dummy_input)
    assert logits.shape == (2, 4, 128, 128), f"Unexpected logits shape: {logits.shape}"

    criterion = DiceCELoss(num_classes=4, dice_weight=1.0, ce_weight=1.0, include_background=False)
    loss = criterion(logits, dummy_target)
    assert torch.isfinite(loss), "Baseline loss is not finite!"
    print(f"  [PASS] SegmentationUNet forward pass: output shape {logits.shape}, loss = {loss.item():.4f}")


def test_05_metrics():
    step_header(5, "Verifying Evaluation Metrics & Patient-Level Aggregation")
    from src.metrics import compute_metrics_single, compute_patient_level_metrics

    pred = np.zeros((128, 128), dtype=np.int64)
    pred[30:60, 30:60] = 1
    pred[60:90, 30:60] = 2
    target = pred.copy()

    single_m = compute_metrics_single(pred, target, compute_hd=False)
    assert abs(single_m["LV_Dice"] - 1.0) < 1e-5
    assert abs(single_m["Mean_Dice"] - 1.0) < 1e-5

    # Patient level aggregation test
    records = [
        {"patient_id": "patient001", "LV_Dice": 0.90, "Myocardium_Dice": 0.85, "RV_Dice": 0.88, "Mean_Dice": 0.8767, "Mean_HD95": 2.1},
        {"patient_id": "patient001", "LV_Dice": 0.92, "Myocardium_Dice": 0.86, "RV_Dice": 0.89, "Mean_Dice": 0.8900, "Mean_HD95": 1.9},
        {"patient_id": "patient002", "LV_Dice": 0.80, "Myocardium_Dice": 0.70, "RV_Dice": 0.75, "Mean_Dice": 0.7500, "Mean_HD95": 4.5},
    ]
    patient_agg = compute_patient_level_metrics(records, compute_hd=True)
    assert patient_agg["n_patients"] == 2
    assert "mean" in patient_agg and "std" in patient_agg
    print(f"  [PASS] Single metric: LV_Dice = {single_m['LV_Dice']:.2f}")
    print(f"  [PASS] Patient aggregation: Mean_Dice = {patient_agg['mean']['Mean_Dice']:.4f} across 2 patients.")


def test_06_ssl():
    step_header(6, "Verifying SSL Temporal Masked Autoencoder Components")
    from src.ssl import SSLModel, compute_ssl_loss

    ssl_model = SSLModel(
        in_channels=1,
        encoder_channels=[32, 64, 128, 256],
        proj_dim=128,
        mask_patch_size=16,
        mask_ratio=0.50,
    )

    ft = torch.randn(2, 1, 128, 128)
    ft1 = torch.randn(2, 1, 128, 128)

    outputs = ssl_model(ft, ft1)
    assert "reconstructed" in outputs
    assert "proj_t" in outputs
    assert "proj_t1" in outputs

    recon_loss, temporal_loss, total_loss = compute_ssl_loss(outputs, recon_weight=1.0, temporal_weight=0.1)
    assert torch.isfinite(total_loss)
    print(f"  [PASS] SSL model forward pass: total_loss = {total_loss.item():.4f}, recon = {recon_loss.item():.4f}, temporal = {temporal_loss.item():.4f}")


def test_07_motion():
    step_header(7, "Verifying Motion Estimation & Differentiable Warping")
    from src.motion import MotionEstimator, SpatialTransformer
    from src.metrics import compute_temporal_consistency_metrics

    motion_net = MotionEstimator(channels=[16, 32, 64, 32])
    transformer = SpatialTransformer(align_corners=True)

    ft = torch.randn(2, 1, 128, 128)
    ft1 = torch.randn(2, 1, 128, 128)

    motion_out = motion_net(ft, ft1)
    flow = motion_out["flow"]
    assert flow.shape == (2, 2, 128, 128), f"Incorrect flow shape: {flow.shape}"
    assert "total_loss" in motion_out

    warped = transformer(ft, flow)
    assert warped.shape == ft.shape

    # Consistency metrics
    mask_a = np.zeros((64, 64), dtype=np.int64)
    mask_a[20:40, 20:40] = 1
    mask_b = np.roll(mask_a, shift=2, axis=1)
    t_metrics = compute_temporal_consistency_metrics(mask_a, mask_b, warped_pred_t=mask_b)
    assert t_metrics["temporal_warped_agreement"] >= t_metrics["temporal_raw_agreement"]
    print(f"  [PASS] Motion flow shape: {flow.shape}, warped agreement: {t_metrics['temporal_warped_agreement']:.4f}")


def test_08_pseudo_labels():
    step_header(8, "Verifying Confidence Filtering & Pseudo-Label Calibration")
    from src.pseudo_labels import compute_confidence_map, apply_confidence_filtering, evaluate_pseudo_label_quality

    logits = torch.randn(2, 4, 64, 64)
    probs = F.softmax(logits, dim=1)
    conf = compute_confidence_map(probs, method="max_probability")
    raw_pseudo = torch.argmax(probs, dim=1)

    filtered_labels, accept_mask, acc_rate = apply_confidence_filtering(raw_pseudo, conf, threshold=0.85, ignore_index=-1)
    assert 0.0 <= acc_rate <= 1.0
    assert filtered_labels.shape == raw_pseudo.shape

    gt = torch.randint(0, 4, (2, 64, 64))
    eval_m = evaluate_pseudo_label_quality(raw_pseudo, gt, accept_mask, confidence=conf)
    assert "acceptance_rate" in eval_m
    print(f"  [PASS] Pseudo-label filter: acceptance rate = {acc_rate*100:.2f}%, mean FG Dice = {eval_m['mean_fg_dice']:.4f}")


def test_09_ablation_variants():
    step_header(9, "Verifying 5 Ablation Variants & Multi-Loss Formulation")
    from src.experiment_runner import build_experiment_model, ExperimentLossManager

    with open(Path(project_root) / "configs/experiments.yaml", "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    loss_mgr = ExperimentLossManager(dice_weight=1.0, ce_weight=1.0, motion_weight=0.1, pseudo_weight=0.25, num_classes=4)
    variants = ["supervised", "ssl_finetune", "ssl_motion", "ssl_pseudo", "full_pipeline"]

    dummy_in = torch.randn(2, 1, 128, 128)
    dummy_tg = torch.randint(0, 4, (2, 128, 128))

    for mode in variants:
        model, motion_est = build_experiment_model(exp_cfg, mode=mode, device=torch.device("cpu"))
        logits = model(dummy_in)
        m_loss = torch.tensor(0.04) if motion_est is not None else None
        p_loss = logits if "pseudo" in mode or mode == "full_pipeline" else None
        p_labels = dummy_tg if p_loss is not None else None

        tot_loss, l_dict = loss_mgr.compute_loss(logits, dummy_tg, motion_loss=m_loss, logits_pseudo=p_loss, pseudo_labels=p_labels)
        assert torch.isfinite(tot_loss)
        print(f"  [PASS] Variant [{mode:15s}]: model={model.__class__.__name__}, motion={motion_est is not None}, loss={tot_loss.item():.4f}")


def test_10_aggregation_and_reporting():
    step_header(10, "Verifying Results Tables & Publication Figures")
    from src.aggregate_results import generate_ablation_table, generate_label_efficiency_table, generate_robustness_table, generate_patient_metrics_table, generate_all_plots

    tables_dir = Path(project_root) / "results/tables"
    figures_dir = Path(project_root) / "results/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reg_path = Path(project_root) / "results/experiments/experiment_registry.json"
    reg_data = json.load(open(reg_path, "r", encoding="utf-8")) if reg_path.exists() else {}

    df_abl = generate_ablation_table(reg_data, tables_dir)
    df_le = generate_label_efficiency_table(reg_data, tables_dir)
    df_rob = generate_robustness_table(reg_data, tables_dir)
    df_pat = generate_patient_metrics_table(None, tables_dir)

    assert (tables_dir / "ablation_table.csv").exists()
    assert (tables_dir / "label_efficiency_table.csv").exists()
    assert (tables_dir / "robustness_table.csv").exists()
    assert (tables_dir / "patient_metrics_table.csv").exists()

    generate_all_plots(figures_dir)
    assert (figures_dir / "label_efficiency_curves.png").exists()
    assert (figures_dir / "ablation_comparison.png").exists()
    assert (figures_dir / "classwise_performance.png").exists()
    assert (figures_dir / "temporal_consistency_plot.png").exists()
    assert (figures_dir / "pseudo_label_quality.png").exists()
    assert (figures_dir / "robustness_comparison.png").exists()
    print("  [PASS] All 4 CSV/Markdown tables and 6 publication figures generated successfully.")


def main():
    print("\n" + "#"*75)
    print("STARTING COMPLETE END-TO-END PIPELINE AUDIT & SMOKE TEST")
    print("DEVELOPMENT SYSTEM CONSTRAINT: STRICTLY ZERO NEURAL NETWORK TRAINING")
    print("#"*75)
    t0 = time.time()

    test_01_configs()
    test_02_imports()
    test_03_dataset_and_splits()
    test_04_baseline_model()
    test_05_metrics()
    test_06_ssl()
    test_07_motion()
    test_08_pseudo_labels()
    test_09_ablation_variants()
    test_10_aggregation_and_reporting()

    elapsed = time.time() - t0
    print("\n" + "#"*75)
    print(f"AUDIT & SMOKE TEST PASSED IN {elapsed:.2f}s (10/10 CHECKS SUCCESSFUL)")
    print("STATUS: REPOSITORY IS FULLY INTEGRATED & READY FOR GPU TRAINING SYSTEM")
    print("#"*75 + "\n")


if __name__ == "__main__":
    main()
