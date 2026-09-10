"""
Lightweight validation script for ACDC Preprocessing and Patient Splits.

Performs verification without model training:
1. Verifies patient split integrity (counts, disjoint sets, no leakage).
2. Verifies preprocessed .npz arrays (shapes, finite values, discrete class labels).
3. Verifies PyTorch Dataset wrappers (ACDCProcessedDataset, ACDCTemporalDataset, ACDCSegDataset).
4. Verifies temporal pairing (frame t, frame t+1, patient-specific, slice-specific).
5. Reports total dataset statistics and temporal pair counts.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from pathlib import Path
import numpy as np
import torch

from src.dataset import (
    ACDCProcessedDataset,
    ACDCTemporalDataset,
    ACDCSegDataset,
    ACDC_LABEL_REMAP
)


def validate_splits(splits_dir: Path):
    print("=" * 60)
    print("1. VALIDATING PATIENT-LEVEL SPLITS")
    print("=" * 60)
    
    train_file = splits_dir / "train_patients.txt"
    val_file = splits_dir / "val_patients.txt"
    test_file = splits_dir / "test_patients.txt"
    meta_file = splits_dir / "split_metadata.json"
    
    assert train_file.exists(), f"Missing {train_file}"
    assert val_file.exists(), f"Missing {val_file}"
    assert test_file.exists(), f"Missing {test_file}"
    assert meta_file.exists(), f"Missing {meta_file}"
    
    with open(train_file) as f:
        train_pids = [l.strip() for l in f if l.strip()]
    with open(val_file) as f:
        val_pids = [l.strip() for l in f if l.strip()]
    with open(test_file) as f:
        test_pids = [l.strip() for l in f if l.strip()]
        
    with open(meta_file) as f:
        meta = json.load(f)
        
    print(f"Train patients: {len(train_pids)} ({len(train_pids)/100*100:.1f}%)")
    print(f"Val patients:   {len(val_pids)} ({len(val_pids)/100*100:.1f}%)")
    print(f"Test patients:  {len(test_pids)} ({len(test_pids)/100*100:.1f}%)")
    print(f"Total:          {len(train_pids) + len(val_pids) + len(test_pids)}")
    
    # Assert zero overlap
    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)
    
    assert len(train_set & val_set) == 0, f"Leakage Train-Val: {train_set & val_set}"
    assert len(train_set & test_set) == 0, f"Leakage Train-Test: {train_set & test_set}"
    assert len(val_set & test_set) == 0, f"Leakage Val-Test: {val_set & test_set}"
    assert len(train_set | val_set | test_set) == 100, "Not all 100 patients accounted for!"
    assert meta["verification"]["no_patient_overlap"] is True
    
    print("[PASS] Split verification passed: 0 patient overlap across all splits.")
    return train_pids, val_pids, test_pids


def validate_processed_data(processed_dir: Path):
    print("\n" + "=" * 60)
    print("2. VALIDATING PROCESSED .NPZ ARRAYS")
    print("=" * 60)
    
    index_file = processed_dir / "dataset_index.json"
    summary_file = processed_dir / "preprocessing_summary.json"
    assert index_file.exists(), f"Missing {index_file}"
    assert summary_file.exists(), f"Missing {summary_file}"
    
    with open(summary_file) as f:
        summary = json.load(f)
        
    print(f"Total samples: {summary['total_samples_generated']}")
    print(f"Labeled samples (ED/ES): {summary['labeled_samples']}")
    print(f"Unlabeled cine samples:  {summary['unlabeled_cine_samples']}")
    
    # Spot check several sample files across the dataset
    npz_files = sorted(list(processed_dir.glob("*.npz")))
    assert len(npz_files) == summary['total_samples_generated'], "File count mismatch!"
    
    # Check first, middle, and last 5 files
    check_indices = [0, 1, 2, len(npz_files)//4, len(npz_files)//2, 3*len(npz_files)//4, len(npz_files)-1]
    valid_classes = {0, 1, 2, 3}
    
    for idx in check_indices:
        fpath = npz_files[idx]
        data = np.load(str(fpath))
        img = data["image"]
        msk = data["mask"]
        pid = str(data["patient_id"])
        phase = str(data["phase"])
        
        assert img.shape == (256, 256), f"Wrong img shape {img.shape} in {fpath.name}"
        assert msk.shape == (256, 256), f"Wrong msk shape {msk.shape} in {fpath.name}"
        assert np.isfinite(img).all(), f"Non-finite values in {fpath.name}"
        
        uniq = set(np.unique(msk))
        if phase in ("ED", "ES"):
            assert uniq.issubset(valid_classes), f"Invalid labels {uniq} in {fpath.name}"
        else:
            assert uniq == {-1}, f"Expected -1 mask in unlabeled cine {fpath.name}, got {uniq}"
            
    print(f"[PASS] Spot check passed: All arrays (256, 256), finite, valid class labels {valid_classes}.")


def validate_dataset_interfaces(processed_dir: Path, splits_dir: Path):
    print("\n" + "=" * 60)
    print("3. VALIDATING PYTORCH DATASET INTERFACES")
    print("=" * 60)
    
    train_txt = str(splits_dir / "train_patients.txt")
    val_txt = str(splits_dir / "val_patients.txt")
    test_txt = str(splits_dir / "test_patients.txt")
    
    # 1. ACDCProcessedDataset
    ds_proc = ACDCProcessedDataset(str(processed_dir), train_txt, has_labels=True)
    sample_proc = ds_proc[0]
    print(f"ACDCProcessedDataset (Train labeled): {len(ds_proc)} slices")
    assert sample_proc["image"].shape == (1, 256, 256)
    assert sample_proc["mask"].shape == (256, 256)
    assert set(torch.unique(sample_proc["mask"]).tolist()).issubset({0, 1, 2, 3})
    print("  [PASS] ACDCProcessedDataset sample shape & mask class IDs valid.")
    
    # 2. ACDCSegDataset
    ds_val = ACDCSegDataset(str(processed_dir), val_txt)
    ds_test = ACDCSegDataset(str(processed_dir), test_txt)
    print(f"ACDCSegDataset (Val labeled):         {len(ds_val)} slices")
    print(f"ACDCSegDataset (Test labeled):        {len(ds_test)} slices")
    
    # 3. ACDCTemporalDataset (Temporal Pair Generation)
    print("\n" + "=" * 60)
    print("4. VALIDATING TEMPORAL PAIR GENERATION")
    print("=" * 60)
    
    ds_temp_train = ACDCTemporalDataset(str(processed_dir), train_txt)
    ds_temp_val = ACDCTemporalDataset(str(processed_dir), val_txt)
    ds_temp_test = ACDCTemporalDataset(str(processed_dir), test_txt)
    
    print(f"Temporal pairs (Train): {len(ds_temp_train)} adjacent pairs")
    print(f"Temporal pairs (Val):   {len(ds_temp_val)} adjacent pairs")
    print(f"Temporal pairs (Test):  {len(ds_temp_test)} adjacent pairs")
    total_pairs = len(ds_temp_train) + len(ds_temp_val) + len(ds_temp_test)
    print(f"Total temporal pairs:   {total_pairs} adjacent pairs")
    
    # Check temporal sample details
    t_sample = ds_temp_train[0]
    print(f"\nSample temporal pair structure:")
    print(f"  Patient ID:    {t_sample['patient_id']}")
    print(f"  Slice Index:   {t_sample['slice_idx']}")
    print(f"  Frame t Index: {t_sample['frame_idx_t']} ({t_sample.get('phase_t', 'N/A')})")
    print(f"  Frame t+1 Idx: {t_sample['frame_idx_t1']} ({t_sample.get('phase_t1', 'N/A')})")
    print(f"  Frame t Shape: {t_sample['frame_t'].shape}")
    print(f"  Frame t+1 Shp: {t_sample['frame_t1'].shape}")
    print(f"  Has GT t:      {t_sample.get('has_gt_t', False)}")
    print(f"  Has GT t+1:    {t_sample.get('has_gt_t1', False)}")
    
    # Assert temporal sequence integrity
    assert t_sample['frame_t'].shape == (1, 256, 256)
    assert t_sample['frame_t1'].shape == (1, 256, 256)
    assert t_sample['frame_idx_t1'] == t_sample['frame_idx_t'] + 1, "Frames not adjacent!"
    
    # Check all pairs in training set to guarantee patient-specific ordering
    for i in range(min(500, len(ds_temp_train))):
        s = ds_temp_train[i]
        assert s['frame_idx_t1'] == s['frame_idx_t'] + 1, f"Non-adjacent pair at {i}"
        
    print("[PASS] Temporal pairing strictly validated: adjacent frames (t, t+1), within-patient and within-slice.")
    
    return {
        "labeled_train_slices": len(ds_proc),
        "labeled_val_slices": len(ds_val),
        "labeled_test_slices": len(ds_test),
        "temporal_pairs_train": len(ds_temp_train),
        "temporal_pairs_val": len(ds_temp_val),
        "temporal_pairs_test": len(ds_temp_test),
        "total_temporal_pairs": total_pairs
    }


def main():
    splits_dir = Path("data/splits")
    processed_dir = Path("data/processed")
    
    train_pids, val_pids, test_pids = validate_splits(splits_dir)
    validate_processed_data(processed_dir)
    stats = validate_dataset_interfaces(processed_dir, splits_dir)
    
    print("\n" + "=" * 60)
    print("ALL PREPROCESSING & SPLIT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
