"""
ACDC Preprocessing Pipeline (Step 2).

Standardizes ACDC Cardiac Cine MRI volumes:
1. Spatial resampling to target in-plane spacing (1.5 x 1.5 mm).
   - Images: Spline interpolation order 3 (bicubic).
   - Masks: Nearest-neighbor interpolation order 0 (preserves discrete labels).
2. Center-crop / zero-pad to 256 x 256.
3. Per-volume Z-score intensity normalization to avoid data leakage.
4. Label remapping to project specification:
   - 0: Background
   - 1: LV cavity
   - 2: Myocardium
   - 3: RV cavity
5. Generates 2D / 2D+t slice representations (.npz) preserving patient IDs and temporal sequence.
6. Stratified patient-level splitting (70% Train / 10% Val / 20% Test) by pathology.
7. Comprehensive sanity checks and visual before/after verification.
"""

import os
import sys

# Ensure project root is on sys.path and prevent src/ from shadowing stdlib modules (e.g. ssl)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# Import reusable utilities from src.dataset
from src.dataset import (
    parse_info_cfg,
    remap_labels,
    resample_slice,
    center_crop_or_pad,
    normalize_intensity,
    ACDC_LABEL_REMAP
)


def process_single_patient_volume(
    pdir: Path,
    processed_dir: Path,
    target_size: Tuple[int, int] = (256, 256),
    target_spacing: Tuple[float, float] = (1.5, 1.5),
    norm_strategy: str = "zscore",
    epsilon: float = 1e-8
) -> Dict[str, Any]:
    """Process a single patient volume into 2D / 2D+t .npz files."""
    pid = pdir.name
    info_file = pdir / "Info.cfg"
    info = parse_info_cfg(str(info_file)) if info_file.exists() else {}
    ed_frame = info.get("ED")
    es_frame = info.get("ES")
    pathology = info.get("Group", "Unknown")
    
    cine_file = pdir / f"{pid}_4d.nii.gz"
    if not cine_file.exists():
        raise FileNotFoundError(f"4D cine missing for {pid}")
        
    nii_4d = nib.load(str(cine_file))
    vol_4d = nii_4d.get_fdata().astype(np.float32)
    pixdim = nii_4d.header.get_zooms()
    orig_spacing = (float(pixdim[0]), float(pixdim[1]))
    orig_shape = vol_4d.shape[:2]
    
    n_slices = vol_4d.shape[2]
    n_frames = vol_4d.shape[3] if len(vol_4d.shape) == 4 else 1
    
    raw_mean = float(np.mean(vol_4d))
    raw_std = float(np.std(vol_4d))
    
    # Check if patient already completely processed
    expected_total = n_slices * n_frames
    existing_patient_files = list(processed_dir.glob(f"{pid}_frame*_slice*.npz"))
    if len(existing_patient_files) == expected_total:
        # Quick verify and build index from existing files
        p_index = []
        p_labeled = 0
        p_unlabeled = 0
        for f in sorted(existing_patient_files):
            data = np.load(str(f))
            msk = data["mask"]
            has_gt = not np.all(msk == -1)
            if has_gt:
                p_labeled += 1
            else:
                p_unlabeled += 1
            p_index.append({
                "file": f.name,
                "patient_id": pid,
                "slice_idx": int(data["slice_idx"]),
                "frame_idx": int(data["frame_idx"]),
                "phase": str(data["phase"]),
                "pathology": str(data["pathology"]),
                "has_gt": bool(has_gt),
                "classes_present": [int(c) for c in np.unique(msk) if c >= 0]
            })
        vol_4d_norm = normalize_intensity(vol_4d, method="zscore", epsilon=epsilon)
        norm_mean = float(np.mean(vol_4d_norm))
        norm_std = float(np.std(vol_4d_norm))
        return {
            "patient_id": pid,
            "pathology": pathology,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "n_slices": n_slices,
            "n_frames": n_frames,
            "orig_shape": list(orig_shape),
            "orig_spacing": list(orig_spacing),
            "labeled_slices": p_labeled,
            "unlabeled_slices": p_unlabeled,
            "total_slices": expected_total,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "index_entries": p_index,
            "skipped": True
        }

    # Normalize per-volume to prevent data leakage
    vol_4d_norm = normalize_intensity(vol_4d, method="zscore", epsilon=epsilon)
    norm_mean = float(np.mean(vol_4d_norm))
    norm_std = float(np.std(vol_4d_norm))
    
    # Load ground truth masks for ED and ES frames
    gt_masks = {}
    for phase_name, f_idx in [("ED", ed_frame), ("ES", es_frame)]:
        if f_idx is not None:
            gt_file = pdir / f"{pid}_frame{int(f_idx):02d}_gt.nii.gz"
            if gt_file.exists():
                raw_gt = nib.load(str(gt_file)).get_fdata().astype(np.int64)
                gt_masks[int(f_idx)] = remap_labels(raw_gt)
                
    p_labeled = 0
    p_unlabeled = 0
    p_index = []
    
    for f_idx in range(n_frames):
        for s_idx in range(n_slices):
            img_slice = vol_4d_norm[:, :, s_idx, f_idx]
            img_res = resample_slice(img_slice, orig_spacing, target_spacing, is_mask=False)
            img_proc = center_crop_or_pad(img_res, target_size, pad_value=0.0).astype(np.float32)
            
            has_gt = (f_idx in gt_masks)
            if has_gt:
                mask_slice = gt_masks[f_idx][:, :, s_idx]
                mask_res = resample_slice(mask_slice.astype(np.float32), orig_spacing, target_spacing, is_mask=True)
                mask_proc = center_crop_or_pad(mask_res.astype(np.int64), target_size, pad_value=0).astype(np.int64)
                p_labeled += 1
            else:
                mask_proc = np.full(target_size, -1, dtype=np.int64)
                p_unlabeled += 1
                
            if f_idx == ed_frame:
                phase_tag = "ED"
            elif f_idx == es_frame:
                phase_tag = "ES"
            else:
                phase_tag = "cine"
                
            fname = f"{pid}_frame{f_idx:02d}_slice{s_idx:02d}.npz"
            fpath = processed_dir / fname
            
            np.savez_compressed(
                str(fpath),
                image=img_proc,
                mask=mask_proc,
                patient_id=pid,
                slice_idx=s_idx,
                frame_idx=f_idx,
                phase=phase_tag,
                pathology=pathology,
                original_shape=np.array(orig_shape),
                original_spacing=np.array(orig_spacing),
                target_spacing=np.array(target_spacing),
                target_size=np.array(target_size)
            )
            
            p_index.append({
                "file": fname,
                "patient_id": pid,
                "slice_idx": s_idx,
                "frame_idx": f_idx,
                "phase": phase_tag,
                "pathology": pathology,
                "has_gt": bool(has_gt),
                "classes_present": [int(c) for c in np.unique(mask_proc) if c >= 0]
            })
            
    return {
        "patient_id": pid,
        "pathology": pathology,
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "n_slices": n_slices,
        "n_frames": n_frames,
        "orig_shape": list(orig_shape),
        "orig_spacing": list(orig_spacing),
        "labeled_slices": p_labeled,
        "unlabeled_slices": p_unlabeled,
        "total_slices": p_labeled + p_unlabeled,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "index_entries": p_index,
        "skipped": False
    }


def load_canonical_splits(splits_dir: Path, available_patient_ids: List[str]) -> Dict[str, List[str]]:
    """Load and validate the repository's fixed patient-level split artifacts."""
    split_specs = {
        "train": ("train_patients.txt", 70),
        "val": ("val_patients.txt", 10),
        "test": ("test_patients.txt", 20),
        "labeled_10": ("labeled_10.txt", 7),
        "labeled_25": ("labeled_25.txt", 17),
        "labeled_50": ("labeled_50.txt", 35),
        "labeled_100": ("labeled_100.txt", 70),
    }

    splits = {}
    for name, (filename, expected_count) in split_specs.items():
        path = splits_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required canonical split file is missing: {path}")
        with open(path, "r") as f:
            patient_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if len(patient_ids) != expected_count:
            raise ValueError(f"{path} has {len(patient_ids)} patients; expected {expected_count}")
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError(f"{path} contains duplicate patient IDs")
        splits[name] = patient_ids

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    available_set = set(available_patient_ids)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Canonical train/val/test splits overlap; refusing to preprocess")
    if train_set | val_set | test_set != available_set:
        raise ValueError("Canonical train/val/test splits do not exactly cover the raw patient cohort")

    labeled_sets = [set(splits[name]) for name in ("labeled_10", "labeled_25", "labeled_50", "labeled_100")]
    if labeled_sets[-1] != train_set:
        raise ValueError("labeled_100.txt must exactly match train_patients.txt")
    if not all(labeled_sets[i] <= labeled_sets[i + 1] for i in range(len(labeled_sets) - 1)):
        raise ValueError("Canonical labeled subsets must be nested: 10% <= 25% <= 50% <= 100%")
    if not all(label_set <= train_set for label_set in labeled_sets):
        raise ValueError("Canonical labeled subsets must be contained in the training split")

    return splits


def run_preprocessing(config_path: str = "configs/preprocessing_config.yaml") -> Dict[str, Any]:
    """Execute complete parallel ACDC preprocessing pipeline."""
    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = Path(config["data"]["raw_dir"]) / "training"
    processed_dir = Path(config["data"]["processed_dir"])
    splits_dir = Path(config["data"]["splits_dir"])
    results_dir = Path("results/preprocessing")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    target_size = tuple(config["spatial"]["target_size"])
    target_spacing = tuple(config["spatial"]["target_spacing"])
    norm_strategy = config["intensity_normalization"]["strategy"]
    epsilon = float(config["intensity_normalization"].get("epsilon", 1e-8))
    
    patient_dirs = sorted([
        d for d in raw_dir.iterdir() 
        if d.is_dir() and d.name.startswith("patient")
    ])
    num_patients = len(patient_dirs)
    print(f"==================================================")
    print(f"Starting Parallel ACDC Preprocessing for {num_patients} patients (6 workers)")
    print(f"Target Size: {target_size}, Target Spacing: {target_spacing} mm")
    print(f"Normalization: {norm_strategy}")
    print(f"==================================================")
    
    if num_patients == 0:
        raise RuntimeError(f"No patient directories found in {raw_dir}")
        
    processed_index = []
    patient_stats = {}
    total_slices_saved = 0
    total_labeled_slices = 0
    total_unlabeled_slices = 0
    raw_means, raw_stds = [], []
    norm_means, norm_stds = [], []
    
    # Run processing with ThreadPoolExecutor (6 workers)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_pid = {
            executor.submit(
                process_single_patient_volume,
                pdir,
                processed_dir,
                target_size,
                target_spacing,
                norm_strategy,
                epsilon
            ): pdir.name for pdir in patient_dirs
        }
        
        for future in as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                res = future.result()
                completed_count += 1
                patient_stats[pid] = res
                processed_index.extend(res["index_entries"])
                total_slices_saved += res["total_slices"]
                total_labeled_slices += res["labeled_slices"]
                total_unlabeled_slices += res["unlabeled_slices"]
                raw_means.append(res["raw_mean"])
                raw_stds.append(res["raw_std"])
                norm_means.append(res["norm_mean"])
                norm_stds.append(res["norm_std"])
                
                status_tag = "SKIPPED (cached)" if res.get("skipped") else "DONE"
                if completed_count % 10 == 0 or completed_count == num_patients:
                    print(f"[{completed_count:03d}/{num_patients:03d}] {pid} {status_tag} | {res['n_slices']} slices x {res['n_frames']} frames ({res['labeled_slices']} labeled, {res['unlabeled_slices']} cine)")
            except Exception as e:
                print(f"ERROR processing {pid}: {e}")
                raise
                
    # Sort index by patient_id, frame_idx, slice_idx
    processed_index.sort(key=lambda x: (x["patient_id"], x["frame_idx"], x["slice_idx"]))
    
    print(f"\nAll {num_patients} patients processed in {time.time() - start_time:.1f}s:")
    print(f"  Total samples generated: {total_slices_saved}")
    print(f"  Labeled samples (ED/ES): {total_labeled_slices}")
    print(f"  Unlabeled cine samples:  {total_unlabeled_slices}")
    
    # Save complete metadata index
    index_file = processed_dir / "dataset_index.json"
    with open(index_file, "w") as f:
        json.dump(processed_index, f, indent=2)
    print(f"Saved dataset index to {index_file}")
    
    # 2. Comprehensive Sanity Checks
    print("\n--- Running Preprocessing Sanity Checks ---")
    valid_classes = {0, 1, 2, 3}
    sanity_errors = []
    
    sample_files = sorted(list(processed_dir.glob("*.npz")))
    checked_count = 0
    for npz_path in sample_files:
        data = np.load(str(npz_path))
        img = data["image"]
        msk = data["mask"]
        
        # Spatial dimensions check
        if img.shape != target_size:
            sanity_errors.append(f"{npz_path.name}: Image shape {img.shape} != {target_size}")
        if msk.shape != target_size:
            sanity_errors.append(f"{npz_path.name}: Mask shape {msk.shape} != {target_size}")
            
        # Numerical validity
        if not np.isfinite(img).all():
            sanity_errors.append(f"{npz_path.name}: Image contains NaN or Inf values")
            
        # Mask integrity
        unique_labels = set(np.unique(msk))
        if not np.all(msk == -1):
            if not unique_labels.issubset(valid_classes):
                sanity_errors.append(f"{npz_path.name}: Invalid class labels detected: {unique_labels}")
        else:
            if unique_labels != {-1}:
                sanity_errors.append(f"{npz_path.name}: Unlabeled mask does not contain -1: {unique_labels}")
                
        checked_count += 1
        
    if sanity_errors:
        print(f"SANITY CHECK FAILED with {len(sanity_errors)} errors:")
        for err in sanity_errors[:10]:
            print(f"  - {err}")
        raise ValueError("Sanity check failed during preprocessing!")
    else:
        print(f"[OK] Sanity checks passed across all {checked_count} processed files:")
        print(f"     - Image and mask spatial dimensions strictly match {target_size}")
        print(f"     - Mask labels strictly remain within {valid_classes} (or -1 for unlabeled)")
        print(f"     - Zero NaNs, Infs, or array corruptions detected")
        print(f"     - Average raw volume mean: {np.mean(raw_means):.2f} +/- {np.std(raw_means):.2f}")
        print(f"     - Average norm volume mean: {np.mean(norm_means):.4f}, std: {np.mean(norm_stds):.4f}")

    # 3. Load the fixed patient-level splits. They are experimental inputs and
    # must never be regenerated or overwritten during image preprocessing.
    print("\n--- Validating Canonical Patient-Level Splits ---")
    pids = sorted(patient_stats.keys())
    canonical_splits = load_canonical_splits(splits_dir, pids)
    train_pids = canonical_splits["train"]
    val_pids = canonical_splits["val"]
    test_pids = canonical_splits["test"]
    
    def count_pathologies(p_list):
        return dict(Counter([patient_stats[p]["pathology"] for p in p_list]))
        
    print("Canonical splits validated and preserved:")
    print(f"  Train: {len(train_pids)} patients {count_pathologies(train_pids)}")
    print(f"  Val:   {len(val_pids)} patients {count_pathologies(val_pids)}")
    print(f"  Test:  {len(test_pids)} patients {count_pathologies(test_pids)}")
    print("  Label subsets: 7 (10%) <= 17 (25%) <= 35 (50%) <= 70 (100%)")

    # 4. Visual Before/After Verification
    print("\n--- Generating Visual Verification Figures ---")
    sample_patients = [pids[0], pids[len(pids)//2], pids[-1]]
    
    for pid in sample_patients:
        pdir = raw_dir / pid
        info = patient_stats[pid]
        ed = info["ed_frame"]
        
        raw_ed_file = pdir / f"{pid}_frame{int(ed):02d}.nii.gz"
        raw_gt_file = pdir / f"{pid}_frame{int(ed):02d}_gt.nii.gz"
        
        raw_img_vol = nib.load(str(raw_ed_file)).get_fdata()
        raw_gt_vol = nib.load(str(raw_gt_file)).get_fdata() if raw_gt_file.exists() else None
        
        mid_slice = info["n_slices"] // 2
        raw_slice = raw_img_vol[:, :, mid_slice]
        raw_mask = raw_gt_vol[:, :, mid_slice] if raw_gt_vol is not None else None
        
        proc_file = processed_dir / f"{pid}_frame{int(ed):02d}_slice{mid_slice:02d}.npz"
        proc_data = np.load(str(proc_file))
        proc_img = proc_data["image"]
        proc_mask = proc_data["mask"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Preprocessing Verification: {pid} (Group: {info['pathology']}, Slice {mid_slice:02d}, ED Frame {ed})", fontsize=14, fontweight='bold')
        
        axes[0, 0].imshow(raw_slice, cmap="gray")
        axes[0, 0].set_title(f"Raw Image\nShape: {raw_slice.shape}, Spacing: ({info['orig_spacing'][0]:.2f}, {info['orig_spacing'][1]:.2f}) mm")
        axes[0, 0].axis("off")
        
        if raw_mask is not None:
            axes[0, 1].imshow(raw_mask, cmap="tab10", vmin=0, vmax=9)
            axes[0, 1].set_title(f"Raw Mask (ACDC Labels)\nUnique: {np.unique(raw_mask).tolist()}")
            axes[0, 1].axis("off")
            
            axes[0, 2].imshow(raw_slice, cmap="gray")
            axes[0, 2].imshow(np.ma.masked_where(raw_mask == 0, raw_mask), cmap="tab10", alpha=0.5, vmin=0, vmax=9)
            axes[0, 2].set_title("Raw Overlay")
            axes[0, 2].axis("off")
        else:
            axes[0, 1].axis("off")
            axes[0, 2].axis("off")
            
        axes[1, 0].imshow(proc_img, cmap="gray")
        axes[1, 0].set_title(f"Processed Image (Z-Score Norm)\nShape: {proc_img.shape}, Spacing: (1.50, 1.50) mm")
        axes[1, 0].axis("off")
        
        from matplotlib.colors import ListedColormap
        spec_cmap = ListedColormap(['black', '#e41a1c', '#4daf4a', '#377eb8'])
        
        axes[1, 1].imshow(proc_mask, cmap=spec_cmap, vmin=0, vmax=3)
        axes[1, 1].set_title(f"Processed Mask (Spec Remapped)\nUnique: {np.unique(proc_mask).tolist()} (1=LV, 2=Myo, 3=RV)")
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(proc_img, cmap="gray")
        axes[1, 2].imshow(np.ma.masked_where(proc_mask == 0, proc_mask), cmap=spec_cmap, alpha=0.5, vmin=0, vmax=3)
        axes[1, 2].set_title("Processed Overlay (256x256)")
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        out_fig = results_dir / f"{pid}_preprocessing_comparison.png"
        plt.savefig(str(out_fig), dpi=200)
        plt.close()
        print(f"  Saved verification figure: {out_fig.name}")
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Intensity Normalization Comparison across Patient Volumes", fontsize=14, fontweight='bold')
    
    axes[0].hist(raw_means, bins=20, color='royalblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Raw Volume Means (Mean={np.mean(raw_means):.1f})")
    axes[0].set_xlabel("Raw Mean Intensity")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(norm_means, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title(f"Normalized Volume Means (Mean={np.mean(norm_means):.4f})")
    axes[1].set_xlabel("Z-Score Normalized Mean")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    norm_plot = results_dir / "intensity_normalization_distribution.png"
    plt.savefig(str(norm_plot), dpi=200)
    plt.close()
    print(f"  Saved intensity comparison plot: {norm_plot.name}")

    elapsed = time.time() - start_time
    summary_data = {
        "preprocessing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "total_patients": len(pids),
        "target_size": list(target_size),
        "target_spacing_mm": list(target_spacing),
        "normalization_strategy": norm_strategy,
        "image_interpolation": "bicubic (spline order 3)",
        "mask_interpolation": "nearest-neighbor (order 0)",
        "total_samples_generated": total_slices_saved,
        "labeled_samples": total_labeled_slices,
        "unlabeled_cine_samples": total_unlabeled_slices,
        "discarded_samples": 0,
        "discard_rationale": "No samples were discarded. All 4D cine frames and ED/ES slices were preserved.",
        "label_mapping": {
            "source_acdc": {"0": "BG", "1": "RV", "2": "Myo", "3": "LV"},
            "target_spec": {"0": "BG", "1": "LV", "2": "Myo", "3": "RV"}
        },
        "patient_splits": {
            "train": len(train_pids),
            "val": len(val_pids),
            "test": len(test_pids)
        },
        "split_provenance": {
            "source": "existing canonical data/splits/*.txt artifacts",
            "canonical_assignments": canonical_splits
        }
    }
    
    summary_file = processed_dir / "preprocessing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved preprocessing summary to {summary_file}")
    
    return summary_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACDC Cardiac Cine MRI Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/preprocessing_config.yaml", help="Path to preprocessing configuration YAML")
    args = parser.parse_args()
    run_preprocessing(config_path=args.config)
