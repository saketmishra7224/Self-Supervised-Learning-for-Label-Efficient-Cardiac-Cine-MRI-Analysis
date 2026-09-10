"""
Evaluation metrics for cardiac segmentation.

Computes:
- Per-class Dice coefficient (LV, Myocardium, RV)
- Mean Dice across foreground classes
- Per-class Hausdorff Distance 95th percentile (HD95)
- Patient-level aggregation (mean ± std)

All metrics operate on numpy arrays for compatibility with medpy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    print("Warning: medpy not installed. HD95 will use fallback implementation.")


# Class names matching project specification
CLASS_NAMES = {0: "Background", 1: "LV", 2: "Myocardium", 3: "RV"}
FOREGROUND_CLASSES = [1, 2, 3]  # LV, Myo, RV


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Dice score in [0, 1]
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    
    if union == 0:
        return 1.0 if np.sum(target) == 0 else 0.0
    
    return float((2 * intersection + smooth) / (union + smooth))


def hausdorff_distance_95(
    pred: np.ndarray, target: np.ndarray, voxel_spacing: Optional[Tuple] = None
) -> float:
    """
    Compute 95th percentile Hausdorff distance.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        voxel_spacing: Physical voxel spacing (for distance in mm)
        
    Returns:
        HD95 in mm (or pixels if no spacing provided)
    """
    # Handle edge cases
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.inf
    
    if HAS_MEDPY:
        return float(medpy_hd95(pred, target, voxelspacing=voxel_spacing))
    else:
        return _fallback_hd95(pred, target, voxel_spacing)


def _fallback_hd95(
    pred: np.ndarray, target: np.ndarray, voxel_spacing: Optional[Tuple] = None
) -> float:
    """Fallback HD95 using scipy distance transform."""
    from scipy.ndimage import distance_transform_edt
    
    pred_border = pred.astype(bool)
    target_border = target.astype(bool)
    
    # Distance from target border to nearest pred
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target_border, sampling=voxel_spacing)
    
    # Directed distances
    d_pred_to_target = dt_target[pred_border]
    d_target_to_pred = dt_pred[target_border]
    
    if len(d_pred_to_target) == 0 or len(d_target_to_pred) == 0:
        return np.inf
    
    # 95th percentile of the combined directed distances
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_metrics_single(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Optional[Tuple] = None,
    compute_hd: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics for a single prediction-target pair.
    
    Args:
        pred: (H, W) integer class predictions
        target: (H, W) integer class ground truth
        voxel_spacing: Physical pixel spacing (for HD95 in mm)
        compute_hd: Whether to compute HD95 (slower)
        
    Returns:
        Dict with Dice and HD95 for each foreground class + mean Dice
    """
    metrics = {}
    dice_values = []
    
    for cls in FOREGROUND_CLASSES:
        cls_name = CLASS_NAMES[cls]
        pred_binary = (pred == cls).astype(np.uint8)
        target_binary = (target == cls).astype(np.uint8)
        
        # Dice
        d = dice_coefficient(pred_binary, target_binary)
        metrics[f"{cls_name}_Dice"] = d
        dice_values.append(d)
        
        # HD95
        if compute_hd:
            h = hausdorff_distance_95(pred_binary, target_binary, voxel_spacing)
            metrics[f"{cls_name}_HD95"] = h
    
    metrics["Mean_Dice"] = float(np.mean(dice_values))
    
    if compute_hd:
        hd_values = [metrics[f"{CLASS_NAMES[c]}_HD95"] for c in FOREGROUND_CLASSES]
        # Filter out inf values for mean HD95
        finite_hd = [h for h in hd_values if np.isfinite(h)]
        metrics["Mean_HD95"] = float(np.mean(finite_hd)) if finite_hd else np.inf
    
    return metrics


def compute_metrics_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    patient_ids: Optional[List[str]] = None,
    voxel_spacing: Optional[Tuple] = None,
    compute_hd: bool = True,
) -> Dict[str, any]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        preds: (N, H, W) integer class predictions
        targets: (N, H, W) integer class ground truth
        patient_ids: Optional list of patient IDs for per-patient grouping
        voxel_spacing: Physical pixel spacing
        compute_hd: Whether to compute HD95
        
    Returns:
        Dict with:
        - per_sample: List of per-sample metrics
        - mean: Mean metrics across samples
        - std: Standard deviation across samples
    """
    n_samples = preds.shape[0]
    per_sample = []
    
    for i in range(n_samples):
        m = compute_metrics_single(preds[i], targets[i], voxel_spacing, compute_hd)
        if patient_ids is not None and i < len(patient_ids):
            m['patient_id'] = patient_ids[i]
        per_sample.append(m)
    
    # Aggregate
    metric_keys = [k for k in per_sample[0].keys() if k != 'patient_id']
    
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in per_sample if np.isfinite(m[key])]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
        else:
            mean_metrics[key] = np.inf
            std_metrics[key] = 0.0
    
    return {
        'per_sample': per_sample,
        'mean': mean_metrics,
        'std': std_metrics,
    }


def compute_patient_level_metrics(
    per_sample_metrics: List[Dict],
    compute_hd: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate slice-level metrics to patient level.
    
    First averages metrics per patient, then computes mean ± std across patients.
    This is the proper evaluation approach to avoid bias toward patients with more slices.
    
    Args:
        per_sample_metrics: List of per-slice metric dicts (must include 'patient_id')
        
    Returns:
        Dict with 'per_patient', 'mean', 'std' keys
    """
    # Group by patient
    patient_metrics = defaultdict(list)
    for m in per_sample_metrics:
        pid = m.get('patient_id', 'unknown')
        patient_metrics[pid].append(m)
    
    # Average within each patient
    per_patient = {}
    metric_keys = [k for k in per_sample_metrics[0].keys() 
                   if k != 'patient_id' and isinstance(per_sample_metrics[0][k], (int, float))]
    
    for pid, slices in patient_metrics.items():
        patient_avg = {}
        for key in metric_keys:
            values = [s[key] for s in slices if np.isfinite(s[key])]
            patient_avg[key] = float(np.mean(values)) if values else np.inf
        per_patient[pid] = patient_avg
    
    # Aggregate across patients
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [pm[key] for pm in per_patient.values() if np.isfinite(pm[key])]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
        else:
            mean_metrics[key] = np.inf
            std_metrics[key] = 0.0
    
    return {
        'per_patient': per_patient,
        'mean': mean_metrics,
        'std': std_metrics,
        'n_patients': len(per_patient),
    }


def format_metrics_table(
    metrics_dict: Dict,
    title: str = "Results",
) -> str:
    """
    Format metrics as a readable table string.
    
    Args:
        metrics_dict: Dict with 'mean' and 'std' keys
        title: Table title
        
    Returns:
        Formatted string
    """
    mean = metrics_dict['mean']
    std = metrics_dict['std']
    
    lines = [f"\n{'='*60}", f" {title}", f"{'='*60}"]
    
    for key in sorted(mean.keys()):
        m = mean[key]
        s = std[key]
        if 'Dice' in key:
            lines.append(f"  {key:20s}: {m:.4f} ± {s:.4f}")
        elif 'HD95' in key:
            if np.isfinite(m):
                lines.append(f"  {key:20s}: {m:.2f} ± {s:.2f} mm")
            else:
                lines.append(f"  {key:20s}: inf")
        else:
            lines.append(f"  {key:20s}: {m:.4f} ± {s:.4f}")
    
    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)


def compute_temporal_consistency_metrics(
    pred_t: np.ndarray,
    pred_t1: np.ndarray,
    warped_pred_t: Optional[np.ndarray] = None,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Evaluate temporal consistency of predictions across consecutive cine frames.
    
    Computes:
    - Frame-to-frame uncompensated agreement and Dice
    - Motion-compensated agreement and Dice (using warped frame t)
    - Motion compensation gain (warped - uncompensated)
    
    Args:
        pred_t: (H, W) or (N, H, W) class predictions at frame t
        pred_t1: (H, W) or (N, H, W) class predictions at frame t+1
        warped_pred_t: Optional (H, W) or (N, H, W) prediction at frame t warped to t+1 space
        num_classes: Number of classes (4)
        
    Returns:
        Dict of temporal consistency metrics
    """
    # 1. Uncompensated direct frame agreement
    raw_agreement = float(np.mean(pred_t == pred_t1))
    
    # Uncompensated foreground Dice
    raw_fg_t = (pred_t > 0).astype(np.uint8)
    raw_fg_t1 = (pred_t1 > 0).astype(np.uint8)
    raw_dice = dice_coefficient(raw_fg_t, raw_fg_t1)
    
    results = {
        'temporal_raw_agreement': raw_agreement,
        'temporal_raw_fg_dice': raw_dice,
    }
    
    # 2. Motion-compensated agreement (if warped predictions provided)
    if warped_pred_t is not None:
        comp_agreement = float(np.mean(warped_pred_t == pred_t1))
        comp_fg_t = (warped_pred_t > 0).astype(np.uint8)
        comp_dice = dice_coefficient(comp_fg_t, raw_fg_t1)
        
        results['temporal_warped_agreement'] = comp_agreement
        results['temporal_warped_fg_dice'] = comp_dice
        results['temporal_motion_gain'] = comp_dice - raw_dice
    
    return results

