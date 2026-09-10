"""
Confidence-filtered pseudo-label generation and evaluation for cardiac cine MRI.

Implements:
1. Multi-metric confidence estimation:
   - Maximum Softmax Probability (MSP)
   - Normalized Shannon Entropy confidence
   - Motion-warped temporal agreement
   - Hybrid confidence metrics
2. Differentiable / configurable filtering engine:
   - Pixel-level confidence thresholding
   - Rejection masking with ignore_index
   - Minimum foreground area filtering
3. Comprehensive pseudo-label quality evaluation against ground truth:
   - Per-class Dice (LV, Myocardium, RV)
   - Mean foreground Dice
   - Acceptance rates (overall and foreground)
   - Confidence distribution statistics
4. Metadata logging and persistent export
5. PseudoLabelDataset wrapper for semi-supervised fine-tuning
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

import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.motion import SpatialTransformer


# ---------------------------------------------------------------------------
# 1. Confidence Metrics
# ---------------------------------------------------------------------------

def compute_confidence_map(
    probs: torch.Tensor,
    method: str = "max_probability",
) -> torch.Tensor:
    """
    Compute pixel-wise confidence map from softmax class probabilities.
    
    Args:
        probs: Softmax probabilities of shape (B, C, H, W)
        method: Metric to use:
            - "max_probability": Maximum softmax probability (MSP) in [1/C, 1.0]
            - "entropy": Normalized Shannon negative entropy in [0.0, 1.0]
            - "hybrid": Minimum of MSP and entropy-based confidence
            
    Returns:
        confidence: Tensor of shape (B, H, W) with confidence scores in [0.0, 1.0]
    """
    B, C, H, W = probs.shape
    
    if method == "max_probability":
        # Maximum Softmax Probability (MSP)
        confidence = torch.max(probs, dim=1)[0]
    
    elif method == "entropy":
        # Normalized Shannon negative entropy:
        # H(P) = -sum(p * log(p)) / log(C) in [0, 1]
        # Confidence = 1 - H(P) (1 = maximally certain, 0 = uniform uncertainty)
        eps = 1e-8
        log_c = math.log(C)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) / log_c
        confidence = torch.clamp(1.0 - entropy, 0.0, 1.0)
    
    elif method == "hybrid":
        msp = torch.max(probs, dim=1)[0]
        eps = 1e-8
        log_c = math.log(C)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) / log_c
        ent_conf = torch.clamp(1.0 - entropy, 0.0, 1.0)
        confidence = torch.min(msp, ent_conf)
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    
    return confidence


def compute_temporal_agreement(
    probs_t: torch.Tensor,
    probs_t1: torch.Tensor,
    flow: torch.Tensor,
    spatial_transformer: Optional[SpatialTransformer] = None,
) -> torch.Tensor:
    """
    Compute motion-warped temporal agreement between adjacent predictions.
    
    Warps probabilities at frame t toward frame t+1 and measures class-probability
    cosine similarity or 1 - L1 disagreement.
    
    Args:
        probs_t: Softmax probabilities at frame t (B, C, H, W)
        probs_t1: Softmax probabilities at frame t+1 (B, C, H, W)
        flow: Dense displacement field from t to t+1 (B, 2, H, W)
        spatial_transformer: Optional SpatialTransformer
        
    Returns:
        agreement: Tensor of shape (B, H, W) in [0.0, 1.0]
    """
    if spatial_transformer is None:
        spatial_transformer = SpatialTransformer(align_corners=True)
    
    warped_probs_t = spatial_transformer(probs_t, flow, mode='bilinear')
    
    # Cosine similarity across probability channel dimension
    dot_product = torch.sum(warped_probs_t * probs_t1, dim=1)
    norm_t = torch.norm(warped_probs_t, p=2, dim=1)
    norm_t1 = torch.norm(probs_t1, p=2, dim=1)
    
    cosine_sim = dot_product / (norm_t * norm_t1 + 1e-8)
    return torch.clamp(cosine_sim, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Confidence Filtering Engine
# ---------------------------------------------------------------------------

def apply_confidence_filtering(
    pseudo_labels: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float = 0.90,
    ignore_index: int = -1,
    min_foreground_pixels: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Filter pseudo-labels using pixel-wise confidence thresholding.
    
    Uncertain pixels (confidence < threshold) are set to `ignore_index`
    so they are omitted during downstream cross-entropy loss calculation.
    
    Args:
        pseudo_labels: Integer label tensor (B, H, W)
        confidence: Confidence tensor (B, H, W)
        threshold: Minimum confidence required to accept a pixel
        ignore_index: Value assigned to rejected pixels (default -1)
        min_foreground_pixels: Minimum accepted foreground pixels required per slice
        
    Returns:
        filtered_labels: Tensor of shape (B, H, W) with rejected pixels as ignore_index
        accept_mask: Binary mask of shape (B, H, W) where 1 = accepted, 0 = rejected
        acceptance_rate: Overall proportion of accepted pixels (float)
    """
    accept_mask = (confidence >= threshold).bool()
    filtered_labels = pseudo_labels.clone()
    filtered_labels[~accept_mask] = ignore_index
    
    if min_foreground_pixels > 0:
        # Zero-out slices with too few accepted foreground pixels (likely spurious)
        for b in range(filtered_labels.shape[0]):
            fg_count = ((filtered_labels[b] > 0) & (filtered_labels[b] != ignore_index)).sum()
            if fg_count < min_foreground_pixels:
                filtered_labels[b] = ignore_index
                accept_mask[b] = False
    
    acceptance_rate = float(accept_mask.float().mean().item())
    return filtered_labels, accept_mask, acceptance_rate


# ---------------------------------------------------------------------------
# 3. Pseudo-Label Quality Evaluation
# ---------------------------------------------------------------------------

def evaluate_pseudo_label_quality(
    pseudo_labels: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    accept_mask: Union[torch.Tensor, np.ndarray],
    confidence: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 4,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """
    Evaluate pseudo-label quality against ground truth segmentations.
    
    Computes acceptance rates, overall pixel accuracy on accepted regions,
    per-class Dice, mean foreground Dice, and confidence distribution statistics.
    
    Args:
        pseudo_labels: (N, H, W) or (H, W) predicted labels
        ground_truth: (N, H, W) or (H, W) true labels
        accept_mask: (N, H, W) or (H, W) binary accepted mask (1=accepted)
        confidence: Optional confidence scores for statistical analysis
        num_classes: Total classes (4: 0=BG, 1=LV, 2=Myo, 3=RV)
        ignore_index: Value indicating masked/ignored pixels
        
    Returns:
        Dictionary of quantitative evaluation metrics.
    """
    if isinstance(pseudo_labels, torch.Tensor):
        pl_np = pseudo_labels.detach().cpu().numpy()
    else:
        pl_np = np.asarray(pseudo_labels)
        
    if isinstance(ground_truth, torch.Tensor):
        gt_np = ground_truth.detach().cpu().numpy()
    else:
        gt_np = np.asarray(ground_truth)
        
    if isinstance(accept_mask, torch.Tensor):
        acc_np = accept_mask.detach().cpu().numpy().astype(bool)
    else:
        acc_np = np.asarray(accept_mask).astype(bool)
    
    # Exclude any GT pixels marked invalid (-1)
    valid_gt = (gt_np != ignore_index)
    acc_valid = acc_np & valid_gt
    
    total_valid = valid_gt.sum()
    accepted_count = acc_valid.sum()
    acceptance_rate = float(accepted_count / max(total_valid, 1))
    
    # Foreground acceptance rate
    gt_fg = valid_gt & (gt_np > 0)
    fg_total = gt_fg.sum()
    fg_accepted = (acc_valid & gt_fg).sum()
    fg_acceptance_rate = float(fg_accepted / max(fg_total, 1))
    
    # Accuracy on accepted pixels
    if accepted_count > 0:
        correct = (pl_np[acc_valid] == gt_np[acc_valid]).sum()
        accuracy_on_accepted = float(correct / accepted_count)
    else:
        accuracy_on_accepted = 0.0
    
    # Overall accuracy across all pixels (without filtering)
    if total_valid > 0:
        raw_correct = (pl_np[valid_gt] == gt_np[valid_gt]).sum()
        raw_accuracy = float(raw_correct / total_valid)
    else:
        raw_accuracy = 0.0
    
    # Compute per-class Dice on accepted pixels
    class_names = {1: "LV", 2: "Myo", 3: "RV"}
    per_class_dice = {}
    fg_dices = []
    
    for c in range(1, num_classes):
        pred_c = acc_valid & (pl_np == c)
        true_c = valid_gt & (gt_np == c)
        
        intersection = (pred_c & true_c).sum()
        cardinality = pred_c.sum() + true_c.sum()
        
        if cardinality > 0:
            dice_c = float((2.0 * intersection) / cardinality)
        else:
            dice_c = 1.0 if (pred_c.sum() == 0 and true_c.sum() == 0) else 0.0
            
        name = class_names.get(c, f"Class_{c}")
        per_class_dice[f"dice_{name.lower()}"] = dice_c
        fg_dices.append(dice_c)
    
    mean_fg_dice = float(np.mean(fg_dices)) if len(fg_dices) > 0 else 0.0
    
    metrics = {
        "acceptance_rate": acceptance_rate,
        "fg_acceptance_rate": fg_acceptance_rate,
        "accuracy_on_accepted": accuracy_on_accepted,
        "raw_accuracy": raw_accuracy,
        "mean_fg_dice": mean_fg_dice,
        **per_class_dice,
    }
    
    # Optional confidence statistics
    if confidence is not None:
        if isinstance(confidence, torch.Tensor):
            conf_np = confidence.detach().cpu().numpy()
        else:
            conf_np = np.asarray(confidence)
            
        metrics["confidence_mean"] = float(conf_np[valid_gt].mean()) if total_valid > 0 else 0.0
        metrics["confidence_median"] = float(np.median(conf_np[valid_gt])) if total_valid > 0 else 0.0
        if accepted_count > 0:
            metrics["confidence_accepted_mean"] = float(conf_np[acc_valid].mean())
        if fg_total > 0:
            metrics["confidence_gt_fg_mean"] = float(conf_np[gt_fg].mean())
            
    return metrics


# ---------------------------------------------------------------------------
# 4. Generator Pipeline Class
# ---------------------------------------------------------------------------

class PseudoLabelGenerator:
    """
    High-level generator for confidence-filtered pseudo-labels.
    
    Wraps model inference, confidence estimation, threshold filtering,
    optional temporal consistency verification, and quality logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        confidence_metric: str = "max_probability",
        confidence_threshold: float = 0.90,
        entropy_threshold: float = 0.25,
        use_temporal_consistency: bool = False,
        temporal_consistency_threshold: float = 0.80,
        ignore_index: int = -1,
        min_foreground_pixels: int = 30,
        motion_estimator: Optional[nn.Module] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.confidence_metric = confidence_metric
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.use_temporal_consistency = use_temporal_consistency
        self.temporal_consistency_threshold = temporal_consistency_threshold
        self.ignore_index = ignore_index
        self.min_foreground_pixels = min_foreground_pixels
        self.motion_estimator = motion_estimator.to(device).eval() if motion_estimator else None
    
    @torch.no_grad()
    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and confidence filtering on a single batch.
        
        Args:
            batch: Batch dictionary with 'image' (B, 1, H, W)
            
        Returns:
            Dict containing:
                - logits: (B, C, H, W)
                - probs: (B, C, H, W)
                - raw_pseudo_labels: (B, H, W)
                - confidence: (B, H, W)
                - filtered_labels: (B, H, W) with ignore_index for rejected
                - accept_mask: (B, H, W) binary mask (1=accepted)
        """
        images = batch['image'].to(self.device)
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        raw_pseudo = torch.argmax(probs, dim=1)
        
        # Primary confidence map
        confidence = compute_confidence_map(probs, method=self.confidence_metric)
        
        # Optional temporal consistency filtering if adjacent frames are provided
        if self.use_temporal_consistency and self.motion_estimator is not None and 'frame_t1' in batch:
            frame_t = images
            frame_t1 = batch['frame_t1'].to(self.device)
            logits_t1 = self.model(frame_t1)
            probs_t1 = F.softmax(logits_t1, dim=1)
            
            motion_out = self.motion_estimator(frame_t, frame_t1)
            flow = motion_out['flow']
            
            agreement = compute_temporal_agreement(
                probs, probs_t1, flow, self.motion_estimator.transformer
            )
            # Modulate confidence by temporal agreement
            confidence = confidence * agreement
        
        filtered_labels, accept_mask, acceptance_rate = apply_confidence_filtering(
            pseudo_labels=raw_pseudo,
            confidence=confidence,
            threshold=self.confidence_threshold,
            ignore_index=self.ignore_index,
            min_foreground_pixels=self.min_foreground_pixels,
        )
        
        return {
            'logits': logits,
            'probs': probs,
            'raw_pseudo_labels': raw_pseudo,
            'confidence': confidence,
            'filtered_labels': filtered_labels,
            'accept_mask': accept_mask,
            'acceptance_rate': acceptance_rate,
        }


# ---------------------------------------------------------------------------
# 5. Pseudo-Label Dataset Wrapper
# ---------------------------------------------------------------------------

class PseudoLabelDataset(Dataset):
    """
    PyTorch Dataset wrapping generated pseudo-labels for downstream training.
    
    Provides:
        - image: input slice (1, H, W)
        - pseudo_label: integer label map (H, W) with low-confidence pixels as ignore_index
        - confidence: confidence weights (H, W)
        - accept_mask: binary indicator of accepted pixels
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        filtered_labels: np.ndarray,
        confidence: np.ndarray,
        accept_mask: np.ndarray,
        transform=None,
    ):
        assert len(base_dataset) == len(filtered_labels), \
            f"Dataset length mismatch: {len(base_dataset)} vs {len(filtered_labels)}"
            
        self.base_dataset = base_dataset
        self.filtered_labels = filtered_labels
        self.confidence = confidence
        self.accept_mask = accept_mask
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.base_dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        
        sample['pseudo_label'] = torch.from_numpy(self.filtered_labels[idx]).long()
        sample['confidence'] = torch.from_numpy(self.confidence[idx]).float()
        sample['accept_mask'] = torch.from_numpy(self.accept_mask[idx]).float()
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# ---------------------------------------------------------------------------
# 6. Metadata Serialization Helper
# ---------------------------------------------------------------------------

def save_pseudo_label_metadata(
    metadata: Dict,
    output_dir: Union[str, Path],
    filename: str = "pseudo_label_metadata.json",
):
    """Save pseudo-label run metadata to disk in JSON format."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Pseudo-label metadata saved to: {out_path}")


# ---------------------------------------------------------------------------
# 7. CLI & Self-Test Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Confidence-Filtered Pseudo-Labeling for Cardiac Cine MRI"
    )
    parser.add_argument(
        "--config", type=str, default="configs/pseudo_labels.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run lightweight smoke test on CPU without training"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override confidence threshold"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if args.threshold is not None:
        config['filtering']['confidence_threshold'] = args.threshold
        
    print(f"Loaded Pseudo-Label configuration from: {config_path}")
    print(f"Confidence metric:    {config['filtering']['confidence_metric']}")
    print(f"Confidence threshold: {config['filtering']['confidence_threshold']}")
    print(f"Temporal consistency: {config['filtering']['use_temporal_consistency']}")
    
    if args.smoke_test:
        print("\n--- Running Pseudo-Label Module Smoke Test ---")
        # Generate dummy logits for verification
        torch.manual_seed(42)
        dummy_logits = torch.randn(2, 4, 256, 256)
        probs = F.softmax(dummy_logits, dim=1)
        raw_pseudo = torch.argmax(probs, dim=1)
        
        conf = compute_confidence_map(probs, method=config['filtering']['confidence_metric'])
        print(f"Confidence map shape: {conf.shape} (mean: {conf.mean().item():.3f})")
        
        filtered, mask, acc_rate = apply_confidence_filtering(
            pseudo_labels=raw_pseudo,
            confidence=conf,
            threshold=config['filtering']['confidence_threshold'],
            ignore_index=config['filtering']['ignore_index'],
        )
        print(f"Filtered labels:      {filtered.shape}")
        print(f"Acceptance mask:      {mask.shape} (rate: {acc_rate*100:.2f}%)")
        
        # Test quality evaluation
        dummy_gt = torch.randint(0, 4, (2, 256, 256))
        metrics = evaluate_pseudo_label_quality(
            pseudo_labels=raw_pseudo,
            ground_truth=dummy_gt,
            accept_mask=mask,
            confidence=conf,
        )
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # Test metadata saving
        meta = {
            "config": config['filtering'],
            "metrics": metrics,
            "samples_evaluated": 2,
        }
        save_pseudo_label_metadata(meta, config['logging']['output_dir'])
        print("--- Pseudo-Label Module Smoke Test PASSED ---")


if __name__ == "__main__":
    main()
