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
import re
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

from src.motion import MotionEstimator, SpatialTransformer
from src.dataset import ACDCTemporalDataset
from src.segmentation_model import build_segmentation_model


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
        # Temporal samples expose frame_t; conventional segmentation samples
        # expose image. Supporting both keeps the generation stage compatible
        # with motion-aware and confidence-only operation.
        images = batch.get('image', batch.get('frame_t')).to(self.device)
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


class UnlabeledTemporalPseudoDataset(Dataset):
    """Training-only intermediate cine frames, each paired with its successor.

    The wrapper deliberately excludes ED/ES frames with ground truth. This
    prevents pseudo labels from replacing known annotations and ensures neither
    validation nor test patients enter teacher inference.
    """

    def __init__(self, processed_dir: str, train_split: str):
        self.base = ACDCTemporalDataset(processed_dir, train_split)
        self.indices = []
        for index, (path_t, _) in enumerate(self.base.pairs):
            with np.load(path_t, allow_pickle=True) as item:
                if 'mask' in item and np.all(item['mask'] == -1):
                    self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[index]]


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


def resolve_teacher_label_fraction(checkpoint_path: Path) -> Optional[int]:
    """Determine the label fraction a teacher checkpoint was trained with.

    Checks explicit checkpoint metadata first, then the run-ID naming
    convention (e.g. supervised_10pct_seed42). Returns None when the
    provenance cannot be established.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(checkpoint, dict):
        if checkpoint.get("label_fraction") is not None:
            try:
                return int(checkpoint["label_fraction"])
            except (TypeError, ValueError):
                pass
        for source in (str(checkpoint.get("experiment_name") or ""), checkpoint_path.stem):
            match = re.search(r"(\d+)pct", source)
            if match:
                return int(match.group(1))
    return None


def validate_teacher_label_fraction(checkpoint_value: str, label_fraction: int) -> None:
    """Fail closed when the teacher's fraction cannot be proven to match."""
    teacher_fraction = resolve_teacher_label_fraction(Path(checkpoint_value))
    if teacher_fraction is None:
        raise ValueError(
            f"Cannot establish the label fraction of teacher checkpoint "
            f"{checkpoint_value}; refusing to generate pseudo-labels for "
            f"{label_fraction}%. Use a teacher checkpoint from the matching "
            "supervised run."
        )
    if teacher_fraction != label_fraction:
        raise ValueError(
            f"Teacher checkpoint {checkpoint_value} was trained with "
            f"{teacher_fraction}% labels, but pseudo-labels were requested "
            f"for {label_fraction}%. Aborting to prevent label leakage."
        )


def load_teacher_model(config: Dict, device: torch.device) -> nn.Module:
    """Load the baseline checkpoint used as the pseudo-label teacher."""
    checkpoint_value = config['model'].get('checkpoint')
    if not checkpoint_value:
        raise ValueError(
            "A label-fraction-matched teacher checkpoint is required. "
            "Pass --teacher-checkpoint from the supervised run for this fraction."
        )
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}. "
            "Train the supervised baseline first, then set model.checkpoint."
        )
    model = build_segmentation_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return model.eval()


def load_motion_estimator(config: Dict, device: torch.device) -> Optional[MotionEstimator]:
    """Load the trained motion dependency only when temporal filtering is enabled."""
    if not config['filtering'].get('use_temporal_consistency', False):
        return None
    checkpoint_path = Path(config['model'].get('motion_checkpoint', 'checkpoints/motion/motion_model_best.pth'))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Temporal filtering is enabled but motion checkpoint is missing: {checkpoint_path}. "
            "Run motion pretraining, or set filtering.use_temporal_consistency to false."
        )
    estimator = MotionEstimator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    estimator.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    return estimator.eval()


def generate_pseudo_labels(config: Dict, device: torch.device, label_fraction: int) -> Dict[str, Union[int, float, str]]:
    """Generate raw and confidence-filtered labels for unlabeled train frames."""
    data_cfg, filter_cfg, log_cfg = config['data'], config['filtering'], config['logging']
    dataset = UnlabeledTemporalPseudoDataset(data_cfg['processed_dir'], data_cfg['train_split'])
    if len(dataset) == 0:
        raise ValueError("No unlabeled intermediate training frames found. Run preprocessing and verify the training split.")
    loader = DataLoader(
        dataset,
        batch_size=config.get('generation', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=device.type == 'cuda',
    )
    model = load_teacher_model(config, device)
    validate_teacher_label_fraction(config['model']['checkpoint'], label_fraction)
    motion_estimator = load_motion_estimator(config, device)
    generator = PseudoLabelGenerator(
        model=model,
        device=device,
        confidence_metric=filter_cfg.get('confidence_metric', 'max_probability'),
        confidence_threshold=filter_cfg.get('confidence_threshold', 0.90),
        entropy_threshold=filter_cfg.get('entropy_threshold', 0.25),
        use_temporal_consistency=filter_cfg.get('use_temporal_consistency', False),
        temporal_consistency_threshold=filter_cfg.get('temporal_consistency_threshold', 0.80),
        ignore_index=filter_cfg.get('ignore_index', -1),
        min_foreground_pixels=filter_cfg.get('min_foreground_pixels', 30),
        motion_estimator=motion_estimator,
    )
    output_dir = Path(log_cfg['output_dir']) / f'{label_fraction}pct'
    labels_dir = output_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    records, accepted_pixels, total_pixels = [], 0, 0
    for batch in tqdm(loader, desc='Generating pseudo-labels'):
        outputs = generator.process_batch(batch)
        raw = outputs['raw_pseudo_labels'].cpu().numpy().astype(np.int16)
        filtered = outputs['filtered_labels'].cpu().numpy().astype(np.int16)
        confidence = outputs['confidence'].cpu().numpy().astype(np.float32)
        accepted = outputs['accept_mask'].cpu().numpy().astype(np.uint8)
        for item_index in range(raw.shape[0]):
            patient_id = batch['patient_id'][item_index]
            slice_idx = int(batch['slice_idx'][item_index])
            frame_idx = int(batch['frame_idx_t'][item_index])
            filename = f"{patient_id}_frame{frame_idx:02d}_slice{slice_idx:02d}.npz"
            np.savez_compressed(
                labels_dir / filename,
                raw_pseudo_label=raw[item_index],
                filtered_pseudo_label=filtered[item_index],
                confidence=confidence[item_index],
                accept_mask=accepted[item_index],
                patient_id=patient_id,
                slice_idx=slice_idx,
                frame_idx=frame_idx,
            )
            item_accepted = int(accepted[item_index].sum())
            accepted_pixels += item_accepted
            total_pixels += int(accepted[item_index].size)
            records.append({'file': f'labels/{filename}', 'patient_id': patient_id, 'slice_idx': slice_idx, 'frame_idx': frame_idx, 'acceptance_rate': item_accepted / accepted[item_index].size})
    manifest_path = output_dir / 'pseudo_label_index.json'
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2)
    metadata = {
        'teacher_checkpoint': str(config['model']['checkpoint']),
        'label_fraction': label_fraction,
        'train_split': str(data_cfg['train_split']),
        'samples_generated': len(records),
        'acceptance_rate': accepted_pixels / max(total_pixels, 1),
        'confidence_metric': filter_cfg.get('confidence_metric', 'max_probability'),
        'confidence_threshold': filter_cfg.get('confidence_threshold', 0.90),
        'uses_temporal_consistency': motion_estimator is not None,
        'manifest': str(manifest_path),
    }
    save_pseudo_label_metadata(metadata, output_dir, log_cfg.get('metadata_file', 'pseudo_label_metadata.json'))
    return metadata


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
        "--device", type=str, default=None,
        help="Device override ('auto', 'cpu', or 'cuda')"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run lightweight smoke test on CPU without training"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override confidence threshold"
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default=None,
        help="Supervised teacher checkpoint trained with this same label fraction"
    )
    parser.add_argument(
        "--label-fraction", type=int, choices=[10, 25, 50, 100], default=None,
        help="Label fraction used to train the teacher; required for leakage-safe output"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if args.threshold is not None:
        config['filtering']['confidence_threshold'] = args.threshold
    if args.teacher_checkpoint is not None:
        config['model']['checkpoint'] = args.teacher_checkpoint
        
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
        return

    requested_device = args.device or config.get('device', 'auto')
    device = torch.device('cuda' if requested_device == 'auto' and torch.cuda.is_available() else 'cpu' if requested_device == 'auto' else requested_device)
    if args.label_fraction is None:
        raise ValueError('--label-fraction is required for real pseudo-label generation.')
    metadata = generate_pseudo_labels(config, device, args.label_fraction)
    print(f"Generated {metadata['samples_generated']:,} pseudo-labels with {metadata['acceptance_rate']:.2%} pixel acceptance.")


if __name__ == "__main__":
    main()
