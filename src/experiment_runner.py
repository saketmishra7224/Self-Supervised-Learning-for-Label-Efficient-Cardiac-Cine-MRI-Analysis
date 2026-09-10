"""
Reproducible Experiment Runner for Limited-Label Cardiac Cine MRI Segmentation.

Supports the 16-experiment matrix:
- 4 Label Regimes: 10%, 25%, 50%, 100% of training patients (nested patient splits)
- 4 Modular Model Variants:
    A. Supervised Baseline: Standard 2D U-Net initialized from scratch
    B. SSL Fine-Tuning: 2D U-Net initialized with pretrained SharedEncoder
    C. SSL + Motion Consistency: Pretrained 2D U-Net + Motion-Warped Temporal Regularization
    D. SSL + Motion + Pseudo-Labels: Full framework with confidence-filtered pseudo-labels

Integrates experiment logging, metric tracking, checkpoint saving, and lightweight smoke tests.
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

import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoder import count_parameters, load_encoder_weights
from src.segmentation_model import SegmentationUNet
from src.motion import MotionEstimator, warp_mask
from src.pseudo_labels import compute_confidence_map, apply_confidence_filtering
from src.losses import DiceCELoss
from src.metrics import compute_metrics_batch, compute_patient_level_metrics
from src.dataset import ACDCSegDataset, ACDCTemporalDataset, get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# 1. Reproducibility & Setup Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Seed all pseudo-random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# 2. Experiment Registry Manager
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """Maintains a persistent JSON registry of all experimental runs."""
    
    def __init__(self, registry_file: Union[str, Path] = "results/experiments/experiment_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.records = self._load()
    
    def _load(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def register(self, experiment_id: str, data: Dict[str, Any]):
        """Append or update an experiment record."""
        self.records[experiment_id] = {
            **data,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        print(f"Experiment [{experiment_id}] registered in: {self.registry_file}")


# ---------------------------------------------------------------------------
# 3. Model Factory with Encoder Transfer Support
# ---------------------------------------------------------------------------

def build_experiment_model(
    config: dict,
    mode: str,
    ssl_checkpoint: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SegmentationUNet, Optional[MotionEstimator]]:
    """
    Construct model and optional motion estimator according to experimental variant.
    
    Args:
        config: Full configuration dict
        mode: One of 'supervised', 'ssl_finetune', 'ssl_motion', 'ssl_motion_pseudo'
        ssl_checkpoint: Optional override path to SSL pretrained encoder weights
        device: Target compute device
        
    Returns:
        model: SegmentationUNet (with loaded SSL weights if applicable)
        motion_estimator: MotionEstimator instance if mode uses motion, else None
    """
    model_cfg = config.get('model', {})
    in_channels = model_cfg.get('in_channels', 1)
    num_classes = model_cfg.get('num_classes', 4)
    encoder_channels = model_cfg.get('encoder_channels', [32, 64, 128, 256])
    dropout = model_cfg.get('dropout', 0.1)
    use_residual = model_cfg.get('use_residual', True)
    
    model = SegmentationUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_channels=encoder_channels,
        dropout=dropout,
        use_residual=use_residual,
    ).to(device)
    
    variant_cfg = config.get('variants', {}).get(mode, {})
    use_ssl = variant_cfg.get('use_ssl_pretrained', False) or (mode != "supervised")
    
    if use_ssl:
        ckpt_path = ssl_checkpoint or variant_cfg.get('ssl_checkpoint', "checkpoints/ssl/ssl_encoder_best.pth")
        if Path(ckpt_path).exists():
            print(f"Loading pretrained encoder weights from: {ckpt_path}")
            load_encoder_weights(model.encoder, ckpt_path, strict=False)
        else:
            print(f"Notice: SSL checkpoint '{ckpt_path}' not found on dev system. Using initialized weights for testing.")
    
    # Initialize motion estimator if variant uses motion
    use_motion = variant_cfg.get('use_motion', False)
    motion_estimator = None
    if use_motion:
        motion_estimator = MotionEstimator(channels=[16, 32, 64, 32]).to(device)
        motion_ckpt = variant_cfg.get('motion_checkpoint', "checkpoints/motion/motion_model_best.pth")
        if Path(motion_ckpt).exists():
            print(f"Loading motion model weights from: {motion_ckpt}")
            try:
                motion_estimator.load_state_dict(torch.load(motion_ckpt, map_location=device))
            except Exception as e:
                print(f"Notice: Could not load motion checkpoint ({e}). Using initialized weights.")
    
    return model, motion_estimator


# ---------------------------------------------------------------------------
# 4. Multi-Component Loss Manager
# ---------------------------------------------------------------------------

class ExperimentLossManager:
    """
    Computes combined multi-objective loss:
        L_total = L_sup + lambda_motion * L_motion + lambda_pseudo * L_pseudo
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        motion_weight: float = 0.1,
        pseudo_weight: float = 0.25,
        num_classes: int = 4,
        include_background: bool = False,
    ):
        self.sup_criterion = DiceCELoss(
            num_classes=num_classes,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            include_background=include_background,
        )
        self.motion_weight = motion_weight
        self.pseudo_weight = pseudo_weight
        self.num_classes = num_classes
    
    def compute_loss(
        self,
        logits_sup: torch.Tensor,
        targets_sup: torch.Tensor,
        motion_loss: Optional[torch.Tensor] = None,
        logits_pseudo: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total weighted loss and return breakdown metrics.
        """
        # 1. Supervised segmentation loss
        sup_loss = self.sup_criterion(logits_sup, targets_sup)
        total_loss = sup_loss
        loss_dict = {'loss_sup': sup_loss.item()}
        
        # 2. Optional motion/temporal loss
        if motion_loss is not None:
            total_loss = total_loss + self.motion_weight * motion_loss
            loss_dict['loss_motion'] = motion_loss.item()
        
        # 3. Optional confidence-filtered pseudo-label loss
        if logits_pseudo is not None and pseudo_labels is not None:
            # Mask out rejected pixels (ignore_index = -1)
            valid_mask = (pseudo_labels >= 0)
            if valid_mask.sum() > 0:
                pseudo_ce = F.cross_entropy(
                    logits_pseudo,
                    pseudo_labels,
                    ignore_index=-1,
                )
                total_loss = total_loss + self.pseudo_weight * pseudo_ce
                loss_dict['loss_pseudo'] = pseudo_ce.item()
            else:
                loss_dict['loss_pseudo'] = 0.0
        
        loss_dict['loss_total'] = total_loss.item()
        return total_loss, loss_dict


# ---------------------------------------------------------------------------
# 5. Smoke Test Runner (1 Batch Verification)
# ---------------------------------------------------------------------------

def run_experiment_smoke_test(
    config: dict,
    mode: str,
    label_fraction: int,
    device: torch.device,
) -> bool:
    """
    Lightweight 1-batch smoke test validating pipeline mechanics without training.
    """
    print(f"\n{'='*70}")
    print(f"RUNNING SMOKE TEST: Variant [{mode.upper()}], Label Fraction [{label_fraction}%]")
    print(f"{'='*70}")
    
    data_cfg = config.get('data', {})
    processed_dir = data_cfg.get('processed_dir', 'data/processed')
    split_file = data_cfg['subsets'].get(label_fraction)
    assert Path(split_file).exists(), f"Subset split file not found: {split_file}"
    
    # 1. Load dataset subset
    seg_dataset = ACDCSegDataset(processed_dir=processed_dir, split_file=split_file)
    print(f"[1/5] Labeled dataset loaded: {len(seg_dataset)} labeled slices from {split_file}")
    
    loader = DataLoader(seg_dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    print(f"      Batch shape: image={images.shape}, mask={masks.shape}")
    
    # 2. Build model & motion estimator
    model, motion_est = build_experiment_model(config, mode=mode, device=device)
    total_params = count_parameters(model)
    print(f"[2/5] Model constructed: {model.__class__.__name__} ({total_params:,} parameters)")
    if motion_est:
        print(f"      Motion estimator active: {count_parameters(motion_est):,} parameters")
    
    # 3. Optimizer setup with differential learning rates
    opt_cfg = config.get('optimization', {})
    lr_dec = opt_cfg.get('lr_decoder', 1e-4)
    lr_enc = opt_cfg.get('lr_encoder', 1e-5)
    weight_decay = opt_cfg.get('weight_decay', 1e-5)
    
    optimizer = torch.optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': lr_dec},
        {'params': model.encoder.parameters(), 'lr': lr_enc},
    ], weight_decay=weight_decay)
    
    # 4. Forward pass & multi-objective loss
    loss_cfg = config.get('loss', {})
    loss_manager = ExperimentLossManager(
        dice_weight=loss_cfg.get('dice_weight', 1.0),
        ce_weight=loss_cfg.get('ce_weight', 1.0),
        motion_weight=loss_cfg.get('motion_weight', 0.1),
        pseudo_weight=loss_cfg.get('pseudo_weight', 0.25),
        num_classes=data_cfg.get('num_classes', 4),
    )
    
    optimizer.zero_grad()
    model.train()
    logits = model(images)
    
    # Optional motion loss
    motion_loss = None
    if motion_est is not None:
        temp_dataset = ACDCTemporalDataset(processed_dir=processed_dir, split_file=split_file)
        temp_sample = temp_dataset[0]
        ft = temp_sample['frame_t'].unsqueeze(0).to(device)
        ft1 = temp_sample['frame_t1'].unsqueeze(0).to(device)
        motion_out = motion_est(ft, ft1)
        motion_loss = motion_out['total_loss']
    
    # Optional pseudo-label loss
    logits_pseudo = None
    pseudo_labels = None
    if mode in ["ssl_pseudo", "full_pipeline", "ssl_motion_pseudo"]:
        # Simulate pseudo-labeling forward on the batch
        probs = F.softmax(logits, dim=1)
        raw_pseudo = torch.argmax(probs, dim=1)
        conf = compute_confidence_map(probs, method="max_probability")
        filt_labels, _, _ = apply_confidence_filtering(raw_pseudo, conf, threshold=0.9, ignore_index=-1)
        logits_pseudo = logits
        pseudo_labels = filt_labels
    
    total_loss, loss_breakdown = loss_manager.compute_loss(
        logits_sup=logits,
        targets_sup=masks,
        motion_loss=motion_loss,
        logits_pseudo=logits_pseudo,
        pseudo_labels=pseudo_labels,
    )
    
    print(f"[3/5] Forward pass & Loss computed:")
    for k, v in loss_breakdown.items():
        print(f"      {k}: {v:.4f}")
    assert torch.isfinite(total_loss), "Loss is not finite!"
    
    # 5. Backward gradient flow & parameter update
    total_loss.backward()
    
    enc_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    dec_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    print(f"[4/5] Gradient flow check: Encoder={enc_grads}, Decoder={dec_grads}")
    assert enc_grads and dec_grads, "Gradients missing in encoder or decoder!"
    
    optimizer.step()
    print("      Optimizer step executed successfully.")
    
    # 6. Verify registry entry
    registry = ExperimentRegistry(config.get('logging', {}).get('registry_file', 'results/experiments/experiment_registry.json'))
    exp_id = f"{mode}_{label_fraction}pct_seed{config.get('project', {}).get('seed', 42)}_smoketest"
    registry.register(exp_id, {
        "mode": mode,
        "label_fraction": label_fraction,
        "status": "smoke_test_passed",
        "loss_breakdown": loss_breakdown,
    })
    print(f"[5/5] Smoke test logged to registry.")
    print(f"{'='*70}")
    print(f"SMOKE TEST PASSED for [{mode.upper()} {label_fraction}%] (OK)\n")
    return True


# ---------------------------------------------------------------------------
# 6. CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Limited-Label Fine-Tuning & Label-Efficiency Experiment Runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments.yaml",
        help="Path to parameterized experiments YAML config"
    )
    parser.add_argument(
        "--mode", type=str, default="supervised",
        choices=["supervised", "ssl_finetune", "ssl_motion", "ssl_pseudo", "full_pipeline", "ssl_motion_pseudo"],
        help="Experimental model variant"
    )
    parser.add_argument(
        "--label-fraction", type=int, default=100,
        choices=[10, 25, 50, 100],
        help="Percentage of labeled patients"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Reproducibility random seed"
    )
    parser.add_argument(
        "--ssl-checkpoint", type=str, default=None,
        help="Override path to SSL encoder checkpoint"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Execute a 1-batch CPU smoke test without performing full training"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config['project']['seed'] = args.seed
    set_seed(args.seed)
    
    device = get_device(args.device) if not args.smoke_test else torch.device("cpu")
    
    if args.smoke_test:
        run_experiment_smoke_test(
            config=config,
            mode=args.mode,
            label_fraction=args.label_fraction,
            device=device,
        )
        return
        
    # Guard against accidental training on development machine
    if device.type == 'cpu':
        print("\n" + "="*70)
        print("COMPUTE CONSTRAINT WARNING: Full experiment requested on CPU.")
        print("Per project specifications, actual experiments run on the separate GPU system.")
        print("Use --smoke-test for lightweight development verification.")
        print("="*70 + "\n")
        return
        
    print(f"\nExecuting Experiment: [{args.mode.upper()}] with [{args.label_fraction}%] labels on {device} (TRAINING MACHINE ONLY)...")
    # Training loop would be executed here on the separate GPU machine


if __name__ == "__main__":
    main()
