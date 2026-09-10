"""
Training utilities: Trainer class, seed management, and helpers.

Provides a flexible Trainer that supports:
- Standard supervised training
- SSL pretraining
- Fine-tuning with pretrained weights
- Mixed precision (AMP)
- Early stopping
- TensorBoard logging
- Checkpoint management
- Seed-based reproducibility
"""

import os
import sys

# Ensure project root is on sys.path and prevent src/ from shadowing stdlib modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import json
import random
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device(preference: str = "auto") -> torch.device:
    """Get compute device."""
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("Using CPU (no GPU detected)")
    else:
        device = torch.device(preference)
    return device


def get_adaptive_batch_size(device: torch.device, default: int = 8) -> int:
    """Adaptively set batch size based on available GPU memory."""
    if device.type != 'cuda':
        return min(default, 4)
    
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    if mem_gb >= 16:
        return 16
    elif mem_gb >= 8:
        return 8
    elif mem_gb >= 6:
        return 4
    else:
        return 2


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 30, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    General-purpose trainer for segmentation and SSL.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Compute device
        scheduler: Optional LR scheduler
        mixed_precision: Whether to use AMP
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
        experiment_name: Name for this experiment
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "results/logs",
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision and device.type == 'cuda'
        self.scaler = GradScaler() if self.mixed_precision else None
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # TensorBoard
        self.writer = None
        if HAS_TB:
            tb_dir = Path(log_dir) / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_dice': [], 'lr': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / max(n_batches, 1)
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []
        all_patient_ids = []
        
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            
            val_loss += loss.item()
            n_batches += 1
            
            # Collect predictions for metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
            if 'patient_id' in batch:
                all_patient_ids.extend(batch['patient_id'])
        
        avg_loss = val_loss / max(n_batches, 1)
        results = {'val_loss': avg_loss}
        
        # Compute segmentation metrics
        if compute_metrics_fn is not None and all_preds:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            metrics = compute_metrics_fn(all_preds, all_targets, all_patient_ids)
            results.update(metrics)
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        early_stopping_patience: int = 30,
        compute_metrics_fn: Optional[Callable] = None,
        monitor_metric: str = "val_dice",
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            compute_metrics_fn: Function to compute validation metrics
            monitor_metric: Metric to monitor for early stopping/checkpointing
            
        Returns:
            Training history dict
        """
        early_stop = EarlyStopping(
            patience=early_stopping_patience,
            mode="max" if "dice" in monitor_metric.lower() else "min",
        )
        
        best_metric = -float('inf') if "dice" in monitor_metric.lower() else float('inf')
        best_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Max epochs: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Monitor: {monitor_metric}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, n_epochs + 1):
            t_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch, compute_metrics_fn)
            
            # LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(monitor_metric, val_metrics['val_loss']))
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['lr'].append(current_lr)
            
            val_dice = val_metrics.get('Mean_Dice', val_metrics.get('val_dice', 0.0))
            self.history['val_dice'].append(val_dice)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('Metrics/val_dice', val_dice, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
            
            # Check for best model
            current_metric = val_metrics.get(monitor_metric, val_dice)
            is_best = False
            if "dice" in monitor_metric.lower():
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    is_best = True
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    is_best = True
            
            # Save checkpoint
            if is_best:
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Print progress
            elapsed = time.time() - t_start
            status = " ★ BEST" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s{status}"
            )
            
            # Early stopping
            if early_stop(current_metric):
                print(f"\nEarly stopping at epoch {epoch}. Best: epoch {best_epoch}")
                break
        
        # Save final checkpoint
        self._save_checkpoint(epoch, val_metrics, is_best=False, tag="final")
        
        # Save history
        history_path = self.checkpoint_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        print(f"\nTraining complete. Best {monitor_metric}: {best_metric:.4f} at epoch {best_epoch}")
        return self.history
    
    def _save_checkpoint(
        self, epoch: int, metrics: Dict, is_best: bool = False, tag: str = ""
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, path)
        
        if tag:
            path = self.checkpoint_dir / f"{self.experiment_name}_{tag}.pth"
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def compute_val_metrics_wrapper(preds, targets, patient_ids=None):
    """Wrapper to compute metrics in Trainer.validate()."""
    from src.metrics import compute_metrics_batch, compute_patient_level_metrics
    
    batch_results = compute_metrics_batch(
        preds, targets, patient_ids, compute_hd=False  # Skip HD95 during training (slow)
    )
    
    # If patient IDs available, compute patient-level
    if patient_ids:
        patient_results = compute_patient_level_metrics(batch_results['per_sample'])
        return patient_results['mean']
    
    return batch_results['mean']


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline 2D U-Net Cardiac Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to YAML config")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name override")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch CPU smoke test and exit without training")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to base_config.yaml if baseline_config.yaml is not found
        fallback = Path("configs/base_config.yaml")
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Command-line overrides
    exp_name = args.experiment_name or config.get("experiment_name", "baseline_unet")
    seed = args.seed if args.seed is not None else config.get("random_seed", config.get("project", {}).get("seed", 42))
    device_pref = args.device or config.get("device", config.get("project", {}).get("device", "auto"))
    
    # Dataset and training configs
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", config.get("baseline", {}))
    loss_cfg = config.get("loss", {})
    log_cfg = config.get("logging", {})
    
    epochs = args.epochs or train_cfg.get("epochs", 200)
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    lr = args.lr or train_cfg.get("learning_rate", train_cfg.get("lr", 1e-4))
    
    # Set reproducibility seed
    set_seed(seed)
    
    # If smoke test requested, force CPU and 1 batch verification
    if args.smoke_test:
        print("=" * 65)
        print("RUNNING BASELINE CPU SMOKE TEST (1 BATCH FORWARD / LOSS / BACKWARD)")
        print("=" * 65)
        device = torch.device("cpu")
        
        from src.dataset import ACDCSegDataset
        from src.segmentation_model import build_segmentation_model
        from src.losses import DiceCELoss
        from src.metrics import compute_metrics_single
        from src.encoder import count_parameters
        
        processed_dir = data_cfg.get("processed_dir", "data/processed")
        splits_dir = data_cfg.get("splits_dir", "data/splits")
        train_split = data_cfg.get("train_split", str(Path(splits_dir) / "train_patients.txt"))
        
        print(f"Loading sample dataset from {train_split}...")
        dataset = ACDCSegDataset(processed_dir=processed_dir, split_file=train_split)
        print(f"Available training labeled slices: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        print(f"Input image tensor shape: {images.shape}, dtype: {images.dtype}")
        print(f"Input mask tensor shape:  {masks.shape}, unique labels: {torch.unique(masks).tolist()}")
        
        # Build model
        model = build_segmentation_model(config).to(device)
        n_params = count_parameters(model)
        print(f"Model instantiated: {model.__class__.__name__} ({n_params:,} parameters)")
        
        # Forward pass
        logits = model(images)
        print(f"Output logits shape:      {logits.shape} (B, num_classes={logits.shape[1]}, H, W)")
        assert logits.shape == (2, 4, 256, 256), f"Unexpected logits shape: {logits.shape}"
        
        # Loss calculation
        criterion = DiceCELoss(
            num_classes=4,
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            include_background=loss_cfg.get("include_background", False),
        )
        loss = criterion(logits, masks)
        print(f"Computed DiceCELoss:      {loss.item():.4f}")
        assert torch.isfinite(loss), "Loss is not finite!"
        
        # Single optimizer step solely to verify backward pass
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Gradient backward step:   SUCCESS (gradient flow verified)")
        
        # Compute metrics on the sample
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = masks.cpu().numpy()
        sample_metrics = compute_metrics_single(preds[0], targets[0], compute_hd=False)
        print(f"Sample Dice metrics:      LV={sample_metrics['LV_Dice']:.4f}, Myo={sample_metrics['Myocardium_Dice']:.4f}, RV={sample_metrics['RV_Dice']:.4f}, Mean={sample_metrics['Mean_Dice']:.4f}")
        
        print("=" * 65)
        print("SMOKE TEST COMPLETE: Pipeline verified on CPU. No full training performed.")
        print("=" * 65)
        return
        
    # Normal training mode (to be executed on GPU system)
    device = get_device(device_pref)
    
    from src.dataset import ACDCSegDataset, get_train_transforms, get_val_transforms
    from src.segmentation_model import build_segmentation_model
    from src.losses import DiceCELoss
    from src.encoder import count_parameters
    
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    splits_dir = data_cfg.get("splits_dir", "data/splits")
    train_split = data_cfg.get("train_split", str(Path(splits_dir) / "train_patients.txt"))
    val_split = data_cfg.get("val_split", str(Path(splits_dir) / "val_patients.txt"))
    num_workers = data_cfg.get("num_workers", 0)
    
    train_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=train_split,
        transform=get_train_transforms(),
    )
    val_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=val_split,
        transform=get_val_transforms(),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    model = build_segmentation_model(config).to(device)
    print(f"Model: SegmentationUNet with {count_parameters(model):,} parameters")
    
    criterion = DiceCELoss(
        num_classes=data_cfg.get("num_classes", 4),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        include_background=loss_cfg.get("include_background", False),
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    
    scheduler_type = train_cfg.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)
    else:
        scheduler = None
        
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=train_cfg.get("mixed_precision", True),
        checkpoint_dir=log_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=log_cfg.get("log_dir", "results/logs"),
        experiment_name=exp_name,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    patience = train_cfg.get("early_stopping_patience", 30)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=epochs,
        early_stopping_patience=patience,
        compute_metrics_fn=compute_val_metrics_wrapper,
        monitor_metric="Mean_Dice",
    )


if __name__ == "__main__":
    main()
