"""
Self-Supervised Learning (SSL) module for cardiac cine MRI.

Implements a temporal representation-learning framework based on:
1. Masked Image Reconstruction: Divides cine frames into non-overlapping patches,
   masks a configurable fraction, and reconstructs pixel intensities from encoder features.
2. Adjacent-Frame Feature Consistency: Enforces temporal feature consistency between
   adjacent cine frames (t, t+1) from the same patient sequence.

The trained encoder can subsequently be transferred to the supervised segmentation model.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Sys.path guard: ensure project root is accessible and prevent src/ from
# shadowing standard library modules (e.g. stdlib 'ssl').
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
from typing import Dict, Tuple, Optional, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoder import SharedEncoder, ProjectionHead, count_parameters
from src.dataset import ACDCTemporalDataset


# ---------------------------------------------------------------------------
# 1. Patch Masker
# ---------------------------------------------------------------------------

class PatchMasker(nn.Module):
    """
    Random patch-based masking mechanism for masked image reconstruction.
    
    Divides an image into non-overlapping patches of size `patch_size x patch_size`
    and randomly masks a specified fraction (`mask_ratio`) of them.
    
    Args:
        patch_size: Square patch side length in pixels (default 16)
        mask_ratio: Fraction of patches to mask out (default 0.50)
    """
    
    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.50):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked input and binary mask.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            masked_x: Input with masked patches zeroed out (B, C, H, W)
            mask: Binary mask tensor (B, 1, H, W) where 1 = masked, 0 = visible
        """
        B, C, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        n_patches = ph * pw
        n_mask = max(1, int(n_patches * self.mask_ratio))
        
        mask = torch.zeros((B, 1, H, W), dtype=x.dtype, device=x.device)
        
        for b in range(B):
            perm = torch.randperm(n_patches, device=x.device)[:n_mask]
            for idx in perm:
                r = (idx // pw) * self.patch_size
                c = (idx % pw) * self.patch_size
                mask[b, 0, r:r + self.patch_size, c:c + self.patch_size] = 1.0
        
        masked_x = x * (1.0 - mask)
        return masked_x, mask


# ---------------------------------------------------------------------------
# 2. Reconstruction Decoder
# ---------------------------------------------------------------------------

class ReconstructionDecoder(nn.Module):
    """
    Lightweight convolutional decoder for masked image reconstruction.
    
    Takes bottleneck features from the SharedEncoder and progressively
    upsamples them via transposed convolutions and residual convolutions
    to reconstruct the full original 2D image.
    
    Args:
        encoder_channels: Channel depths from encoder stages (e.g. [32, 64, 128, 256])
        out_channels: Output image channels (1 for grayscale cardiac MRI)
    """
    
    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        out_channels: int = 1,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        
        reversed_ch = list(reversed(encoder_channels))
        
        layers = []
        for i in range(len(reversed_ch) - 1):
            in_ch = reversed_ch[i]
            out_ch = reversed_ch[i + 1]
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            ])
        
        # Final 1x1 convolution mapping to input image channels
        layers.append(nn.Conv2d(reversed_ch[-1], out_channels, kernel_size=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct 2D image from encoder bottleneck representation."""
        return self.decoder(features)


# ---------------------------------------------------------------------------
# 3. SSL Model
# ---------------------------------------------------------------------------

class SSLModel(nn.Module):
    """
    Self-Supervised Learning model for cardiac cine MRI.
    
    Combines:
    1. SharedEncoder: 4-stage residual CNN encoder to be transferred downstream.
    2. PatchMasker: Configurable patch-level input masking.
    3. ReconstructionDecoder: Lightweight decoder for masked reconstruction.
    4. ProjectionHead: Non-linear projection MLP for temporal feature consistency.
    
    Args:
        in_channels: Input channels (1 for grayscale cine MRI)
        encoder_channels: Channel dimensions for each encoder level
        proj_dim: Projection embedding dimension for temporal consistency
        mask_patch_size: Square patch side length in pixels
        mask_ratio: Fraction of patches to mask
        dropout: Encoder dropout rate
        use_residual: Whether to use residual connections in the encoder
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: Optional[List[int]] = None,
        proj_dim: int = 128,
        mask_patch_size: int = 16,
        mask_ratio: float = 0.50,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        
        self.encoder_channels = encoder_channels
        self.proj_dim = proj_dim
        
        # 1. Shared encoder (transferred to downstream segmentation model)
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            dropout=dropout,
            use_residual=use_residual,
        )
        
        # 2. Patch masker
        self.masker = PatchMasker(
            patch_size=mask_patch_size,
            mask_ratio=mask_ratio,
        )
        
        # 3. Reconstruction decoder (lightweight, not transferred)
        self.recon_decoder = ReconstructionDecoder(
            encoder_channels=encoder_channels,
            out_channels=in_channels,
        )
        
        # 4. Projection head for temporal feature consistency (not transferred)
        self.projection = ProjectionHead(
            in_channels=encoder_channels[-1],
            proj_dim=proj_dim,
        )
    
    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SSL pretraining.
        
        Args:
            frame_t: Current frame tensor (B, 1, H, W)
            frame_t1: Temporally adjacent next frame tensor (B, 1, H, W), optional
            
        Returns:
            Dict containing:
                - reconstructed: Reconstructed image for frame_t (B, 1, H, W)
                - mask: Binary patch mask used for frame_t (B, 1, H, W)
                - original: Original unmasked frame_t (B, 1, H, W)
                - masked_input: Input with masked patches zeroed out (B, 1, H, W)
                - proj_t: L2-normalized projection vector for frame_t (B, proj_dim)
                - proj_t1: L2-normalized projection vector for frame_t1 (B, proj_dim)
                - features_t: Bottleneck feature map for frame_t
                - features_t1: Bottleneck feature map for frame_t1
        """
        results = {}
        
        # === Task 1: Masked Reconstruction on frame_t ===
        masked_input, mask = self.masker(frame_t)
        bottleneck_masked, _ = self.encoder(masked_input)
        reconstructed = self.recon_decoder(bottleneck_masked)
        
        # Ensure reconstructed spatial dimensions strictly match input dimensions
        if reconstructed.shape[2:] != frame_t.shape[2:]:
            reconstructed = F.interpolate(
                reconstructed,
                size=frame_t.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        
        results['reconstructed'] = reconstructed
        results['mask'] = mask
        results['original'] = frame_t
        results['masked_input'] = masked_input
        
        # === Task 2: Temporal Consistency between frame_t and frame_t1 ===
        if frame_t1 is not None:
            # Clean (unmasked) forward pass through encoder for feature representations
            bottleneck_t, _ = self.encoder(frame_t)
            bottleneck_t1, _ = self.encoder(frame_t1)
            
            # Map bottleneck features to normalized embedding space
            proj_t = self.projection(bottleneck_t)
            proj_t1 = self.projection(bottleneck_t1)
            
            results['proj_t'] = proj_t
            results['proj_t1'] = proj_t1
            results['features_t'] = bottleneck_t
            results['features_t1'] = bottleneck_t1
        
        return results
    
    def get_encoder_state_dict(self) -> dict:
        """Extract encoder weights for transfer to segmentation model."""
        return self.encoder.state_dict()
    
    def save_encoder(self, path: str):
        """Save encoder weights to disk for downstream transfer."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'channels': self.encoder_channels,
        }, path)
        print(f"Pretrained encoder saved to {path}")


# ---------------------------------------------------------------------------
# 4. SSL Loss Computation
# ---------------------------------------------------------------------------

def compute_ssl_loss(
    results: Dict[str, torch.Tensor],
    recon_weight: float = 1.0,
    temporal_weight: float = 0.1,
    recon_loss_type: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total SSL pretraining loss:
        L_ssl = lambda_recon * L_recon + lambda_temporal * L_temporal
    
    Args:
        results: Dictionary returned by SSLModel.forward
        recon_weight: Scalar weight lambda_recon
        temporal_weight: Scalar weight lambda_temporal
        recon_loss_type: "l1" (default) or "mse", computed on masked pixels
        
    Returns:
        recon_loss: Reconstruction loss scalar tensor
        temporal_loss: Temporal consistency loss scalar tensor
        total_loss: Weighted combined SSL loss scalar tensor
    """
    reconstructed = results['reconstructed']
    original = results['original']
    mask = results['mask']
    
    # 1. Masked Reconstruction Loss (computed specifically on masked pixels M=1)
    mask_sum = mask.sum()
    if mask_sum > 0:
        if recon_loss_type.lower() == "mse":
            recon_loss = F.mse_loss(reconstructed * mask, original * mask, reduction='sum') / mask_sum
        else:
            recon_loss = F.l1_loss(reconstructed * mask, original * mask, reduction='sum') / mask_sum
    else:
        recon_loss = F.l1_loss(reconstructed, original)
    
    # 2. Adjacent-Frame Temporal Consistency Loss
    # Since proj_t and proj_t1 are L2-normalized vectors on the unit sphere,
    # MSE(z_t, z_t1) = 2 - 2 * cos_sim(z_t, z_t1), directly maximizing cosine similarity.
    if 'proj_t' in results and 'proj_t1' in results:
        temporal_loss = F.mse_loss(results['proj_t'], results['proj_t1'])
    else:
        temporal_loss = torch.tensor(0.0, device=reconstructed.device)
    
    total_loss = recon_weight * recon_loss + temporal_weight * temporal_loss
    return recon_loss, temporal_loss, total_loss


# ---------------------------------------------------------------------------
# 5. SSL Trainer & Pipeline
# ---------------------------------------------------------------------------

class SSLTrainer:
    """
    Trainer for Self-Supervised Temporal Pretraining.
    
    Manages optimizer, LR scheduling, mixed precision (AMP), logging,
    checkpointing (full model + standalone encoder), and resume support.
    """
    
    def __init__(
        self,
        model: SSLModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        recon_weight: float = 1.0,
        temporal_weight: float = 0.1,
        recon_loss_type: str = "l1",
        mixed_precision: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = "checkpoints/ssl",
        output_dir: str = "results/ssl",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.recon_weight = recon_weight
        self.temporal_weight = temporal_weight
        self.recon_loss_type = recon_loss_type
        self.mixed_precision = mixed_precision and (device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.scheduler = scheduler
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'total_loss': [],
            'recon_loss': [],
            'temporal_loss': [],
            'lr': [],
        }
        self.start_epoch = 1
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch across all adjacent cine frame pairs."""
        self.model.train()
        total_loss_acc = 0.0
        recon_loss_acc = 0.0
        temp_loss_acc = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"SSL Epoch {epoch:3d}", leave=False)
        for batch in pbar:
            frame_t = batch['frame_t'].to(self.device)
            frame_t1 = batch['frame_t1'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    results = self.model(frame_t, frame_t1)
                    recon_loss, temp_loss, loss = compute_ssl_loss(
                        results,
                        recon_weight=self.recon_weight,
                        temporal_weight=self.temporal_weight,
                        recon_loss_type=self.recon_loss_type,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(frame_t, frame_t1)
                recon_loss, temp_loss, loss = compute_ssl_loss(
                    results,
                    recon_weight=self.recon_weight,
                    temporal_weight=self.temporal_weight,
                    recon_loss_type=self.recon_loss_type,
                )
                loss.backward()
                self.optimizer.step()
            
            total_loss_acc += loss.item()
            recon_loss_acc += recon_loss.item()
            temp_loss_acc += temp_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'temp': f"{temp_loss.item():.4f}",
            })
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        n = max(n_batches, 1)
        return {
            'total_loss': total_loss_acc / n,
            'recon_loss': recon_loss_acc / n,
            'temporal_loss': temp_loss_acc / n,
            'lr': float(self.optimizer.param_groups[0]['lr']),
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint with resume support."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.get_encoder_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save current epoch checkpoint
        ckpt_path = self.checkpoint_dir / f"ssl_checkpoint_epoch{epoch}.pth"
        torch.save(state, ckpt_path)
        
        # Save standalone encoder for transfer learning
        self.model.save_encoder(str(self.checkpoint_dir / f"ssl_encoder_epoch{epoch}.pth"))
        
        if is_best:
            best_path = self.checkpoint_dir / "ssl_best.pth"
            torch.save(state, best_path)
            self.model.save_encoder(str(self.checkpoint_dir / "ssl_encoder_best.pth"))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from saved checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.scaler is not None and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"Resumed SSL training from epoch {self.start_epoch} (checkpoint: {checkpoint_path})")
    
    def run_smoke_test(self, dataloader: DataLoader) -> Dict[str, float]:
        """Lightweight 1-batch smoke test to verify execution, loss, and gradients."""
        print("\n--- Running Lightweight SSL Smoke Test (1 Batch) ---")
        self.model.train()
        batch = next(iter(dataloader))
        frame_t = batch['frame_t'].to(self.device)
        frame_t1 = batch['frame_t1'].to(self.device)
        
        print(f"Input frame_t shape: {frame_t.shape}, frame_t1 shape: {frame_t1.shape}")
        self.optimizer.zero_grad()
        
        results = self.model(frame_t, frame_t1)
        recon = results['reconstructed']
        mask = results['mask']
        proj_t = results['proj_t']
        proj_t1 = results['proj_t1']
        
        print(f"Mask shape: {mask.shape} (masked fraction: {mask.mean().item():.3f})")
        print(f"Reconstructed shape: {recon.shape}")
        print(f"Projection embeddings: {proj_t.shape}, L2 norms: t={torch.norm(proj_t, dim=-1).mean().item():.3f}, t1={torch.norm(proj_t1, dim=-1).mean().item():.3f}")
        
        recon_loss, temp_loss, total_loss = compute_ssl_loss(
            results,
            recon_weight=self.recon_weight,
            temporal_weight=self.temporal_weight,
            recon_loss_type=self.recon_loss_type,
        )
        print(f"Reconstruction Loss: {recon_loss.item():.4f}")
        print(f"Temporal Loss:       {temp_loss.item():.4f}")
        print(f"Total SSL Loss:      {total_loss.item():.4f}")
        
        # Verify gradient flow
        total_loss.backward()
        
        has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.encoder.parameters())
        has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.recon_decoder.parameters())
        has_proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.projection.parameters())
        
        print(f"Gradient Check: Encoder={has_encoder_grad}, Decoder={has_decoder_grad}, ProjectionHead={has_proj_grad}")
        assert has_encoder_grad, "Encoder has no gradients!"
        assert has_decoder_grad, "Reconstruction decoder has no gradients!"
        assert has_proj_grad, "Projection head has no gradients!"
        
        self.optimizer.step()
        print("Optimizer step completed successfully!")
        print("--- SSL Smoke Test PASSED ---\n")
        
        return {
            'recon_loss': recon_loss.item(),
            'temporal_loss': temp_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def train(self, dataloader: DataLoader, n_epochs: int, save_interval: int = 10) -> Dict:
        """Full SSL training loop (for separate GPU training system)."""
        print(f"\n{'='*70}")
        print(f"Starting SSL Pretraining (TRAINING MACHINE ONLY)")
        print(f"Device: {self.device} | AMP: {self.mixed_precision}")
        print(f"Epochs: {self.start_epoch} -> {n_epochs} | Batch size: {dataloader.batch_size}")
        print(f"Recon weight: {self.recon_weight} | Temporal weight: {self.temporal_weight}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, n_epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)
            
            self.history['epoch'].append(epoch)
            for k in ['total_loss', 'recon_loss', 'temporal_loss', 'lr']:
                self.history[k].append(metrics[k])
            
            is_best = metrics['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['total_loss']
            
            print(
                f"Epoch {epoch:3d}/{n_epochs:3d} | "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Recon: {metrics['recon_loss']:.4f} | "
                f"Temporal: {metrics['temporal_loss']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                f"{' [BEST]' if is_best else ''}"
            )
            
            if epoch % save_interval == 0 or is_best or epoch == n_epochs:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Save final model, encoder, and training history
        torch.save(self.model.state_dict(), self.checkpoint_dir / "ssl_final.pth")
        self.model.save_encoder(str(self.checkpoint_dir / "ssl_encoder_final.pth"))
        
        history_path = self.output_dir / "ssl_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\nSSL Pretraining completed in {elapsed / 60:.1f} minutes.")
        print(f"Encoder checkpoint saved to: {self.checkpoint_dir / 'ssl_encoder_best.pth'}")
        return self.history


# ---------------------------------------------------------------------------
# 6. Builder Helpers & CLI
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Ensure deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_ssl_model(config: dict) -> SSLModel:
    """Instantiate SSLModel from configuration dict."""
    model_cfg = config.get('model', {})
    return SSLModel(
        in_channels=model_cfg.get('in_channels', 1),
        encoder_channels=model_cfg.get('encoder_channels', [32, 64, 128, 256]),
        proj_dim=model_cfg.get('proj_dim', 128),
        mask_patch_size=model_cfg.get('mask_patch_size', 16),
        mask_ratio=model_cfg.get('mask_ratio', 0.50),
        dropout=model_cfg.get('dropout', 0.1),
        use_residual=model_cfg.get('use_residual', True),
    )


def create_ssl_pipeline(config: dict, device_override: Optional[str] = None):
    """Construct model, dataset, dataloader, optimizer, and trainer."""
    seed = config.get('random_seed', 42)
    set_seed(seed)
    
    # Device selection
    if device_override:
        device_str = device_override
    else:
        device_str = config.get('device', 'auto')
    
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    # Build dataset
    data_cfg = config.get('data', {})
    processed_dir = data_cfg.get('processed_dir', 'data/processed')
    train_split = data_cfg.get('train_split', 'data/splits/train_patients.txt')
    
    dataset = ACDCTemporalDataset(
        processed_dir=processed_dir,
        split_file=train_split,
    )
    
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 16)
    if device.type == 'cpu':
        batch_size = min(batch_size, 4)
    
    num_workers = data_cfg.get('num_workers', 0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    
    # Build model
    model = build_ssl_model(config)
    
    # Optimizer
    lr = float(train_cfg.get('learning_rate', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    epochs = train_cfg.get('epochs', 100)
    scheduler_type = train_cfg.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # Loss weights & trainer
    loss_cfg = config.get('loss', {})
    log_cfg = config.get('logging', {})
    
    trainer = SSLTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        recon_weight=loss_cfg.get('recon_weight', 1.0),
        temporal_weight=loss_cfg.get('temporal_weight', 0.1),
        recon_loss_type=loss_cfg.get('recon_loss_type', 'l1'),
        mixed_precision=train_cfg.get('mixed_precision', True),
        scheduler=scheduler,
        checkpoint_dir=log_cfg.get('checkpoint_dir', 'checkpoints/ssl'),
        output_dir=log_cfg.get('output_dir', 'results/ssl'),
    )
    
    return model, dataloader, trainer, device


def main():
    parser = argparse.ArgumentParser(
        description="Self-Supervised Temporal Pretraining for Cardiac Cine MRI (ACDC)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/ssl.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a lightweight 1-batch smoke test on CPU without training"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override initial learning rate"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    model, dataloader, trainer, device = create_ssl_pipeline(config, device_override=args.device)
    
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    print(f"\nSSL Model initialized on {device}")
    print(f"Total trainable parameters:   {total_params:,}")
    print(f"Shared encoder parameters:    {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"Temporal dataset pairs:       {len(dataloader.dataset):,}")
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if args.smoke_test:
        trainer.run_smoke_test(dataloader)
        return
    
    # Guard against accidental training on development machine
    if device.type == 'cpu':
        print("\n" + "="*70)
        print("WARNING: You are about to run full SSL pretraining on CPU.")
        print("Per project specifications, model training belongs on the separate GPU system.")
        print("Run with --smoke-test for development validation.")
        print("="*70 + "\n")
    
    epochs = config['training'].get('epochs', 100)
    save_interval = config['training'].get('save_interval', 10)
    trainer.train(dataloader, n_epochs=epochs, save_interval=save_interval)


if __name__ == "__main__":
    main()
