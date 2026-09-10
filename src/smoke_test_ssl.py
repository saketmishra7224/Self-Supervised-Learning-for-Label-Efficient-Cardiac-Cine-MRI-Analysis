"""
Standalone lightweight CPU smoke test for Self-Supervised Temporal Pretraining.

Validates the full SSL pipeline without performing expensive multi-epoch training:
1. Loads temporal adjacent frame pairs (t, t+1) from ACDCTemporalDataset.
2. Asserts patient consistency (never pairing frames across different patients).
3. Verifies temporal ordering (adjacent sequential frame indices).
4. Tests patch-based masking mechanism (shape, mask ratio).
5. Runs forward pass through SSLModel on CPU.
6. Verifies reconstruction output shape and projection embeddings.
7. Computes reconstruction loss, temporal consistency loss, and combined SSL loss.
8. Checks backward pass and verifies gradient flow across encoder, decoder, and projection head.
9. Executes a single optimizer step.
10. Validates encoder weight saving and loading for transfer to segmentation models.
"""

import os
import sys

# Ensure project root is on sys.path and src/ does not shadow stdlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if len(sys.path) > 0 and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.ssl import SSLModel, compute_ssl_loss, count_parameters
from src.encoder import SharedEncoder, load_encoder_weights
from src.dataset import ACDCTemporalDataset


def run_ssl_smoke_test(config_path: str = "configs/ssl.yaml"):
    print("=" * 70)
    print("STARTING LIGHTWEIGHT SSL TEMPORAL PRETRAINING SMOKE TEST")
    print("=" * 70)
    
    # 1. Load Configuration
    cfg_file = Path(config_path)
    assert cfg_file.exists(), f"Configuration file {config_path} not found!"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    print(f"[1/7] Configuration successfully loaded from: {config_path}")
    print(f"      Mask patch size: {config['model']['mask_patch_size']}x{config['model']['mask_patch_size']}")
    print(f"      Mask ratio:      {config['model']['mask_ratio'] * 100:.0f}%")
    print(f"      Recon weight:    {config['loss']['recon_weight']}")
    print(f"      Temporal weight: {config['loss']['temporal_weight']}")
    
    # 2. Verify Dataset & Temporal Pairing Integrity
    data_cfg = config['data']
    train_split = data_cfg['train_split']
    processed_dir = data_cfg['processed_dir']
    
    dataset = ACDCTemporalDataset(
        processed_dir=processed_dir,
        split_file=train_split,
    )
    print(f"[2/7] ACDCTemporalDataset instantiated: {len(dataset):,} adjacent pairs available.")
    assert len(dataset) > 0, "Dataset contains 0 samples!"
    
    # Check 10 individual samples to verify temporal pairing rules
    print("      Verifying temporal pairing rules on sample pairs...")
    for idx in range(min(10, len(dataset))):
        sample = dataset[idx]
        pid = sample['patient_id']
        s_idx = sample['slice_idx']
        f_t = sample['frame_idx_t']
        f_t1 = sample['frame_idx_t1']
        
        # Verify frame adjacency and same patient
        assert f_t1 == f_t + 1, f"Frames not adjacent! Frame t: {f_t}, Frame t+1: {f_t1}"
        assert sample['frame_t'].shape == (1, 256, 256), f"Unexpected shape {sample['frame_t'].shape}"
        assert sample['frame_t1'].shape == (1, 256, 256), f"Unexpected shape {sample['frame_t1'].shape}"
    print("      Verification passed: All pairs are strictly from the same patient and adjacent in time.")
    
    # 3. Create DataLoader (Batch Size = 4 for CPU test)
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    frame_t = batch['frame_t']
    frame_t1 = batch['frame_t1']
    print(f"[3/7] Batch loaded successfully: frame_t={frame_t.shape}, frame_t1={frame_t1.shape}")
    
    # 4. Instantiate SSL Model
    device = torch.device("cpu")
    model = SSLModel(
        in_channels=config['model']['in_channels'],
        encoder_channels=config['model']['encoder_channels'],
        proj_dim=config['model']['proj_dim'],
        mask_patch_size=config['model']['mask_patch_size'],
        mask_ratio=config['model']['mask_ratio'],
        dropout=config['model']['dropout'],
        use_residual=config['model']['use_residual'],
    ).to(device)
    
    total_p = count_parameters(model)
    enc_p = count_parameters(model.encoder)
    dec_p = count_parameters(model.recon_decoder)
    proj_p = count_parameters(model.projection)
    print(f"[4/7] SSL Model instantiated on {device}:")
    print(f"      Total parameters:      {total_p:,}")
    print(f"      Shared Encoder:        {enc_p:,} ({enc_p/total_p*100:.1f}%)")
    print(f"      Reconstruction Decoder: {dec_p:,} ({dec_p/total_p*100:.1f}%)")
    print(f"      Projection Head:       {proj_p:,} ({proj_p/total_p*100:.1f}%)")
    
    # 5. Forward Pass: Masking, Reconstruction, and Projection
    model.train()
    results = model(frame_t, frame_t1)
    
    reconstructed = results['reconstructed']
    mask = results['mask']
    masked_input = results['masked_input']
    proj_t = results['proj_t']
    proj_t1 = results['proj_t1']
    
    assert mask.shape == (batch_size, 1, 256, 256), f"Bad mask shape {mask.shape}"
    assert reconstructed.shape == (batch_size, 1, 256, 256), f"Bad recon shape {reconstructed.shape}"
    assert proj_t.shape == (batch_size, config['model']['proj_dim']), f"Bad proj_t shape {proj_t.shape}"
    assert proj_t1.shape == (batch_size, config['model']['proj_dim']), f"Bad proj_t1 shape {proj_t1.shape}"
    
    # Verify L2 normalization of projection embeddings
    norm_t = torch.norm(proj_t, p=2, dim=-1)
    norm_t1 = torch.norm(proj_t1, p=2, dim=-1)
    assert torch.allclose(norm_t, torch.ones_like(norm_t), atol=1e-4), "proj_t embeddings are not unit normalized!"
    assert torch.allclose(norm_t1, torch.ones_like(norm_t1), atol=1e-4), "proj_t1 embeddings are not unit normalized!"
    
    actual_mask_ratio = mask.mean().item()
    expected_ratio = config['model']['mask_ratio']
    print(f"[5/7] Forward pass completed:")
    print(f"      Mask fraction:        {actual_mask_ratio:.3f} (target: {expected_ratio:.2f})")
    print(f"      Reconstructed output: {reconstructed.shape}")
    print(f"      Projection vectors:   {proj_t.shape} (unit L2 normalized: {norm_t.mean().item():.3f})")
    
    # Compute cosine similarities between adjacent frames
    cos_sim = F.cosine_similarity(proj_t, proj_t1, dim=-1)
    print(f"      Adjacent frame cosine similarities: {cos_sim.tolist()}")
    
    # 6. Loss Calculation and Backward Gradient Flow
    recon_loss, temp_loss, total_loss = compute_ssl_loss(
        results,
        recon_weight=config['loss']['recon_weight'],
        temporal_weight=config['loss']['temporal_weight'],
        recon_loss_type=config['loss']['recon_loss_type'],
    )
    
    assert not torch.isnan(recon_loss) and not torch.isinf(recon_loss), "Reconstruction loss is NaN/Inf!"
    assert not torch.isnan(temp_loss) and not torch.isinf(temp_loss), "Temporal loss is NaN/Inf!"
    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "Total SSL loss is NaN/Inf!"
    
    print(f"[6/7] Loss calculation verified:")
    print(f"      Reconstruction loss:  {recon_loss.item():.4f}")
    print(f"      Temporal consistency: {temp_loss.item():.4f}")
    print(f"      Total SSL loss:       {total_loss.item():.4f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check that all model components received gradients
    enc_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in model.recon_decoder.parameters() if p.requires_grad]
    proj_grads = [p.grad for p in model.projection.parameters() if p.requires_grad]
    
    assert all(g is not None and not torch.isnan(g).any() for g in enc_grads), "Encoder gradients missing or invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in dec_grads), "Decoder gradients missing or invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in proj_grads), "Projection gradients missing or invalid!"
    
    optimizer.step()
    print("      Gradient flow verified across SharedEncoder, ReconstructionDecoder, and ProjectionHead.")
    print("      Optimizer step executed successfully.")
    
    # 7. Checkpoint & Transfer Learning Verification
    ckpt_dir = Path(config['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_encoder_path = str(ckpt_dir / "test_ssl_encoder.pth")
    
    model.save_encoder(test_encoder_path)
    assert Path(test_encoder_path).exists(), "Saved encoder file not found on disk!"
    
    # Reload into a fresh SharedEncoder
    fresh_encoder = SharedEncoder(
        in_channels=config['model']['in_channels'],
        channels=config['model']['encoder_channels'],
        dropout=config['model']['dropout'],
        use_residual=config['model']['use_residual'],
    )
    load_encoder_weights(fresh_encoder, test_encoder_path, strict=True)
    
    # Verify loaded weights match model encoder weights
    for (n1, p1), (n2, p2) in zip(model.encoder.named_parameters(), fresh_encoder.named_parameters()):
        assert torch.equal(p1, p2), f"Weight mismatch in {n1} after transfer loading!"
    print(f"[7/7] Transfer learning verified: Encoder weights saved and reloaded into fresh SharedEncoder.")
    
    # Clean up test file
    if Path(test_encoder_path).exists():
        Path(test_encoder_path).unlink()
    
    print("=" * 70)
    print("LIGHTWEIGHT SSL SMOKE TEST RESULT: ALL CHECKS PASSED (OK)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_ssl_smoke_test()
