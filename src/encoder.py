"""
Shared encoder for SSL pretraining and segmentation.

Lightweight CNN encoder based on a standard U-Net encoder path.
Designed to be:
- Used standalone for SSL pretraining (encoder + SSL heads)
- Plugged into a full U-Net for segmentation (encoder + decoder)

Architecture: 4 down-sampling stages with residual connections.
Channels: [32, 64, 128, 256] (lightweight for cardiac MRI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResConvBlock(nn.Module):
    """Residual double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class SharedEncoder(nn.Module):
    """
    Shared encoder for SSL pretraining and segmentation.
    
    Returns multi-scale feature maps at each resolution level
    for skip connections in the U-Net decoder.
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        channels: Channel sizes for each encoder level
        dropout: Dropout rate
        use_residual: Whether to use residual connections in conv blocks
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        
        self.channels = channels
        self.n_levels = len(channels)
        
        Block = ResConvBlock if use_residual else ConvBlock
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_ch = in_channels
        for i, ch in enumerate(channels):
            self.encoders.append(Block(prev_ch, ch, dropout if i > 0 else 0.0))
            if i < len(channels) - 1:
                self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning bottleneck features and skip connections.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            bottleneck: Features at lowest resolution (B, channels[-1], H/8, W/8)
            skips: List of feature maps at each level [level0, level1, level2]
                   (excluding bottleneck). Used for skip connections in decoder.
        """
        skips = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < self.n_levels - 1:
                skips.append(x)
                x = self.pools[i](x)
        
        return x, skips
    
    def get_feature_channels(self) -> List[int]:
        """Return channel sizes at each level (for decoder construction)."""
        return self.channels.copy()
    
    def get_bottleneck_channels(self) -> int:
        """Return number of channels at the bottleneck."""
        return self.channels[-1]


class ProjectionHead(nn.Module):
    """
    Projection head for SSL feature comparison.
    
    Maps encoder features to a lower-dimensional space for
    temporal consistency loss computation.
    """
    
    def __init__(self, in_channels: int, proj_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, proj_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space. Returns (B, proj_dim)."""
        return F.normalize(self.head(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_encoder_weights(
    encoder: SharedEncoder,
    checkpoint_path: str,
    strict: bool = False,
) -> SharedEncoder:
    """
    Load pretrained weights into encoder.
    
    Handles the case where the checkpoint was saved from an SSL model
    that wraps the encoder (extracts encoder.* keys).
    
    Args:
        encoder: SharedEncoder instance
        checkpoint_path: Path to saved checkpoint
        strict: Whether to require exact key matching
    
    Returns:
        Encoder with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        # Extract encoder keys from full model
        state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('encoder.'):
                state_dict[k[len('encoder.'):]] = v
    elif 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('encoder.'):
                state_dict[k[len('encoder.'):]] = v
    else:
        state_dict = checkpoint
    
    encoder.load_state_dict(state_dict, strict=strict)
    print(f"Loaded encoder weights from {checkpoint_path}")
    return encoder
