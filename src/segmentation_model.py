"""
Segmentation model: U-Net with shared encoder + decoder.

Uses the SharedEncoder from encoder.py and adds a symmetric decoder
with skip connections. Supports loading pretrained encoder weights
from SSL pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .encoder import SharedEncoder, ConvBlock, ResConvBlock, load_encoder_weights


class UNetDecoder(nn.Module):
    """
    U-Net decoder with skip connections.
    
    Mirrors the encoder structure with transposed convolutions
    for upsampling and concatenation of skip features.
    
    Args:
        encoder_channels: Channel sizes from encoder (e.g., [32, 64, 128, 256])
        num_classes: Number of output segmentation classes
        dropout: Dropout rate
        use_residual: Whether to use residual conv blocks
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        
        Block = ResConvBlock if use_residual else ConvBlock
        
        # Decoder stages (reverse order, excluding bottleneck)
        reversed_channels = list(reversed(encoder_channels))
        
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After concat with skip: out_ch (from upconv) + out_ch (from skip) = 2*out_ch
            self.decoders.append(
                Block(out_ch * 2, out_ch, dropout if i < len(reversed_channels) - 2 else 0.0)
            )
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, 1)
    
    def forward(
        self, bottleneck: torch.Tensor, skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, C_bottleneck, H/8, W/8)
            skips: List of skip features [level0, level1, level2] from encoder
            
        Returns:
            Logits: (B, num_classes, H, W)
        """
        x = bottleneck
        
        # Reverse skips to match decoder order
        skips = list(reversed(skips))
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            # Handle size mismatch from non-divisible dimensions
            skip = skips[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.final_conv(x)


class SegmentationUNet(nn.Module):
    """
    Full U-Net segmentation model.
    
    Combines SharedEncoder with UNetDecoder. Supports:
    - Random initialization for baseline
    - Loading pretrained encoder from SSL
    - Freezing/unfreezing encoder layers
    
    Args:
        in_channels: Input channels (1 for grayscale MRI)
        num_classes: Number of segmentation classes
        encoder_channels: Channel sizes per level
        dropout: Dropout rate
        use_residual: Whether to use residual blocks
        pretrained_encoder_path: Optional path to pretrained encoder weights
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        encoder_channels: List[int] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
        pretrained_encoder_path: Optional[str] = None,
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            dropout=dropout,
            use_residual=use_residual,
        )
        
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            dropout=dropout,
            use_residual=use_residual,
        )
        
        self.num_classes = num_classes
        
        # Load pretrained encoder if provided
        if pretrained_encoder_path:
            self.load_pretrained_encoder(pretrained_encoder_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 1, H, W)
            
        Returns:
            Logits: (B, num_classes, H, W)
        """
        bottleneck, skips = self.encoder(x)
        logits = self.decoder(bottleneck, skips)
        return logits
    
    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """Load SSL-pretrained encoder weights."""
        self.encoder = load_encoder_weights(self.encoder, checkpoint_path, strict)
    
    def freeze_encoder(self):
        """Freeze encoder parameters (for initial fine-tuning phase)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    
    def get_encoder_params(self):
        """Return encoder parameters (for differential learning rates)."""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Return decoder parameters (for differential learning rates)."""
        return self.decoder.parameters()
    
    def get_features(self, x: torch.Tensor):
        """Get encoder features without decoding (for SSL/analysis)."""
        return self.encoder(x)


def build_segmentation_model(
    config: dict,
    pretrained_path: Optional[str] = None,
) -> SegmentationUNet:
    """
    Build segmentation model from config dictionary.
    
    Args:
        config: Configuration dictionary (from YAML)
        pretrained_path: Optional path to pretrained encoder
    
    Returns:
        SegmentationUNet model
    """
    model_cfg = config.get('model', config.get('baseline', {}))
    data_cfg = config.get('data', {})
    
    in_channels = model_cfg.get('in_channels', 1)
    num_classes = model_cfg.get('num_classes', data_cfg.get('num_classes', 4))
    encoder_channels = model_cfg.get('encoder_channels', [32, 64, 128, 256])
    dropout = model_cfg.get('dropout', 0.1)
    use_residual = model_cfg.get('use_residual', True)
    
    model = SegmentationUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_channels=encoder_channels,
        dropout=dropout,
        use_residual=use_residual,
        pretrained_encoder_path=pretrained_path,
    )
    
    return model
