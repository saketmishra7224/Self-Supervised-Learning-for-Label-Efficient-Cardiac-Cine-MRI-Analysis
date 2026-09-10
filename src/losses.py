"""
Loss functions for all project stages.

Includes:
- DiceCELoss: Combined Dice + Cross-Entropy for supervised segmentation
- MaskedReconstructionLoss: L1/L2 for masked image reconstruction (SSL)
- TemporalConsistencyLoss: Feature-level consistency between adjacent frames (SSL)
- MotionConsistencyLoss: Penalizes temporal disagreement in warped predictions
- PseudoLabelLoss: Confidence-weighted supervision from pseudo-labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.
    
    Computes per-class Dice and averages (excluding background optionally).
    
    Args:
        num_classes: Number of segmentation classes
        include_background: Whether to include background in loss
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, H, W) integer class labels
            
        Returns:
            Scalar Dice loss (1 - mean Dice)
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        start_class = 0 if self.include_background else 1
        
        dice_scores = []
        for c in range(start_class, self.num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]
            
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())
        
        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss.
    
    This is the standard loss for cardiac MRI segmentation.
    
    Args:
        num_classes: Number of segmentation classes
        dice_weight: Weight for Dice component
        ce_weight: Weight for CE component
        include_background: Whether to include background in Dice
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes, include_background)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, H, W) integer class labels
            
        Returns:
            Weighted sum of Dice and CE losses
        """
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class MaskedReconstructionLoss(nn.Module):
    """
    Loss for masked image reconstruction in SSL pretraining.
    
    Computes L1 or L2 loss only on masked regions.
    
    Args:
        loss_type: "l1" or "l2"
    """
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reconstructed: (B, 1, H, W) reconstructed image
            original: (B, 1, H, W) original image
            mask: (B, 1, H, W) binary mask (1 = masked/reconstruct, 0 = visible)
            
        Returns:
            Scalar reconstruction loss on masked regions
        """
        diff = reconstructed - original
        
        if self.loss_type == "l1":
            loss = torch.abs(diff)
        else:
            loss = diff ** 2
        
        # Apply mask: only compute loss on masked regions
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Feature-level temporal consistency loss for SSL.
    
    Encourages encoder features from adjacent cardiac frames to be
    similar (since cardiac anatomy changes smoothly between frames).
    
    Supports:
    - MSE: Direct feature distance
    - Cosine: Cosine similarity (projected features)
    
    Args:
        loss_type: "mse" or "cosine"
        temperature: Temperature for cosine similarity
    """
    
    def __init__(self, loss_type: str = "mse", temperature: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(
        self,
        features_t: torch.Tensor,
        features_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features_t: Encoder features from frame t
            features_t1: Encoder features from frame t+1
            
        Returns:
            Scalar temporal consistency loss
        """
        if self.loss_type == "mse":
            return F.mse_loss(features_t, features_t1)
        elif self.loss_type == "cosine":
            # Cosine similarity loss (1 - cos_sim)
            cos_sim = F.cosine_similarity(features_t, features_t1, dim=1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MotionConsistencyLoss(nn.Module):
    """
    Motion consistency loss for temporal agreement.
    
    Penalizes disagreement between:
    1. Warped features/predictions from frame t
    2. Direct features/predictions from frame t+1
    
    Also includes smoothness regularization on the displacement field.
    
    Args:
        feature_weight: Weight for feature-level consistency
        mask_weight: Weight for mask/prediction-level consistency
        smooth_weight: Weight for displacement field smoothness
    """
    
    def __init__(
        self,
        feature_weight: float = 1.0,
        mask_weight: float = 1.0,
        smooth_weight: float = 0.01,
    ):
        super().__init__()
        self.feature_weight = feature_weight
        self.mask_weight = mask_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self,
        warped_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        warped_pred: Optional[torch.Tensor] = None,
        target_pred: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            warped_features: Features from frame_t warped by flow
            target_features: Features directly computed from frame_{t+1}
            warped_pred: Predictions from frame_t warped by flow
            target_pred: Predictions directly computed from frame_{t+1}
            flow: (B, 2, H, W) displacement field
            
        Returns:
            Scalar motion consistency loss
        """
        loss = torch.tensor(0.0, device=self._get_device(
            warped_features, target_features, warped_pred, target_pred, flow
        ))
        
        # Feature-level consistency
        if warped_features is not None and target_features is not None:
            loss += self.feature_weight * F.mse_loss(warped_features, target_features)
        
        # Prediction-level consistency (soft Dice between warped and direct predictions)
        if warped_pred is not None and target_pred is not None:
            warped_prob = F.softmax(warped_pred, dim=1)
            target_prob = F.softmax(target_pred, dim=1)
            loss += self.mask_weight * F.mse_loss(warped_prob, target_prob)
        
        # Smoothness regularization on flow
        if flow is not None:
            loss += self.smooth_weight * self._smoothness_loss(flow)
        
        return loss
    
    def _smoothness_loss(self, flow: torch.Tensor) -> torch.Tensor:
        """Total variation regularization on displacement field."""
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        return dx.mean() + dy.mean()
    
    def _get_device(self, *tensors):
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return 'cpu'


class PseudoLabelLoss(nn.Module):
    """
    Confidence-weighted pseudo-label loss.
    
    Applies cross-entropy supervision from pseudo-labels, weighted
    by per-pixel confidence scores. Low-confidence regions contribute
    less to the loss.
    
    Args:
        weight: Overall weight for pseudo-label loss (λ_pseudo)
        threshold: Minimum confidence to include a pixel (hard threshold)
    """
    
    def __init__(self, weight: float = 0.25, threshold: float = 0.0):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
    
    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model predictions
            pseudo_labels: (B, H, W) pseudo-label class indices
            confidence: (B, H, W) per-pixel confidence scores [0, 1]
            
        Returns:
            Weighted pseudo-label loss
        """
        # Apply threshold mask
        mask = (confidence >= self.threshold).float()
        
        # Pixel-wise cross-entropy (unreduced)
        ce = F.cross_entropy(logits, pseudo_labels, reduction='none')  # (B, H, W)
        
        # Weight by confidence and mask
        weighted_ce = ce * confidence * mask
        
        # Normalize by number of accepted pixels
        n_accepted = mask.sum() + 1e-8
        loss = weighted_ce.sum() / n_accepted
        
        return self.weight * loss


class CombinedLoss(nn.Module):
    """
    Combined loss for the full pipeline.
    
    L_total = L_seg + λ_motion * L_motion + λ_pseudo * L_pseudo
    
    Args:
        num_classes: Number of segmentation classes
        motion_weight: λ_motion
        pseudo_weight: λ_pseudo
        confidence_threshold: Threshold for pseudo-label acceptance
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        motion_weight: float = 0.1,
        pseudo_weight: float = 0.25,
        confidence_threshold: float = 0.9,
    ):
        super().__init__()
        self.seg_loss = DiceCELoss(num_classes)
        self.motion_loss = MotionConsistencyLoss()
        self.pseudo_loss = PseudoLabelLoss(pseudo_weight, confidence_threshold)
        self.motion_weight = motion_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        motion_kwargs: Optional[dict] = None,
        pseudo_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Returns dict with individual and total losses for logging.
        """
        losses = {}
        
        # Supervised segmentation loss
        losses['seg'] = self.seg_loss(logits, targets)
        losses['total'] = losses['seg']
        
        # Motion consistency
        if motion_kwargs is not None:
            losses['motion'] = self.motion_loss(**motion_kwargs)
            losses['total'] += self.motion_weight * losses['motion']
        
        # Pseudo-label loss
        if pseudo_kwargs is not None:
            losses['pseudo'] = self.pseudo_loss(
                pseudo_kwargs['logits'],
                pseudo_kwargs['pseudo_labels'],
                pseudo_kwargs['confidence'],
            )
            losses['total'] += losses['pseudo']
        
        return losses
