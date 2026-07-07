"""Loss functions for 1D well-log semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Combined per-point CE/Focal/Dice loss for logits shaped (B, C, L)."""

    def __init__(
        self,
        mode="ce_focal_dice",
        weight=None,
        ignore_index=-100,  #
        focal_gamma=2.0,
        ce_weight=1.0,
        focal_weight=0.5,
        dice_weight=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.mode = mode
        allowed_modes = {"ce", "ce_focal", "ce_dice", "ce_focal_dice"}
        if self.mode not in allowed_modes:
            raise ValueError(f"Unknown segmentation loss mode '{mode}'. Expected one of {sorted(allowed_modes)}.")
        self.ignore_index = int(ignore_index)  
        self.focal_gamma = float(focal_gamma)
        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def _ce_per_point(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def _focal_loss(self, logits, target, valid_mask, sample_weight=None):
        ce = self._ce_per_point(logits, target)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.focal_gamma) * ce
        weights = valid_mask.to(loss.dtype)
        if sample_weight is not None:
            weights = weights * sample_weight.view(-1, 1).to(loss.dtype)
        denom = weights.sum().clamp_min(1)
        return (loss * weights).sum() / denom

    def _dice_loss(self, logits, target, valid_mask, sample_weight=None):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        safe_target = target.clamp_min(0)
        target_1h = F.one_hot(safe_target, num_classes=num_classes).permute(0, 2, 1).to(probs.dtype)
        valid = valid_mask.unsqueeze(1).to(probs.dtype)
        if sample_weight is not None:
            valid = valid * sample_weight.view(-1, 1, 1).to(probs.dtype)
        probs = probs * valid
        target_1h = target_1h * valid

        dims = (0, 2)
        intersection = (probs * target_1h).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_1h.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        present = target_1h.sum(dim=dims) > 0
        if present.any():
            dice = dice[present]
        return 1.0 - dice.mean()

    def forward(self, logits, target, sample_weight=None):
        if logits.ndim != 3 or target.ndim != 2:
            raise ValueError(
                "SegmentationLoss expects logits (B, C, L) and target (B, L); "
                f"got {tuple(logits.shape)} and {tuple(target.shape)}."
            )
        if logits.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"Prediction length {logits.shape[-1]} does not match target length {target.shape[-1]}."
            )

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0

        total = logits.new_zeros(())
        mode_parts = set(self.mode.split("_"))

        if "ce" in mode_parts:
            ce = self._ce_per_point(logits, target)
            if sample_weight is not None:
                weights = valid_mask.to(ce.dtype) * sample_weight.view(-1, 1).to(ce.dtype)
            else:
                weights = valid_mask.to(ce.dtype)
            denom = weights.sum().clamp_min(1)
            total = total + self.ce_weight * (ce * weights).sum() / denom

        if "focal" in mode_parts:
            total = total + self.focal_weight * self._focal_loss(
                logits, target, valid_mask, sample_weight=sample_weight
            )

        if "dice" in mode_parts:
            total = total + self.dice_weight * self._dice_loss(
                logits, target, valid_mask, sample_weight=sample_weight
            )

        return total
