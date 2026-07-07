"""Supervised contrastive and related metric-learning losses."""

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Expects L2-normalized feature vectors. Samples with no positive pair in the
    batch contribute zero loss (standard practice).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D), L2-normalized embeddings
            labels:   (B,) integer class indices
        Returns:
            Scalar loss; 0 if B < 2 or no class has >= 2 samples in the batch.
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Exclude self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Cosine similarity logits (features already normalized)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        # Numerical stability: subtract row max before exp
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positive pairs per anchor
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Only anchors with at least one positive contribute
        valid = (mask.sum(1) > 0).float()
        if valid.sum() < 1:
            return features.new_zeros(())

        loss = -(mean_log_prob_pos * valid).sum() / valid.sum()
        return loss
