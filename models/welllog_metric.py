"""ConvNeXt wrapper with optional projection head for supervised contrastive learning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head used by SupCon."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WellLogMetricModel(nn.Module):
    """Wrap a ConvNeXt backbone with an optional SupCon projection head.

    Public API:
        forward(x)        -> logits (compatible with existing eval / linear infer)
        forward_train(x)  -> (logits, z_proj) for CE + SupCon training
        encode(x)         -> L2-normalized embedding for prototype infer
    """

    def __init__(self, backbone: nn.Module, embedding_dim: int = 128, use_supcon: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_supcon = bool(use_supcon)
        feat_dim = backbone.head.in_features
        self.projection_head = (
            ProjectionHead(feat_dim, embedding_dim) if self.use_supcon else None
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone(x, lengths=lengths)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding for prototype classification."""
        # backbone -> projection_head 投影
        h = self._features(x)  # backbone的输出特征
        if self.projection_head is not None:
            z = self.projection_head(h)  # prototype需要进行特征投影，得到投影后的特征z，用于计算相似度
        else:
            z = h
        return F.normalize(z, dim=-1)

    def forward_train(self, x: torch.Tensor):
        """Return (logits, projected_embedding). embedding is None if SupCon disabled."""
        h = self._features(x)  # backbone的输出特征
        logits = self.backbone.head(h)  # 分类linear头输出logits,用于计算CE损失
        if self.projection_head is not None:
            z = F.normalize(self.projection_head(h), dim=-1)  # 投影后的特征z进行L2归一化，用于计算SupCon损失
        else:
            z = None
        return logits, z

# 将普通convnext的权重映射到WellLogMetricModel的backbone.*键
# 相当于在普通convnext的权重前加上backbone.前缀
def adapt_checkpoint_state_dict(state_dict):
    """Map plain ConvNeXt keys to WellLogMetricModel backbone.* keys when needed."""
    if any(k.startswith("backbone.") for k in state_dict):
        return state_dict
    if any(k.startswith("projection_head.") for k in state_dict):
        return state_dict
    return {f"backbone.{k}": v for k, v in state_dict.items()}
