"""Class prototype construction and prototype-based classification."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def build_class_prototypes(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute L2-normalized class prototypes from training embeddings.

    Returns:
        prototypes: (num_classes, D)
        counts:     (num_classes,) number of samples per class
    """
    if not hasattr(model, "encode"):
        raise AttributeError("Model must implement encode() for prototype building.")

    num_classes = len(dataset.label_map)
    feat_dim = None
    proto_sum = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    for batch in loader:
        samples, targets = batch[0], batch[1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                z = model.encode(samples)
        else:
            z = model.encode(samples)

        if feat_dim is None:
            feat_dim = z.shape[1]
            proto_sum = torch.zeros(num_classes, feat_dim, device=device, dtype=z.dtype)

        for cls in targets.unique():
            cls_int = int(cls.item())
            mask = targets == cls
            proto_sum[cls_int] += z[mask].sum(dim=0)
            counts[cls_int] += int(mask.sum().item())

    if proto_sum is None:
        raise ValueError("No samples found when building class prototypes.")

    prototypes = proto_sum.clone()
    for k in range(num_classes):
        if counts[k] > 0:
            prototypes[k] = prototypes[k] / counts[k].float()
        else:
            print(f"Warning: class index {k} has zero training samples; prototype left as zero vector.")
    prototypes = F.normalize(prototypes, dim=-1)
    return prototypes, counts


def save_prototypes(
    path: str,
    prototypes: torch.Tensor,
    label_map: Dict[str, int],
    counts: Optional[torch.Tensor] = None,
    embedding_dim: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "prototypes": prototypes.cpu(),
        "label_map": dict(label_map),
        "embedding_dim": embedding_dim or prototypes.shape[1],
    }
    if counts is not None:
        payload["counts"] = counts.cpu()
    torch.save(payload, path)


def load_prototypes(path: str, device: torch.device) -> Tuple[torch.Tensor, Dict[str, int]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prototype file not found: {path}")
    payload = torch.load(path, map_location=device)
    prototypes = payload["prototypes"].to(device)
    label_map = payload["label_map"]
    return prototypes, label_map


def predict_by_prototype(embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Nearest-prototype classification via cosine similarity.

    Args:
        embeddings: (B, D), L2-normalized
        prototypes: (C, D), L2-normalized
    Returns:
        pred: (B,) long tensor
    """
    logits = torch.matmul(embeddings, prototypes.T)
    return logits.argmax(dim=1)
