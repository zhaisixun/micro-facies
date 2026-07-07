#!/usr/bin/env python3
"""
Analyze separability of sedimentary microfacies in ConvNeXt Encoder feature space.

Extract the deepest encoder feature map, apply global average pooling, then run
PCA / t-SNE visualization and clustering metrics on the validation (test) set.

Example:
    python scripts/analyze_encoder_features.py \\
        --checkpoint ./outputs/seg_w128/checkpoint-best.pth \\
        --output_dir ./outputs/encoder_features \\
        --xlsx_path ./facies-gr-diff0607-用GR-CNL-DEN.xlsx \\
        --auto_split_wells true --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from timm.models import create_model

# Project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.convnext  # noqa: F401
import models.convnext1d  # noqa: F401
import utils
from datasets import build_dataset
from main import _num_feature_cols, _resolve_welllog_model, get_args_parser
from welllog.well_split import auto_split_wells


CLASS_LABELS: Dict[int, str] = {
    0: "0",
    1: "1-河口坝",
    2: "2-席状砂",
    3: "3",
}

CLASS_COLORS: Dict[int, str] = {
    0: "#4477AA",
    1: "#EE6677",
    2: "#228833",
    3: "#CCBB44",
}


def _setup_sci_style() -> None:
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK TC",
                "DejaVu Sans",
                "Arial",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
        }
    )


def _unwrap_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "module"):
        core = model.module
        return core.backbone if hasattr(core, "backbone") else core
    return model


@torch.no_grad()
def extract_encoder_features(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ignore_index: int = -100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract GAP-pooled deepest encoder features and one label per window."""
    model.eval()
    backbone = _unwrap_backbone(model)

    if not hasattr(backbone, "forward_features"):
        raise AttributeError("Model backbone has no forward_features().")

    feat_dim = None
    feats: List[np.ndarray] = []
    labels: List[int] = []

    for images, targets, _weights in data_loader:
        images = images.to(device, non_blocking=True)
        enc = backbone.forward_features(images)

        if enc.ndim == 2:
            pooled = enc
        elif enc.ndim == 3:
            pooled = enc.mean(dim=-1)
        else:
            raise ValueError(f"Unexpected encoder output shape: {tuple(enc.shape)}")

        if feat_dim is None:
            feat_dim = pooled.shape[1]
            print(f"Encoder feature dim: {feat_dim}")

        if targets.ndim == 2:
            center = targets.shape[1] // 2
            window_labels = targets[:, center]
        else:
            window_labels = targets

        pooled_np = pooled.detach().cpu().numpy()
        window_labels_np = window_labels.detach().cpu().numpy()

        for i in range(pooled_np.shape[0]):
            lbl = int(window_labels_np[i])
            if lbl == ignore_index:
                continue
            feats.append(pooled_np[i])
            labels.append(lbl)

    if not feats:
        raise ValueError("No valid features extracted. Check ignore_index and dataset labels.")

    F = np.stack(feats, axis=0).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    print(f"Collected features: F.shape={F.shape}, labels.shape={y.shape}")
    return F, y


def load_checkpoint_model(args, device: torch.device) -> torch.nn.Module:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required.")

    model_name, model_warnings = _resolve_welllog_model(args)
    for msg in model_warnings or []:
        print(f"[model] Warning: {msg}")
    print(f"[model] Using architecture: {model_name}")

    model_kwargs = dict(
        in_chans=_num_feature_cols(args),
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )
    if (
        getattr(args, "task_mode", "classification") == "segmentation"
        and getattr(args, "seg_decoder", "uper") == "uper"
    ):
        model_kwargs["decoder_channels"] = args.decoder_channels

    model = create_model(model_name, pretrained=False, **model_kwargs)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    current = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            filtered[k] = v
        else:
            print(f"[checkpoint] skip key (missing or shape mismatch): {k}")

    utils.load_state_dict(model, filtered)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    return model


def print_class_counts(y: np.ndarray, title: str = "Class counts") -> None:
    counts = Counter(int(v) for v in y)
    print(f"\n=== {title} ===")
    for cls in sorted(counts):
        name = CLASS_LABELS.get(cls, str(cls))
        print(f"  class {cls} ({name}): {counts[cls]}")
    print(f"  total: {len(y)}")


def subsample(
    X: np.ndarray, y: np.ndarray, max_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], y[idx]


def plot_embedding(
    coords: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    classes = sorted(set(int(v) for v in y))
    for cls in classes:
        mask = y == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLORS.get(cls, None),
            label=CLASS_LABELS.get(cls, str(cls)),
            s=14,
            alpha=0.75,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Facies", loc="best", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {save_path}")


def compute_class_centers(F: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    centers = {}
    for cls in sorted(set(int(v) for v in y)):
        centers[cls] = F[y == cls].mean(axis=0)
    return centers


def print_class_centers(centers: Dict[int, np.ndarray]) -> None:
    print("\n=== Class feature centers (first 8 dims) ===")
    for cls, center in centers.items():
        name = CLASS_LABELS.get(cls, str(cls))
        preview = ", ".join(f"{v:.4f}" for v in center[:8])
        print(f"  class {cls} ({name}): [{preview}, ...]  (dim={center.shape[0]})")


def print_inter_class_distance_matrix(centers: Dict[int, np.ndarray]) -> np.ndarray:
    classes = sorted(centers.keys())
    dist = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            dist[i, j] = np.linalg.norm(centers[ci] - centers[cj])

    header = "Inter-class Euclidean distance matrix"
    print(f"\n=== {header} ===")
    name_row = [CLASS_LABELS.get(c, str(c)) for c in classes]
    col_w = max(10, max(len(n) for n in name_row) + 2)
    print(" " * col_w + "".join(f"{n:>{col_w}}" for n in name_row))
    for i, ci in enumerate(classes):
        row_name = CLASS_LABELS.get(ci, str(ci))
        row = "".join(f"{dist[i, j]:>{col_w}.4f}" for j in range(len(classes)))
        print(f"{row_name:>{col_w}}{row}")
    return dist


def run_analysis(
    F: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    seed: int = 42,
    max_tsne_samples: int = 10000,
    tsne_perplexity: float = 30.0,
) -> None:
    _setup_sci_style()
    os.makedirs(output_dir, exist_ok=True)

    print_class_counts(y)

    scaler = StandardScaler()
    F_scaled = scaler.fit_transform(F)

    np.save(os.path.join(output_dir, "encoder_features.npy"), F_scaled)
    np.save(os.path.join(output_dir, "encoder_labels.npy"), y)
    print(f"Saved arrays -> {output_dir}/encoder_features.npy, encoder_labels.npy")

    centers = compute_class_centers(F_scaled, y)
    print_class_centers(centers)
    dist_mat = print_inter_class_distance_matrix(centers)
    np.save(os.path.join(output_dir, "inter_class_distance.npy"), dist_mat)

    if len(set(int(v) for v in y)) > 1 and len(y) > len(set(int(v) for v in y)):
        sil = silhouette_score(F_scaled, y, metric="euclidean")
        ch = calinski_harabasz_score(F_scaled, y)
        print("\n=== Clustering metrics (on all samples) ===")
        print(f"  Silhouette Score: {sil:.4f}  (range [-1, 1], higher is better)")
        print(f"  Calinski-Harabasz Score: {ch:.4f}  (higher is better)")
    else:
        print("\n=== Clustering metrics skipped (need >=2 classes and enough samples) ===")

    F_vis, y_vis = subsample(F_scaled, y, max_samples=max_tsne_samples, seed=seed)
    print(f"\nVisualization subset: {F_vis.shape[0]} / {F.shape[0]} samples")

    pca = PCA(n_components=2, random_state=seed)
    F_pca = pca.fit_transform(F_vis)
    evr = pca.explained_variance_ratio_
    print("\n=== PCA explained variance ratio ===")
    print(f"  PC1: {evr[0]:.4f} ({evr[0] * 100:.2f}%)")
    print(f"  PC2: {evr[1]:.4f} ({evr[1] * 100:.2f}%)")
    print(f"  PC1+PC2 cumulative: {evr.sum():.4f} ({evr.sum() * 100:.2f}%)")

    plot_embedding(
        F_pca,
        y_vis,
        title="PCA of ConvNeXt Encoder Features",
        xlabel=f"PC1 ({evr[0] * 100:.1f}%)",
        ylabel=f"PC2 ({evr[1] * 100:.1f}%)",
        save_path=os.path.join(output_dir, "pca_encoder_features.png"),
    )

    n_vis = F_vis.shape[0]
    perplexity = min(tsne_perplexity, max(5.0, (n_vis - 1) / 3.0))
    print(f"\nRunning t-SNE (n={n_vis}, perplexity={perplexity:.1f}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
    )
    F_tsne = tsne.fit_transform(F_vis)

    plot_embedding(
        F_tsne,
        y_vis,
        title="t-SNE of ConvNeXt Encoder Features",
        xlabel="t-SNE dimension 1",
        ylabel="t-SNE dimension 2",
        save_path=os.path.join(output_dir, "tsne_encoder_features.png"),
    )

    print("\nEncoder feature analysis complete.")


def prepare_args(analysis_args: argparse.Namespace):
    """Mirror main.py well-log dataset setup."""
    if analysis_args.data_set != "WELLLOG_XLSX":
        raise ValueError("This script currently supports data_set=WELLLOG_XLSX only.")

    if not analysis_args.xlsx_path:
        raise ValueError("--xlsx_path is required.")

    if analysis_args.auto_split_wells:
        train_wells, val_wells = auto_split_wells(
            analysis_args.xlsx_path,
            label_col=analysis_args.label_col,
            val_ratio=analysis_args.val_ratio,
            seed=analysis_args.seed,
        )
        analysis_args.train_wells = ",".join(train_wells)
        analysis_args.val_wells = ",".join(val_wells)
        print(f"[auto_split_wells] train ({len(train_wells)}): {train_wells}")
        print(f"[auto_split_wells] val   ({len(val_wells)}):   {val_wells}")
    elif not analysis_args.train_wells or not analysis_args.val_wells:
        raise ValueError("Set --train_wells/--val_wells or enable --auto_split_wells true.")

    if analysis_args.task_mode == "segmentation":
        if analysis_args.input_size != analysis_args.window_size:
            print(
                f"[segmentation] align input_size {analysis_args.input_size} "
                f"-> window_size {analysis_args.window_size}"
            )
            analysis_args.input_size = analysis_args.window_size

    return analysis_args


def get_analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Analyze ConvNeXt encoder feature separability",
        parents=[get_args_parser()],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. checkpoint-best.pth)",
    )
    parser.add_argument(
        "--max_tsne_samples",
        type=int,
        default=10000,
        help="Max samples for PCA/t-SNE visualization",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (auto-capped by sample size)",
    )
    parser.add_argument(
        "--feature_label_mode",
        type=str,
        default="center",
        choices=["center"],
        help="How to assign one label per window when task_mode=segmentation",
    )
    return parser


def main() -> None:
    parser = get_analysis_parser()
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = "./outputs/encoder_features"
    args = prepare_args(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train split builds label_map; val split reuses it.
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    del dataset_train

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    print(f"Validation windows: {len(dataset_val)}")

    model = load_checkpoint_model(args, device)
    F, y = extract_encoder_features(
        model,
        data_loader,
        device,
        ignore_index=args.ignore_index,
    )
    run_analysis(
        F=F,
        y=y,
        output_dir=args.output_dir,
        seed=args.seed,
        max_tsne_samples=args.max_tsne_samples,
        tsne_perplexity=args.tsne_perplexity,
    )


if __name__ == "__main__":
    main()
