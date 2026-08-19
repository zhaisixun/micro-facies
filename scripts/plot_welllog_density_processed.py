#!/usr/bin/env python3
"""
Plot probability density distributions of GR / CNL / DEN curves after
``_load_single_well`` and ``load_welllog_store`` processing
(median imputation + per-well z-score normalization).

Example:
    python scripts/plot_welllog_density_processed.py \\
        --xlsx_path ./facies-gr-diff0614-用GR-CNL-DEN.xlsx \\
        --output_dir ./outputs/welllog_density_processed \\
        --all_wells true
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from openpyxl import load_workbook
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from welllog.dataset import load_welllog_store
from welllog.well_split import auto_split_wells


def _setup_sci_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Noto Serif CJK SC",
                "Noto Serif CJK TC",
                "DejaVu Serif",
                "serif",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
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


def _parse_feature_cols(feature_cols: str) -> List[str]:
    return [c.strip() for c in feature_cols.split(",") if c.strip()]


def _list_xlsx_wells(xlsx_path: str, label_col: str) -> List[str]:
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    wells = []
    for sheet_name in wb.sheetnames:
        if sheet_name.lower() == "mapping":
            continue
        ws = wb[sheet_name]
        header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
        if label_col in header:
            wells.append(sheet_name)
    wb.close()
    return sorted(wells)


def _load_processed_curve_values(
    args: SimpleNamespace,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {well_name: {curve_name: 1d float array}} from load_welllog_store."""
    feature_cols = _parse_feature_cols(args.feature_cols)

    if args.all_wells:
        all_wells = _list_xlsx_wells(args.xlsx_path, args.label_col)
        if not all_wells:
            raise ValueError(f"No wells found in '{args.xlsx_path}'.")
        args.train_wells = ",".join(all_wells)
        args.val_wells = ""

    train_wells, label_map, train_store = load_welllog_store(args, is_train=True)
    well_store = dict(train_store)

    val_wells = [w.strip() for w in args.val_wells.split(",") if w.strip()]
    if val_wells:
        _, _, val_store = load_welllog_store(args, is_train=False, label_map=label_map)
        well_store.update(val_store)

    well_data: Dict[str, Dict[str, np.ndarray]] = {}
    for well in sorted(well_store.keys()):
        feat = well_store[well]["feat"]
        if feat.shape[1] != len(feature_cols):
            raise ValueError(
                f"Feature count mismatch for well '{well}': "
                f"expected {len(feature_cols)}, got {feat.shape[1]}."
            )
        well_data[well] = {
            col: feat[:, j].astype(np.float64) for j, col in enumerate(feature_cols)
        }

    loaded_wells = sorted(well_data.keys())
    print(f"Loaded {len(loaded_wells)} wells: {loaded_wells}")
    if train_wells:
        print(f"  train ({len(train_wells)}): {train_wells}")
    if val_wells:
        print(f"  val   ({len(val_wells)}): {val_wells}")
    return well_data


def _kde_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(values)
    return kde(x_grid)


def _build_x_grid(all_vals: np.ndarray, kde_points: int) -> np.ndarray:
    x_min, x_max = np.percentile(all_vals, [0.5, 99.5])
    if x_max <= x_min:
        x_min, x_max = all_vals.min(), all_vals.max()
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    return np.linspace(x_min - pad, x_max + pad, kde_points)


def plot_density_distributions(
    well_data: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    output_dir: str,
    xlsx_basename: str,
    kde_points: int = 512,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    wells = sorted(well_data.keys())
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(wells))]
    processed_tag = "processed (median fill + z-score)"

    for col in feature_cols:
        col_values = []
        for well in wells:
            vals = well_data[well].get(col)
            if vals is not None and vals.size > 0:
                col_values.append(vals)

        if not col_values:
            print(f"[skip] No valid data for column '{col}'.")
            continue

        all_vals = np.concatenate(col_values)
        x_grid = _build_x_grid(all_vals, kde_points)

        fig, ax = plt.subplots(figsize=(8, 5))
        plotted_wells = 0

        for well, color in zip(wells, colors):
            values = well_data[well].get(col)
            if values is None or values.size < 2:
                continue
            density = _kde_curve(values, x_grid)
            ax.plot(x_grid, density, color=color, linewidth=1.2, alpha=0.85, label=well)
            plotted_wells += 1

        ax.set_xlabel(f"{col} (z-score)")
        ax.set_ylabel("Probability density")
        ax.set_title(
            f"{col} probability density — {processed_tag} ({plotted_wells} wells)"
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            ncol=1,
        )

        fig.tight_layout()
        out_png = os.path.join(output_dir, f"{col}_density_processed_all_wells.png")
        out_pdf = os.path.join(output_dir, f"{col}_density_processed_all_wells.pdf")
        fig.savefig(out_png)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")

    n_cols = len(feature_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), squeeze=False)
    for ax, col in zip(axes[0], feature_cols):
        col_values = []
        for well in wells:
            vals = well_data[well].get(col)
            if vals is not None and vals.size > 0:
                col_values.append(vals)
        if not col_values:
            ax.set_visible(False)
            continue

        all_vals = np.concatenate(col_values)
        x_grid = _build_x_grid(all_vals, kde_points)

        for well, color in zip(wells, colors):
            values = well_data[well].get(col)
            if values is None or values.size < 2:
                continue
            density = _kde_curve(values, x_grid)
            ax.plot(x_grid, density, color=color, linewidth=1.0, alpha=0.8)

        ax.set_xlabel(f"{col} (z-score)")
        ax.set_ylabel("Probability density")
        ax.set_title(col)

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.2, label=well)
        for well, color in zip(wells, colors)
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        frameon=True,
    )
    fig.suptitle(
        f"Processed well-log density — {xlsx_basename}\n({processed_tag})",
        y=1.04,
    )
    fig.tight_layout()
    combo_png = os.path.join(output_dir, "all_curves_density_processed_all_wells.png")
    combo_pdf = os.path.join(output_dir, "all_curves_density_processed_all_wells.pdf")
    fig.savefig(combo_png, bbox_inches="tight")
    fig.savefig(combo_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combo_png}")
    print(f"Saved: {combo_pdf}")


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot probability density of GR/CNL/DEN after dataset preprocessing "
            "(_load_single_well + load_welllog_store)."
        )
    )
    p.add_argument(
        "--xlsx_path",
        default="../facies-gr-diff0614-用GR-CNL-DEN.xlsx",
        type=str,
        help="xlsx path (one sheet per well)",
    )
    p.add_argument(
        "--feature_cols",
        default="GR,CNL,DEN",
        type=str,
        help="comma-separated curve column names",
    )
    p.add_argument(
        "--label_col",
        default="facies",
        type=str,
        help="label column name in xlsx sheets",
    )
    p.add_argument(
        "--depth_col",
        default="DEPT",
        type=str,
        help="depth column name in xlsx sheets",
    )
    p.add_argument(
        "--train_wells",
        default="",
        type=str,
        help="comma-separated train wells (ignored when --all_wells or --auto_split_wells)",
    )
    p.add_argument(
        "--val_wells",
        default="",
        type=str,
        help="comma-separated val wells (ignored when --all_wells or --auto_split_wells)",
    )
    p.add_argument(
        "--all_wells",
        action="store_true",
        help="load every well sheet in the xlsx (overrides train/val split)",
    )
    p.add_argument(
        "--auto_split_wells",
        action="store_true",
        help="auto split wells into train/val before loading (uses --val_ratio)",
    )
    p.add_argument(
        "--val_ratio",
        default=0.2,
        type=float,
        help="val fraction when --auto_split_wells is set",
    )
    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for --auto_split_wells",
    )
    p.add_argument(
        "--output_dir",
        default="./outputs/welllog_density_processed",
        type=str,
        help="directory to save figures",
    )
    p.add_argument(
        "--kde_points",
        default=512,
        type=int,
        help="number of points on the KDE evaluation grid",
    )
    return p


def main() -> None:
    cli_args = get_args_parser().parse_args()
    _setup_sci_style()

    feature_cols = _parse_feature_cols(cli_args.feature_cols)
    if not feature_cols:
        raise ValueError("--feature_cols must contain at least one column name.")

    xlsx_path = os.path.abspath(cli_args.xlsx_path)
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    store_args = SimpleNamespace(
        xlsx_path=xlsx_path,
        feature_cols=cli_args.feature_cols,
        label_col=cli_args.label_col,
        depth_col=cli_args.depth_col,
        train_wells=cli_args.train_wells,
        val_wells=cli_args.val_wells,
    )

    if cli_args.auto_split_wells and not cli_args.all_wells:
        train_wells, val_wells = auto_split_wells(
            xlsx_path,
            cli_args.label_col,
            val_ratio=cli_args.val_ratio,
            seed=cli_args.seed,
        )
        store_args.train_wells = ",".join(train_wells)
        store_args.val_wells = ",".join(val_wells)
        print(f"[auto_split_wells] train ({len(train_wells)}): {train_wells}")
        print(f"[auto_split_wells] val   ({len(val_wells)}): {val_wells}")

    store_args.all_wells = cli_args.all_wells

    well_data = _load_processed_curve_values(store_args)
    if not well_data:
        raise ValueError(f"No wells loaded from '{xlsx_path}'.")

    plot_density_distributions(
        well_data=well_data,
        feature_cols=feature_cols,
        output_dir=cli_args.output_dir,
        xlsx_basename=os.path.basename(xlsx_path),
        kde_points=cli_args.kde_points,
    )


if __name__ == "__main__":
    main()
