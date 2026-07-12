#!/usr/bin/env python3
"""
Hyperparameter experiment runner for well-log segmentation.

Varies loss_mode, window_size, feature_cols, seg_decoder, well_input_mode (OFAT by default),
runs main.py for each configuration, and aggregates Acc@1 / mIoU / F1 into a CSV table.

Example (one-factor-at-a-time, recommended):
    python scripts/run_hyperparam_experiments.py \\
        --xlsx_path ./facies-gr-diff0607-用GR-CNL-DEN.xlsx \\
        --base_output_dir ./outputs/hparam_exp \\
        --epochs 100 --seed 42

Example (full grid):
    python scripts/run_hyperparam_experiments.py \\
        --mode grid \\
        --loss_modes ce,ce_focal_dice \\
        --window_sizes 64,128 \\
        --feature_cols_list "GR,CNL,DEN|GR,GR_diff1,GR_diff2" \\
        --seg_decoders lite,uper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    exp_id: str
    loss_mode: str
    window_size: int
    feature_cols: str
    seg_decoder: str
    well_input_mode: str
    output_dir: str
    acc: Optional[float] = None
    miou: Optional[float] = None
    f1: Optional[float] = None
    best_epoch_acc: Optional[float] = None
    best_epoch_miou: Optional[float] = None
    returncode: Optional[int] = None
    status: str = "pending"
    error: str = ""


def _split_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _split_feature_sets(text: str) -> List[str]:
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts if parts else ["GR,CNL,DEN"]


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def build_baseline() -> Dict[str, str]:
    return {
        "loss_mode": "ce_focal_dice",
        "window_size": "128",
        "feature_cols": "GR,CNL,DEN",
        "seg_decoder": "uper",
        "well_input_mode": "sliding_window",
    }


def generate_ofat_experiments(
    baseline: Dict[str, str],
    loss_modes: List[str],
    window_sizes: List[int],
    feature_cols_list: List[str],
    seg_decoders: List[str],
    well_input_modes: List[str],
    base_output_dir: str,
) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []
    exp_idx = 0

    def add(
        exp_id: str,
        loss_mode: str,
        window_size: int,
        feature_cols: str,
        seg_decoder: str,
        well_input_mode: str,
    ):
        nonlocal exp_idx
        out = os.path.join(base_output_dir, f"{exp_idx:02d}_{_safe_tag(exp_id)}")
        experiments.append(
            ExperimentConfig(
                exp_id=exp_id,
                loss_mode=loss_mode,
                window_size=window_size,
                feature_cols=feature_cols,
                seg_decoder=seg_decoder,
                well_input_mode=well_input_mode,
                output_dir=out,
            )
        )
        exp_idx += 1

    add(
        "baseline",
        baseline["loss_mode"],
        int(baseline["window_size"]),
        baseline["feature_cols"],
        baseline["seg_decoder"],
        baseline["well_input_mode"],
    )

    for lm in loss_modes:
        if lm == baseline["loss_mode"]:
            continue
        add(
            f"loss_{lm}",
            lm,
            int(baseline["window_size"]),
            baseline["feature_cols"],
            baseline["seg_decoder"],
            baseline["well_input_mode"],
        )

    for ws in window_sizes:
        if ws == int(baseline["window_size"]):
            continue
        add(
            f"window_{ws}",
            baseline["loss_mode"],
            ws,
            baseline["feature_cols"],
            baseline["seg_decoder"],
            baseline["well_input_mode"],
        )

    for fc in feature_cols_list:
        if fc == baseline["feature_cols"]:
            continue
        tag = _safe_tag(fc.replace(",", "-"))
        add(
            f"feat_{tag}",
            baseline["loss_mode"],
            int(baseline["window_size"]),
            fc,
            baseline["seg_decoder"],
            baseline["well_input_mode"],
        )

    for sd in seg_decoders:
        if sd == baseline["seg_decoder"]:
            continue
        add(
            f"decoder_{sd}",
            baseline["loss_mode"],
            int(baseline["window_size"]),
            baseline["feature_cols"],
            sd,
            baseline["well_input_mode"],
        )

    for wim in well_input_modes:
        if wim == baseline["well_input_mode"]:
            continue
        add(
            f"input_{wim}",
            baseline["loss_mode"],
            int(baseline["window_size"]),
            baseline["feature_cols"],
            baseline["seg_decoder"],
            wim,
        )

    return experiments


def generate_grid_experiments(
    loss_modes: List[str],
    window_sizes: List[int],
    feature_cols_list: List[str],
    seg_decoders: List[str],
    well_input_modes: List[str],
    base_output_dir: str,
) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []
    combos = product(loss_modes, window_sizes, feature_cols_list, seg_decoders, well_input_modes)
    for i, (lm, ws, fc, sd, wim) in enumerate(combos):
        exp_id = (
            f"grid_lm-{lm}_ws-{ws}_fc-{_safe_tag(fc.replace(',', '-'))}_sd-{sd}_wim-{wim}"
        )
        experiments.append(
            ExperimentConfig(
                exp_id=exp_id,
                loss_mode=lm,
                window_size=ws,
                feature_cols=fc,
                seg_decoder=sd,
                well_input_mode=wim,
                output_dir=os.path.join(base_output_dir, f"{i:02d}_{_safe_tag(exp_id)}"),
            )
        )
    return experiments


def build_main_command(exp: ExperimentConfig, args: argparse.Namespace) -> List[str]:
    ws = exp.window_size
    cmd = [
        args.python,
        str(ROOT / "main.py"),
        "--data_set", "WELLLOG_XLSX",
        "--xlsx_path", args.xlsx_path,
        "--feature_cols", exp.feature_cols,
        "--label_col", args.label_col,
        "--depth_col", args.depth_col,
        "--task_mode", "segmentation",
        "--use_1d_conv", "true",
        "--seg_decoder", exp.seg_decoder,
        "--loss_mode", exp.loss_mode,
        "--well_input_mode", exp.well_input_mode,
        "--model", args.model,
        "--window_size", str(ws),
        "--input_size", str(ws),
        "--window_stride", str(args.window_stride),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_epochs", str(args.warmup_epochs),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--use_amp", "true" if args.use_amp else "false",
        "--auto_split_wells", "true" if args.auto_split_wells else "false",
        "--val_ratio", str(args.val_ratio),
        "--class_weight", "true" if args.class_weight else "false",
        "--focal_gamma", str(args.focal_gamma),
        "--ce_weight", str(args.ce_weight),
        "--focal_weight", str(args.focal_weight),
        "--dice_weight", str(args.dice_weight),
        "--best_metric", args.best_metric,
        "--mixup", "0",
        "--cutmix", "0",
        "--smoothing", "0",
        "--dist_eval", "false",
        "--save_ckpt", "true",
        "--auto_resume", "false",
        "--output_dir", exp.output_dir,
        "--decoder_channels", str(args.decoder_channels),
    ]
    if args.class_weights:
        cmd.extend(["--class_weights", args.class_weights])
    cmd.extend([
        "--seg_oversample", "true" if args.seg_oversample else "false",
        "--seg_oversample_boost", str(args.seg_oversample_boost),
    ])
    return cmd


def parse_log_txt(log_path: str, best_metric: str = "miou") -> Tuple[Optional[float], Optional[float]]:
    if not os.path.isfile(log_path):
        return None, None

    best_acc = None
    best_miou = None
    best_score = None

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = entry.get("test_acc1")
            miou = entry.get("test_miou")
            if acc is None:
                continue
            score = miou if best_metric == "miou" else acc
            if score is None:
                score = acc
            if best_score is None or score > best_score:
                best_score = score
                best_acc = float(acc)
                best_miou = float(miou) if miou is not None else None
    return best_acc, best_miou


def parse_train_log(train_log_path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.isfile(train_log_path):
        return None, None

    acc = None
    miou = None
    pattern = re.compile(
        r"Final full-well accuracy.*?([0-9.]+)%(?:\s+mIoU:\s+([0-9.]+)%)?",
        re.IGNORECASE,
    )
    with open(train_log_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                acc = float(m.group(1))
                if m.group(2) is not None:
                    miou = float(m.group(2))
    return acc, miou


def parse_f1_score(output_dir: str, train_log_path: str) -> Optional[float]:
    """Parse macro-averaged F1 from segmentation_report.txt or train.log."""
    f1_pattern = re.compile(r"^\s*macro avg\s+\S+\s+\S+\s+([0-9.]+)", re.MULTILINE)

    report_path = os.path.join(output_dir, "segmentation_report.txt")
    for path in (report_path, train_log_path):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        m = f1_pattern.search(text)
        if m:
            return float(m.group(1)) * 100.0
    return None


def run_one_experiment(exp: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    Path(exp.output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_main_command(exp, args)
    train_log = os.path.join(exp.output_dir, "train.log")
    cmd_path = os.path.join(exp.output_dir, "command.txt")
    with open(cmd_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")

    print(f"\n{'=' * 72}")
    print(f"[{exp.exp_id}] output -> {exp.output_dir}")
    print(f"  loss_mode={exp.loss_mode}  window_size={exp.window_size}")
    print(f"  feature_cols={exp.feature_cols}  seg_decoder={exp.seg_decoder}")
    print(f"  well_input_mode={exp.well_input_mode}")
    print(f"{'=' * 72}")

    if args.dry_run:
        exp.status = "dry_run"
        return exp

    with open(train_log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))

    exp.returncode = proc.returncode
    if proc.returncode != 0:
        exp.status = "failed"
        exp.error = f"main.py exited with code {proc.returncode}"
        return exp

    fw_acc, fw_miou = parse_train_log(train_log)
    ep_acc, ep_miou = parse_log_txt(os.path.join(exp.output_dir, "log.txt"), best_metric=args.best_metric)

    exp.best_epoch_acc = ep_acc
    exp.best_epoch_miou = ep_miou
    exp.acc = fw_acc if fw_acc is not None else ep_acc
    exp.miou = fw_miou if fw_miou is not None else ep_miou
    exp.f1 = parse_f1_score(exp.output_dir, train_log)
    exp.status = "ok" if exp.acc is not None else "no_metrics"
    if exp.status == "no_metrics":
        exp.error = "Could not parse Acc/mIoU from train.log or log.txt"

    acc_s = f"{exp.acc:.2f}%" if exp.acc is not None else "N/A"
    miou_s = f"{exp.miou:.2f}%" if exp.miou is not None else "N/A"
    f1_s = f"{exp.f1:.2f}%" if exp.f1 is not None else "N/A"
    print(f"[{exp.exp_id}] done  acc={acc_s}  miou={miou_s}  f1={f1_s}  status={exp.status}")
    return exp


def save_results(experiments: List[ExperimentConfig], out_csv: str, out_md: str) -> None:
    fieldnames = [
        "exp_id",
        "loss_mode",
        "window_size",
        "feature_cols",
        "seg_decoder",
        "well_input_mode",
        "acc",
        "miou",
        "f1",
        "best_epoch_acc",
        "best_epoch_miou",
        "status",
        "returncode",
        "output_dir",
        "error",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = asdict(exp)
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# Hyperparameter Experiment Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| exp_id | loss_mode | window_size | feature_cols | seg_decoder | well_input_mode | Acc@1 (%) | mIoU (%) | F1 (%) | status |",
        "|:------:|:---------:|:-----------:|:-------------|:-----------:|:---------------:|:---------:|:--------:|:------:|:------:|",
    ]
    for exp in experiments:
        acc = f"{exp.acc:.2f}" if exp.acc is not None else "N/A"
        miou = f"{exp.miou:.2f}" if exp.miou is not None else "N/A"
        f1 = f"{exp.f1:.2f}" if exp.f1 is not None else "N/A"
        lines.append(
            f"| {exp.exp_id} | {exp.loss_mode} | {exp.window_size} | {exp.feature_cols} | "
            f"{exp.seg_decoder} | {exp.well_input_mode} | {acc} | {miou} | {f1} | {exp.status} |"
        )
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "y", "1")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run hyperparameter experiments and summarize Acc/mIoU")

    p.add_argument("--mode", choices=["ofat", "grid"], default="ofat",
                   help="ofat: baseline + one-factor changes; grid: full Cartesian product")
    p.add_argument("--base_output_dir", default="./outputs/hparam_experiments")
    p.add_argument("--summary_csv", default="", help="default: <base_output_dir>/experiments_summary.csv")
    p.add_argument("--summary_md", default="", help="default: <base_output_dir>/experiments_summary.md")
    p.add_argument("--dry_run", action="store_true", help="print planned runs without training")
    p.add_argument("--python", default=sys.executable)

    p.add_argument("--loss_modes", default="ce,ce_focal,ce_dice,ce_focal_dice")
    p.add_argument("--window_sizes", default="64,128,256")
    p.add_argument("--feature_cols_list", default="GR,CNL,DEN|GR,GR_diff1,GR_diff2|GR,CNL",
                   help='pipe-separated feature sets, e.g. "GR,CNL,DEN|GR,GR_diff1,GR_diff2"')
    p.add_argument("--seg_decoders", default="lite,uper")
    p.add_argument(
        "--well_input_modes",
        default="sliding_window,whole_well",
        help="comma-separated well input modes: sliding_window (滑窗) or whole_well (整口井)",
    )

    p.add_argument("--xlsx_path", default="../facies-gr-diff0614-用GR-CNL-DEN.xlsx")
    p.add_argument("--label_col", default="facies")
    p.add_argument("--depth_col", default="DEPT")
    p.add_argument("--model", default="convnext_tiny")
    p.add_argument("--window_stride", default=0, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--warmup_epochs", default=20, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--auto_split_wells", type=_str2bool, default=True)
    p.add_argument("--no_auto_split_wells", dest="auto_split_wells", action="store_false")
    p.add_argument("--val_ratio", default=0.2, type=float)
    p.add_argument("--class_weight", type=_str2bool, default=True)
    p.add_argument("--no_class_weight", dest="class_weight", action="store_false")
    p.add_argument("--class_weights", default="")
    p.add_argument("--seg_oversample", type=_str2bool, default=False)
    p.add_argument("--seg_oversample_boost", default=5.0, type=float)
    p.add_argument("--decoder_channels", default=256, type=int)
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--ce_weight", default=1.0, type=float)
    p.add_argument("--focal_weight", default=2.0, type=float)
    p.add_argument("--dice_weight", default=0.5, type=float)
    p.add_argument("--best_metric", default="miou", choices=["acc1", "miou"])
    return p


def main() -> None:
    args = get_parser().parse_args()
    loss_modes = _split_list(args.loss_modes)
    window_sizes = [int(x) for x in _split_list(args.window_sizes)]
    feature_cols_list = _split_feature_sets(args.feature_cols_list)
    seg_decoders = _split_list(args.seg_decoders)
    well_input_modes = _split_list(args.well_input_modes)
    baseline = build_baseline()

    if args.mode == "ofat":
        experiments = generate_ofat_experiments(
            baseline,
            loss_modes,
            window_sizes,
            feature_cols_list,
            seg_decoders,
            well_input_modes,
            args.base_output_dir,
        )
    else:
        experiments = generate_grid_experiments(
            loss_modes,
            window_sizes,
            feature_cols_list,
            seg_decoders,
            well_input_modes,
            args.base_output_dir,
        )

    print(f"Planned experiments: {len(experiments)}  (mode={args.mode})")
    for exp in experiments:
        print(
            f"  - {exp.exp_id}: loss={exp.loss_mode}, ws={exp.window_size}, "
            f"feat={exp.feature_cols}, decoder={exp.seg_decoder}, "
            f"well_input_mode={exp.well_input_mode}"
        )

    results: List[ExperimentConfig] = []
    for exp in experiments:
        results.append(run_one_experiment(exp, args))

    summary_csv = args.summary_csv or os.path.join(args.base_output_dir, "experiments_summary.csv")
    summary_md = args.summary_md or os.path.join(args.base_output_dir, "experiments_summary.md")
    save_results(results, summary_csv, summary_md)

    print(f"\n{'=' * 72}")
    print("Experiment summary")
    print(f"{'=' * 72}")
    header = (
        f"{'exp_id':<18} {'loss_mode':<14} {'ws':>4} {'decoder':<6} {'input':<14} "
        f"{'Acc':>8} {'mIoU':>8} {'F1':>8} {'status':<10}"
    )
    print(header)
    print("-" * len(header))
    for exp in results:
        acc_s = f"{exp.acc:.2f}" if exp.acc is not None else "N/A"
        miou_s = f"{exp.miou:.2f}" if exp.miou is not None else "N/A"
        f1_s = f"{exp.f1:.2f}" if exp.f1 is not None else "N/A"
        print(
            f"{exp.exp_id:<18} {exp.loss_mode:<14} {exp.window_size:>4} {exp.seg_decoder:<6} "
            f"{exp.well_input_mode:<14} {acc_s:>7}% {miou_s:>7}% {f1_s:>7}% {exp.status:<10}"
        )
    print(f"\nSaved -> {summary_csv}")
    print(f"Saved -> {summary_md}")


if __name__ == "__main__":
    main()
