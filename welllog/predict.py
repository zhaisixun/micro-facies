import csv
from typing import List, Optional, Union

import torch
import torch.nn.functional as F


def min_segment_filter(preds: List[Union[int, None]], min_len: int) -> List[Union[int, None]]:
    """Merge contiguous runs shorter than ``min_len`` into the longer neighbour segment."""
    if min_len <= 1 or len(preds) == 0:
        return preds

    preds = list(preds)
    changed = True
    while changed:
        changed = False
        segs = []
        i = 0
        while i < len(preds):
            j = i
            while j < len(preds) and preds[j] == preds[i]:
                j += 1
            segs.append([i, j, preds[i]])
            i = j
        for k, (start, end, label) in enumerate(segs):
            seg_len = end - start
            if seg_len < min_len:
                left_len = segs[k - 1][1] - segs[k - 1][0] if k > 0 else 0
                right_len = segs[k + 1][1] - segs[k + 1][0] if k < len(segs) - 1 else 0
                if left_len == 0 and right_len == 0:
                    continue
                new_label = segs[k - 1][2] if left_len >= right_len else segs[k + 1][2]
                if new_label != label:
                    for idx in range(start, end):
                        preds[idx] = new_label
                    changed = True
                    break
    return preds


def _infer_all_points(dataset_val, model, device, use_amp, min_segment_length):
    """共用推理核心：对每口 val 井做全井逐点非对称窗口推理，返回 (n_correct, n_labeled)。"""
    wsize = dataset_val.window_size
    isize = dataset_val.input_size
    half = wsize // 2
    n_correct_total = 0
    n_labeled_total = 0

    for well in dataset_val.target_wells:
        store = dataset_val._well_store[well]
        feat = store["feat"]
        labels = store["labels"]
        valid_mask = store["valid_mask"]
        n = len(feat)

        preds: List[Optional[int]] = []
        for i in range(n):
            if not valid_mask[i]:
                preds.append(None)
                continue
            start = max(0, min(i - half, n - wsize))
            x_win = feat[start : start + wsize]
            x = torch.from_numpy(x_win.T).unsqueeze(-1).float()
            x = F.interpolate(
                x.unsqueeze(0),
                size=(isize, isize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            x_in = x.unsqueeze(0).to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x_in)
            else:
                logits = model(x_in)
            preds.append(logits.argmax(dim=1).item())

        if min_segment_length > 1:
            pad = -1
            flat = [p if p is not None else pad for p in preds]
            smoothed = min_segment_filter(flat, min_segment_length)
            preds = [None if s == pad else s for s in smoothed]

        for i in range(n):
            true_int = labels[i]
            pred_int = preds[i]
            if true_int is not None and pred_int is not None:
                n_labeled_total += 1
                if pred_int == true_int:
                    n_correct_total += 1

    return n_correct_total, n_labeled_total


@torch.no_grad()
def evaluate_full_well(dataset_val, model, device, use_amp=False, min_segment_length=1):
    """全井逐点准确率评估（与 save_predictions_csv 完全同口径）。

    返回值格式与 engine.evaluate() 兼容：{'acc1': float, 'acc5': float, 'loss': float}
    acc1 = acc5 = 全井正确点数 / 有标签点数 * 100
    """
    model.eval()
    n_correct, n_labeled = _infer_all_points(dataset_val, model, device, use_amp, min_segment_length)
    acc = (n_correct / n_labeled * 100.0) if n_labeled > 0 else 0.0
    print(f"* Full-well Acc@1 {acc:.3f}  ({n_correct}/{n_labeled} labeled points)")
    return {"acc1": acc, "acc5": acc, "loss": 0.0}


@torch.no_grad()
def save_predictions_csv(
    dataset_val,
    model: torch.nn.Module,
    device: torch.device,
    output_path: str,
    use_amp: bool = False,
    min_segment_length: int = 1,
) -> None:
    """全井深度点导出 CSV：非对称滑动窗（无边距填充），每条预测均来自模型。"""
    model.eval()

    inv_map = dataset_val.inv_label_map
    wsize = dataset_val.window_size
    isize = dataset_val.input_size
    half = wsize // 2
    all_rows = []

    for well in dataset_val.target_wells:
        store = dataset_val._well_store[well]
        feat = store["feat"]   # 特征
        labels = store["labels"]   # 标签
        depths = store["depths"]   # 深度
        valid_mask = store["valid_mask"]
        n = len(feat)    # 样本数

        preds: List[Optional[int]] = []
        for i in range(n):   # 遍历每个样本
            if not valid_mask[i]:
                preds.append(None)
                continue
            start = max(0, min(i - half, n - wsize))   # 窗口起始位置
            x_win = feat[start : start + wsize]   # 窗口数据
            x = torch.from_numpy(x_win.T).unsqueeze(-1).float()
            x = F.interpolate(
                x.unsqueeze(0),
                size=(isize, isize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            x_in = x.unsqueeze(0).to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x_in)
            else:
                logits = model(x_in)
            preds.append(logits.argmax(dim=1).item())

        if min_segment_length > 1:
            pad = -1
            flat = [p if p is not None else pad for p in preds]
            smoothed = min_segment_filter(flat, min_segment_length)
            preds = [None if s == pad else s for s in smoothed]

        for i in range(n):
            true_int = labels[i]
            pred_int = preds[i]
            true_label = inv_map.get(true_int, true_int) if true_int is not None else ""
            pred_label = inv_map.get(pred_int, pred_int) if pred_int is not None else ""
            correct = (
                int(pred_int == true_int)
                if (true_int is not None and pred_int is not None)
                else ""
            )
            all_rows.append({
                "well": well,
                "depth": depths[i],
                "true_facies": true_label,
                "pred_facies": pred_label,
                "correct": correct,
            })

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["well", "depth", "true_facies", "pred_facies", "correct"])
        w.writeheader()
        w.writerows(all_rows)

    labeled = [r for r in all_rows if r["correct"] != ""]
    n_correct = sum(r["correct"] for r in labeled)
    acc_pct = (n_correct / len(labeled) * 100.0) if labeled else 0.0
    print(
        f"Predictions saved -> {output_path}  "
        f"(total={len(all_rows)} rows, labeled={len(labeled)}, "
        f"acc={acc_pct:.2f}%, "
        f"min_segment_length={min_segment_length})"
    )
