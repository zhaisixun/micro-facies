import csv
import os
import numpy as np
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from welllog.input_utils import prepare_welllog_batch


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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

# 模型编码器
def model_encode(model: torch.nn.Module, batch_t: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized embeddings from WellLogMetricModel or plain ConvNeXt."""
    core = _unwrap_model(model)   
    if hasattr(core, "encode"):
        return core.encode(batch_t)  # backbone 输出特征 -> projection_head 投影
    h = core.forward_features(batch_t)  # backbone 输出特征
    return F.normalize(h, dim=-1)  


def _infer_well_batched(
    feat: np.ndarray,
    valid_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    wsize: int,
    isize: int,
    infer_batch_size: int,
    infer_mode: str = "linear",
    prototypes: Optional[torch.Tensor] = None,
    use_1d_conv: bool = False,
) -> List[Optional[int]]:
    """方案2：批量推理单口井的所有有效深度点（非对称滑动窗）。

    infer_mode:
        linear    - argmax over classification head logits (default)
        prototype - nearest class prototype in embedding space

    返回长度为 n（总深度点数）的预测列表，无效点为 None。
    """
    if infer_mode not in ("linear", "prototype"):
        raise ValueError(f"Unknown infer_mode: {infer_mode}")
    if infer_mode == "prototype" and prototypes is None:
        raise ValueError("prototype infer_mode requires prototypes tensor.")

    half = wsize // 2
    n = len(feat)

    # 第一步：预收集所有有效点的窗口（CPU numpy，避免逐点 to(device)）
    valid_indices: List[int] = []
    windows: List[np.ndarray] = []
    for i in range(n):
        if not valid_mask[i]:
            continue
        start = max(0, min(i - half, n - wsize))
        # feat[start:start+wsize] shape: (wsize, C)
        # .T -> (C, wsize)，与原来 x_win.T.unsqueeze(-1) 保持一致
        windows.append(feat[start: start + wsize].T)
        valid_indices.append(i)

    if not valid_indices:
        return [None] * n

    # 第二步：按 infer_batch_size 分批 forward
    pred_classes: List[int] = []
    for b in range(0, len(windows), infer_batch_size):
        batch_np = windows[b: b + infer_batch_size]
        batch_t = prepare_welllog_batch(
            np.stack(batch_np, axis=0),
            input_size=isize,
            use_1d_conv=use_1d_conv,
        )
        batch_t = batch_t.to(device, non_blocking=True)
        if infer_mode == "linear":
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(batch_t)
            else:
                logits = model(batch_t)
            pred_classes.extend(logits.argmax(dim=1).tolist())
        else:  # prototype 推理模式， 先经过模型编码器，得到投影后的特征z
            if use_amp:
                with torch.cuda.amp.autocast():
                    z = model_encode(model, batch_t)
            else:
                z = model_encode(model, batch_t)  
            from welllog.prototype import predict_by_prototype
            pred_classes.extend(predict_by_prototype(z, prototypes).tolist())

    # 第三步：映射回完整深度轴（无效点保持 None）
    preds: List[Optional[int]] = [None] * n
    for idx, cls in zip(valid_indices, pred_classes):
        preds[idx] = cls

    return preds


def _center_weights(length: int) -> np.ndarray:
    pos = np.arange(length, dtype=np.float32)
    center = (length - 1) / 2.0
    denom = max(center, 1.0)
    weights = 1.0 - np.abs(pos - center) / denom
    return np.clip(weights, 0.1, 1.0).astype(np.float32)


def _infer_well_segmentation(
    feat: np.ndarray,
    valid_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    wsize: int,
    isize: int,
    infer_batch_size: int,
    num_classes: int,
    infer_stride: int = 0,
    infer_fusion: str = "weighted_center",
    use_1d_conv: bool = True,
) -> List[Optional[int]]:
    """Sliding-window dense inference with overlap fusion for 1D segmentation."""
    if infer_fusion not in ("mean", "weighted_center", "vote"):
        raise ValueError(f"Unknown infer_fusion: {infer_fusion}")

    n = len(feat)
    if n == 0:
        return []
    stride = infer_stride if infer_stride > 0 else max(wsize // 4, 1)
    if n >= wsize:
        starts = list(range(0, n - wsize + 1, stride))
        if starts[-1] != n - wsize:
            starts.append(n - wsize)
        feat_for_windows = feat
    else:
        pad_len = wsize - n
        feat_for_windows = np.pad(feat, ((0, pad_len), (0, 0)), mode="edge")
        starts = [0]

    windows = [feat_for_windows[start : start + wsize].T for start in starts]
    prob_sum = np.zeros((n, num_classes), dtype=np.float64)
    weight_sum = np.zeros((n,), dtype=np.float64)
    base_weights = _center_weights(wsize) if infer_fusion == "weighted_center" else np.ones(wsize, dtype=np.float32)

    for b in range(0, len(windows), infer_batch_size):
        batch_np = windows[b : b + infer_batch_size]
        batch_starts = starts[b : b + infer_batch_size]
        batch_t = prepare_welllog_batch(
            np.stack(batch_np, axis=0),
            input_size=isize,
            use_1d_conv=use_1d_conv,
            task_mode="segmentation",
        ).to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch_t)
        else:
            logits = model(batch_t)

        if logits.ndim != 3:
            raise ValueError(f"Segmentation model must return (B, C, L), got {tuple(logits.shape)}")
        if logits.shape[-1] != wsize:
            logits = F.interpolate(logits, size=wsize, mode="linear", align_corners=False)
        probs = F.softmax(logits, dim=1).permute(0, 2, 1).detach().cpu().numpy()

        for win_probs, start in zip(probs, batch_starts):
            usable = min(wsize, n - start)
            if usable <= 0:
                continue
            if infer_fusion == "vote":
                cls = win_probs[:usable].argmax(axis=1)
                fused = np.zeros((usable, num_classes), dtype=np.float64)
                fused[np.arange(usable), cls] = 1.0
            else:
                fused = win_probs[:usable].astype(np.float64)
            weights = base_weights[:usable].astype(np.float64)
            prob_sum[start : start + usable] += fused * weights[:, None]
            weight_sum[start : start + usable] += weights

    preds: List[Optional[int]] = [None] * n
    covered = weight_sum > 0
    final_probs = np.zeros_like(prob_sum)
    final_probs[covered] = prob_sum[covered] / weight_sum[covered, None]
    for idx in range(n):
        if valid_mask[idx] and covered[idx]:
            preds[idx] = int(final_probs[idx].argmax())
    return preds


@torch.no_grad()
def _infer_well_whole(  # 全井段推理
    feat: np.ndarray,
    valid_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    use_1d_conv: bool = True,
) -> List[Optional[int]]:
    """Single forward pass on the full normalized well sequence."""
    if not use_1d_conv:
        raise ValueError("whole_well inference requires use_1d_conv=true.")

    n = len(feat)  #
    if n == 0:
        return []

    batch_t = prepare_welllog_batch(
        feat.T[np.newaxis, ...],
        input_size=n,
        use_1d_conv=True,
        task_mode="segmentation",
    ).to(device, non_blocking=True)

    if use_amp:
        with torch.cuda.amp.autocast():
            logits = model(batch_t)
    else:
        logits = model(batch_t)

    if logits.ndim != 3:
        raise ValueError(f"Segmentation model must return (B, C, L), got {tuple(logits.shape)}")
    if logits.shape[-1] != n:
        logits = F.interpolate(logits, size=n, mode="linear", align_corners=False)

    pred_classes = logits.argmax(dim=1).squeeze(0).detach().cpu().tolist()
    preds: List[Optional[int]] = [None] * n
    for i in range(n):
        if valid_mask[i]:
            preds[i] = int(pred_classes[i])
    return preds


def _resolve_well_inference(
    feat: np.ndarray,
    valid_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    *,
    well_input_mode: str,
    task_mode: str,
    wsize: int,
    isize: int,
    infer_batch_size: int,
    num_classes: int,
    infer_mode: str = "linear",
    prototypes: Optional[torch.Tensor] = None,
    infer_stride: int = 0,
    infer_fusion: str = "weighted_center",
    use_1d_conv: bool = False,
) -> List[Optional[int]]:
    if well_input_mode == "whole_well" and task_mode == "segmentation":
        return _infer_well_whole(
            feat, valid_mask, model, device, use_amp, use_1d_conv=use_1d_conv
        )
    if task_mode == "segmentation":
        return _infer_well_segmentation(
            feat, valid_mask, model, device, use_amp,
            wsize, isize, infer_batch_size,
            num_classes=num_classes,
            infer_stride=infer_stride,
            infer_fusion=infer_fusion,
            use_1d_conv=use_1d_conv,
        )
    return _infer_well_batched(
        feat, valid_mask, model, device, use_amp,
        wsize, isize, infer_batch_size,
        infer_mode=infer_mode,
        prototypes=prototypes,
        use_1d_conv=use_1d_conv,
    )


def print_and_save_classification_report(
    y_true: List[int],
    y_pred: List[int],
    inv_label_map: Dict[int, str],
    report_path: Optional[str] = None,
    digits: int = 4,
) -> str:
    """Print and optionally save a sklearn classification report for test predictions."""
    if not y_true:
        print("Classification report skipped: no labeled test points.")
        return ""

    labels = sorted(inv_label_map.keys())
    target_names = [str(inv_label_map[i]) for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=digits,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    header = "Classification Report (test set)"
    print(f"\n{header}\n")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print("labels:", target_names)
    print(cm)

    text = (
        f"{header}\n\n"
        f"{report}\n"
        "Confusion Matrix (rows=true, cols=pred)\n"
        f"labels: {target_names}\n"
        f"{cm}\n"
    )
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Classification report saved -> {report_path}")
    return text


def mean_iou_score(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    """Mean IoU over classes present in either prediction or target."""
    if not y_true:
        return 0.0
    ious = []
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    for cls in range(num_classes):
        true_mask = true_arr == cls
        pred_mask = pred_arr == cls
        union = np.logical_or(true_mask, pred_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(true_mask, pred_mask).sum()
        ious.append(float(intersection) / float(union))
    return float(np.mean(ious) * 100.0) if ious else 0.0


@torch.no_grad()
def evaluate_and_save_csv(
    dataset_val,
    model: torch.nn.Module,
    device: torch.device,
    output_path: str,
    use_amp: bool = False,
    min_segment_length: int = 1,
    infer_batch_size: int = 256,
    infer_mode: str = "linear",
    prototypes: Optional[torch.Tensor] = None,
    infer_stride: int = 0,
    infer_fusion: str = "weighted_center",
):
    """方案1+2：单次批量推理，同时计算全井准确率并写 CSV。

    返回值格式与 engine.evaluate() 兼容：{'acc1': float, 'acc5': float, 'loss': float}
    """
    model.eval()

    inv_map = dataset_val.inv_label_map
    wsize = dataset_val.window_size
    isize = dataset_val.input_size
    use_1d_conv = getattr(dataset_val, "use_1d_conv", False)
    task_mode = getattr(dataset_val, "task_mode", "classification")
    well_input_mode = getattr(dataset_val, "well_input_mode", "sliding_window")

    all_rows = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    n_correct_total = 0
    n_labeled_total = 0

    print(f"* Full-well task_mode={task_mode} infer_mode={infer_mode} well_input_mode={well_input_mode}")

    for well in dataset_val.target_wells:
        store = dataset_val._well_store[well]
        feat = store["feat"]
        labels = store["labels"]
        depths = store["depths"]
        valid_mask = store["valid_mask"]
        n = len(feat)

        preds = _resolve_well_inference(
            feat, valid_mask, model, device, use_amp,
            well_input_mode=well_input_mode,
            task_mode=task_mode,
            wsize=wsize,
            isize=isize,
            infer_batch_size=infer_batch_size,
            num_classes=len(inv_map),
            infer_mode=infer_mode,
            prototypes=prototypes,
            infer_stride=infer_stride,
            infer_fusion=infer_fusion,
            use_1d_conv=use_1d_conv,
        )

        # 后处理：短段合并
        if min_segment_length > 1:
            pad = -1
            flat = [p if p is not None else pad for p in preds]
            smoothed = min_segment_filter(flat, min_segment_length)
            preds = [None if s == pad else s for s in smoothed]

        # 统计准确率 + 组装 CSV 行（单次循环）
        for i in range(n):
            true_int = labels[i]
            pred_int = preds[i]

            if true_int is not None and pred_int is not None:
                n_labeled_total += 1
                y_true_all.append(true_int)
                y_pred_all.append(pred_int)
                if pred_int == true_int:
                    n_correct_total += 1

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

    acc = (n_correct_total / n_labeled_total * 100.0) if n_labeled_total > 0 else 0.0
    miou = mean_iou_score(y_true_all, y_pred_all, len(inv_map))
    print(
        f"* Full-well Acc@1 {acc:.3f}  mIoU {miou:.3f}  "
        f"({n_correct_total}/{n_labeled_total} labeled points)"
    )

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["well", "depth", "true_facies", "pred_facies", "correct"])
        w.writeheader()
        w.writerows(all_rows)

    print(
        f"Predictions saved -> {output_path}  "
        f"(total={len(all_rows)} rows, labeled={n_labeled_total}, "
        f"acc={acc:.2f}%, miou={miou:.2f}%, "
        f"min_segment_length={min_segment_length})"
    )

    report_name = "segmentation_report.txt" if task_mode == "segmentation" else "classification_report.txt"
    report_path = os.path.join(os.path.dirname(output_path) or ".", report_name)
    if task_mode != "segmentation" and infer_mode == "prototype":
        report_path = os.path.join(
            os.path.dirname(output_path) or ".", "classification_report_prototype.txt"
        )
    print_and_save_classification_report(y_true_all, y_pred_all, inv_map, report_path=report_path)
    return {"acc1": acc, "acc5": acc, "miou": miou, "loss": 0.0}


@torch.no_grad()
def evaluate_full_well(
    dataset_val,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
    min_segment_length: int = 1,
    infer_batch_size: int = 256,
    report_path: Optional[str] = None,
    infer_mode: str = "linear",
    prototypes: Optional[torch.Tensor] = None,
    infer_stride: int = 0,
    infer_fusion: str = "weighted_center",
):
    """全井逐点准确率评估（批量推理版，与 evaluate_and_save_csv 口径一致）。

    返回值格式与 engine.evaluate() 兼容：{'acc1': float, 'acc5': float, 'loss': float}
    用于 per-epoch 快速评估（--eval_full_well_each_epoch True）。
    """
    model.eval()

    inv_map = dataset_val.inv_label_map
    wsize = dataset_val.window_size
    isize = dataset_val.input_size
    use_1d_conv = getattr(dataset_val, "use_1d_conv", False)
    task_mode = getattr(dataset_val, "task_mode", "classification")
    well_input_mode = getattr(dataset_val, "well_input_mode", "sliding_window")
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    n_correct_total = 0
    n_labeled_total = 0

    print(f"* Full-well task_mode={task_mode} infer_mode={infer_mode} well_input_mode={well_input_mode}")

    for well in dataset_val.target_wells:
        store = dataset_val._well_store[well]
        feat = store["feat"]
        labels = store["labels"]
        valid_mask = store["valid_mask"]
        n = len(feat)

        preds = _resolve_well_inference(
            feat, valid_mask, model, device, use_amp,
            well_input_mode=well_input_mode,
            task_mode=task_mode,
            wsize=wsize,
            isize=isize,
            infer_batch_size=infer_batch_size,
            num_classes=len(inv_map),
            infer_mode=infer_mode,
            prototypes=prototypes,
            infer_stride=infer_stride,
            infer_fusion=infer_fusion,
            use_1d_conv=use_1d_conv,
        )

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
                y_true_all.append(true_int)
                y_pred_all.append(pred_int)
                if pred_int == true_int:
                    n_correct_total += 1

    acc = (n_correct_total / n_labeled_total * 100.0) if n_labeled_total > 0 else 0.0
    miou = mean_iou_score(y_true_all, y_pred_all, len(inv_map))
    print(
        f"* Full-well Acc@1 {acc:.3f}  mIoU {miou:.3f}  "
        f"({n_correct_total}/{n_labeled_total} labeled points)"
    )
    print_and_save_classification_report(y_true_all, y_pred_all, inv_map, report_path=report_path)
    return {"acc1": acc, "acc5": acc, "miou": miou, "loss": 0.0}


@torch.no_grad()
def save_predictions_csv(
    dataset_val,
    model: torch.nn.Module,
    device: torch.device,
    output_path: str,
    use_amp: bool = False,
    min_segment_length: int = 1,
    infer_batch_size: int = 256,
    infer_mode: str = "linear",
    prototypes: Optional[torch.Tensor] = None,
    infer_stride: int = 0,
    infer_fusion: str = "weighted_center",
) -> None:
    """向后兼容接口：内部调用 evaluate_and_save_csv（单次批量推理）。"""
    evaluate_and_save_csv(
        dataset_val, model, device, output_path,
        use_amp=use_amp,
        min_segment_length=min_segment_length,
        infer_batch_size=infer_batch_size,
        infer_mode=infer_mode,
        prototypes=prototypes,
        infer_stride=infer_stride,
        infer_fusion=infer_fusion,
    )
