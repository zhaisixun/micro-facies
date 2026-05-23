from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from openpyxl import load_workbook
from torch.utils.data import Dataset


class WellLogSlidingWindowDataset(Dataset):
    """Well-log facies dataset using data-level sliding windows."""

    def __init__(self, args, is_train, label_map=None):
        super().__init__()
        self.input_size = args.input_size
        self.window_size = args.window_size
        self.window_stride = args.window_stride if args.window_stride > 0 else args.window_size // 2
        self.feature_cols = [c.strip() for c in args.feature_cols.split(",")]
        self.label_col = args.label_col
        self.depth_col = args.depth_col
        self.label_map = dict(label_map) if label_map is not None else {}
        self.class_counts = Counter()

        train_wells = [w.strip() for w in args.train_wells.split(",") if w.strip()]
        val_wells = [w.strip() for w in args.val_wells.split(",") if w.strip()]
        self.target_wells = train_wells if is_train else val_wells
        if len(self.target_wells) == 0:
            raise ValueError("No wells provided. Please set --train_wells/--val_wells.")

        wb = load_workbook(args.xlsx_path, data_only=True, read_only=True)
        available_wells = set(wb.sheetnames)
        for well in self.target_wells:
            if well not in available_wells:
                raise ValueError(f"Well '{well}' not found in xlsx sheets.")

        self.samples = []  # 存储每个样本的数据
        self._well_store = {}   # 存储每个井的数据
        half = self.window_size // 2

        # Pre-scan: build a sorted, stable label_map from all training wells
        # so that the mapping is deterministic regardless of row / well order.
        if label_map is None:
            all_raw_labels = set()
            for well in self.target_wells:
                ws = wb[well]
                header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
                col_to_idx_pre = {k: i for i, k in enumerate(header)}
                if self.label_col not in col_to_idx_pre:
                    continue
                label_idx_pre = col_to_idx_pre[self.label_col]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    v = row[label_idx_pre]
                    if v is not None and str(v).strip() != "":
                        all_raw_labels.add(str(v).strip())
            for i, lab in enumerate(sorted(all_raw_labels)):
                self.label_map[lab] = i

        for well in self.target_wells:
            ws = wb[well]
            header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
            col_to_idx = {k: i for i, k in enumerate(header)}
            for col in self.feature_cols + [self.label_col]:
                if col not in col_to_idx:
                    raise ValueError(f"Column '{col}' not found in sheet '{well}'.")

            rows = list(ws.iter_rows(min_row=2, values_only=True))

            feat = np.zeros((len(rows), len(self.feature_cols)), dtype=np.float32)  # 特征矩阵
            depths = []
            labels = []
            valid_mask = np.ones((len(rows),), dtype=bool)
            depth_idx = col_to_idx.get(self.depth_col)

            for i, row in enumerate(rows):
                depths.append(row[depth_idx] if depth_idx is not None else None)
                for j, col in enumerate(self.feature_cols):
                    v = row[col_to_idx[col]]
                    if v is None:
                        feat[i, j] = np.nan
                    else:
                        feat[i, j] = float(v)
                raw_label = row[col_to_idx[self.label_col]]
                if raw_label is None or str(raw_label).strip() == "":
                    valid_mask[i] = False
                    labels.append(None)
                else:
                    lab = str(raw_label).strip()
                    if lab not in self.label_map:
                        valid_mask[i] = False
                        labels.append(None)
                        continue
                    labels.append(self.label_map[lab])

            for j in range(feat.shape[1]):
                col_data = feat[:, j]
                col_median = np.nanmedian(col_data)
                if np.isnan(col_median):
                    col_median = 0.0
                col_data[np.isnan(col_data)] = col_median
                feat[:, j] = col_data

            feat_mean = feat.mean(axis=0, keepdims=True)
            feat_std = feat.std(axis=0, keepdims=True)
            feat_std[feat_std < 1e-6] = 1.0
            feat = (feat - feat_mean) / feat_std

            self._well_store[well] = {
                "feat": feat,   # 
                "labels": labels,   # 标签
                "depths": depths,   # 深度
                "valid_mask": valid_mask,   
            }

            if len(rows) < self.window_size:
                continue
            for center in range(half, len(rows) - (self.window_size - half) + 1, self.window_stride):
                if not valid_mask[center]:
                    continue
                start = center - half
                x_win = feat[start : start + self.window_size]
                y = labels[center]
                self.samples.append((x_win, y))
                self.class_counts[y] += 1

        if len(self.samples) == 0:
            raise ValueError("No valid samples built. Check well split and window size.")

    @property
    def inv_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def iter_dense(self, well):
        if well not in self._well_store:
            raise KeyError(f"Well '{well}' not found in this dataset split.")
        store = self._well_store[well]
        feat = store["feat"]
        labels = store["labels"]
        depths = store["depths"]
        valid_mask = store["valid_mask"]
        n = len(feat)
        half = self.window_size // 2
        for center in range(half, n - (self.window_size - half) + 1):
            if not valid_mask[center]:
                continue
            x_win = feat[center - half : center - half + self.window_size]
            x = torch.from_numpy(x_win.T).unsqueeze(-1).float()
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            yield depths[center], x, labels[center]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_win, y = self.samples[idx]
        x = torch.from_numpy(x_win.T).unsqueeze(-1).float()
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return x, torch.tensor(y, dtype=torch.long)
