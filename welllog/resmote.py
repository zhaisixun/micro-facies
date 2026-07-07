"""Well-constrained RESMOTE oversampling for imbalanced well-log facies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.neighbors import LocalOutlierFactor
except ImportError:  # pragma: no cover
    LocalOutlierFactor = None


@dataclass
class SyntheticPoint:
    feat: np.ndarray
    depth: float
    label: int


def parse_target_classes(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def count_points_by_class(well_store: dict, real_only: bool = True) -> Counter:
    counts: Counter = Counter()
    for store in well_store.values():
        labels = store["labels"]
        valid_mask = store["valid_mask"]
        is_synthetic = store.get("is_synthetic")
        for i, label in enumerate(labels):
            if not valid_mask[i] or label is None:
                continue
            if real_only and is_synthetic is not None and is_synthetic[i]:
                continue
            counts[int(label)] += 1
    return counts


def _resolve_rng(args):
    seed = getattr(args, "resmote_seed", -1)
    if seed is None or int(seed) < 0:
        seed = getattr(args, "seed", 42)
    return np.random.default_rng(int(seed))


def _resolve_target_count(current: int, majority: int, args) -> int:
    absolute = int(getattr(args, "resmote_target_count", 0) or 0)
    if absolute > 0:
        return max(current, absolute)
    ratio = float(getattr(args, "resmote_target_ratio", 0.15))
    return max(current, int(round(majority * ratio)))


def _well_class_indices(store: dict, class_id: int, real_only: bool = True) -> List[int]:
    indices: List[int] = []
    labels = store["labels"]
    valid_mask = store["valid_mask"]
    is_synthetic = store.get("is_synthetic")
    for i, label in enumerate(labels):
        if not valid_mask[i] or label is None:
            continue
        if int(label) != int(class_id):
            continue
        if real_only and is_synthetic is not None and is_synthetic[i]:
            continue
        depth = store["depths"][i]
        if depth is None:
            continue
        indices.append(i)
    return indices


def _lof_keep_mask(feat: np.ndarray, contamination: float) -> np.ndarray:
    if LocalOutlierFactor is None:
        raise ImportError(
            "scikit-learn is required for --resmote_lof. "
            "Install sklearn or disable --resmote_lof."
        )
    n = len(feat)
    if n < 3:
        return np.ones(n, dtype=bool)
    cont = min(max(float(contamination), 0.01), 0.49)
    lof = LocalOutlierFactor(n_neighbors=min(5, n - 1), contamination=cont)
    labels = lof.fit_predict(feat)
    return labels == 1


def _neighbor_candidates(
    seed_idx: int,
    feat_pool: np.ndarray,
    depth_pool: np.ndarray,
    k: int,
    depth_radius: float,
    same_class_mask: Optional[np.ndarray] = None,
) -> List[int]:
    seed_depth = depth_pool[seed_idx]
    depth_diff = np.abs(depth_pool - seed_depth)
    in_radius = depth_diff <= float(depth_radius)
    if same_class_mask is not None:
        in_radius = in_radius & same_class_mask
    in_radius[seed_idx] = False
    candidate_idx = np.where(in_radius)[0]
    if len(candidate_idx) == 0:
        in_radius = depth_diff <= float(depth_radius)
        in_radius[seed_idx] = False
        candidate_idx = np.where(in_radius)[0]
    if len(candidate_idx) == 0:
        return []
    dists = np.linalg.norm(feat_pool[candidate_idx] - feat_pool[seed_idx], axis=1)
    order = np.argsort(dists)
    top = candidate_idx[order[: min(k, len(candidate_idx))]]
    return top.tolist()


def _smote_interpolate(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    depth_a: float,
    depth_b: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    lam = float(rng.random())
    feat_new = feat_a + lam * (feat_b - feat_a)
    depth_new = float(depth_a + lam * (depth_b - depth_a))
    return feat_new.astype(np.float32), depth_new


def _generate_batch(
    feat_pool: np.ndarray,
    depth_pool: np.ndarray,
    label: int,
    n_samples: int,
    k: int,
    depth_radius: float,
    rng: np.random.Generator,
) -> List[SyntheticPoint]:
    if len(feat_pool) < 2 or n_samples <= 0:
        return []

    generated: List[SyntheticPoint] = []
    n_pool = len(feat_pool)
    for _ in range(n_samples):
        seed_idx = int(rng.integers(0, n_pool))
        neighbors = _neighbor_candidates(
            seed_idx,
            feat_pool,
            depth_pool,
            k=k,
            depth_radius=depth_radius,
        )
        if not neighbors:
            continue
        neighbor_idx = int(neighbors[int(rng.integers(0, len(neighbors)))])
        feat_new, depth_new = _smote_interpolate(
            feat_pool[seed_idx],
            feat_pool[neighbor_idx],
            float(depth_pool[seed_idx]),
            float(depth_pool[neighbor_idx]),
            rng,
        )
        generated.append(SyntheticPoint(feat=feat_new, depth=depth_new, label=int(label)))
    return generated


def _resmote_for_well_class(
    store: dict,
    class_id: int,
    n_needed: int,
    k: int,
    depth_radius: float,
    iterations: int,
    use_lof: bool,
    lof_contamination: float,
    rng: np.random.Generator,
) -> List[SyntheticPoint]:
    indices = _well_class_indices(store, class_id, real_only=True)
    if len(indices) < 2:
        return []

    feat = store["feat_raw"][indices].astype(np.float32)
    depths = np.array([float(store["depths"][i]) for i in indices], dtype=np.float64)

    keep = np.ones(len(indices), dtype=bool)
    if use_lof:
        keep = _lof_keep_mask(feat, lof_contamination)
    if keep.sum() < 2:
        keep = np.ones(len(indices), dtype=bool)

    feat = feat[keep]
    depths = depths[keep]
    if len(feat) < 2:
        return []

    iterations = max(int(iterations), 1)
    per_iter = max(1, int(np.ceil(n_needed / iterations)))
    all_generated: List[SyntheticPoint] = []
    working_feat = feat.copy()
    working_depth = depths.copy()

    for iter_idx in range(iterations):
        remaining = n_needed - len(all_generated)
        if remaining <= 0:
            break
        n_this = per_iter if iter_idx < iterations - 1 else remaining
        batch = _generate_batch(
            working_feat,
            working_depth,
            class_id,
            n_this,
            k=k,
            depth_radius=depth_radius,
            rng=rng,
        )
        if not batch:
            break
        all_generated.extend(batch)
        if iter_idx < iterations - 1:
            working_feat = np.stack([p.feat for p in batch], axis=0)
            working_depth = np.array([p.depth for p in batch], dtype=np.float64)
        else:
            # Final merge round: combine real minority points with all synthetics so far.
            merged_feat = np.concatenate(
                [feat, np.stack([p.feat for p in all_generated], axis=0)],
                axis=0,
            )
            merged_depth = np.concatenate(
                [depths, np.array([p.depth for p in all_generated], dtype=np.float64)],
            )
            if len(all_generated) < n_needed:
                extra = _generate_batch(
                    merged_feat,
                    merged_depth,
                    class_id,
                    n_needed - len(all_generated),
                    k=k,
                    depth_radius=depth_radius,
                    rng=rng,
                )
                all_generated.extend(extra)

    return all_generated[:n_needed]


def _insert_synthetic_points(store: dict, points: Sequence[SyntheticPoint]) -> int:
    if not points:
        return 0

    feat = store["feat_raw"]
    labels = store["labels"]
    depths = store["depths"]
    valid_mask = store["valid_mask"]
    is_synthetic = store["is_synthetic"]

    new_feat = [feat]
    new_labels = list(labels)
    new_depths = list(depths)
    new_valid = list(valid_mask)
    new_syn = list(is_synthetic)

    min_depth = min(d for d in depths if d is not None) if depths else None
    max_depth = max(d for d in depths if d is not None) if depths else None

    inserted = 0
    for pt in points:
        if min_depth is not None and pt.depth < min_depth:
            continue
        if max_depth is not None and pt.depth > max_depth:
            continue
        new_feat.append(pt.feat.reshape(1, -1))
        new_labels.append(int(pt.label))
        new_depths.append(float(pt.depth))
        new_valid.append(True)
        new_syn.append(True)
        inserted += 1

    if inserted == 0:
        return 0

    merged_feat = np.concatenate(new_feat, axis=0)
    order = np.argsort(
        np.array([d if d is not None else np.inf for d in new_depths], dtype=np.float64),
        kind="stable",
    )

    store["feat_raw"] = merged_feat[order]
    store["labels"] = [new_labels[i] for i in order]
    store["depths"] = [new_depths[i] for i in order]
    store["valid_mask"] = np.array([new_valid[i] for i in order], dtype=bool)
    store["is_synthetic"] = np.array([new_syn[i] for i in order], dtype=bool)
    return inserted


def _normalize_well_store(well_store: dict) -> None:
    for store in well_store.values():
        feat_raw = store["feat_raw"]
        real_mask = ~store["is_synthetic"]
        if real_mask.any():
            ref = feat_raw[real_mask]
        else:
            ref = feat_raw
        mean = ref.mean(axis=0, keepdims=True)
        std = ref.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        store["norm_mean"] = mean
        store["norm_std"] = std
        store["feat"] = ((feat_raw - mean) / std).astype(np.float32)


def apply_resmote(well_store: dict, args) -> dict:
    """Augment minority classes in-place on raw features; normalization runs afterward."""
    target_classes = parse_target_classes(getattr(args, "resmote_classes", "1,2"))
    if not target_classes:
        return {"enabled": False}

    rng = _resolve_rng(args)
    k = int(getattr(args, "resmote_k", 5))
    depth_radius = float(getattr(args, "resmote_depth_radius", 5.0))
    iterations = int(getattr(args, "resmote_iterations", 3))
    use_lof = bool(getattr(args, "resmote_lof", False))
    lof_contamination = float(getattr(args, "resmote_lof_contamination", 0.05))

    before = count_points_by_class(well_store, real_only=True)
    if not before:
        return {"enabled": True, "before": dict(before), "after": {}, "inserted": 0}

    majority = max(before.values())
    class_targets = {
        cls: _resolve_target_count(before.get(cls, 0), majority, args)
        for cls in target_classes
    }
    need_by_class = {
        cls: max(0, class_targets[cls] - before.get(cls, 0))
        for cls in target_classes
    }

    inserted_total = 0
    per_class_inserted: Counter = Counter()
    warnings: List[str] = []

    for cls in target_classes:
        n_need = need_by_class[cls]
        if n_need <= 0:
            continue
        n_have = before.get(cls, 0)
        if n_have == 0:
            warnings.append(f"class {cls}: no real training points, skipped")
            continue

        well_counts = Counter()
        for well, store in well_store.items():
            well_counts[well] = len(_well_class_indices(store, cls, real_only=True))

        allocated = 0
        wells_with_class = [w for w, c in well_counts.items() if c > 0]
        for well_idx, well in enumerate(wells_with_class):
            n_well = well_counts[well]
            if well_idx == len(wells_with_class) - 1:
                n_well_need = n_need - allocated
            else:
                n_well_need = int(round(n_need * n_well / n_have))
            allocated += n_well_need

            if n_well_need <= 0:
                continue
            store = well_store[well]
            generated = _resmote_for_well_class(
                store,
                cls,
                n_well_need,
                k=k,
                depth_radius=depth_radius,
                iterations=iterations,
                use_lof=use_lof,
                lof_contamination=lof_contamination,
                rng=rng,
            )
            n_inserted = _insert_synthetic_points(store, generated)
            inserted_total += n_inserted
            per_class_inserted[cls] += n_inserted
            if n_inserted < n_well_need:
                warnings.append(
                    f"well={well} class={cls}: requested {n_well_need}, inserted {n_inserted}"
                )

    _normalize_well_store(well_store)
    after = count_points_by_class(well_store, real_only=False)

    return {
        "enabled": True,
        "target_classes": target_classes,
        "before": dict(before),
        "after": dict(after),
        "targets": class_targets,
        "inserted": inserted_total,
        "inserted_by_class": dict(per_class_inserted),
        "warnings": warnings,
    }
