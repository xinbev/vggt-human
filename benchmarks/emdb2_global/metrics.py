"""Human3R/GVHMR-style global trajectory metrics implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalMetricResult:
    w_mpjpe_mm: np.ndarray
    wa_mpjpe_mm: np.ndarray
    rte_percent: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "W-MPJPE_mm": _finite_mean(self.w_mpjpe_mm),
            "WA-MPJPE_mm": _finite_mean(self.wa_mpjpe_mm),
            "RTE_percent": _finite_mean(self.rte_percent),
        }


def evaluate_global_metrics(
    target_joints_world: np.ndarray,
    pred_joints_world: np.ndarray,
    chunk_length: int = 100,
    root_index: int = 0,
) -> GlobalMetricResult:
    """Evaluate Human3R global metrics on one matched single-person sequence.

    W-MPJPE uses one Sim(3) estimated from the first two frames of each chunk.
    WA-MPJPE estimates Sim(3) from every joint in the complete chunk. RTE uses
    one rigid SE(3) trajectory alignment over the complete sequence and divides
    absolute root error by total GT root displacement.
    """
    target = _validate_joints(target_joints_world, "target_joints_world")
    pred = _validate_joints(pred_joints_world, "pred_joints_world")
    if target.shape != pred.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target.shape} vs {pred.shape}")
    if target.shape[0] < 2:
        raise ValueError("Human3R W-MPJPE and RTE require at least two matched frames")
    if not 0 <= int(root_index) < target.shape[1]:
        raise ValueError(f"root_index={root_index} is outside J={target.shape[1]}")
    chunk_length = max(int(chunk_length), 2)

    w_errors: list[np.ndarray] = []
    wa_errors: list[np.ndarray] = []
    for start in range(0, target.shape[0], chunk_length):
        end = min(target.shape[0], start + chunk_length)
        target_chunk = target[start:end]
        pred_chunk = pred[start:end]
        w_aligned = first_two_frame_align(target_chunk, pred_chunk)
        wa_aligned = global_align(target_chunk, pred_chunk)
        w_errors.append(joint_position_error(target_chunk, w_aligned) * 1000.0)
        wa_errors.append(joint_position_error(target_chunk, wa_aligned) * 1000.0)

    rte = root_translation_error(
        target[:, int(root_index)],
        pred[:, int(root_index)],
        fixed_scale=True,
    ) * 100.0
    return GlobalMetricResult(
        w_mpjpe_mm=np.concatenate(w_errors, axis=0),
        wa_mpjpe_mm=np.concatenate(wa_errors, axis=0),
        rte_percent=rte,
    )


def first_two_frame_align(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    target = _validate_joints(target, "target")
    pred = _validate_joints(pred, "pred")
    frame_count = min(2, target.shape[0])
    scale, rotation, translation = align_points(
        target[:frame_count].reshape(-1, 3),
        pred[:frame_count].reshape(-1, 3),
        fixed_scale=False,
    )
    return apply_similarity(pred, scale, rotation, translation)


def global_align(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    target = _validate_joints(target, "target")
    pred = _validate_joints(pred, "pred")
    scale, rotation, translation = align_points(
        target.reshape(-1, 3),
        pred.reshape(-1, 3),
        fixed_scale=False,
    )
    return apply_similarity(pred, scale, rotation, translation)


def root_translation_error(
    target_root: np.ndarray,
    pred_root: np.ndarray,
    fixed_scale: bool = True,
) -> np.ndarray:
    target = _validate_points(target_root, "target_root")
    pred = _validate_points(pred_root, "pred_root")
    if target.shape != pred.shape:
        raise ValueError(f"Root trajectory shape mismatch: {target.shape} vs {pred.shape}")
    if target.shape[0] < 2:
        raise ValueError("RTE requires at least two matched frames")
    scale, rotation, translation = align_points(target, pred, fixed_scale=fixed_scale)
    pred_aligned = apply_similarity(pred, scale, rotation, translation)
    total_displacement = np.linalg.norm(np.diff(target, axis=0), axis=-1).sum()
    if not np.isfinite(total_displacement) or total_displacement <= 1e-8:
        raise ValueError("RTE is undefined because GT root trajectory displacement is zero")
    return np.linalg.norm(target - pred_aligned, axis=-1) / total_displacement


def align_points(
    target: np.ndarray,
    pred: np.ndarray,
    fixed_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment matching Human3R's ``align_pcl(Y, X)`` convention."""
    target = _validate_points(target, "target")
    pred = _validate_points(pred, "pred")
    if target.shape != pred.shape:
        raise ValueError(f"Point shape mismatch: {target.shape} vs {pred.shape}")
    count = float(target.shape[0])
    target_mean = target.sum(axis=0) / count
    pred_mean = pred.sum(axis=0) / count
    target_centered = target - target_mean
    pred_centered = pred - pred_mean
    correlation = target_centered.T @ pred_centered / count
    u, singular, vh = np.linalg.svd(correlation)
    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh.T) < 0.0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vh
    if fixed_scale:
        scale = 1.0
    else:
        variance = np.square(pred_centered).sum() / count
        if variance <= 1e-12:
            raise ValueError("Similarity alignment is undefined for zero-variance prediction points")
        scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_mean - scale * (rotation @ pred_mean)
    return scale, rotation, translation


def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return scale * np.einsum("ij,...j->...i", rotation, value) + translation


def joint_position_error(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(target) - np.asarray(pred), axis=-1).mean(axis=-1)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform [...,4,4], got {transform.shape}")
    if transform.ndim == 2:
        return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]
    if transform.shape[0] != points.shape[0]:
        raise ValueError(f"Per-frame transform mismatch: {transform.shape[0]} vs {points.shape[0]}")
    return np.einsum("fij,fnj->fni", transform[:, :3, :3], points) + transform[:, None, :3, 3]


def _validate_joints(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [F,J,3], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _validate_points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _finite_mean(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    valid = np.isfinite(array)
    return float(array[valid].mean()) if valid.any() else float("nan")
