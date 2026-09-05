#!/usr/bin/env python3
"""Compute Human3R-compatible ATE from TUM trajectories.

Human3R calls ``evo.main_ape.ape`` with translation-part APE, ``align=True``
and ``correct_scale=True``.  The implementation below reproduces that
Sim(3)-aligned translation RMSE with NumPy, so this benchmark does not require
the optional ``evo`` package.

Prediction files are normally ``<pred-root>/<sequence>/pred_traj.txt``.  The
released Human3R helper writes quaternion columns as ``qw qx qy qz`` even
though the filename is called TUM format; therefore that order is the default
and can be changed with ``--prediction-quaternion-order xyzw``.  Quaternion
values do not affect translation ATE, but validating them catches malformed
files early.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def read_trajectory(path: Path, quaternion_order: str = "xyzw") -> tuple[np.ndarray, np.ndarray]:
    """Read timestamp and position columns from a TUM trajectory file."""

    timestamps: list[float] = []
    positions: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").replace("\t", " ").split()
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: expected timestamp x y z [qx qy qz qw]")
            try:
                values = [float(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric trajectory field") from exc
            timestamps.append(values[0])
            positions.append(values[1:4])
            if len(values) >= 8:
                if quaternion_order not in {"xyzw", "wxyz"}:
                    raise ValueError(f"Unsupported quaternion order: {quaternion_order}")
                quaternion = np.asarray(values[4:8], dtype=np.float64)
                if float(np.linalg.norm(quaternion)) <= np.finfo(np.float64).eps:
                    raise ValueError(f"{path}:{line_number}: zero-norm quaternion")
    if not positions:
        raise ValueError(f"Trajectory is empty: {path}")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(timestamp_array).all() or not np.isfinite(position_array).all():
        raise ValueError(f"Trajectory contains NaN/Inf: {path}")
    return timestamp_array, position_array


def associate_by_timestamp(
    reference_timestamps: np.ndarray,
    estimate_timestamps: np.ndarray,
    max_difference: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one nearest timestamp association, matching evo's TUM sync."""

    candidates: list[tuple[float, int, int]] = []
    for ref_index, ref_time in enumerate(reference_timestamps):
        estimate_index = int(np.searchsorted(estimate_timestamps, ref_time))
        for candidate in (estimate_index - 1, estimate_index, estimate_index + 1):
            if 0 <= candidate < estimate_timestamps.size:
                difference = abs(float(ref_time - estimate_timestamps[candidate]))
                if difference <= max_difference:
                    candidates.append((difference, ref_index, candidate))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_ref: set[int] = set()
    used_estimate: set[int] = set()
    ref_indices: list[int] = []
    estimate_indices: list[int] = []
    for _, ref_index, estimate_index in candidates:
        if ref_index in used_ref or estimate_index in used_estimate:
            continue
        used_ref.add(ref_index)
        used_estimate.add(estimate_index)
        ref_indices.append(ref_index)
        estimate_indices.append(estimate_index)
    order = np.argsort(np.asarray(ref_indices, dtype=np.int64))
    return np.asarray(ref_indices, dtype=np.int64)[order], np.asarray(estimate_indices, dtype=np.int64)[order]


def sim3_align(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Align row-vector positions with a similarity transform (Umeyama)."""

    if estimate.shape != reference.shape or estimate.ndim != 2 or estimate.shape[1] != 3:
        raise ValueError(f"Expected matching [N,3] trajectories, got {estimate.shape} and {reference.shape}")
    if estimate.shape[0] < 3:
        raise ValueError("Sim(3) alignment needs at least three associated poses")
    estimate_mean = estimate.mean(axis=0, keepdims=True)
    reference_mean = reference.mean(axis=0, keepdims=True)
    estimate_zero = estimate - estimate_mean
    reference_zero = reference - reference_mean
    covariance = estimate_zero.T @ reference_zero / estimate.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    denominator = float(np.sum(estimate_zero * estimate_zero))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate estimated trajectory: zero centered variance")
    scale = float(np.sum(singular_values * np.diag(correction)) * estimate.shape[0] / denominator)
    aligned = scale * (estimate_zero @ rotation) + reference_mean
    return aligned, scale


def compute_ate(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    aligned, scale = sim3_align(estimate, reference)
    residual = np.linalg.norm(aligned - reference, axis=1)
    return float(np.sqrt(np.mean(residual * residual))), scale


def resolve_prediction_files(pred_root: Path, sequence: str | None) -> dict[str, Path]:
    if pred_root.is_file():
        if sequence is None:
            sequence = pred_root.parent.name
        return {sequence: pred_root}
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {pred_root}")
    files = sorted(pred_root.rglob("pred_traj.txt"))
    if sequence:
        files = [path for path in files if path.parent.name == sequence or path.parent.relative_to(pred_root).parts[:-1] == (sequence,)]
    if not files:
        raise FileNotFoundError(f"No pred_traj.txt found below {pred_root}")
    selected: dict[str, Path] = {}
    for path in files:
        key = path.parent.name
        if key in selected and selected[key] != path:
            raise ValueError(f"Multiple prediction files resolve to sequence {key!r}: {selected[key]}, {path}")
        selected[key] = path
    return selected


def resolve_groundtruth(dataset_root: Path, sequence: str, length: int | None, count: int) -> Path:
    sequence_root = dataset_root / sequence
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Dataset sequence does not exist: {sequence_root}")
    candidates: list[Path] = []
    if length is not None:
        candidates.append(sequence_root / f"groundtruth_{length}.txt")
    candidates.append(sequence_root / "groundtruth.txt")
    candidates.extend(sorted(sequence_root.glob("groundtruth_*.txt")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            _, positions = read_trajectory(candidate)
        except (OSError, ValueError):
            continue
        if length is not None and candidate.name == f"groundtruth_{length}.txt":
            return candidate
        if positions.shape[0] == count:
            return candidate
    available = ", ".join(path.name for path in sorted(sequence_root.glob("groundtruth*.txt")))
    raise FileNotFoundError(f"Could not resolve GT for {sequence!r} (length={length}, prediction poses={count}); available: {available}")


def evaluate_prediction_root(
    dataset_root: Path,
    pred_root: Path,
    length: int | None,
    quaternion_order: str,
    association: str,
    max_time_difference: float,
    sequence: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_files = resolve_prediction_files(pred_root, sequence)
    rows: list[dict[str, Any]] = []
    for sequence_name, prediction_file in sorted(prediction_files.items()):
        pred_timestamps, pred_positions = read_trajectory(prediction_file, quaternion_order=quaternion_order)
        groundtruth_file = resolve_groundtruth(dataset_root, sequence_name, length, pred_positions.shape[0])
        gt_timestamps, gt_positions = read_trajectory(groundtruth_file)
        use_index = association == "index" or (association == "auto" and pred_positions.shape[0] == gt_positions.shape[0])
        if association == "index" and pred_positions.shape[0] != gt_positions.shape[0]:
            raise ValueError(
                f"Index association requires equal pose counts for {sequence_name}: "
                f"prediction={pred_positions.shape[0]}, groundtruth={gt_positions.shape[0]}"
            )
        if use_index:
            gt_indices = np.arange(min(pred_positions.shape[0], gt_positions.shape[0]), dtype=np.int64)
            pred_indices = gt_indices.copy()
        else:
            gt_indices, pred_indices = associate_by_timestamp(gt_timestamps, pred_timestamps, max_time_difference)
        if gt_indices.size < 3:
            raise ValueError(f"Only {gt_indices.size} poses associated for {sequence_name}; need at least 3 for Sim(3) ATE")
        ate, scale = compute_ate(gt_positions[gt_indices], pred_positions[pred_indices])
        rows.append(
            {
                "sequence": sequence_name,
                "prediction_file": str(prediction_file),
                "groundtruth_file": str(groundtruth_file),
                "prediction_poses": int(pred_positions.shape[0]),
                "groundtruth_poses": int(gt_positions.shape[0]),
                "associated_poses": int(gt_indices.size),
                "ate_rmse_m": ate,
                "alignment_scale": scale,
                "association": "index" if use_index else "timestamp",
            }
        )
    if not rows:
        raise RuntimeError(f"No sequences evaluated from {pred_root}")
    summary: dict[str, Any] = {
        "benchmark": "human3r_tum_dynamics_ate_v1",
        "protocol": "translation APE RMSE after Sim(3) alignment (Human3R/evo align=True, correct_scale=True)",
        "dataset_root": str(dataset_root),
        "prediction_root": str(pred_root),
        "length": length,
        "sequence_count": len(rows),
        "ate_rmse_m_mean_over_sequences": float(np.mean([row["ate_rmse_m"] for row in rows])),
        "ate_rmse_m_median_over_sequences": float(np.median([row["ate_rmse_m"] for row in rows])),
        "associated_pose_count": int(sum(row["associated_poses"] for row in rows)),
        "prediction_quaternion_order": quaternion_order,
        "association_mode": association,
        "max_time_difference_s": max_time_difference,
    }
    return summary, rows


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({**summary, "sequences": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "sequence_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Prepared TUM-Dynamics root")
    parser.add_argument("--pred-root", required=True, type=Path, help="Human3R output root or a single pred_traj.txt")
    parser.add_argument("--length", type=int, default=None, help="Prefix length used for GT selection, e.g. 500")
    parser.add_argument("--sequence", default=None, help="Evaluate one sequence only")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz", help="Human3R's saved pred_traj uses wxyz; standard TUM files use xyzw")
    parser.add_argument("--association", choices=("auto", "index", "timestamp"), default="auto", help="Use index when lengths agree (Human3R behavior), otherwise timestamp association")
    parser.add_argument("--max-time-difference", type=float, default=0.02)
    args = parser.parse_args()
    if args.max_time_difference <= 0:
        parser.error("--max-time-difference must be positive")
    summary, rows = evaluate_prediction_root(
        args.dataset_root,
        args.pred_root,
        args.length,
        args.prediction_quaternion_order,
        args.association,
        args.max_time_difference,
        args.sequence,
    )
    summary.update({"output_dir": str(args.output_dir)})
    write_outputs(args.output_dir, summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
