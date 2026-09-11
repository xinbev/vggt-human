#!/usr/bin/env python
"""Compare cached VGGT/HSI camera trajectories with native EMDB GT camera poses."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.metrics import align_points, apply_similarity  # noqa: E402
from scripts.vis.full_viewer_cache_io import load_full_sequence_viewer_cache  # noqa: E402


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
COLORS = {"gt": "#8A8A8A", "raw": "#F05A78", "hsi": "#58A5F7"}


def main() -> None:
    args = parse_args()
    cache_dir = resolve_path(args.cache_dir)
    gt_pkl = resolve_path(args.gt_pkl)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, viewer_args, manifest_path = load_full_sequence_viewer_cache(cache_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gt = load_emdb_camera(gt_pkl)
    frame_indices, index_source = resolve_source_frame_indices(
        manifest=manifest,
        scene=scene,
        viewer_args=viewer_args,
        gt_frame_count=gt["camera_to_world"].shape[0],
        frame_index_offset=int(args.frame_index_offset),
    )
    raw_pred = trajectory_from_scene(scene, "raw")
    hsi_pred = trajectory_from_scene(scene, "hsi")
    if not (len(frame_indices) == raw_pred.shape[0] == hsi_pred.shape[0]):
        raise ValueError(
            f"Frame association mismatch: indices={len(frame_indices)} raw={raw_pred.shape} hsi={hsi_pred.shape}"
        )

    valid = np.ones(len(frame_indices), dtype=bool)
    if bool(args.good_frames_only):
        valid &= gt["good_frames_mask"][frame_indices]
    valid &= np.isfinite(raw_pred).all(axis=1)
    valid &= np.isfinite(hsi_pred).all(axis=1)
    gt_centers = gt["camera_to_world"][frame_indices, :3, 3]
    valid &= np.isfinite(gt_centers).all(axis=1)
    if int(valid.sum()) < 3:
        raise ValueError(f"Need at least three valid matched camera poses, got {int(valid.sum())}")

    frame_indices = frame_indices[valid]
    gt_centers = gt_centers[valid].astype(np.float64, copy=False)
    raw_pred = raw_pred[valid].astype(np.float64, copy=False)
    hsi_pred = hsi_pred[valid].astype(np.float64, copy=False)

    raw_se3, raw_se3_scale = align_trajectory(raw_pred, gt_centers, fixed_scale=True)
    hsi_se3, hsi_se3_scale = align_trajectory(hsi_pred, gt_centers, fixed_scale=True)
    raw_sim3, raw_sim3_scale = align_trajectory(raw_pred, gt_centers, fixed_scale=False)
    hsi_sim3, hsi_sim3_scale = align_trajectory(hsi_pred, gt_centers, fixed_scale=False)
    axes_name = str(args.plot_axes).lower()
    if axes_name == "auto":
        axes_name = choose_plot_axes(gt_centers)
    if len(axes_name) != 2 or any(axis not in AXIS_INDEX for axis in axes_name):
        raise ValueError(f"--plot-axes must be xy, xz, yz, or auto; got {args.plot_axes!r}")

    metrics = {
        "sequence": str(gt["name"]),
        "gt_pkl": str(gt_pkl),
        "cache_manifest": str(manifest_path),
        "cache_frames_dir": str(getattr(viewer_args, "frames_dir", "")),
        "frame_index_source": index_source,
        "frame_index_offset": int(args.frame_index_offset),
        "good_frames_only": bool(args.good_frames_only),
        "matched_frames": int(len(frame_indices)),
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "plot_axes": axes_name,
        "raw_vggt": trajectory_metrics(gt_centers, raw_pred, raw_se3, raw_sim3, raw_sim3_scale),
        "hsi_metric": trajectory_metrics(gt_centers, hsi_pred, hsi_se3, hsi_sim3, hsi_sim3_scale),
        "alignment_note": (
            "SE3 preserves predicted scale; Sim3 estimates rotation, translation, and scale. "
            "ATE is translation RMSE after Sim3."
        ),
        "camera_convention": {
            "gt": "EMDB camera.extrinsics is T_w2c; camera center comes from inverse(T_w2c)",
            "prediction": "cached raw_camera/hsi_camera position is camera center in predicted world coordinates",
        },
    }
    metrics_path = output_dir / "camera_trajectory_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "camera_trajectory_aligned.csv"
    write_frame_csv(csv_path, frame_indices, gt_centers, raw_se3, hsi_se3, raw_sim3, hsi_sim3)

    title = str(args.title or gt["name"])
    paper_png = output_dir / "camera_trajectory_sim3_paper.png"
    paper_pdf = output_dir / "camera_trajectory_sim3_paper.pdf"
    diagnostic_png = output_dir / "camera_trajectory_alignment_diagnostic.png"
    plot_paper_figure(paper_png, paper_pdf, title, axes_name, gt_centers, hsi_sim3, metrics)
    plot_diagnostic_figure(
        diagnostic_png,
        title,
        axes_name,
        gt_centers,
        raw_se3,
        hsi_se3,
        raw_sim3,
        hsi_sim3,
        metrics,
    )
    print(
        json.dumps(
            {
                "paper_png": str(paper_png),
                "paper_pdf": str(paper_pdf),
                "diagnostic_png": str(diagnostic_png),
                "metrics": str(metrics_path),
                "frames_csv": str(csv_path),
                "matched_frames": int(len(frame_indices)),
                "hsi_ate_rmse_m": metrics["hsi_metric"]["sim3_ate_rmse_m"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gt-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot-axes", choices=["xy", "xz", "yz", "auto"], default="xz")
    parser.add_argument("--frame-index-offset", type=int, default=0)
    parser.add_argument("--good-frames-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_emdb_camera(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing EMDB GT pickle: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)  # noqa: S301 - user-provided native EMDB annotation.
    if not isinstance(payload, dict) or not isinstance(payload.get("camera"), dict):
        raise TypeError(f"Invalid native EMDB payload: {path}")
    frame_count = int(payload["n_frames"])
    world_to_camera = np.asarray(payload["camera"]["extrinsics"], dtype=np.float64)
    if world_to_camera.shape != (frame_count, 4, 4):
        raise ValueError(f"EMDB camera.extrinsics must be [{frame_count},4,4], got {world_to_camera.shape}")
    good = np.asarray(payload.get("good_frames_mask", np.ones(frame_count)), dtype=bool).reshape(-1)
    if good.shape != (frame_count,):
        raise ValueError(f"EMDB good_frames_mask must have shape [{frame_count}], got {good.shape}")
    return {
        "name": str(payload.get("name", path.stem)),
        "camera_to_world": np.linalg.inv(world_to_camera),
        "good_frames_mask": good,
    }


def trajectory_from_scene(scene: dict[str, Any], kind: str) -> np.ndarray:
    key = "camera_trajectory_raw" if kind == "raw" else "camera_trajectory_hsi"
    value = scene.get(key)
    if value is not None:
        trajectory = np.asarray(value, dtype=np.float64).reshape(-1, 3)
        if trajectory.shape[0] == len(scene["frames"]):
            return trajectory
    camera_key = "raw_camera" if kind == "raw" else "hsi_camera"
    return np.stack(
        [np.asarray(frame[camera_key]["position"], dtype=np.float64).reshape(3) for frame in scene["frames"]],
        axis=0,
    )


def resolve_source_frame_indices(
    manifest: dict[str, Any],
    scene: dict[str, Any],
    viewer_args: Any,
    gt_frame_count: int,
    frame_index_offset: int,
) -> tuple[np.ndarray, str]:
    records = manifest["frames"]
    explicit = np.asarray([int(record.get("source_frame_index", -1)) for record in records], dtype=np.int64)
    if explicit.size and bool((explicit >= 0).all()):
        indices = explicit + int(frame_index_offset)
        validate_frame_indices(indices, gt_frame_count)
        return indices, "full_cache_manifest.source_frame_index"

    output_dir = Path(str(getattr(viewer_args, "output_dir", ""))).expanduser()
    summary_path = output_dir / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary.get("frame_selection", {}).get("selected_source_indices")
        if isinstance(selected, list) and len(selected) == len(records):
            indices = np.asarray(selected, dtype=np.int64) + int(frame_index_offset)
            validate_frame_indices(indices, gt_frame_count)
            return indices, "run_summary.frame_selection.selected_source_indices"

    frame_ids = [str(record.get("frame_id", "")) for record in records]
    parsed = [trailing_integer(frame_id) for frame_id in frame_ids]
    if all(value is not None for value in parsed):
        indices = np.asarray(parsed, dtype=np.int64) + int(frame_index_offset)
        validate_frame_indices(indices, gt_frame_count)
        return indices, "numeric suffix of cached frame_id"

    selected_args = getattr(viewer_args, "selected_source_indices", None)
    if isinstance(selected_args, list) and len(selected_args) == len(records):
        indices = np.asarray(selected_args, dtype=np.int64) + int(frame_index_offset)
        validate_frame_indices(indices, gt_frame_count)
        return indices, "cached viewer_args.selected_source_indices"
    raise ValueError(
        "Cannot associate cached frames with native EMDB indices. Regenerate the full cache with the current exporter "
        "or pass --frame-index-offset if frame IDs use a fixed offset."
    )


def trailing_integer(value: str) -> int | None:
    match = re.search(r"(\d+)$", value)
    return None if match is None else int(match.group(1))


def validate_frame_indices(indices: np.ndarray, gt_frame_count: int) -> None:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("No cached frame indices were resolved")
    if bool((indices < 0).any()) or bool((indices >= gt_frame_count).any()):
        raise IndexError(
            f"Resolved frame indices fall outside EMDB [0,{gt_frame_count - 1}]: "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )
    if indices.size > 1 and not bool((np.diff(indices) > 0).all()):
        raise ValueError("Resolved frame indices must be strictly increasing")


def align_trajectory(pred: np.ndarray, gt: np.ndarray, fixed_scale: bool) -> tuple[np.ndarray, float]:
    scale, rotation, translation = align_points(gt, pred, fixed_scale=fixed_scale)
    return apply_similarity(pred, scale, rotation, translation), float(scale)


def trajectory_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    pred_se3: np.ndarray,
    pred_sim3: np.ndarray,
    sim3_scale: float,
) -> dict[str, float]:
    return {
        "se3_rmse_m": rmse(pred_se3, gt),
        "sim3_ate_rmse_m": rmse(pred_sim3, gt),
        "sim3_alignment_scale": float(sim3_scale),
        "pred_path_length_m_before_alignment": path_length(pred),
        "gt_path_length_m": path_length(gt),
        "path_length_ratio_pred_over_gt": path_length(pred) / max(path_length(gt), 1e-12),
    }


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    error = np.linalg.norm(first - second, axis=1)
    return float(np.sqrt(np.mean(np.square(error))))


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def choose_plot_axes(gt: np.ndarray) -> str:
    ranges = np.ptp(gt, axis=0)
    indices = np.argsort(ranges)[-2:]
    return "".join("xyz"[index] for index in sorted(indices.tolist()))


def project(points: np.ndarray, axes_name: str) -> np.ndarray:
    return points[:, [AXIS_INDEX[axes_name[0]], AXIS_INDEX[axes_name[1]]]]


def plot_paper_figure(
    png_path: Path,
    pdf_path: Path,
    title: str,
    axes_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axis = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    draw_trajectory(axis, project(gt, axes_name), "Ground Truth", COLORS["gt"], linestyle="--")
    draw_trajectory(axis, project(pred, axes_name), "VGGT-Omega (Sim3 aligned)", COLORS["hsi"])
    axis.set_title(title, fontsize=12, fontweight="semibold")
    style_axis(axis, axes_name)
    axis.text(
        0.02,
        0.02,
        f"ATE {metrics['hsi_metric']['sim3_ate_rmse_m']:.3f} m · N={metrics['matched_frames']}",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=8)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_diagnostic_figure(
    output_path: Path,
    title: str,
    axes_name: str,
    gt: np.ndarray,
    raw_se3: np.ndarray,
    hsi_se3: np.ndarray,
    raw_sim3: np.ndarray,
    hsi_sim3: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, raw, hsi, subtitle in (
        (axes[0], raw_se3, hsi_se3, "SE(3) aligned · scale preserved"),
        (axes[1], raw_sim3, hsi_sim3, "Sim(3) aligned · shape / ATE"),
    ):
        draw_trajectory(axis, project(gt, axes_name), "Ground Truth", COLORS["gt"], linestyle="--")
        draw_trajectory(axis, project(raw, axes_name), "Raw VGGT", COLORS["raw"], linewidth=1.5)
        draw_trajectory(axis, project(hsi, axes_name), "HSI metric", COLORS["hsi"], linewidth=2.0)
        axis.set_title(subtitle, fontsize=10, fontweight="semibold")
        style_axis(axis, axes_name)
    axes[0].text(
        0.02,
        0.02,
        f"HSI SE3 RMSE {metrics['hsi_metric']['se3_rmse_m']:.3f} m",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#555555",
    )
    axes[1].text(
        0.02,
        0.02,
        f"HSI ATE {metrics['hsi_metric']['sim3_ate_rmse_m']:.3f} m",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#555555",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="semibold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_trajectory(
    axis: Any,
    points: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    axis.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    axis.scatter(points[0, 0], points[0, 1], color=color, marker="o", s=20, zorder=5)
    axis.scatter(points[-1, 0], points[-1, 1], color=color, marker="X", s=28, zorder=5)


def style_axis(axis: Any, axes_name: str) -> None:
    axis.set_xlabel(f"{axes_name[0].upper()} (m)", fontsize=9)
    axis.set_ylabel(f"{axes_name[1].upper()} (m)", fontsize=9)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, color="#E7E7E7", linewidth=0.7)
    axis.tick_params(labelsize=8, colors="#555555")
    for spine in axis.spines.values():
        spine.set_color("#D0D0D0")


def write_frame_csv(
    path: Path,
    frame_indices: np.ndarray,
    gt: np.ndarray,
    raw_se3: np.ndarray,
    hsi_se3: np.ndarray,
    raw_sim3: np.ndarray,
    hsi_sim3: np.ndarray,
) -> None:
    fields = ["frame_index"] + [f"{name}_{axis}" for name in ("gt", "raw_se3", "hsi_se3", "raw_sim3", "hsi_sim3") for axis in "xyz"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row_index, frame_index in enumerate(frame_indices.tolist()):
            row: dict[str, Any] = {"frame_index": int(frame_index)}
            for name, values in (
                ("gt", gt),
                ("raw_se3", raw_se3),
                ("hsi_se3", hsi_se3),
                ("raw_sim3", raw_sim3),
                ("hsi_sim3", hsi_sim3),
            ):
                for axis_index, axis_name in enumerate("xyz"):
                    row[f"{name}_{axis_name}"] = float(values[row_index, axis_index])
            writer.writerow(row)


if __name__ == "__main__":
    main()
