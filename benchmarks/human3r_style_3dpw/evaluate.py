#!/usr/bin/env python3
"""Evaluate NLF and optional V2 with a Human3R-style raw 3DPW protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.human3r_style_3dpw.data import (  # noqa: E402
    ThreeDPWTestSequence,
    decode_gt_camera_space,
    frame_path,
    gt_camera_parameters,
    load_processed_frame,
    load_test_sequences,
    raw_openpose_2d,
)
from benchmarks.human3r_style_3dpw.metrics import human3r_camera_metrics, match_by_2d_joints  # noqa: E402
from vggt_omega.integrations.nlf_smpl_provider import NLFSMPLProvider  # noqa: E402
from vggt_omega.models import PoseStabilizerConfig, PoseTemporalStabilizer  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


class Totals:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, values: dict[str, torch.Tensor]) -> None:
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().sum().cpu())
            self.counts[key] = self.counts.get(key, 0) + int(value.numel())

    def summary(self) -> dict[str, Any]:
        return {**{key: total / max(self.counts[key], 1) for key, total in sorted(self.sums.items())}, "count": dict(self.counts)}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    root = Path(args.threedpw_root or require_path(cfg, "datasets.threedpw_root"))
    sequences = load_test_sequences(root, args.sequence_filter)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nlf = build_nlf(cfg).to(device).eval()
    smpl_layers = {
        "neutral": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="neutral").to(device).eval(),
        "male": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="male").to(device).eval(),
        "female": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="female").to(device).eval(),
    }
    temporal = load_temporal(args.temporal_checkpoint, device) if args.temporal_checkpoint else None
    print_manifest(args, cfg, root, sequences, device, temporal)
    totals = {"nlf_base": Totals(), "nlf_pose_temporal": Totals()}
    rows: list[dict[str, Any]] = []
    component_totals = {name: Totals() for name in COMPONENT_NAMES} if args.component_diagnostics else None
    component_rows: list[dict[str, Any]] = []
    coverage = {"sequences": 0, "gt_people": 0, "matched_people": 0, "false_positives": 0, "temporal_applied_matches": 0}
    for sequence in sequences:
        result = infer_sequence(sequence, root, nlf, cfg, device, args)
        temporal_pose, temporal_applied = refine_tracks(result, temporal, int(cfg["matching"]["temporal_window"]))
        sequence_stats = evaluate_sequence(
            sequence, result, temporal_pose, temporal_applied, smpl_layers, cfg, device,
            totals, rows, component_totals, component_rows,
        )
        for key, value in sequence_stats.items():
            coverage[key] = coverage.get(key, 0) + int(value)
        coverage["sequences"] += 1
        print(
            f"[sequence] {sequence.name}: gt={sequence_stats['gt_people']} match={sequence_stats['matched_people']} "
            f"fp={sequence_stats['false_positives']} temporal={sequence_stats['temporal_applied_matches']}",
            flush=True,
        )
    precision = coverage["matched_people"] / max(coverage["matched_people"] + coverage["false_positives"], 1)
    recall = coverage["matched_people"] / max(coverage["gt_people"], 1)
    coverage.update({"precision": precision, "recall": recall, "temporal_applied_rate": coverage["temporal_applied_matches"] / max(coverage["matched_people"], 1)})
    summary = {
        "benchmark": "human3r_style_3dpw_test_v1",
        "input_protocol": "raw 3DPW test RGB + exact per-sequence GT K -> NLF detector -> optional PoseTemporalStabilizerV2",
        "gt_protocol": "raw 3DPW test pkl; gender-specific SMPL; Human3R-style camera-coordinate conversion",
        "matching_protocol": "project predicted SMPL joints with processed GT K; greedy common OpenPose/SMPL 2D-joint association",
        "metric_protocol": "Human3R-style: pelvis[1,2]-aligned MPJPE/PVE; similarity PA-MPJPE/PA-PVE; metric unaligned diagnostics",
        "nlf_checkpoint": str(cfg.get("checkpoints", {}).get("nlf_smpl", "")),
        "temporal_checkpoint": args.temporal_checkpoint or None,
        "sequence_filter": args.sequence_filter,
        "coverage": coverage,
        **{name: values.summary() for name, values in totals.items()},
    }
    if component_totals is not None:
        summary["component_diagnostics"] = {
            "description": "Counterfactual local-body proxies. They are non-additive; compare them to nlf_base to identify whether pose, beta, or neutral-vs-gender representation dominates.",
            **{name: values.summary() for name, values in component_totals.items()},
            "rows_csv": str(output / "component_rows.csv"),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rows(output / "rows.csv", rows)
    if component_totals is not None:
        write_component_rows(output / "component_rows.csv", component_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threedpw-root", default="")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--config", default="benchmarks/human3r_style_3dpw/config.yaml")
    parser.add_argument("--temporal-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/human3r_style_3dpw")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--component-diagnostics", action="store_true", help="Record pose/beta counterfactual error decomposition per matched person-frame")
    return parser.parse_args()


def build_nlf(cfg: dict[str, Any]) -> NLFSMPLProvider:
    data = cfg["data"]
    return NLFSMPLProvider(
        model_path=str(cfg.get("checkpoints", {}).get("nlf_smpl", "")),
        third_party_root=str(cfg.get("third_party", {}).get("nlf_root", "third_party/nlf")),
        model_name="smpl",
        use_detector=True,
        require_boxes=False,
        internal_batch_size=int(data["nlf_internal_batch_size"]),
        num_aug=int(data["nlf_num_aug"]),
        detector_threshold=float(data["nlf_detector_threshold"]),
        detector_nms_iou_threshold=float(data["nlf_detector_nms_iou_threshold"]),
    )


def load_temporal(path: str, device: torch.device) -> PoseTemporalStabilizer:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") not in {"smpl_temporal_stabilizer_v2_pose_mixture", "smpl_temporal_stabilizer_v2_pose_e0"}:
        raise ValueError(f"Unsupported V2 pose checkpoint: {checkpoint.get('format')!r}")
    model = PoseTemporalStabilizer(PoseStabilizerConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def infer_sequence(sequence: ThreeDPWTestSequence, root: Path, nlf: NLFSMPLProvider, cfg: dict[str, Any], device: torch.device, args: argparse.Namespace) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    total_frames = sequence.length if args.max_frames <= 0 else min(sequence.length, args.max_frames)
    k_original = np.asarray(sequence.metadata["cam_intrinsics"], dtype=np.float32).reshape(3, 3)
    images, intrinsics, openpose = [], [], []
    for frame_index in range(total_frames):
        image, k, geometry = load_processed_frame(frame_path(root, sequence, frame_index), k_original, int(cfg["data"]["image_resolution"]), int(cfg["data"]["patch_size"]), str(cfg["data"]["resize_mode"]))
        images.append(image); intrinsics.append(k); openpose.append(raw_openpose_2d(sequence, frame_index, geometry))
    if len({tuple(image.shape) for image in images}) != 1:
        raise RuntimeError("This benchmark currently requires one processed image shape per sequence")
    batch_size = int(cfg["data"]["nlf_batch_size"])
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in ("pred_pose_6d", "pred_betas", "pred_transl_cam", "pred_confs", "pred_boxes")}
    for start in range(0, total_frames, batch_size):
        end = min(total_frames, start + batch_size)
        prediction = nlf.forward_with_intrinsics(torch.stack(images[start:end]).unsqueeze(0).to(device), torch.stack(intrinsics[start:end]).unsqueeze(0).to(device), max_humans=20)
        for key in parts:
            parts[key].append(prediction[key][0].detach())
    out: dict[str, Any] = {"openpose": openpose, "intrinsics": intrinsics}
    out.update({key.removeprefix("pred_"): torch.cat(value, dim=0) for key, value in parts.items()})
    valid = out["confs"][..., 0] > 0.0
    out["tracks"] = BaseSMPLTrackAssigner().assign(out["boxes"].unsqueeze(0), out["betas"].unsqueeze(0), out["transl_cam"].unsqueeze(0), out["confs"].unsqueeze(0), query_mask=valid.unsqueeze(0))
    return out


def refine_tracks(result: dict[str, Any], temporal: PoseTemporalStabilizer | None, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    pose = result["pose_6d"]
    refined, applied = pose.clone(), torch.zeros(pose.shape[:2], dtype=torch.bool, device=pose.device)
    if temporal is None:
        return refined, applied
    if window != temporal.config.window_size:
        raise ValueError(f"Benchmark window={window}, temporal checkpoint window={temporal.config.window_size}")
    ids, mask = result["tracks"]["assigned_track_ids"][0], result["tracks"]["assigned_track_mask"][0]
    for track_id in torch.unique(ids[mask]).tolist():
        same = (ids == track_id) & mask
        valid = same.sum(dim=1) == 1
        slots = same.long().argmax(dim=1)
        starts = list(range(0, max(0, pose.shape[0] - window + 1)))
        for offset in range(0, len(starts), 128):
            selected = starts[offset : offset + 128]
            frame_grid = torch.stack([torch.arange(start, start + window, device=pose.device) for start in selected])
            window_pose = pose[frame_grid, slots[frame_grid]]
            window_valid = valid[frame_grid]
            output = temporal(window_pose, window_valid)
            centre = window // 2
            for local_index, start in enumerate(selected):
                if bool(output["context_valid"][local_index, centre]):
                    frame = start + centre
                    refined[frame, slots[frame]] = output["refined_pose_6d"][local_index, centre]
                    applied[frame, slots[frame]] = True
    return refined, applied


COMPONENT_NAMES = ("full_pred_neutral", "pred_pose_gt_beta", "gt_pose_pred_beta", "gt_pose_gt_beta_neutral")


def evaluate_sequence(sequence: ThreeDPWTestSequence, result: dict[str, Any], temporal_pose: torch.Tensor, temporal_applied: torch.Tensor, smpl: dict[str, SMPLLayer], cfg: dict[str, Any], device: torch.device, totals: dict[str, Totals], rows: list[dict[str, Any]], component_totals: dict[str, Totals] | None, component_rows: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"gt_people": 0, "matched_people": 0, "false_positives": 0, "temporal_applied_matches": 0}
    for frame_index in range(result["pose_6d"].shape[0]):
        gt_vertices, gt_joints, gt_people = decode_gt_camera_space(sequence, frame_index, smpl, device)
        stats["gt_people"] += int(gt_people.numel())
        if not gt_people.numel():
            continue
        valid_pred = result["confs"][frame_index, :, 0] > 0.0
        pred_index = torch.nonzero(valid_pred, as_tuple=False).reshape(-1)
        if not pred_index.numel():
            continue
        base_vertices, base_joints = decode_pred(result["pose_6d"][frame_index, pred_index], result["betas"][frame_index, pred_index], result["transl_cam"][frame_index, pred_index], smpl["neutral"])
        # Raw OpenPose labels remain on CPU until the matched valid GT people
        # are known.  ``gt_people`` originates from GPU SMPL decoding, so use
        # a CPU index for the CPU label tensor before moving that small subset.
        openpose = result["openpose"][frame_index][gt_people.detach().cpu()].to(device)
        matched_local, matched_gt, false_pos = match_by_2d_joints(
            base_joints,
            openpose,
            result["intrinsics"][frame_index].to(device),
            int(cfg["matching"]["min_keypoints"]),
            float(cfg["matching"]["min_confidence"]),
            float(cfg["matching"].get("min_bbox_iou", 0.05)),
        )
        stats["false_positives"] += int(false_pos)
        if not matched_local.numel():
            continue
        original_pred = pred_index[matched_local]
        temporal_vertices, temporal_joints = decode_pred(temporal_pose[frame_index, original_pred], result["betas"][frame_index, original_pred], result["transl_cam"][frame_index, original_pred], smpl["neutral"])
        base_metrics = human3r_camera_metrics(base_joints[matched_local], gt_joints[matched_gt], base_vertices[matched_local], gt_vertices[matched_gt])
        temporal_metrics = human3r_camera_metrics(temporal_joints, gt_joints[matched_gt], temporal_vertices, gt_vertices[matched_gt])
        totals["nlf_base"].add(base_metrics); totals["nlf_pose_temporal"].add(temporal_metrics)
        components = None
        if component_totals is not None:
            gt_pose, gt_beta, _, _ = gt_camera_parameters(sequence, frame_index, device)
            pred_pose = rot6d_to_axis_angle(result["pose_6d"][frame_index, original_pred].reshape(-1, 24, 6)).reshape(-1, 72)
            pred_beta = result["betas"][frame_index, original_pred]
            gt_pose_matched, gt_beta_matched = gt_pose[matched_gt], gt_beta[matched_gt]
            component_meshes = {
                "full_pred_neutral": decode_local(pred_pose, pred_beta, smpl["neutral"]),
                "pred_pose_gt_beta": decode_local(pred_pose, gt_beta_matched, smpl["neutral"]),
                "gt_pose_pred_beta": decode_local(gt_pose_matched, pred_beta, smpl["neutral"]),
                "gt_pose_gt_beta_neutral": decode_local(gt_pose_matched, gt_beta_matched, smpl["neutral"]),
            }
            components = {}
            for name, (vertices, joints) in component_meshes.items():
                metrics = human3r_camera_metrics(joints, gt_joints[matched_gt], vertices, gt_vertices[matched_gt])
                component_totals[name].add(metrics)
                components[name] = metrics
        for local in range(original_pred.numel()):
            is_temporal = bool(temporal_applied[frame_index, original_pred[local]])
            rows.append({"sequence": sequence.name, "frame": frame_index, "pred_index": int(original_pred[local]), "gt_person": int(gt_people[matched_gt[local]]), "temporal_applied": int(is_temporal), **{f"base_{key}": float(value[local].detach().cpu()) for key, value in base_metrics.items()}, **{f"temporal_{key}": float(value[local].detach().cpu()) for key, value in temporal_metrics.items()}})
            if components is not None:
                component_rows.append({
                    "sequence": sequence.name,
                    "frame": frame_index,
                    "pred_index": int(original_pred[local]),
                    "gt_person": int(gt_people[matched_gt[local]]),
                    **{f"{name}_{key}": float(value[local].detach().cpu()) for name, metrics in components.items() for key, value in metrics.items()},
                })
            stats["matched_people"] += 1
            stats["temporal_applied_matches"] += int(is_temporal)
    return stats


def decode_pred(pose6d: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor, layer: SMPLLayer) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, joints = layer(rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72).float(), betas.float())
    return vertices + transl[:, None], joints[:, :24] + transl[:, None]


def decode_local(pose: torch.Tensor, betas: torch.Tensor, layer: SMPLLayer) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, joints = layer(pose.float(), betas.float())
    return vertices, joints[:, :24]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sequence", "frame", "pred_index", "gt_person", "temporal_applied"] + [f"{branch}_{metric}" for branch in ("base", "temporal") for metric in ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm", "pa_pve_mm", "metric_mpjpe_mm", "metric_pve_mm", "root_error_mm")]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_component_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sequence", "frame", "pred_index", "gt_person"] + [
        f"{name}_{metric}"
        for name in COMPONENT_NAMES
        for metric in ("mpjpe_mm", "pve_mm", "pa_mpjpe_mm", "pa_pve_mm", "metric_mpjpe_mm", "metric_pve_mm", "root_error_mm")
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def print_manifest(args: argparse.Namespace, cfg: dict[str, Any], root: Path, sequences: list[ThreeDPWTestSequence], device: torch.device, temporal: PoseTemporalStabilizer | None) -> None:
    print("========== Human3R-style raw 3DPW test benchmark ==========")
    print(f"device: {device}; raw root: {root}; test sequences: {len(sequences)}")
    print(f"NLF checkpoint: {cfg.get('checkpoints', {}).get('nlf_smpl')}; num_aug={cfg['data']['nlf_num_aug']}; detector_thr={cfg['data']['nlf_detector_threshold']}")
    print(f"V2 checkpoint: {args.temporal_checkpoint or 'disabled'}")
    print("GT: gender-specific SMPL + Human3R-style camera conversion; matching: projected 2D common joints")


if __name__ == "__main__":
    main()
