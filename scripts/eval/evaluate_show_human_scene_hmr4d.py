#!/usr/bin/env python3
"""Evaluate project predictions with SHOW HS-V/HS-CF on 3DPW or EMDB-1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hmr4d_smpl_metrics import (  # noqa: E402
    box_iou_cxcywh,
    extract_gt_smpl,
    move_to_device,
)
from scripts.eval.evaluate_show_human_scene_3dpw import (  # noqa: E402
    MetricAccumulator,
    canonical_batch_depth,
    metric_config_to_json,
    resolve_branch,
    resolve_output_dir,
    select_branch_vertices,
    validate_evaluation_config,
)
from scripts.train.train_smpl import apply_overrides, build_model  # noqa: E402
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn  # noqa: E402
from vggt_omega.evaluation import compute_human_scene_consistency, render_mesh_silhouette  # noqa: E402
from vggt_omega.evaluation.rich_physical_grounding import (  # noqa: E402
    CascadeConfig,
    configure_reference_cascade_model,
    load_evaluation_weights,
    run_metric_cascade,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args)
    branches, mask_source, metric_config = validate_evaluation_config(config["human_scene_evaluation"])
    if mask_source != "gt_smpl_projection":
        raise ValueError("The HMR4D-support evaluator currently requires mask_source=gt_smpl_projection")
    if int(args.batch_size) != 1:
        raise ValueError("Reference cascade evaluation requires --batch-size=1 because windows may have different lengths")
    cascade_config = build_cascade_config(config)
    config, checkpoint_config_report = configure_reference_cascade_model(
        config,
        args.checkpoint,
        max_humans=int(args.max_humans),
    )

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config).to(device)
    checkpoint_audit = load_evaluation_weights(
        model=model,
        baseline_checkpoint=require_path(config, "checkpoints.vggt_baseline"),
        stage2_checkpoint=Path(args.checkpoint).expanduser(),
        scale_checkpoint=Path(args.scale_checkpoint).expanduser(),
        device=device,
    )
    model.eval()
    smpl_model_dir = require_path(config, "assets.smpl_model_dir")
    # Project predictions use the neutral SMPL topology, matching the model's
    # existing inference path.  GT meshes are decoded with the annotated
    # dataset gender so that the evaluation-region silhouette is not widened
    # or narrowed by a neutral-body approximation.
    smpl = SMPLLayer(smpl_model_dir, gender="neutral").to(device).eval()
    gt_smpl_by_gender = {
        gender: SMPLLayer(smpl_model_dir, gender=gender).to(device).eval()
        for gender in ("neutral", "male", "female")
    }
    faces = torch.as_tensor(smpl.faces, dtype=torch.int64, device=device)

    dataset = build_dataset(config, args)
    indices = list(range(len(dataset)))
    if int(args.max_windows) > 0:
        indices = indices[: int(args.max_windows)]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=hmr4d_eval_collate_fn,
        drop_last=False,
    )
    accumulators = {branch: MetricAccumulator() for branch in branches}
    rows: list[dict[str, Any]] = []
    processed = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        predictions, _ = run_metric_cascade(model, smpl, batch["images"], cascade_config)
        evaluate_batch(
            predictions,
            batch,
            smpl,
            gt_smpl_by_gender,
            faces,
            branches,
            metric_config,
            accumulators,
            rows,
        )
        processed += int(batch["images"].shape[0])
        if args.log_interval > 0 and processed % int(args.log_interval) == 0:
            print(f"[show-hs] dataset={args.dataset} processed_windows={processed}", flush=True)

    branch_summaries = {branch: accumulators[branch].summary() for branch in branches}
    primary_branch = "refined" if "refined" in branch_summaries else branches[0]
    rows_path = output_dir / f"{args.dataset}_show_human_scene_rows.csv"
    summary_path = output_dir / f"{args.dataset}_show_human_scene_summary.json"
    write_csv(rows_path, rows)
    summary = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "scale_checkpoint": str(args.scale_checkpoint),
        "checkpoint_load": checkpoint_audit,
        "checkpoint_model_config": checkpoint_config_report,
        "num_windows": processed,
        "num_person_frames": len(rows) // max(len(branches), 1),
        "num_sequences_evaluated": len({str(row["vid"]) for row in rows}),
        "frames_root": str(dataset.frames_root),
        "mask_source": mask_source,
        "primary_branch": primary_branch,
        "table3": branch_summaries[primary_branch]["table3_show_code"],
        "protocol": {
            "metric": "SHOW released HS-V/HS-CF implementation",
            "region": "GT SMPL mesh projection; identical region protocol must be used for every compared method",
            "inference": "accepted analytic-coarse + residual-scale + Stage-2 human-scene alignment cascade",
            "windowing": "non-overlapping windows; final shorter window retained; every source frame counted once",
            "cascade": {
                "confidence_threshold": cascade_config.confidence_threshold,
                "coarse_min_anchor_pixels": cascade_config.coarse_min_anchor_pixels,
                "coarse_scale_min": cascade_config.coarse_scale_min,
                "coarse_scale_max": cascade_config.coarse_scale_max,
                "coarse_anchor_stride": cascade_config.coarse_anchor_stride,
                "coarse_fallback": cascade_config.coarse_fallback,
                "effective_affine_mode": cascade_config.effective_affine_mode,
            },
            **metric_config_to_json(metric_config),
        },
        "branches": branch_summaries,
        "rows_csv": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["3dpw", "emdb1"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale-checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--model-config", default="configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml")
    parser.add_argument("--eval-config", default="configs/eval_show_human_scene_hmr4d.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/show_human_scene_hmr4d")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.model_config))
    config = deep_update(config, load_yaml_config(args.eval_config))
    config = apply_overrides(config, args.override)
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    return config


def build_cascade_config(config: dict[str, Any]) -> CascadeConfig:
    cascade = config.get("reference_cascade", {})
    return CascadeConfig(
        confidence_threshold=float(cascade.get("confidence_threshold", 0.05)),
        coarse_min_anchor_pixels=int(cascade.get("coarse_min_anchor_pixels", 32)),
        coarse_scale_min=float(cascade.get("coarse_scale_min", 0.10)),
        coarse_scale_max=float(cascade.get("coarse_scale_max", 10.0)),
        coarse_anchor_stride=int(cascade.get("coarse_anchor_stride", 8)),
        coarse_fallback=str(cascade.get("coarse_fallback", "sequence_median")),
        effective_affine_mode=str(cascade.get("effective_affine_mode", "clip_median")),
    )


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> HMR4DSupportEvalDataset:
    data_cfg = config.get("data", {})
    support_key = "datasets.emdb_hmr4d_support_root" if args.dataset == "emdb1" else "datasets.threedpw_hmr4d_support_root"
    support_root = args.support_root or require_path(config, support_key)
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    return HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=None,
        sequence_length=int(args.sequence_length or data_cfg.get("sequence_length", 1)),
        stride=int(args.stride),
        image_size=int(data_cfg.get("image_size", 518)),
        image_resolution=int(data_cfg.get("image_resolution", 512)),
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=1,
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        full_sequence=False,
        non_overlapping_windows=bool(data_cfg.get("non_overlapping_windows", True)),
        one_window_per_sequence=bool(getattr(args, "one_window_per_sequence", False)),
    )


@torch.no_grad()
def evaluate_batch(
    predictions: dict[str, Any],
    batch: dict[str, Any],
    smpl: SMPLLayer,
    gt_smpl_by_gender: dict[str, SMPLLayer],
    faces: torch.Tensor,
    branches: tuple[str, ...],
    metric_config,
    accumulators: dict[str, MetricAccumulator],
    rows: list[dict[str, Any]],
) -> None:
    gt = extract_gt_smpl(batch["eval_label"], predictions["pred_confs"].device)
    gt_vertices_world = decode_gendered_gt_vertices(gt, batch["eval_label"], gt_smpl_by_gender)
    gt_vertices_world = gt_vertices_world + gt["transl"][..., None, :]
    transforms = torch.as_tensor(batch["eval_label"]["T_w2c"], device=gt_vertices_world.device, dtype=gt_vertices_world.dtype)
    gt_vertices_cam = apply_transform(gt_vertices_world, transforms)

    branch_values = {branch: resolve_branch(predictions, branch, smpl) for branch in branches}
    reference_depth = next(iter(branch_values.values()))["depth"]
    confidence_batch = canonical_batch_depth(predictions["depth_conf"])
    confidence = confidence_batch.reshape(-1, *confidence_batch.shape[-2:])
    batch_size, num_frames = batch["images"].shape[:2]
    image_hw = tuple(int(value) for value in reference_depth.shape[-2:])
    _, intrinsics = encoding_to_camera(predictions["pose_enc"].detach().float(), image_size_hw=image_hw, build_intrinsics=True)
    intrinsics_flat = intrinsics.reshape(-1, 3, 3)
    gt_intrinsics_flat = batch["K_scal3r"].reshape(-1, 3, 3).to(intrinsics_flat)
    query_indices = choose_frame_query_indices(predictions, batch)
    eval_mask = batch.get("eval_mask", torch.ones(batch_size, num_frames, device=reference_depth.device)).bool()

    for batch_index in range(batch_size):
        vid = str(batch["meta"]["vid"][batch_index])
        frame_ids = batch["meta"]["frame_indices"][batch_index]
        for frame_offset in range(num_frames):
            if not bool(eval_mask[batch_index, frame_offset]):
                continue
            query_idx = int(query_indices[batch_index, frame_offset].detach().cpu())
            flat_frame = batch_index * num_frames + frame_offset
            human_mask = render_mesh_silhouette(
                gt_vertices_cam[batch_index, frame_offset],
                gt_intrinsics_flat[flat_frame],
                image_hw=image_hw,
                faces=faces,
                backend=metric_config.visibility_backend,
            )
            for branch, values in branch_values.items():
                metric = compute_human_scene_consistency(
                    select_branch_vertices(values, flat_frame, query_idx, branch),
                    values["depth"][flat_frame],
                    intrinsics_flat[flat_frame],
                    human_mask,
                    faces=faces,
                    scene_valid_mask=confidence[flat_frame] > float(metric_config.depth_conf_threshold),
                    config=metric_config,
                )
                accumulators[branch].add(metric)
                rows.append(
                    {
                        "dataset": str(batch["meta"]["dataset_key"][batch_index]),
                        "vid": vid,
                        "frame_index": int(frame_ids[frame_offset]),
                        "query_idx": query_idx,
                        "branch": branch,
                        "mask_source": "gt_smpl_projection",
                        "mask_intrinsics_source": "K_scal3r",
                        "body_source": values["body_source"],
                        "depth_source": values["depth_source"],
                        **metric,
                    }
                )


def choose_frame_query_indices(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> torch.Tensor:
    """Associate the annotated person independently in every frame.

    NLF detector slots are confidence ordered and are not guaranteed to keep a
    stable query index across a long evaluation window.  A clip-level argmax
    can therefore score the wrong person on 3DPW multi-person sequences.
    """

    return choose_frame_query_matches(predictions, batch)["query_indices"]


def choose_frame_query_matches(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return frame-wise GT-box association diagnostics for one target track.

    A 3DPW HMR4D record is a person track (for example ``flat_guitar_01_0``),
    not a multi-person video label.  The selected query slot may change over
    time, but every selection must overlap that record's GT/preprocessed box.
    """

    confs = predictions["pred_confs"]
    fallback = confs[..., 0].argmax(dim=-1)
    nan_iou = torch.full_like(fallback, float("nan"), dtype=torch.float32)
    no_target = torch.zeros_like(fallback, dtype=torch.bool)
    pred_boxes = predictions.get("pred_boxes")
    gt_boxes = batch.get("gt_boxes")
    boxes_mask = batch.get("boxes_mask")
    if not all(isinstance(value, torch.Tensor) for value in (pred_boxes, gt_boxes, boxes_mask)):
        return {
            "query_indices": fallback,
            "target_iou": nan_iou,
            "has_gt_box": no_target,
            "used_gt_box": no_target,
        }
    iou = box_iou_cxcywh(pred_boxes[:, :, :, None, :], gt_boxes[:, :, None, :, :])
    valid_targets = boxes_mask.bool()
    iou = iou.masked_fill(~valid_targets[:, :, None, :], -1.0)
    per_query = iou.max(dim=-1).values
    matched = per_query.argmax(dim=-1)
    has_target = valid_targets.any(dim=-1)
    query_indices = torch.where(has_target, matched, fallback)
    selected_iou = per_query.gather(-1, query_indices[..., None]).squeeze(-1)
    selected_iou = torch.where(has_target, selected_iou, nan_iou)
    return {
        "query_indices": query_indices,
        "target_iou": selected_iou,
        "has_gt_box": has_target,
        "used_gt_box": has_target,
    }


def apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames = points.shape[:2]
    if transform.ndim == 2:
        transform = transform.reshape(1, 1, 4, 4).expand(batch_size, num_frames, -1, -1)
    elif transform.ndim == 3:
        if transform.shape[0] == batch_size:
            transform = transform[:, None].expand(-1, num_frames, -1, -1)
        elif batch_size == 1 and transform.shape[0] == num_frames:
            transform = transform[None]
        else:
            raise ValueError(f"Cannot broadcast T_w2c {tuple(transform.shape)} to points {tuple(points.shape)}")
    if transform.ndim != 4 or tuple(transform.shape[:2]) != (batch_size, num_frames):
        raise ValueError(f"Expected T_w2c [B,S,4,4], got {tuple(transform.shape)}")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return torch.einsum("bsij,bsvj->bsvi", rotation, points) + translation[..., None, :]


def decode_gendered_gt_vertices(
    gt: dict[str, torch.Tensor],
    eval_label: dict[str, Any],
    gt_smpl_by_gender: dict[str, SMPLLayer],
) -> torch.Tensor:
    """Decode world-space GT SMPL using each sequence's annotated gender."""

    batch_size, num_frames = gt["poses"].shape[:2]
    raw_genders = eval_label.get("gender", ["neutral"] * batch_size)
    if isinstance(raw_genders, (str, bytes)):
        raw_genders = [raw_genders] * batch_size
    if not isinstance(raw_genders, (list, tuple)) or len(raw_genders) != batch_size:
        raise ValueError(f"Expected one GT gender per batch item, got {raw_genders!r}")

    vertices = []
    for batch_index, raw_gender in enumerate(raw_genders):
        if isinstance(raw_gender, (list, tuple)) and len(raw_gender) == 1:
            raw_gender = raw_gender[0]
        if isinstance(raw_gender, bytes):
            raw_gender = raw_gender.decode("utf-8")
        gender = str(raw_gender).strip().lower()
        if gender not in gt_smpl_by_gender:
            gender = "neutral"
        batch_vertices, _ = gt_smpl_by_gender[gender](
            gt["poses"][batch_index].reshape(-1, 72).float(),
            gt["betas"][batch_index].reshape(-1, 10).float(),
        )
        vertices.append(batch_vertices.reshape(num_frames, -1, 3))
    return torch.stack(vertices, dim=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["dataset"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
