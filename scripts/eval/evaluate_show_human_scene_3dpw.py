#!/usr/bin/env python3
"""Evaluate SHOW HS-V/HS-CF on the project 3DPW prediction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_3dpw_smpl_base_metrics import (  # noqa: E402
    build_dataset,
    collect_matches,
    flatten_prediction,
    move_to_device,
    select_indices,
)
from scripts.train.train_smpl import (  # noqa: E402
    apply_overrides,
    build_model,
    extract_state_dict,
    forward_model,
    load_initial_checkpoint,
)
from vggt_omega.data import threedpw_collate_fn  # noqa: E402
from vggt_omega.evaluation import (  # noqa: E402
    HumanSceneConsistencyConfig,
    compute_human_scene_consistency,
    render_mesh_silhouette,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.training.hungarian_losses import flatten_smpl_targets  # noqa: E402
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


TABLE_METRICS = ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")
OFFICIAL_SHOW_METRICS = (
    "xy_scale_mse_p5_95",
    "xy_scale_mse_p10_90",
    "chamfer_norm_pct_p5_95",
    "chamfer_norm_pct_p10_90",
)
ALLOWED_BRANCHES = frozenset({"base", "refined"})
ALLOWED_VISIBILITY_BACKENDS = frozenset({"pytorch3d", "vertex_zbuffer"})
ALLOWED_MASK_SOURCES = frozenset({"gt_smpl_projection", "sam2_patch"})


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.invalid: dict[str, int] = {}

    def add(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if not key.startswith(("hs_", "chamfer_", "xy_scale_mse_")):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1
        if not bool(values.get("valid", False)):
            reason = str(values.get("invalid_reason", "unknown") or "unknown")
            self.invalid[reason] = self.invalid.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        means = {
            key: self.sums[key] / self.counts[key]
            for key in sorted(self.sums)
            if self.counts.get(key, 0) > 0
        }
        return {
            "means": means,
            "counts": dict(sorted(self.counts.items())),
            "invalid": dict(sorted(self.invalid.items())),
            "table3_show_code": {key: means.get(key) for key in TABLE_METRICS},
            "official_show_names": {key: means.get(key) for key in OFFICIAL_SHOW_METRICS},
        }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    eval_cfg = config["human_scene_evaluation"]
    branches, mask_source, metric_config = validate_evaluation_config(eval_cfg)

    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    checkpoint_audit = load_evaluation_checkpoint(
        model,
        Path(args.checkpoint).expanduser(),
        device,
        required_prefixes=required_checkpoint_prefixes(config, branches),
    )
    model.eval()
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir"), gender="neutral").to(device).eval()
    faces = torch.as_tensor(smpl.faces, dtype=torch.int64, device=device)

    dataset = build_dataset(config, args)
    indices = select_indices(args, len(dataset))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=threedpw_collate_fn,
        drop_last=False,
    )
    matcher = HungarianSMPLMatcher(
        cost_conf=0.5,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_kpts=0.0,
        require_boxes=True,
        require_j2ds=False,
    )
    accumulators = {branch: MetricAccumulator() for branch in branches}
    rows: list[dict[str, Any]] = []
    processed = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        predictions = forward_model(model, batch, config)
        batch_indices = indices[processed : processed + int(batch["images"].shape[0])]
        evaluate_batch(
            predictions,
            batch,
            matcher,
            smpl,
            faces,
            branches,
            mask_source,
            metric_config,
            accumulators,
            rows,
            batch_indices,
        )
        processed += int(batch["images"].shape[0])
        if args.log_interval > 0 and processed % int(args.log_interval) == 0:
            print(f"[show-hs] processed={processed}", flush=True)

    rows_path = output_dir / "show_human_scene_rows.csv"
    summary_path = output_dir / "show_human_scene_summary.json"
    write_csv(rows_path, rows)
    branch_summaries = {branch: accumulators[branch].summary() for branch in branches}
    primary_branch = "refined" if "refined" in branch_summaries else branches[0]
    summary = {
        "dataset": "3dpw",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_load": checkpoint_audit,
        "model_config": str(args.model_config),
        "evaluation_config": str(args.eval_config),
        "num_windows": int(processed),
        "num_person_frame_branch_rows": len(rows),
        "primary_branch": primary_branch,
        "table3": branch_summaries[primary_branch]["table3_show_code"],
        "mask_source": mask_source,
        "reference": {
            "project": "SHOW official codebase",
            "files": [
                "hmr4d/utils/eval/eval_utils.py::compute_camcoord_human_scene_synchronize_metrics",
                "hmr4d/model/gvhmr/callbacks/metric_3dpw.py::ThreeDPWMetricMocap",
            ],
            "adaptation": "adapted rewrite; no runtime import from .paper",
        },
        "protocol": {
            "show_code_hs_cf": "scene-to-rendered-body mean nearest distance / mean masked-scene depth * 100",
            "hs_v": "mean absolute x/y population-variance difference (Eq. 20)",
            "invalid_policy": metric_config.invalid_policy,
            "person_aggregation": "unweighted mean over matched person-frames, matching SHOW's concatenated frame mean",
            "branch_contract": {
                "base": "diagnostic unless raw depth is already in the SMPL metric scale",
                "refined": "primary project branch; uses HSI metric depth and refined SMPL when available",
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--model-config", default="configs/infer_smpl_hsi_v3_trstr_spatial.yaml")
    parser.add_argument("--eval-config", default="configs/eval_show_human_scene_3dpw.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/show_human_scene_3dpw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--subset-indices-csv", default="")
    parser.add_argument("--subset-index-column", default="dataset_index")
    parser.add_argument("--subset-unique", action=argparse.BooleanOptionalAction, default=True)
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
    data_cfg = config.setdefault("data", {})
    mask_source = str(config.get("human_scene_evaluation", {}).get("mask_source", "gt_smpl_projection"))
    if mask_source == "sam2_patch" and not bool(data_cfg.get("require_sam2_patch_masks", False)):
        raise ValueError("SHOW HS metrics require independent per-person masks; set data.require_sam2_patch_masks=true")
    return config


def validate_evaluation_config(
    eval_cfg: dict[str, Any],
) -> tuple[tuple[str, ...], str, HumanSceneConsistencyConfig]:
    branches = tuple(str(item).strip().lower() for item in coerce_sequence(eval_cfg.get("branches", ["refined"])))
    if not branches:
        raise ValueError("human_scene_evaluation.branches must not be empty")
    if len(set(branches)) != len(branches):
        raise ValueError(f"human_scene_evaluation.branches contains duplicates: {branches}")
    unsupported_branches = sorted(set(branches) - ALLOWED_BRANCHES)
    if unsupported_branches:
        raise ValueError(
            "human_scene_evaluation.branches only supports base/refined; "
            f"got {unsupported_branches}"
        )

    trims = tuple(float(item) for item in coerce_sequence(eval_cfg.get("percentile_trims", [5, 10])))
    if not trims:
        raise ValueError("human_scene_evaluation.percentile_trims must not be empty")
    if len(set(trims)) != len(trims):
        raise ValueError(f"human_scene_evaluation.percentile_trims contains duplicates: {trims}")
    invalid_trims = [trim for trim in trims if not math.isfinite(trim) or not (0.0 <= trim < 50.0)]
    if invalid_trims:
        raise ValueError(
            "Each percentile trim must satisfy 0 <= trim < 50; "
            f"got {invalid_trims}"
        )

    backend = str(eval_cfg.get("visibility_backend", "pytorch3d")).strip().lower()
    if backend not in ALLOWED_VISIBILITY_BACKENDS:
        raise ValueError(
            "human_scene_evaluation.visibility_backend must be pytorch3d or vertex_zbuffer; "
            f"got {backend!r}"
        )

    invalid_policy = str(eval_cfg.get("invalid_policy", "show_zero")).strip().lower()
    if invalid_policy not in {"show_zero", "skip_nan"}:
        raise ValueError(
            "human_scene_evaluation.invalid_policy must be show_zero or skip_nan; "
            f"got {invalid_policy!r}"
        )

    mask_source = str(eval_cfg.get("mask_source", "gt_smpl_projection")).strip().lower()
    if mask_source not in ALLOWED_MASK_SOURCES:
        raise ValueError(
            "human_scene_evaluation.mask_source must be gt_smpl_projection or sam2_patch; "
            f"got {mask_source!r}"
        )
    if mask_source == "gt_smpl_projection" and backend != "pytorch3d":
        raise ValueError("mask_source=gt_smpl_projection requires visibility_backend=pytorch3d")

    positive_ints = {
        "scene_point_stride": int(eval_cfg.get("scene_point_stride", 1)),
        "distance_chunk_size": int(eval_cfg.get("distance_chunk_size", 2048)),
        "min_body_points": int(eval_cfg.get("min_body_points", 1)),
        "min_scene_points": int(eval_cfg.get("min_scene_points", 1)),
        "min_points_for_percentile": int(eval_cfg.get("min_points_for_percentile", 20)),
    }
    non_positive = {key: value for key, value in positive_ints.items() if value <= 0}
    if non_positive:
        raise ValueError(f"Human-scene evaluation integer settings must be positive: {non_positive}")

    metric_config = HumanSceneConsistencyConfig(
        percentile_trims=trims,
        visibility_backend=backend,
        invalid_policy=invalid_policy,
        compute_paper_audit=bool(eval_cfg.get("compute_paper_audit", False)),
        depth_conf_threshold=float(eval_cfg.get("depth_conf_threshold", 0.05)),
        **positive_ints,
    )
    return branches, mask_source, metric_config


def coerce_sequence(value: Any) -> tuple[Any, ...]:
    """Accept YAML lists and comma-separated CLI override values."""

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    outputs_root = (ROOT / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"Evaluation outputs must stay under {outputs_root}, got {resolved}") from exc
    return resolved


def required_checkpoint_prefixes(config: dict[str, Any], branches: tuple[str, ...]) -> tuple[str, ...]:
    if "refined" not in branches:
        return ()
    model_cfg = config.get("model", {})
    prefixes = []
    if bool(model_cfg.get("enable_hsi_refine", False)):
        prefixes.append("hsi_refinement_head.")
    if bool(model_cfg.get("enable_hsi_trstr", False)):
        prefixes.append("hsi_trstr_head.")
    return tuple(prefixes)


def load_evaluation_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    required_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Overlay experiment weights and reject missing/random refined heads."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model_keys = set(model.state_dict())
    matched_keys = tuple(key for key in state_dict if key in model_keys)
    if not matched_keys:
        raise RuntimeError(f"Checkpoint has no tensors matching this model configuration: {checkpoint_path}")
    missing_checkpoint_prefixes = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state_dict)
    ]
    missing_model_prefixes = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in matched_keys)
    ]
    if missing_checkpoint_prefixes or missing_model_prefixes:
        raise RuntimeError(
            "Refined evaluation would use missing or randomly initialized modules. "
            f"checkpoint_missing={missing_checkpoint_prefixes}, model_mismatch={missing_model_prefixes}, "
            f"checkpoint={checkpoint_path}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    audit = {
        "path": str(checkpoint_path),
        "checkpoint_tensor_count": len(state_dict),
        "matched_tensor_count": len(matched_keys),
        "missing_model_key_count": len(missing),
        "unexpected_checkpoint_key_count": len(unexpected),
        "required_prefixes": list(required_prefixes),
    }
    print(
        f"[ckpt] loaded {checkpoint_path} matched={len(matched_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return audit


@torch.no_grad()
def evaluate_batch(
    predictions: dict[str, Any],
    batch: dict[str, torch.Tensor],
    matcher: HungarianSMPLMatcher,
    smpl: SMPLLayer,
    faces: torch.Tensor,
    branches: tuple[str, ...],
    mask_source: str,
    metric_config: HumanSceneConsistencyConfig,
    accumulators: dict[str, MetricAccumulator],
    rows: list[dict[str, Any]],
    selected_indices: list[int],
) -> None:
    if mask_source == "sam2_patch" and (
        "smpl_query_patch_masks" not in batch or "smpl_query_patch_masks_valid" not in batch
    ):
        raise KeyError("Batch is missing SAM2 patch masks required for human-region point maps")
    pred_confs = flatten_prediction(predictions["pred_confs"], 3)
    pred_boxes = flatten_prediction(predictions["pred_boxes"], 3)
    targets = flatten_smpl_targets(batch, device=pred_confs.device)
    indices = matcher({"pred_confs": pred_confs, "pred_boxes": pred_boxes}, targets)
    matched = collect_matches(indices, targets, pred_confs.device)
    if matched["frame_idx"].numel() == 0:
        return

    gt_poses = rot6d_to_axis_angle(matched["pose_6d"].reshape(-1, 24, 6)).reshape(-1, 72)
    gt_vertices, _ = smpl(gt_poses.float(), matched["betas"].float())
    gt_vertices_cam = gt_vertices.to(matched["transl_cam"]) + matched["transl_cam"][:, None, :]

    branch_values = {branch: resolve_branch(predictions, branch, smpl) for branch in branches}
    reference_depth = next(iter(branch_values.values()))["depth"]
    confidence = canonical_batch_depth(predictions["depth_conf"])
    image_hw = tuple(int(x) for x in reference_depth.shape[-2:])
    _, intrinsics = encoding_to_camera(
        predictions["pose_enc"].detach().float(),
        image_size_hw=image_hw,
        build_intrinsics=True,
    )
    intrinsics_flat = intrinsics.reshape(-1, 3, 3)
    gt_intrinsics_flat = batch["K_scal3r"].reshape(-1, 3, 3).to(intrinsics_flat)
    mask_flat = None
    mask_valid_flat = None
    if mask_source == "sam2_patch":
        mask_flat = batch["smpl_query_patch_masks"].reshape(
            -1, batch["smpl_query_patch_masks"].shape[-2], batch["smpl_query_patch_masks"].shape[-1]
        )
        mask_valid_flat = batch["smpl_query_patch_masks_valid"].reshape(
            -1, batch["smpl_query_patch_masks_valid"].shape[-1]
        )
    dense_ids = batch["person_ids"].reshape(-1, batch["person_ids"].shape[-1])
    dense_people = batch["smpl_mask"].reshape(-1, batch["smpl_mask"].shape[-1])
    num_frames = int(batch["images"].shape[1])
    patch_size = int(batch.get("patch_size", torch.tensor(16, device=pred_confs.device)).reshape(-1)[0].item())
    input_hw = tuple(int(x) for x in batch["images"].shape[-2:])

    for match_index in range(int(matched["frame_idx"].numel())):
        flat_frame = int(matched["frame_idx"][match_index].detach().cpu())
        query_idx = int(matched["src_idx"][match_index].detach().cpu())
        person_id = int(matched["person_ids"][match_index].detach().cpu())
        if mask_source == "gt_smpl_projection":
            full_mask = render_mesh_silhouette(
                gt_vertices_cam[match_index],
                gt_intrinsics_flat[flat_frame],
                image_hw=image_hw,
                faces=faces,
            )
        else:
            assert mask_flat is not None and mask_valid_flat is not None
            target_slot = find_dense_person_slot(dense_ids[flat_frame], dense_people[flat_frame], person_id)
            if target_slot is None:
                continue
            patch_mask = mask_flat[flat_frame, target_slot]
            patch_mask_valid = bool(mask_valid_flat[flat_frame, target_slot])
            full_mask = patch_mask_to_image_mask(patch_mask, input_hw, image_hw, patch_size) if patch_mask_valid else None
        batch_index = flat_frame // num_frames
        frame_offset = flat_frame % num_frames
        dataset_index = int(selected_indices[batch_index]) if batch_index < len(selected_indices) else -1

        for branch, values in branch_values.items():
            if full_mask is None:
                metric = invalid_mask_metrics(metric_config, metric_config.visibility_backend, "missing_sam2_mask")
            else:
                metric = compute_human_scene_consistency(
                    values["vertices"][flat_frame, query_idx],
                    values["depth"][flat_frame],
                    intrinsics_flat[flat_frame],
                    full_mask,
                    faces=faces,
                    scene_valid_mask=confidence.reshape(-1, *confidence.shape[-2:])[flat_frame]
                    > float(metric_config.depth_conf_threshold),
                    config=metric_config,
                )
            accumulators[branch].add(metric)
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "frame_offset": frame_offset,
                    "query_idx": query_idx,
                    "person_id": person_id,
                    "branch": branch,
                    "mask_source": mask_source,
                    "mask_intrinsics_source": "K_scal3r" if mask_source == "gt_smpl_projection" else "image_mask",
                    "body_source": values["body_source"],
                    "depth_source": values["depth_source"],
                    **metric,
                }
            )


@torch.no_grad()
def resolve_branch(predictions: dict[str, Any], branch: str, smpl: SMPLLayer) -> dict[str, Any]:
    raw_depth = canonical_batch_depth(predictions["depth"])
    if branch == "base":
        pose_key, beta_key, transl_key = "pred_poses", "pred_betas", "pred_transl_cam"
        depth = raw_depth
        depth_source = "depth"
    elif branch == "refined":
        required = ("hsi_refined_pred_poses", "hsi_refined_pred_betas", "hsi_refined_pred_transl_cam")
        if all(isinstance(predictions.get(key), torch.Tensor) for key in required):
            pose_key, beta_key, transl_key = required
        elif isinstance(predictions.get("hsi_trstr_refined_pred_transl_cam"), torch.Tensor):
            pose_key, beta_key, transl_key = "pred_poses", "pred_betas", "hsi_trstr_refined_pred_transl_cam"
        else:
            raise KeyError("Requested refined branch but no HSI/TRSTR refined SMPL tensors are present")
        if isinstance(predictions.get("hsi_translation_depth"), torch.Tensor):
            depth = canonical_batch_depth(predictions["hsi_translation_depth"])
            depth_source = "hsi_translation_depth"
        elif isinstance(predictions.get("hsi_scene_scale"), torch.Tensor) and isinstance(predictions.get("hsi_scene_depth_bias"), torch.Tensor):
            scale = predictions["hsi_scene_scale"].to(raw_depth).reshape(*raw_depth.shape[:2], -1)[..., 0]
            bias = predictions["hsi_scene_depth_bias"].to(raw_depth).reshape(*raw_depth.shape[:2], -1)[..., 0]
            depth = raw_depth * scale[..., None, None] + bias[..., None, None]
            depth_source = "depth*hsi_scene_scale+hsi_scene_depth_bias"
        else:
            depth = raw_depth
            depth_source = "depth (no refined scene affine present)"
    else:
        raise ValueError(f"Unsupported evaluation branch: {branch!r}")

    poses = predictions[pose_key].detach()
    betas = predictions[beta_key].detach()
    translation = predictions[transl_key].detach()
    shape = poses.shape[:3]
    vertices, _ = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
    vertices = vertices.reshape(*shape, vertices.shape[-2], 3).to(translation) + translation[..., None, :]
    return {
        "vertices": vertices.reshape(-1, vertices.shape[-2], 3),
        "depth": depth.reshape(-1, *depth.shape[-2:]),
        "body_source": f"{pose_key}+{beta_key}+{transl_key}",
        "depth_source": depth_source,
    }


def canonical_batch_depth(value: torch.Tensor) -> torch.Tensor:
    depth = value
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim == 3:
        depth = depth[:, None]
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {tuple(value.shape)}")
    return depth.detach().float()


def patch_mask_to_image_mask(
    patch_mask: torch.Tensor,
    input_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    grid_h = int(input_hw[0]) // max(int(patch_size), 1)
    grid_w = int(input_hw[1]) // max(int(patch_size), 1)
    if int(patch_mask.numel()) != grid_h * grid_w:
        raise ValueError(
            f"Patch mask has {patch_mask.numel()} entries but image/patch geometry requires {grid_h}x{grid_w}"
        )
    grid = patch_mask.reshape(1, 1, grid_h, grid_w).float()
    return F.interpolate(grid, size=depth_hw, mode="nearest")[0, 0] > 0.5


def find_dense_person_slot(ids: torch.Tensor, valid: torch.Tensor, person_id: int) -> int | None:
    hits = torch.nonzero(valid.bool() & (ids.long() == int(person_id)), as_tuple=False).flatten()
    return int(hits[0].detach().cpu()) if hits.numel() else None


def invalid_mask_metrics(
    config: HumanSceneConsistencyConfig,
    visibility_backend: str,
    reason: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "valid": False,
        "invalid_reason": reason,
        "visibility_backend": visibility_backend,
        "body_point_count": 0,
        "scene_point_count_unfiltered": 0,
    }
    official_value = 0.0 if config.invalid_policy == "show_zero" else math.nan
    for trim in config.percentile_trims:
        label = str(int(trim)) if float(trim).is_integer() else str(trim).replace(".", "p")
        high = 100.0 - float(trim)
        high_label = str(int(high)) if high.is_integer() else str(high).replace(".", "p")
        suffix = f"p{label}_{high_label}"
        out[f"valid_{label}"] = False
        out[f"valid_show_code{label}"] = False
        if config.compute_paper_audit:
            out[f"valid_paper{label}"] = False
        out[f"invalid_reason_{label}"] = reason
        out[f"scene_point_count_{label}"] = 0
        out[f"hs_cf{label}"] = official_value
        out[f"hs_v{label}"] = official_value
        out[f"chamfer_scene_to_body_m{label}"] = official_value
        out[f"chamfer_abs_{suffix}"] = official_value
        out[f"chamfer_norm_pct_{suffix}"] = official_value
        out[f"xy_scale_mse_{suffix}"] = official_value
        if config.compute_paper_audit:
            for name in ("hs_cf_paper", "hs_cf_paper_pct", "chamfer_body_to_scene_m"):
                out[f"{name}{label}"] = math.nan
    return out


def metric_config_to_json(config: HumanSceneConsistencyConfig) -> dict[str, Any]:
    return {
        "percentile_trims": list(config.percentile_trims),
        "visibility_backend": config.visibility_backend,
        "invalid_policy": config.invalid_policy,
        "compute_paper_audit": config.compute_paper_audit,
        "depth_conf_threshold": config.depth_conf_threshold,
        "scene_point_stride": config.scene_point_stride,
        "distance_chunk_size": config.distance_chunk_size,
        "min_body_points": config.min_body_points,
        "min_scene_points": config.min_scene_points,
        "min_points_for_percentile": config.min_points_for_percentile,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["dataset_index"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
