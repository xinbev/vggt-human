import argparse
import hashlib
import copy
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
#==============================================================================
# Compatibility patch for old chumpy on Python 3.11+
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
#==============================================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data import BedlamDataset, HFBedlamDataset, ThreeDPWDataset, bedlam_collate_fn, hf_bedlam_collate_fn, threedpw_collate_fn
from vggt_omega.data.geometry import resolve_image_size_config
from vggt_omega.models import VGGTOmega
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training import HungarianSMPLLoss, HungarianSMPLMatcher, SMPLSlotLoss
from vggt_omega.training.config import deep_update, load_yaml_config, require_path
from vggt_omega.training.hsi_stage2_v4_noise import apply_deterministic_v4_noise
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices
from vggt_omega.utils.rotation import axis_angle_to_rotmat, rot6d_to_axis_angle, rot6d_to_rotmat


_SMPL_NOISE_CACHE: dict[tuple[str, str], SMPLLayer] = {}


def main() -> None:
    args = parse_args()
    path_config = load_yaml_config(args.path_config)
    train_config = load_yaml_config(args.train_config)
    config = deep_update(path_config, train_config)
    config = apply_overrides(config, args.override)

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    train_loader = build_loader(config, split=config["data"]["train_split"], shuffle=True, role="train")
    val_loader = None
    if config["data"].get("val_split"):
        try:
            val_loader = build_loader(config, split=config["data"]["val_split"], shuffle=False, role="val")
        except FileNotFoundError as exc:
            print(f"[warn] validation split skipped: {exc}")
    topk_state = init_topk_state(output_dir, config)

    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    apply_freeze_policy(model, config)
    teacher_model = build_teacher_model(config, device)
    criterion = build_criterion(config).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters after applying freeze policy")
    print_trainable_summary(model)
    validate_trainable_prefix_contract(model, config)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )

    start_epoch = 0
    global_step = 0
    resume_path = str(config.get("checkpoint", {}).get("resume", "") or "")
    if resume_path:
        start_epoch, global_step = resume_training_checkpoint(model, optimizer, resume_path, device, config)
        if bool(config.get("checkpoint", {}).get("reset_epoch", False)):
            start_epoch = 0
            global_step = 0
            print("[ckpt] reset resume epoch/global_step to 0")
    overlay_path = str(config.get("checkpoint", {}).get("overlay", "") or "")
    if overlay_path:
        load_overlay_checkpoint(model, overlay_path, device, config)
    frozen_hashes = hash_model_prefixes(model, normalize_string_list(config.get("checkpoint", {}).get("frozen_hash_prefixes", [])))

    epochs = int(config["optim"]["epochs"])
    train_started_at = time.monotonic()
    total_train_steps = max((epochs - start_epoch) * len(train_loader), 1)
    print(
        "[data] "
        f"train_samples={len(train_loader.dataset)} "
        f"batch_size={int(config['optim']['batch_size'])} "
        f"steps_per_epoch={len(train_loader)} "
        f"start_epoch={start_epoch} total_epochs={epochs} "
        f"remaining_steps={total_train_steps}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        global_step, train_losses = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            global_step=global_step,
            config=config,
            teacher_model=teacher_model,
            progress_start_time=train_started_at,
            progress_total_steps=total_train_steps,
            progress_done_offset=(epoch - start_epoch) * len(train_loader),
            total_epochs=epochs,
        )
        assert_model_prefix_hashes(model, frozen_hashes)
        monitor_losses = train_losses
        val_losses: dict[str, float] | None = None
        if val_loader is not None and (epoch + 1) % int(config["optim"].get("val_interval", 1)) == 0:
            val_losses = validate(model, criterion, val_loader, device, epoch, config, teacher_model=teacher_model)
            monitor_losses = val_losses
            maybe_save_topk_checkpoint(
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                output_dir,
                topk_state,
                monitor_losses,
                source="val",
            )
        elif bool(config.get("checkpoint", {}).get("save_top_k_from_train", True)):
            maybe_save_topk_checkpoint(
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                output_dir,
                topk_state,
                monitor_losses,
                source="train",
            )
        metrics_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": {key: float(value) for key, value in train_losses.items()},
            "val": ({key: float(value) for key, value in val_losses.items()} if val_losses is not None else None),
            "frozen_hashes": frozen_hashes,
        }
        save_json(metrics_payload, output_dir / f"metrics_epoch_{epoch + 1:04d}.json")
        save_json(metrics_payload, output_dir / "metrics_latest.json")
        if should_save_checkpoint(config, epoch + 1, epochs):
            checkpoint_cfg = config.get("checkpoint", {})
            if bool(checkpoint_cfg.get("save_epoch_checkpoint", True)):
                save_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
            if bool(checkpoint_cfg.get("save_latest", True)):
                save_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir / "checkpoint_latest.pt")
            if bool(checkpoint_cfg.get("save_final", False)) and (epoch + 1) == epochs:
                save_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir / "checkpoint_final.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGGT-Omega SMPL query head on BEDLAM-style data")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.key=value")
    return parser.parse_args()


def build_loader(config: dict[str, Any], split: str, shuffle: bool, role: str = "train") -> DataLoader:
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "bedlam")).lower()
    if dataset_name in {"3dpw", "threedpw"}:
        return build_3dpw_loader(config, split=split, shuffle=shuffle)
    if dataset_name in {"hf_bedlam", "hfb_edlam", "bedlam_hf"}:
        return build_hf_bedlam_loader(config, split=split, shuffle=shuffle)
    if dataset_name != "bedlam":
        raise ValueError(f"Unsupported data.dataset: {data_cfg.get('dataset')!r}")
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    root = require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"))
    boxes_root = None
    if data_cfg.get("boxes_root_key"):
        boxes_root = require_path(config, data_cfg["boxes_root_key"], allow_empty=not bool(data_cfg.get("require_boxes", False)))
    manifest = str(data_cfg.get(f"{role}_sequence_manifest", "") or "").strip()
    contact_teacher_root = str(data_cfg.get("contact_teacher_root", "") or "").strip()
    if not contact_teacher_root and data_cfg.get("contact_teacher_root_key"):
        contact_teacher_root = require_path(
            config,
            data_cfg["contact_teacher_root_key"],
            allow_empty=not bool(data_cfg.get("require_contact_teacher", False)),
        )
    dataset = BedlamDataset(
        root=root,
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_depth=bool(data_cfg.get("require_depth", False)),
        boxes_root=boxes_root,
        require_boxes=bool(data_cfg.get("require_boxes", False)),
        query_source=str(data_cfg.get("query_source", "persons")),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
        sequence_manifest=manifest or None,
        contact_teacher_root=contact_teacher_root or None,
        require_contact_teacher=bool(data_cfg.get("require_contact_teacher", False)),
        contact_only_windows=bool(data_cfg.get(f"{role}_contact_only", False)),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split, role=role)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        collate_fn=bedlam_collate_fn,
        drop_last=shuffle,
        **build_dataloader_runtime_kwargs(data_cfg),
    )


def build_3dpw_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    dataset = ThreeDPWDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.threedpw_root")),
        annotation_root=require_path(config, data_cfg.get("annotation_root_key", "datasets.threedpw_smpl_base_root")),
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", 2)),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        sam2_patch_masks_root=resolve_optional_data_path(config, data_cfg, "sam2_patch_masks_root", "sam2_patch_masks_root_key"),
        require_sam2_patch_masks=bool(data_cfg.get("require_sam2_patch_masks", False)),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        collate_fn=threedpw_collate_fn,
        drop_last=shuffle,
        **build_dataloader_runtime_kwargs(data_cfg),
    )


def resolve_optional_data_path(
    config: dict[str, Any],
    data_cfg: dict[str, Any],
    value_key: str,
    path_key: str,
) -> str:
    value = str(data_cfg.get(value_key, "") or "").strip()
    if value:
        return value
    key = str(data_cfg.get(path_key, "") or "").strip()
    if not key:
        return ""
    return require_path(config, key, allow_empty=not bool(data_cfg.get("require_sam2_patch_masks", False)))


def build_hf_bedlam_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    dataset = HFBedlamDataset(
        images_root=require_path(config, data_cfg.get("images_root_key", "datasets.hf_bedlam_images_root")),
        npz_root=require_path(config, data_cfg.get("npz_root_key", "datasets.hf_bedlam_npz_root")),
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", 20)),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        bbox_expand=float(data_cfg.get("bbox_expand", 0.15)),
        transl_add_cam_ext=bool(data_cfg.get("transl_add_cam_ext", True)),
        skip_missing_images=bool(data_cfg.get("skip_missing_images", True)),
        max_npz_files=int(data_cfg.get("max_npz_files", 0) or 0),
        max_frames=int(data_cfg.get("max_frames", 0) or 0),
        sam2_patch_masks_root=resolve_optional_data_path(config, data_cfg, "sam2_patch_masks_root", "sam2_patch_masks_root_key"),
        sam2_patch_masks_split=str(data_cfg.get("sam2_patch_masks_split", split or "train")),
        require_sam2_patch_masks=bool(data_cfg.get("require_sam2_patch_masks", False)),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        collate_fn=hf_bedlam_collate_fn,
        drop_last=shuffle,
        **build_dataloader_runtime_kwargs(data_cfg),
    )


def build_dataloader_runtime_kwargs(data_cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        prefetch_factor = int(data_cfg.get("prefetch_factor", 4) or 0)
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def maybe_subset_dataset(dataset: Any, data_cfg: dict[str, Any], split: str, role: str | None = None) -> Any | Subset:
    train_split = str(data_cfg.get("train_split", split))
    resolved_role = role or ("train" if split == train_split else "val")
    if resolved_role == "val" and not bool(data_cfg.get("subset_apply_to_val", False)):
        return dataset
    subset_csv = str(data_cfg.get("subset_indices_csv", "") or "").strip()
    if not subset_csv:
        return dataset
    subset_path = Path(subset_csv).expanduser()
    if not subset_path.is_file():
        raise FileNotFoundError(f"data.subset_indices_csv not found: {subset_path}")
    column = str(data_cfg.get("subset_index_column", "dataset_index"))
    unique = bool(data_cfg.get("subset_unique", False))
    repeat = max(int(data_cfg.get("subset_repeat", 1) or 1), 1)
    max_samples = int(data_cfg.get("subset_max_samples", 0) or 0)
    indices = read_subset_indices_csv(subset_path, column)
    if unique:
        seen = set()
        indices = [idx for idx in indices if not (idx in seen or seen.add(idx))]
    valid = [idx for idx in indices if 0 <= idx < len(dataset)]
    skipped = len(indices) - len(valid)
    if not valid:
        raise ValueError(f"No valid dataset indices found in {subset_path} column={column!r} for split={split!r}")
    expanded = valid * repeat
    if max_samples > 0:
        expanded = expanded[:max_samples]
    print(
        "[data] subset "
        f"split={split} csv={subset_path} column={column} rows={len(indices)} "
        f"valid={len(valid)} skipped={skipped} unique={unique} repeat={repeat} final={len(expanded)}",
        flush=True,
    )
    return Subset(dataset, expanded)


def read_subset_indices_csv(path: Path, column: str) -> list[int]:
    indices: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV {path} does not contain required column {column!r}")
        for row in reader:
            raw = str(row.get(column, "")).strip()
            if not raw:
                continue
            try:
                indices.append(int(float(raw)))
            except ValueError as exc:
                raise ValueError(f"Invalid dataset index {raw!r} in {path}") from exc
    return indices


def build_model(config: dict[str, Any]) -> VGGTOmega:
    model_cfg = config["model"]
    if config.get("loss", {}).get("type", "slot") != "hungarian" and int(model_cfg["num_smpl_queries"]) != int(config["data"]["max_humans"]):
        raise ValueError("model.num_smpl_queries must equal data.max_humans for slot-aligned SMPLSlotLoss")
    return VGGTOmega(
        patch_size=int(model_cfg["patch_size"]),
        embed_dim=int(model_cfg["embed_dim"]),
        enable_camera=bool(model_cfg.get("enable_camera", False)),
        enable_depth=bool(model_cfg.get("enable_depth", False)),
        enable_alignment=bool(model_cfg.get("enable_alignment", False)),
        enable_smpl=bool(model_cfg.get("enable_smpl", True)),
        num_smpl_queries=int(model_cfg["num_smpl_queries"]),
        smpl_predict_boxes=bool(model_cfg.get("predict_boxes", False)),
        smpl_bbox_mode=str(model_cfg.get("smpl_bbox_mode", "direct")),
        smpl_predict_id_embed=bool(model_cfg.get("predict_id_embed", False)),
        smpl_id_embed_dim=int(model_cfg.get("id_embed_dim", 256)),
        smpl_id_hidden_dim=int(model_cfg.get("id_hidden_dim", 512)),
        smpl_id_feature_mode=str(model_cfg.get("id_feature_mode", "query")),
        smpl_return_aux=bool(model_cfg.get("smpl_return_aux", False)),
        smpl_translation_output_mode=str(model_cfg.get("smpl_translation_output_mode", "direct")),
        smpl_translation_decode_hidden_dim=int(model_cfg.get("smpl_translation_decode_hidden_dim", 512)),
        smpl_translation_decode_max_log_depth_delta=float(model_cfg.get("smpl_translation_decode_max_log_depth_delta", 1.00)),
        smpl_translation_decode_max_ray_delta_m=float(model_cfg.get("smpl_translation_decode_max_ray_delta_m", 1.50)),
        smpl_translation_decode_max_tangent_offset_m=float(model_cfg.get("smpl_translation_decode_max_tangent_offset_m", 1.00)),
        smpl_translation_decode_human_height_prior_m=float(model_cfg.get("smpl_translation_decode_human_height_prior_m", 1.70)),
        smpl_enable_translation_refine=bool(model_cfg.get("smpl_enable_translation_refine", False)),
        smpl_translation_refine_hidden_dim=int(model_cfg.get("smpl_translation_refine_hidden_dim", 512)),
        smpl_translation_refine_max_ray_delta_m=float(model_cfg.get("smpl_translation_refine_max_ray_delta_m", 0.60)),
        smpl_translation_refine_max_tangent_delta_m=float(model_cfg.get("smpl_translation_refine_max_tangent_delta_m", 0.35)),
        smpl_translation_refine_max_log_depth_delta=float(model_cfg.get("smpl_translation_refine_max_log_depth_delta", 0.50)),
        smpl_translation_refine_max_box_prior_weight=float(model_cfg.get("smpl_translation_refine_max_box_prior_weight", 0.50)),
        smpl_translation_refine_human_height_prior_m=float(model_cfg.get("smpl_translation_refine_human_height_prior_m", 1.70)),
        smpl_translation_refine_use_log_depth=bool(model_cfg.get("smpl_translation_refine_use_log_depth", True)),
        smpl_query_box_prior=bool(model_cfg.get("smpl_query_box_prior", False)),
        smpl_query_patch_pool=bool(model_cfg.get("smpl_query_patch_pool", False)),
        smpl_query_patch_pool_expand=float(model_cfg.get("smpl_query_patch_pool_expand", 0.10)),
        smpl_query_patch_pool_mode=str(model_cfg.get("smpl_query_patch_pool_mode", "box")),
        smpl_query_mask_min_patch_count=int(model_cfg.get("smpl_query_mask_min_patch_count", 4)),
        smpl_query_mask_fallback_to_box=bool(model_cfg.get("smpl_query_mask_fallback_to_box", True)),
        smpl_track_assignment_mode=str(model_cfg.get("smpl_track_assignment_mode", "gt")),
        smpl_use_external_track_prior=bool(model_cfg.get("smpl_use_external_track_prior", True)),
        smpl_enable_post_track_temporal_translation=bool(model_cfg.get("smpl_enable_post_track_temporal_translation", False)),
        smpl_track_assign_max_age=int(model_cfg.get("smpl_track_assign_max_age", 90)),
        smpl_track_assign_min_quality=float(model_cfg.get("smpl_track_assign_min_quality", 0.25)),
        smpl_track_assign_max_center_distance_norm=float(model_cfg.get("smpl_track_assign_max_center_distance_norm", 0.25)),
        smpl_track_assign_max_transl_distance_m=float(model_cfg.get("smpl_track_assign_max_transl_distance_m", 1.50)),
        smpl_track_assign_max_beta_l1=float(model_cfg.get("smpl_track_assign_max_beta_l1", 0.30)),
        smpl_track_assign_external_iou_min=float(model_cfg.get("smpl_track_assign_external_iou_min", 0.50)),
        smpl_track_assign_id_weight=float(model_cfg.get("smpl_track_assign_id_weight", 0.0)),
        smpl_track_assign_max_id_distance=float(model_cfg.get("smpl_track_assign_max_id_distance", 0.70)),
        smpl_enable_temporal_translation=bool(model_cfg.get("smpl_enable_temporal_translation", False)),
        smpl_temporal_translation_hidden_dim=int(model_cfg.get("smpl_temporal_translation_hidden_dim", 512)),
        smpl_temporal_translation_max_velocity_delta_m=float(model_cfg.get("smpl_temporal_translation_max_velocity_delta_m", 0.25)),
        smpl_temporal_translation_gate_bias=float(model_cfg.get("smpl_temporal_translation_gate_bias", 2.5)),
        smpl_temporal_translation_use_world=bool(model_cfg.get("smpl_temporal_translation_use_world", True)),
        enable_hsi_refine=bool(model_cfg.get("enable_hsi_refine", False)),
        hsi_hidden_dim=int(model_cfg.get("hsi_hidden_dim", 512)),
        hsi_num_layers=int(model_cfg.get("hsi_num_layers", 5)),
        hsi_num_heads=int(model_cfg.get("hsi_num_heads", 8)),
        hsi_num_iters=int(model_cfg.get("hsi_num_iters", 3)),
        hsi_scene_window=int(model_cfg.get("hsi_scene_window", 3)),
        hsi_probe_mode=str(model_cfg.get("hsi_probe_mode", "projected")),
        hsi_affine_probe_mode=str(model_cfg.get("hsi_affine_probe_mode", "projected")),
        hsi_probe_window=int(model_cfg.get("hsi_probe_window", 9)),
        hsi_probe_blend=float(model_cfg.get("hsi_probe_blend", 1.0)),
        hsi_use_delta_gate=bool(model_cfg.get("hsi_use_delta_gate", False)),
        hsi_enable_temporal_momentum=bool(model_cfg.get("hsi_enable_temporal_momentum", False)),
        hsi_temporal_momentum_decay=float(model_cfg.get("hsi_temporal_momentum_decay", 0.7)),
        hsi_temporal_momentum_detach=bool(model_cfg.get("hsi_temporal_momentum_detach", True)),
        hsi_temporal_momentum_use_track_ids=bool(model_cfg.get("hsi_temporal_momentum_use_track_ids", True)),
        hsi_track_quality_min=float(model_cfg.get("hsi_track_quality_min", 0.25)),
        hsi_track_gap_max=int(model_cfg.get("hsi_track_gap_max", 30)),
        hsi_scene_affine_mode=str(model_cfg.get("hsi_scene_affine_mode", "per_frame")),
        hsi_scene_affine_ema_alpha=float(model_cfg.get("hsi_scene_affine_ema_alpha", 0.25)),
        hsi_scene_log_scale_min=float(model_cfg.get("hsi_scene_log_scale_min", -5.0)),
        hsi_scene_log_scale_max=float(model_cfg.get("hsi_scene_log_scale_max", 5.0)),
        hsi_transl_delta_scale=float(model_cfg.get("hsi_transl_delta_scale", 0.05)),
        hsi_transl_delta_mode=str(model_cfg.get("hsi_transl_delta_mode", "xyz")),
        hsi_use_affine_depth_for_transl=bool(model_cfg.get("hsi_use_affine_depth_for_transl", False)),
        hsi_affine_depth_detach=bool(model_cfg.get("hsi_affine_depth_detach", True)),
        enable_hsi_human_scene_align=bool(model_cfg.get("enable_hsi_human_scene_align", False)),
        hsi_align_hidden_dim=int(model_cfg.get("hsi_align_hidden_dim", 256)),
        hsi_align_num_sample_vertices=int(model_cfg.get("hsi_align_num_sample_vertices", 96)),
        hsi_align_local_window=int(model_cfg.get("hsi_align_local_window", 7)),
        hsi_align_max_ray_delta_m=float(model_cfg.get("hsi_align_max_ray_delta_m", 0.35)),
        hsi_align_max_tangent_delta_m=float(model_cfg.get("hsi_align_max_tangent_delta_m", 0.12)),
        hsi_align_use_delta_gate=bool(model_cfg.get("hsi_align_use_delta_gate", True)),
        hsi_align_gate_application_mode=str(model_cfg.get("hsi_align_gate_application_mode", "sigmoid_v1")),
        hsi_align_gate_deadzone=float(model_cfg.get("hsi_align_gate_deadzone", 0.50)),
        hsi_align_overwrite_refined=bool(model_cfg.get("hsi_align_overwrite_refined", True)),
        hsi_align_base_source=str(model_cfg.get("hsi_align_base_source", "hsi_refined")),
        hsi_align_max_correspondence_distance_m=float(model_cfg.get("hsi_align_max_correspondence_distance_m", 0.35)),
        hsi_align_gt_max_correspondence_distance_m=float(model_cfg.get("hsi_align_gt_max_correspondence_distance_m", 3.5)),
        hsi_align_min_depth_confidence=float(model_cfg.get("hsi_align_min_depth_confidence", 0.0)),
        hsi_align_residual_mad_multiplier=float(model_cfg.get("hsi_align_residual_mad_multiplier", 3.0)),
        hsi_align_max_depth_m=float(model_cfg.get("hsi_align_max_depth_m", 20.0)),
        hsi_align_feature_version=str(model_cfg.get("hsi_align_feature_version", "legacy_mean_v1")),
        hsi_align_delta_parameterization=str(model_cfg.get("hsi_align_delta_parameterization", "learned_v1")),
        hsi_align_robust_ray_gain=float(model_cfg.get("hsi_align_robust_ray_gain", 1.0)),
        enable_hsi_translation_refine_v4=bool(model_cfg.get("enable_hsi_translation_refine_v4", False)),
        hsi_v4_hidden_dim=int(model_cfg.get("hsi_v4_hidden_dim", 256)),
        hsi_v4_num_sample_vertices=int(model_cfg.get("hsi_v4_num_sample_vertices", 128)),
        hsi_v4_local_window=int(model_cfg.get("hsi_v4_local_window", 7)),
        hsi_v4_min_correspondences=int(model_cfg.get("hsi_v4_min_correspondences", 12)),
        hsi_v4_max_ray_ratio=float(model_cfg.get("hsi_v4_max_ray_ratio", 0.25)),
        hsi_v4_ray_parameterization=str(model_cfg.get("hsi_v4_ray_parameterization", "residual_gain")),
        hsi_v4_max_ray_gain=float(model_cfg.get("hsi_v4_max_ray_gain", 4.0)),
        hsi_v4_max_tangent_delta_m=float(model_cfg.get("hsi_v4_max_tangent_delta_m", 0.12)),
        hsi_v4_max_correspondence_distance_m=float(model_cfg.get("hsi_v4_max_correspondence_distance_m", 3.5)),
        hsi_v4_residual_mad_multiplier=float(model_cfg.get("hsi_v4_residual_mad_multiplier", 3.0)),
        hsi_v4_max_depth_m=float(model_cfg.get("hsi_v4_max_depth_m", 20.0)),
        hsi_v4_phase=str(model_cfg.get("hsi_v4_phase", "correction")),
        hsi_v4_gate_threshold=float(model_cfg.get("hsi_v4_gate_threshold", 0.5)),
        hsi_v4_overwrite_refined=bool(model_cfg.get("hsi_v4_overwrite_refined", True)),
        enable_hsi_contact_refine=bool(model_cfg.get("enable_hsi_contact_refine", False)),
        hsi_contact_hidden_dim=int(model_cfg.get("hsi_contact_hidden_dim", 256)),
        hsi_contact_sole_vertices_per_foot=int(model_cfg.get("hsi_contact_sole_vertices_per_foot", 48)),
        hsi_contact_support_window=int(model_cfg.get("hsi_contact_support_window", 21)),
        hsi_contact_support_min_points=int(model_cfg.get("hsi_contact_support_min_points", 24)),
        hsi_contact_support_max_rmse_m=float(model_cfg.get("hsi_contact_support_max_rmse_m", 0.05)),
        hsi_contact_support_max_depth_m=float(model_cfg.get("hsi_contact_support_max_depth_m", 20.0)),
        hsi_contact_max_root_normal_delta_m=float(model_cfg.get("hsi_contact_max_root_normal_delta_m", 0.12)),
        hsi_contact_max_hip_delta_deg=float(model_cfg.get("hsi_contact_max_hip_delta_deg", 4.0)),
        hsi_contact_max_knee_delta_deg=float(model_cfg.get("hsi_contact_max_knee_delta_deg", 8.0)),
        hsi_contact_max_ankle_delta_deg=float(model_cfg.get("hsi_contact_max_ankle_delta_deg", 10.0)),
        hsi_contact_use_temporal_velocity=bool(model_cfg.get("hsi_contact_use_temporal_velocity", False)),
        hsi_contact_max_velocity_m=float(model_cfg.get("hsi_contact_max_velocity_m", 0.25)),
        hsi_contact_overwrite_refined=bool(model_cfg.get("hsi_contact_overwrite_refined", True)),
        enable_hsi_grounding=bool(model_cfg.get("enable_hsi_grounding", False)),
        hsi_grounding_hidden_dim=int(model_cfg.get("hsi_grounding_hidden_dim", 192)),
        hsi_grounding_sole_vertices_per_foot=int(model_cfg.get("hsi_grounding_sole_vertices_per_foot", 48)),
        hsi_grounding_exclusion_vertices=int(model_cfg.get("hsi_grounding_exclusion_vertices", 0)),
        hsi_grounding_support_window=int(model_cfg.get("hsi_grounding_support_window", 31)),
        hsi_grounding_support_min_points=int(model_cfg.get("hsi_grounding_support_min_points", 32)),
        hsi_grounding_support_max_rmse_m=float(model_cfg.get("hsi_grounding_support_max_rmse_m", 0.05)),
        hsi_grounding_support_max_depth_m=float(model_cfg.get("hsi_grounding_support_max_depth_m", 20.0)),
        hsi_grounding_support_max_point_depth_delta_m=float(
            model_cfg.get("hsi_grounding_support_max_point_depth_delta_m", 0.75)
        ),
        hsi_grounding_target_clearance_m=float(model_cfg.get("hsi_grounding_target_clearance_m", 0.0)),
        hsi_grounding_clearance_deadzone_m=float(model_cfg.get("hsi_grounding_clearance_deadzone_m", 0.025)),
        hsi_grounding_max_root_delta_m=float(model_cfg.get("hsi_grounding_max_root_delta_m", 0.12)),
        hsi_grounding_gate_threshold=float(model_cfg.get("hsi_grounding_gate_threshold", 0.5)),
        hsi_grounding_hard_gate_eval=bool(model_cfg.get("hsi_grounding_hard_gate_eval", True)),
        hsi_grounding_overwrite_refined=bool(model_cfg.get("hsi_grounding_overwrite_refined", True)),
        hsi_grounding_min_depth_confidence=float(model_cfg.get("hsi_grounding_min_depth_confidence", 0.0)),
        smpl_model_dir=str(config.get("assets", {}).get("smpl_model_dir", "")),
        smpl_provider=str(model_cfg.get("smpl_provider", "internal")),
        nlf_model_path=str(model_cfg.get("nlf_model_path", config.get("checkpoints", {}).get("nlf_smpl", ""))),
        nlf_third_party_root=str(model_cfg.get("nlf_third_party_root", config.get("third_party", {}).get("nlf_root", "third_party/nlf"))),
        nlf_model_name=str(model_cfg.get("nlf_model_name", "smpl")),
        nlf_use_detector=bool(model_cfg.get("nlf_use_detector", False)),
        nlf_require_boxes=bool(model_cfg.get("nlf_require_boxes", True)),
        nlf_internal_batch_size=int(model_cfg.get("nlf_internal_batch_size", 64)),
        nlf_num_aug=int(model_cfg.get("nlf_num_aug", 1)),
        nlf_detector_threshold=float(model_cfg.get("nlf_detector_threshold", 0.3)),
        nlf_detector_nms_iou_threshold=float(model_cfg.get("nlf_detector_nms_iou_threshold", 0.7)),
        nlf_max_detections=int(model_cfg.get("nlf_max_detections", 150)),
        image_size=int(config.get("data", {}).get("image_resolution", config.get("data", {}).get("image_size", 512))),
        freeze_dense_head=bool(model_cfg.get("freeze_dense_head", False)),
        freeze_aggregator_forward=bool(model_cfg.get("freeze_aggregator_forward", False)),
    )


def build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    loss_cfg = dict(config.get("loss", {}))
    loss_type = str(loss_cfg.pop("type", "slot"))
    if loss_type == "hungarian":
        if "smpl_model_dir" not in loss_cfg:
            smpl_model_dir = config.get("assets", {}).get("smpl_model_dir")
            if smpl_model_dir:
                loss_cfg["smpl_model_dir"] = smpl_model_dir
        match_cfg = config.get("matching", {})
        data_image_resolution = int(config.get("data", {}).get("image_resolution", config.get("data", {}).get("image_size", 512)))
        matcher = HungarianSMPLMatcher(
            cost_conf=float(match_cfg.get("cost_conf", 1.0)),
            cost_bbox=float(match_cfg.get("cost_bbox", 5.0)),
            cost_giou=float(match_cfg.get("cost_giou", 2.0)),
            cost_kpts=float(match_cfg.get("cost_kpts", 0.0)),
            j2ds_norm_scale=float(data_image_resolution),
            require_boxes=bool(config.get("data", {}).get("require_boxes", True)),
            require_j2ds=False,
        )
        return HungarianSMPLLoss(matcher=matcher, **loss_cfg)
    if loss_type == "slot":
        return SMPLSlotLoss(**loss_cfg)
    raise ValueError(f"Unsupported loss.type: {loss_type}")


def apply_freeze_policy(model: torch.nn.Module, config: dict[str, Any]) -> None:
    model_cfg = config.get("model", {})
    camera_head = getattr(model, "camera_head", None)
    if camera_head is not None and bool(model_cfg.get("freeze_camera_head", False)):
        freeze_module(camera_head)
    dense_head = getattr(model, "dense_head", None)
    if dense_head is not None and bool(model_cfg.get("freeze_dense_head", False)):
        freeze_module(dense_head)
    smpl_head = getattr(model, "smpl_head", None)
    if smpl_head is not None and bool(model_cfg.get("freeze_smpl_head", False)):
        freeze_module(smpl_head)
    if smpl_head is not None and bool(model_cfg.get("freeze_smpl_translation", False)):
        regression_head = getattr(smpl_head, "regression_head", None)
        for name in ("transl_cam_heads", "translation_decode_heads", "translation_refiner"):
            module = getattr(regression_head, name, None) if regression_head is not None else None
            if module is not None:
                freeze_module(module)
        temporal_translation_refiner = getattr(smpl_head, "temporal_translation_refiner", None)
        if temporal_translation_refiner is not None:
            freeze_module(temporal_translation_refiner)
    if smpl_head is not None and bool(model_cfg.get("train_smpl_translation_heads", False)):
        regression_head = getattr(smpl_head, "regression_head", None)
        transl_heads = getattr(regression_head, "transl_cam_heads", None)
        if transl_heads is None:
            raise ValueError("train_smpl_translation_heads=true requires SMPL transl_cam_heads")
        unfreeze_module(transl_heads)
    if smpl_head is not None and bool(model_cfg.get("train_smpl_translation_refiner", False)):
        regression_head = getattr(smpl_head, "regression_head", None)
        translation_refiner = getattr(regression_head, "translation_refiner", None)
        if translation_refiner is None:
            raise ValueError("train_smpl_translation_refiner=true requires model.smpl_enable_translation_refine=true")
        unfreeze_module(translation_refiner)
    if smpl_head is not None and bool(model_cfg.get("train_smpl_translation_decode_heads", False)):
        regression_head = getattr(smpl_head, "regression_head", None)
        translation_decode_heads = getattr(regression_head, "translation_decode_heads", None)
        if translation_decode_heads is None:
            raise ValueError(
                "train_smpl_translation_decode_heads=true requires "
                "model.smpl_translation_output_mode=ray_offset_depth or simple_ray_depth"
            )
        unfreeze_module(translation_decode_heads)
    if smpl_head is not None and bool(model_cfg.get("train_smpl_temporal_translation", False)):
        temporal_translation_refiner = getattr(smpl_head, "temporal_translation_refiner", None)
        if temporal_translation_refiner is None:
            raise ValueError(
                "train_smpl_temporal_translation=true requires "
                "model.smpl_enable_temporal_translation=true or model.smpl_enable_post_track_temporal_translation=true"
            )
        unfreeze_module(temporal_translation_refiner)
    if smpl_head is not None and bool(model_cfg.get("train_smpl_box_heads", False)):
        regression_head = getattr(smpl_head, "regression_head", None)
        for name in ("box_heads", "box_delta_heads"):
            module = getattr(regression_head, name, None)
            if module is not None:
                unfreeze_module(module)
    hsi_head = getattr(model, "hsi_refinement_head", None)
    if hsi_head is not None:
        if bool(model_cfg.get("freeze_hsi_scene_affine", False)):
            for name in ("scale_delta", "bias_delta"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
        if bool(model_cfg.get("freeze_hsi_backbone", False)):
            for name in ("scene_projs", "token_mlp", "blocks"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
        train_last_blocks = int(model_cfg.get("train_hsi_last_blocks", 0) or 0)
        if train_last_blocks > 0:
            blocks = getattr(hsi_head, "blocks", None)
            if blocks is not None:
                for block in list(blocks)[-train_last_blocks:]:
                    unfreeze_module(block)
        if bool(model_cfg.get("freeze_hsi_betas_delta", False)):
            module = getattr(hsi_head, "betas_delta", None)
            if module is not None:
                freeze_module(module)
        if bool(model_cfg.get("train_hsi_transl_only", False)):
            for name in ("pose_delta", "betas_delta", "contact_head"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
        if bool(model_cfg.get("train_hsi_scene_affine_only", False)):
            for name in ("pose_delta", "betas_delta", "transl_delta", "contact_head", "delta_gate"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
        if bool(model_cfg.get("train_hsi_smpl_delta_only", False)):
            for name in ("scale_delta", "bias_delta", "contact_head"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
    hsi_align_head = getattr(model, "hsi_human_scene_align_head", None)
    if hsi_align_head is not None:
        if bool(model_cfg.get("freeze_hsi_human_scene_align", False)):
            freeze_module(hsi_align_head)
        if bool(model_cfg.get("train_hsi_human_scene_align_only", False)):
            for module_name in (
                "aggregator",
                "camera_head",
                "dense_head",
                "smpl_head",
                "nlf_smpl_provider",
                "hsi_refinement_head",
            ):
                module = getattr(model, module_name, None)
                if module is not None:
                    freeze_module(module)
            unfreeze_module(hsi_align_head)
    hsi_contact_head = getattr(model, "hsi_contact_refine_head", None)
    if hsi_contact_head is not None:
        if bool(model_cfg.get("freeze_hsi_contact_refine", False)):
            freeze_module(hsi_contact_head)
        if bool(model_cfg.get("train_hsi_contact_refine_only", False)):
            for module_name in (
                "aggregator",
                "camera_head",
                "dense_head",
                "smpl_head",
                "nlf_smpl_provider",
                "hsi_refinement_head",
                "hsi_human_scene_align_head",
            ):
                module = getattr(model, module_name, None)
                if module is not None:
                    freeze_module(module)
            unfreeze_module(hsi_contact_head)
        if bool(model_cfg.get("freeze_hsi_contact_pose_branch", False)):
            freeze_module(hsi_contact_head.lower_pose_head)
        if bool(model_cfg.get("freeze_hsi_contact_root_branch", False)):
            freeze_module(hsi_contact_head.root_normal_head)
    hsi_v4_head = getattr(model, "hsi_translation_refine_v4_head", None)
    if hsi_v4_head is not None:
        if bool(model_cfg.get("freeze_hsi_translation_refine_v4", False)):
            freeze_module(hsi_v4_head)
        if bool(model_cfg.get("train_hsi_v4_correction_only", False)):
            for module_name in (
                "aggregator",
                "camera_head",
                "dense_head",
                "smpl_head",
                "nlf_smpl_provider",
                "hsi_refinement_head",
                "hsi_human_scene_align_head",
                "hsi_contact_refine_head",
            ):
                module = getattr(model, module_name, None)
                if module is not None:
                    freeze_module(module)
            freeze_module(hsi_v4_head)
            unfreeze_module(hsi_v4_head.correction_trunk)
            unfreeze_module(hsi_v4_head.correction_head)
    hsi_grounding_head = getattr(model, "hsi_grounding_head", None)
    if hsi_grounding_head is not None:
        if bool(model_cfg.get("freeze_hsi_grounding", False)):
            freeze_module(hsi_grounding_head)
        if bool(model_cfg.get("train_hsi_grounding_only", False)):
            for module_name in (
                "aggregator",
                "camera_head",
                "dense_head",
                "smpl_head",
                "nlf_smpl_provider",
                "hsi_refinement_head",
                "hsi_human_scene_align_head",
                "hsi_translation_refine_v4_head",
                "hsi_contact_refine_head",
            ):
                module = getattr(model, module_name, None)
                if module is not None:
                    freeze_module(module)
            unfreeze_module(hsi_grounding_head)
            freeze_module(hsi_grounding_head.smpl)
    if not bool(model_cfg.get("freeze_aggregator", False)):
        return
    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        return
    aggregator.eval()
    for _, param in aggregator.named_parameters():
        param.requires_grad = False
    if bool(model_cfg.get("train_smpl_query_token", False)) and hasattr(aggregator, "smpl_query_token"):
        aggregator.smpl_query_token.requires_grad = True
    if bool(model_cfg.get("train_smpl_box_prior_embed", False)) and getattr(aggregator, "smpl_box_prior_embed", None) is not None:
        aggregator.smpl_box_prior_embed.train()
        for param in aggregator.smpl_box_prior_embed.parameters():
            param.requires_grad = True
    if bool(model_cfg.get("train_smpl_patch_pool_embed", False)) and getattr(aggregator, "smpl_patch_pool_embed", None) is not None:
        aggregator.smpl_patch_pool_embed.train()
        for param in aggregator.smpl_patch_pool_embed.parameters():
            param.requires_grad = True


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for _, param in module.named_parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for _, param in module.named_parameters():
        param.requires_grad = True


def print_trainable_summary(model: torch.nn.Module, max_names: int = 40) -> None:
    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    total = sum(count for _, count in trainable)
    print(f"[trainable] tensors={len(trainable)} params={total:,}")
    for name, count in trainable[:max_names]:
        print(f"[trainable] {name} ({count:,})")
    if len(trainable) > max_names:
        print(f"[trainable] ... {len(trainable) - max_names} more tensors")


def validate_trainable_prefix_contract(model: torch.nn.Module, config: dict[str, Any]) -> None:
    optim_cfg = config.get("optim", {})
    allowed = normalize_string_list(optim_cfg.get("allowed_trainable_prefixes", []))
    required = normalize_string_list(optim_cfg.get("required_trainable_prefixes", []))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if allowed:
        unexpected = [name for name in trainable if not any(name.startswith(prefix) for prefix in allowed)]
        if unexpected:
            preview = "\n".join(f"  - {name}" for name in unexpected[:20])
            raise RuntimeError(f"Trainable parameters violate allowed prefix contract:\n{preview}")
    for prefix in required:
        matches = [name for name in trainable if name.startswith(prefix)]
        if not matches:
            raise RuntimeError(f"Required trainable prefix has no parameters: {prefix!r}")
        print(f"[trainable] required prefix active: {prefix} tensors={len(matches)}")


def validate_required_gradient_prefixes(model: torch.nn.Module, config: dict[str, Any]) -> None:
    prefixes = normalize_string_list(config.get("optim", {}).get("required_gradient_prefixes", []))
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        if not gradients:
            raise RuntimeError(f"Required gradient prefix has no trainable parameters: {prefix!r}")
        finite_nonzero = any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and bool((gradient.detach().abs() > 0).any())
            for gradient in gradients
        )
        if not finite_nonzero:
            raise RuntimeError(f"Required gradient prefix has no finite non-zero gradient: {prefix!r}")
        print(f"[grad] required prefix verified: {prefix}")


def load_initial_checkpoint(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    ckpt_cfg = config.get("checkpoint", {})
    if not ckpt_cfg.get("load_vggt_baseline", False):
        return
    checkpoint_path = require_path(config, "checkpoints.vggt_baseline", allow_empty=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(ckpt_cfg.get("strict", False)))
    print(f"[ckpt] loaded baseline: {checkpoint_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def build_teacher_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module | None:
    teacher_cfg = config.get("teacher", {})
    if not bool(teacher_cfg.get("enabled", False)):
        return None
    checkpoint_path = str(teacher_cfg.get("checkpoint", "") or "")
    if not checkpoint_path:
        raise ValueError("teacher.enabled=true requires teacher.checkpoint")
    teacher_config = copy.deepcopy(config)
    model_overrides = teacher_cfg.get("model_overrides", {})
    if isinstance(model_overrides, dict) and model_overrides:
        teacher_config["model"] = deep_update(teacher_config.get("model", {}), model_overrides)
    teacher = build_model(teacher_config).to(device)
    if bool(teacher_cfg.get("load_vggt_baseline", False)):
        load_initial_checkpoint(teacher, teacher_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = teacher.load_state_dict(state_dict, strict=bool(teacher_cfg.get("strict", False)))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[teacher] loaded frozen HSI teacher: {checkpoint_path}")
    print(f"[teacher] missing={len(missing)} unexpected={len(unexpected)}")
    return teacher


def resume_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    ckpt_cfg = config.get("checkpoint", {})
    skip_prefixes = normalize_string_list(ckpt_cfg.get("resume_skip_prefixes", []))
    if skip_prefixes:
        before = len(state_dict)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in skip_prefixes)
        }
        print(f"[ckpt] skipped_by_prefix={before - len(state_dict)} prefixes={skip_prefixes}")
    strict = bool(ckpt_cfg.get("resume_strict", True))
    if not strict:
        state_dict, shape_report = make_state_dict_loadable(
            state_dict,
            model.state_dict(),
            adapt_query_tensors=bool(ckpt_cfg.get("resume_adapt_query_tensors", True)),
        )
    else:
        shape_report = {"adapted": [], "skipped": []}
    validate_required_checkpoint_prefixes(
        checkpoint_path,
        state_dict,
        model.state_dict(),
        normalize_string_list(ckpt_cfg.get("resume_required_prefixes", [])),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if bool(ckpt_cfg.get("resume_optimizer", True)):
        if "optimizer" not in checkpoint:
            raise ValueError(f"Checkpoint has no optimizer state: {checkpoint_path}")
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[ckpt] resumed model: {checkpoint_path}")
    print(f"[ckpt] resume_missing={len(missing)} resume_unexpected={len(unexpected)} strict={strict}")
    if shape_report["adapted"]:
        print("[ckpt] adapted shape-mismatched tensors:")
        for item in shape_report["adapted"][:20]:
            print(f"  - {item}")
        if len(shape_report["adapted"]) > 20:
            print(f"  ... {len(shape_report['adapted']) - 20} more")
    if shape_report["skipped"]:
        print("[ckpt] skipped shape-mismatched tensors:")
        for item in shape_report["skipped"][:20]:
            print(f"  - {item}")
        if len(shape_report["skipped"]) > 20:
            print(f"  ... {len(shape_report['skipped']) - 20} more")
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def validate_required_checkpoint_prefixes(
    checkpoint_path: str,
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    prefixes: list[str],
) -> None:
    for prefix in prefixes:
        expected = {key: value for key, value in model_state.items() if key.startswith(prefix)}
        if not expected:
            raise ValueError(f"Required checkpoint prefix does not exist in current model: {prefix!r}")
        missing = [
            key
            for key, target in expected.items()
            if key not in checkpoint_state or tuple(checkpoint_state[key].shape) != tuple(target.shape)
        ]
        if missing:
            preview = "\n".join(f"  - {key}" for key in missing[:20])
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incomplete for required prefix {prefix!r}: "
                f"missing_or_mismatched={len(missing)}\n{preview}"
            )
        print(f"[ckpt] required prefix loaded: {prefix} tensors={len(expected)}")


def load_overlay_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    config: dict[str, Any],
) -> None:
    """Overlay selected module prefixes without changing epoch or optimizer."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    ckpt_cfg = config.get("checkpoint", {})
    prefixes = normalize_string_list(ckpt_cfg.get("overlay_prefixes", []))
    required = normalize_string_list(ckpt_cfg.get("overlay_required_prefixes", prefixes))
    if not prefixes:
        raise ValueError("checkpoint.overlay requires checkpoint.overlay_prefixes")
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }
    model_state = model.state_dict()
    state_dict, report = make_state_dict_loadable(state_dict, model_state, adapt_query_tensors=False)
    validate_required_checkpoint_prefixes(checkpoint_path, state_dict, model_state, required)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] overlaid modules: {checkpoint_path}")
    print(
        f"[ckpt] overlay_tensors={len(state_dict)} missing={len(missing)} "
        f"unexpected={len(unexpected)} skipped={len(report['skipped'])}"
    )


def hash_model_prefixes(model: torch.nn.Module, prefixes: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    state = model.state_dict()
    for prefix in prefixes:
        digest = hashlib.sha256()
        keys = sorted(key for key in state if key.startswith(prefix))
        if not keys:
            raise ValueError(f"Frozen hash prefix does not exist in model: {prefix!r}")
        for key in keys:
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        hashes[prefix] = digest.hexdigest()
        print(f"[freeze-hash] {prefix}={hashes[prefix]}")
    return hashes


def assert_model_prefix_hashes(model: torch.nn.Module, expected: dict[str, str]) -> None:
    if not expected:
        return
    current = hash_model_prefixes(model, list(expected))
    changed = [prefix for prefix, digest in expected.items() if current[prefix] != digest]
    if changed:
        raise RuntimeError(f"Frozen model prefixes changed during training: {changed}")


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def make_state_dict_loadable(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    adapt_query_tensors: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    loadable: dict[str, torch.Tensor] = {}
    adapted: list[str] = []
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None or not isinstance(value, torch.Tensor):
            loadable[key] = value
            continue
        if tuple(value.shape) == tuple(target.shape):
            loadable[key] = value
            continue
        adapted_value = None
        if adapt_query_tensors:
            adapted_value = adapt_tensor_by_slicing(value, target)
        if adapted_value is not None:
            loadable[key] = adapted_value
            adapted.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
        else:
            skipped.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
    return loadable, {"adapted": adapted, "skipped": skipped}


def adapt_tensor_by_slicing(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.ndim != target.ndim or source.dtype != target.dtype:
        return None
    slices: list[slice] = []
    changed = False
    for src_size, dst_size in zip(source.shape, target.shape):
        if src_size == dst_size:
            slices.append(slice(None))
        elif src_size > dst_size:
            slices.append(slice(0, dst_size))
            changed = True
        else:
            return None
    if not changed:
        return None
    return source[tuple(slices)].clone()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _strip_module_prefix(checkpoint[key])
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return _strip_module_prefix(checkpoint)
    raise ValueError("Could not find a model state_dict in checkpoint")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    teacher_model: torch.nn.Module | None = None,
    progress_start_time: float | None = None,
    progress_total_steps: int | None = None,
    progress_done_offset: int = 0,
    total_epochs: int | None = None,
) -> tuple[int, dict[str, float]]:
    model.train()
    apply_freeze_policy(model, config)
    log_interval = int(config["optim"].get("log_interval", 10))
    grad_clip_norm = float(config["optim"].get("grad_clip_norm", 0.0))
    log_style = str(config.get("optim", {}).get("log_style", "full")).lower()
    max_steps_per_epoch = int(config["optim"].get("max_steps_per_epoch", 0) or 0)
    totals: dict[str, float] = {}
    count = 0
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_model(model, batch, config, epoch=epoch)
        if teacher_model is not None:
            attach_teacher_predictions(predictions, forward_teacher_model(teacher_model, batch, config))
        losses = criterion(predictions, batch)
        loss = losses["loss_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={step}: {loss.item()}")
        loss.backward()
        if step == 0:
            validate_required_gradient_prefixes(model, config)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        for key, value in scalarize_losses(losses).items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        global_step += 1
        if global_step % log_interval == 0:
            if log_style in {"progress", "compact"}:
                line = format_progress_log(
                    prefix="train",
                    epoch=epoch,
                    total_epochs=total_epochs or int(config["optim"]["epochs"]),
                    step=step,
                    steps=len(loader),
                    global_step=global_step,
                    losses=losses,
                    started_at=progress_start_time,
                    total_steps=progress_total_steps,
                    done_steps=progress_done_offset + step + 1,
                    config=config,
                )
                if log_style == "progress":
                    print("\r\033[K" + line, end="", flush=True)
                else:
                    print(line, flush=True)
            else:
                print(format_log("train", epoch, step, len(loader), global_step, losses), flush=True)
        if max_steps_per_epoch > 0 and count >= max_steps_per_epoch:
            print(f"[train] max_steps_per_epoch reached: {max_steps_per_epoch}", flush=True)
            break
    if log_style == "progress":
        print("", flush=True)
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    if averaged:
        print(format_epoch_summary("train-epoch", epoch, averaged, config), flush=True)
    return global_step, averaged


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    teacher_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    worst_rows: list[dict[str, Any]] = []
    v4_records: list[dict[str, Any]] = []
    max_val_steps = int(config.get("optim", {}).get("max_val_steps", 0) or 0)
    for batch_idx, batch in enumerate(loader):
        if max_val_steps > 0 and batch_idx >= max_val_steps:
            print(f"[val] max_val_steps reached: {max_val_steps}")
            break
        batch = move_to_device(batch, device)
        predictions = forward_model(model, batch, config, epoch=epoch)
        if teacher_model is not None:
            attach_teacher_predictions(predictions, forward_teacher_model(teacher_model, batch, config))
        losses = criterion(predictions, batch)
        v4_records.extend(collect_hsi_v4_validation_records(predictions, batch, config))
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        configured_monitor = str(config.get("checkpoint", {}).get("monitor", ""))
        selection_key = configured_monitor if configured_monitor in losses else "metric_stage2_selection"
        selection = losses.get(selection_key)
        dataset_indices = batch.get("dataset_index")
        if isinstance(selection, torch.Tensor) and isinstance(dataset_indices, torch.Tensor):
            score = float(selection.detach().cpu())
            for dataset_index in dataset_indices.detach().cpu().reshape(-1).tolist():
                worst_rows.append(
                    {
                        "selection_metric": selection_key,
                        "selection_value": score,
                        "dataset_index": int(dataset_index),
                        "sample_path": resolve_dataset_sample_path(loader.dataset, int(dataset_index)),
                    }
                )
        count += 1
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    if v4_records:
        v4_metrics = reduce_hsi_v4_validation_records(v4_records, config)
        averaged.update(v4_metrics)
        write_jsonl(
            v4_records,
            Path(config["experiment"]["output_dir"]) / f"v4_val_people_epoch_{epoch + 1:04d}.jsonl",
        )
        save_json(
            {"epoch": epoch + 1, "num_records": len(v4_records), "metrics": v4_metrics},
            Path(config["experiment"]["output_dir"]) / f"v4_val_metrics_epoch_{epoch + 1:04d}.json",
        )
    val_log_style = str(config.get("optim", {}).get("val_log_style", "compact")).lower()
    if val_log_style in {"full", "verbose"}:
        print(format_log("val", epoch, 0, len(loader), 0, averaged))
    else:
        print(format_epoch_summary("val-epoch", epoch, averaged, config), flush=True)
    worst_rows.sort(key=lambda row: float(row["selection_value"]), reverse=True)
    save_json(
        {"epoch": epoch + 1, "worst_samples": worst_rows[:50]},
        Path(config["experiment"]["output_dir"]) / f"val_worst_samples_epoch_{epoch + 1:04d}.json",
    )
    return averaged


def collect_hsi_v4_validation_records(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate = predictions.get("hsi_v4_candidate_pred_transl_cam")
    base = predictions.get("hsi_v4_base_pred_transl_cam")
    eligible = predictions.get("hsi_v4_geometry_eligible")
    if not all(isinstance(value, torch.Tensor) for value in (candidate, base, eligible)):
        return []
    target = batch.get("gt_transl_cam", batch.get("gt_cam_trans"))
    valid = batch.get("smpl_mask")
    if not isinstance(target, torch.Tensor) or not isinstance(valid, torch.Tensor):
        return []
    valid = valid.bool()
    boxes_mask = batch.get("boxes_mask")
    if isinstance(boxes_mask, torch.Tensor):
        valid = valid & boxes_mask.bool()
    clean = predictions.get("transl_noise_is_clean")
    category = predictions.get("transl_noise_category")
    dataset_indices = batch.get("dataset_index")
    if not isinstance(clean, torch.Tensor):
        clean = torch.linalg.norm(base - target, dim=-1, keepdim=True) <= 1e-8
    if not isinstance(category, torch.Tensor):
        category = torch.full_like(clean, -1, dtype=torch.long)
    if not isinstance(dataset_indices, torch.Tensor):
        dataset_indices = torch.arange(base.shape[0], device=base.device)

    ray = torch.nn.functional.normalize(base, dim=-1, eps=1e-6)
    expected_delta = target.to(dtype=base.dtype) - base
    candidate_delta = candidate - base
    expected_ray = (expected_delta * ray).sum(dim=-1)
    candidate_ray = (candidate_delta * ray).sum(dim=-1)
    expected_tangent = expected_delta - expected_ray[..., None] * ray
    candidate_tangent = candidate_delta - candidate_ray[..., None] * ray
    base_l2 = torch.linalg.norm(expected_delta, dim=-1)
    candidate_l2 = torch.linalg.norm(candidate.to(dtype=base.dtype) - target.to(dtype=base.dtype), dim=-1)
    tangent_base_l1 = expected_tangent.abs().mean(dim=-1)
    tangent_refined_l1 = (candidate_tangent - expected_tangent).abs().mean(dim=-1)
    threshold = float(config.get("loss", {}).get("hsi_v4_active_threshold_m", 0.05))
    noise_names = ("clean", "ray", "tangent", "combined")

    records: list[dict[str, Any]] = []
    for batch_idx, frame_idx, person_idx in valid.nonzero(as_tuple=False).detach().cpu().tolist():
        category_id = int(category[batch_idx, frame_idx, person_idx, 0].detach().cpu())
        expected_ray_value = float(expected_ray[batch_idx, frame_idx, person_idx].detach().cpu())
        candidate_ray_value = float(candidate_ray[batch_idx, frame_idx, person_idx].detach().cpu())
        base_value = float(base_l2[batch_idx, frame_idx, person_idx].detach().cpu())
        candidate_value = float(candidate_l2[batch_idx, frame_idx, person_idx].detach().cpu())
        is_clean = bool(clean[batch_idx, frame_idx, person_idx, 0].detach().cpu() > 0.5)
        records.append(
            {
                "dataset_index": int(dataset_indices[batch_idx].detach().cpu()),
                "frame_offset": int(frame_idx),
                "person_slot": int(person_idx),
                "noise_category": noise_names[category_id] if 0 <= category_id < len(noise_names) else "unknown",
                "clean": is_clean,
                "active": (not is_clean) and base_value >= threshold,
                "eligible": bool(eligible[batch_idx, frame_idx, person_idx, 0].detach().cpu() > 0.5),
                "base_depth_m": float(torch.linalg.norm(base[batch_idx, frame_idx, person_idx]).detach().cpu()),
                "base_l2_m": base_value,
                "candidate_l2_m": candidate_value,
                "improved": candidate_value < base_value,
                "expected_ray_m": expected_ray_value,
                "candidate_ray_m": candidate_ray_value,
                "ray_sign_correct": abs(expected_ray_value) <= 1e-6
                or (
                    abs(candidate_ray_value) > 1e-8
                    and math.copysign(1.0, expected_ray_value) == math.copysign(1.0, candidate_ray_value)
                ),
                "tangent_base_l1_m": float(tangent_base_l1[batch_idx, frame_idx, person_idx].detach().cpu()),
                "tangent_refined_l1_m": float(tangent_refined_l1[batch_idx, frame_idx, person_idx].detach().cpu()),
            }
        )
    return records


def reduce_hsi_v4_validation_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, float]:
    active = [record for record in records if bool(record["active"])]
    if not active:
        return {
            "metric_hsi_v4_geometry_eligibility_coverage": 0.0,
            "metric_hsi_v4_base_active_l2_median": 0.0,
            "metric_hsi_v4_candidate_active_l2_median": 0.0,
            "metric_hsi_v4_candidate_active_l2_p90": 0.0,
            "metric_hsi_v4_candidate_improvement_rate": 0.0,
            "metric_hsi_v4_candidate_ray_sign_acc": 0.0,
            "metric_hsi_v4_candidate_tangent_l1_delta": 0.0,
            "metric_hsi_v4_selection": float("inf"),
        }
    base_values = [float(record["base_l2_m"]) for record in active]
    candidate_values = [float(record["candidate_l2_m"]) for record in active]
    coverage = sum(bool(record["eligible"]) for record in active) / len(active)
    improvement = sum(bool(record["improved"]) for record in active) / len(active)
    ray_items = [record for record in active if abs(float(record["expected_ray_m"])) > 1e-6]
    ray_sign = sum(bool(record["ray_sign_correct"]) for record in ray_items) / max(len(ray_items), 1)
    tangent_delta = sum(
        float(record["tangent_refined_l1_m"]) - float(record["tangent_base_l1_m"])
        for record in active
    ) / len(active)
    base_median = percentile(base_values, 0.50)
    candidate_median = percentile(candidate_values, 0.50)
    candidate_p90 = percentile(candidate_values, 0.90)
    selection = candidate_median + 0.25 * candidate_p90 + 0.5 * (1.0 - coverage)
    return {
        "metric_hsi_v4_geometry_eligibility_coverage": coverage,
        "metric_hsi_v4_base_active_l2_median": base_median,
        "metric_hsi_v4_candidate_active_l2_median": candidate_median,
        "metric_hsi_v4_candidate_active_l2_p90": candidate_p90,
        "metric_hsi_v4_candidate_improvement_rate": improvement,
        "metric_hsi_v4_candidate_ray_sign_acc": ray_sign,
        "metric_hsi_v4_candidate_tangent_l1_delta": tangent_delta,
        "metric_hsi_v4_selection": selection,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def resolve_dataset_sample_path(dataset: Any, dataset_index: int) -> str:
    base = dataset
    resolved_index = int(dataset_index)
    if isinstance(dataset, Subset):
        base = dataset.dataset
    sequences = getattr(base, "_sequences", None)
    index = getattr(base, "_index", None)
    split = getattr(base, "split", "")
    if not isinstance(sequences, list) or not isinstance(index, list) or not (0 <= resolved_index < len(index)):
        return f"dataset_index={dataset_index}"
    seq_idx, start_idx = index[resolved_index]
    seq_dir, frame_ids = sequences[seq_idx]
    return f"{split}/{Path(seq_dir).name}/rgb/{frame_ids[start_idx]}.png"


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    epoch: int = 0,
) -> dict[str, torch.Tensor]:
    model_cfg = config.get("model", {})
    provider = str(model_cfg.get("smpl_provider", "internal")).lower()
    needs_query_inputs = bool(model_cfg.get("smpl_query_box_prior", False)) or provider in {"nlf", "gt_perturbed"}
    if not needs_query_inputs:
        return model(batch["images"])
    boxes = batch.get("smpl_query_boxes", batch.get("gt_boxes"))
    mask = batch.get("smpl_query_boxes_mask", batch.get("boxes_mask"))
    if model.training:
        prior_cfg = config.get("training_prior", {})
        if prior_cfg:
            boxes, mask = make_noisy_box_prior(
                boxes,
                mask,
                center_noise=float(prior_cfg.get("center_noise", 0.0)),
                size_noise=float(prior_cfg.get("size_noise", 0.0)),
                drop_prob=float(prior_cfg.get("drop_prob", 0.0)),
            )
    smpl_override_outputs = (
        build_smpl_override_outputs(batch, config, epoch=epoch, is_training=model.training)
        if should_use_gt_smpl_override(model, config, epoch=epoch)
        else None
    )
    geometry = resolve_hsi_geometry_inputs(
        batch,
        config,
        using_gt_override=smpl_override_outputs is not None,
    )
    predictions = model(
        batch["images"],
        smpl_query_boxes=boxes,
        smpl_query_boxes_mask=mask,
        smpl_query_patch_masks=batch.get("smpl_query_patch_masks"),
        smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
        smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
        external_track_ids=batch.get("external_track_ids"),
        external_track_mask=batch.get("external_track_mask"),
        external_track_confidence=batch.get("external_track_confidence"),
        smpl_override_outputs=smpl_override_outputs,
        hsi_intrinsics_override=geometry["intrinsics"],
        hsi_depth_override=geometry["depth"],
        hsi_depth_is_metric=geometry["depth_is_metric"],
        hsi_geometry_mode=geometry["mode"],
    )
    if smpl_override_outputs is not None:
        predictions["stage2_gt_override_active"] = torch.ones((), device=batch["images"].device)
    return predictions


@torch.no_grad()
def forward_teacher_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    predictions = forward_model(model, batch, config)
    if was_training:
        model.train()
    return predictions


def attach_teacher_predictions(predictions: dict[str, torch.Tensor], teacher_predictions: dict[str, torch.Tensor]) -> None:
    for key in (
        "hsi_refined_pred_pose_6d",
        "hsi_refined_pred_poses",
        "hsi_refined_pred_betas",
        "hsi_refined_pred_transl_cam",
        "hsi_scene_scale",
        "hsi_scene_depth_bias",
    ):
        value = teacher_predictions.get(key)
        if value is not None:
            predictions[f"teacher_{key}"] = value.detach()


def make_noisy_box_prior(
    boxes: torch.Tensor | None,
    mask: torch.Tensor | None,
    center_noise: float,
    size_noise: float,
    drop_prob: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if boxes is None or mask is None:
        return boxes, mask
    noisy = boxes.clone()
    prior_mask = mask.bool().clone()
    if center_noise > 0:
        center_delta = noisy[..., :2].new_empty(noisy[..., :2].shape).uniform_(-center_noise, center_noise)
        noisy[..., :2] = noisy[..., :2] + center_delta * prior_mask[..., None].to(dtype=noisy.dtype)
    if size_noise > 0:
        size_scale = noisy[..., 2:].new_empty(noisy[..., 2:].shape).uniform_(1.0 - size_noise, 1.0 + size_noise)
        noisy[..., 2:] = noisy[..., 2:] * torch.where(prior_mask[..., None], size_scale, torch.ones_like(size_scale))
    if drop_prob > 0:
        keep = torch.rand(prior_mask.shape, device=prior_mask.device) >= drop_prob
        prior_mask = prior_mask & keep
    return noisy.clamp(0.0, 1.0), prior_mask


def should_use_gt_smpl_override(model: torch.nn.Module, config: dict[str, Any], epoch: int = 0) -> bool:
    model_provider = str(config.get("model", {}).get("smpl_provider", "internal")).lower()
    if model_provider == "gt_perturbed":
        return True
    prior_cfg = config.get("training_prior", {})
    schedule = parse_float_schedule(
        prior_cfg.get("smpl_gt_override_prob_schedule", prior_cfg.get("smpl_gt_override_prob", 0.0))
    )
    prob = schedule[min(max(int(epoch), 0), len(schedule) - 1)] if schedule else 0.0
    if prob <= 0.0:
        return False
    if not model.training:
        return False
    return random.random() < min(max(prob, 0.0), 1.0)


def resolve_hsi_geometry_inputs(
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    using_gt_override: bool,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    mode = str(model_cfg.get("hsi_geometry_mode", "") or "").lower()
    if not mode:
        legacy_source = str(model_cfg.get("hsi_camera_source", "vggt") or "vggt").lower()
        if legacy_source == "gt":
            return {
                "mode": "legacy_gt_camera",
                "depth": None,
                "intrinsics": batch.get("K_scal3r"),
                "depth_is_metric": False,
            }
        if legacy_source == "mixed":
            return {
                "mode": "legacy_gt_camera" if using_gt_override else "real_inference",
                "depth": None,
                "intrinsics": batch.get("K_scal3r") if using_gt_override else None,
                "depth_is_metric": False,
            }
        mode = "real_inference"
    if mode == "mixed":
        mode = "gt_metric" if using_gt_override else "real_inference"
    if mode == "gt_metric":
        if not using_gt_override:
            raise ValueError("hsi_geometry_mode='gt_metric' requires a GT SMPL override in the same batch")
        depth = batch.get("gt_depth")
        intrinsics = batch.get("K_scal3r")
        if depth is None or intrinsics is None:
            raise ValueError("gt_metric geometry requires batch['gt_depth'] and batch['K_scal3r']")
        return {"mode": mode, "depth": depth, "intrinsics": intrinsics, "depth_is_metric": True}
    if mode == "real_inference":
        if using_gt_override:
            raise ValueError("GT SMPL override cannot be paired with real_inference geometry")
        return {"mode": mode, "depth": None, "intrinsics": None, "depth_is_metric": False}
    raise ValueError(f"Unsupported model.hsi_geometry_mode: {mode!r}")


def build_smpl_override_outputs(
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    epoch: int = 0,
    is_training: bool = True,
) -> dict[str, torch.Tensor]:
    pose6d = batch["gt_pose_6d"].float()
    betas = batch["gt_betas"].float()
    clean_transl = batch.get("gt_transl_cam", batch["gt_cam_trans"]).float()
    valid = batch.get("smpl_mask")
    if valid is None:
        valid = torch.ones(clean_transl.shape[:-1], dtype=torch.bool, device=clean_transl.device)
    else:
        valid = valid.bool()
    boxes_mask = batch.get("boxes_mask")
    if boxes_mask is not None:
        valid = valid & boxes_mask.bool()
    perturb_mode = str(config.get("training_prior", {}).get("smpl_perturb_mode", "translation") or "translation").lower()
    if perturb_mode == "translation":
        prior_cfg = config.get("training_prior", {})
        noise_contract = str(prior_cfg.get("smpl_translation_noise_contract", "legacy_random") or "legacy_random").lower()
        if noise_contract == "v4_deterministic":
            dataset_indices = batch.get("dataset_index")
            if dataset_indices is None:
                raise ValueError("V4 deterministic translation noise requires batch['dataset_index']")
            noise_epoch = int(epoch) if is_training else int(prior_cfg.get("smpl_v4_validation_epoch", 0) or 0)
            noisy_transl, noise_ratio, tangent_noise, clean_mask, noise_category = apply_deterministic_v4_noise(
                clean_transl,
                valid,
                dataset_indices,
                seed=int(prior_cfg.get("smpl_v4_noise_seed", 42) or 42),
                epoch=noise_epoch,
            )
        elif noise_contract == "legacy_random":
            noisy_transl, noise_ratio, tangent_noise, clean_mask = apply_smpl_translation_noise(
                clean_transl,
                valid,
                config,
                epoch=epoch,
            )
            noise_category = torch.full_like(clean_mask, -1, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported training_prior.smpl_translation_noise_contract: {noise_contract!r}")
        noisy_pose6d = pose6d
        contact_noise_signed = clean_transl.new_zeros(*clean_transl.shape[:-1], 1)
    elif perturb_mode in {"contact_root", "contact_pose"}:
        noisy_pose6d, noisy_transl, contact_noise_signed, clean_mask = apply_smpl_contact_noise(
            pose6d,
            betas,
            clean_transl,
            valid,
            batch,
            config,
            mode=perturb_mode,
        )
        noise_ratio = clean_transl.new_ones(*clean_transl.shape[:-1], 1)
        tangent_noise = clean_transl.new_zeros(*clean_transl.shape[:-1], 2)
        noise_category = torch.full_like(clean_mask, -1, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported training_prior.smpl_perturb_mode: {perturb_mode!r}")
    poses = rot6d_to_axis_angle(noisy_pose6d.reshape(-1, 24, 6)).reshape(*pose6d.shape[:-1], 72)
    conf = valid.to(dtype=pose6d.dtype).unsqueeze(-1)
    boxes = batch.get("smpl_query_boxes", batch.get("gt_boxes"))
    outputs = {
        "pred_pose_6d": noisy_pose6d,
        "pred_poses": poses,
        "pred_betas": betas,
        "pred_transl_cam": noisy_transl,
        "pred_confs": conf,
        "pred_cam": noisy_transl,
        "base_pred_transl_cam": noisy_transl,
        "base_clean_pred_transl_cam": clean_transl,
        "perturbed_pred_transl_cam": noisy_transl,
        "transl_noise_ratio": noise_ratio,
        "transl_noise_tangent_m": tangent_noise,
        "transl_noise_is_clean": clean_mask,
        "transl_noise_category": noise_category,
        "contact_noise_signed_m": contact_noise_signed,
        "base_clean_pred_pose_6d": pose6d,
        "gt_smpl_provider_mask": valid,
    }
    if boxes is not None:
        outputs["pred_boxes"] = boxes.to(device=pose6d.device, dtype=pose6d.dtype)
    return outputs


def apply_smpl_translation_noise(
    transl: torch.Tensor,
    valid: torch.Tensor,
    config: dict[str, Any],
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prior_cfg = config.get("training_prior", {})
    schedule = parse_float_schedule(prior_cfg.get("smpl_transl_ray_noise_schedule", "0.0"))
    ratio = schedule[min(max(int(epoch), 0), len(schedule) - 1)] if schedule else 0.0
    clean_prob = min(max(float(prior_cfg.get("smpl_transl_ray_noise_clean_prob", 0.0) or 0.0), 0.0), 1.0)
    mode = str(prior_cfg.get("smpl_transl_ray_noise_mode", "uniform") or "uniform").lower()
    if ratio <= 0.0:
        ray_scale = torch.ones(*transl.shape[:-1], 1, dtype=transl.dtype, device=transl.device)
    elif mode == "uniform":
        clip_shape = (transl.shape[0], 1, transl.shape[2], 1) if transl.ndim == 4 else (*transl.shape[:-1], 1)
        eps = transl.new_empty(*clip_shape).uniform_(-float(ratio), float(ratio))
        ray_scale = 1.0 + eps.expand(*transl.shape[:-1], 1)
    elif mode in {"normal", "gaussian"}:
        clip_shape = (transl.shape[0], 1, transl.shape[2], 1) if transl.ndim == 4 else (*transl.shape[:-1], 1)
        eps = transl.new_empty(*clip_shape).normal_(mean=0.0, std=float(ratio) / 2.0).clamp(-float(ratio), float(ratio))
        ray_scale = 1.0 + eps.expand(*transl.shape[:-1], 1)
    else:
        raise ValueError(f"Unsupported training_prior.smpl_transl_ray_noise_mode: {mode!r}")
    tangent_schedule = parse_float_schedule(prior_cfg.get("smpl_transl_tangent_noise_schedule_m", "0.0"))
    tangent_max = tangent_schedule[min(max(int(epoch), 0), len(tangent_schedule) - 1)] if tangent_schedule else 0.0
    clip_shape_2 = (transl.shape[0], 1, transl.shape[2], 2) if transl.ndim == 4 else (*transl.shape[:-1], 2)
    tangent_coeff = transl.new_empty(*clip_shape_2).uniform_(-float(tangent_max), float(tangent_max))
    tangent_coeff = tangent_coeff.expand(*transl.shape[:-1], 2)
    if clean_prob > 0.0:
        clean_shape = (transl.shape[0], 1, transl.shape[2], 1) if transl.ndim == 4 else (*transl.shape[:-1], 1)
        noisy_slot = (torch.rand(*clean_shape, device=transl.device) >= clean_prob).expand(*transl.shape[:-1], 1)
    else:
        noisy_slot = torch.ones(*transl.shape[:-1], 1, dtype=torch.bool, device=transl.device)
    valid_f = valid.unsqueeze(-1).to(dtype=transl.dtype)
    ray_scale = 1.0 + (ray_scale - 1.0) * noisy_slot.to(dtype=transl.dtype) * valid_f
    tangent_coeff = tangent_coeff * noisy_slot.to(dtype=transl.dtype) * valid_f
    ray, tangent_x, tangent_y = _translation_camera_basis(transl)
    del ray
    tangent_delta = tangent_coeff[..., :1] * tangent_x + tangent_coeff[..., 1:] * tangent_y
    noisy = transl * ray_scale + tangent_delta
    clean_mask = (~noisy_slot | ~valid.unsqueeze(-1)).to(dtype=transl.dtype)
    return noisy, ray_scale, tangent_coeff, clean_mask


def _translation_camera_basis(transl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ray = torch.nn.functional.normalize(transl, dim=-1, eps=1e-6)
    x_axis = torch.zeros_like(ray)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(ray)
    y_axis[..., 1] = 1.0
    tangent_x = x_axis - (x_axis * ray).sum(dim=-1, keepdim=True) * ray
    fallback = y_axis - (y_axis * ray).sum(dim=-1, keepdim=True) * ray
    tangent_x = torch.where(torch.linalg.norm(tangent_x, dim=-1, keepdim=True) > 1e-4, tangent_x, fallback)
    tangent_x = torch.nn.functional.normalize(tangent_x, dim=-1, eps=1e-6)
    tangent_y = torch.nn.functional.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1, eps=1e-6)
    return ray, tangent_x, tangent_y


def apply_smpl_contact_noise(
    pose6d: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    valid: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prior_cfg = config.get("training_prior", {})
    teacher_valid = batch.get("contact_teacher_valid")
    contact_label = batch.get("contact_label")
    plane_normal = batch.get("contact_plane_normal_cam")
    if teacher_valid is None or contact_label is None or plane_normal is None:
        raise ValueError(f"{mode} perturbation requires contact teacher sidecars in the batch")
    active_foot = teacher_valid.bool() & contact_label.bool()
    active_person = active_foot.any(dim=-1) & valid
    normal_weights = active_foot.to(dtype=transl.dtype)
    support_normal = (plane_normal.to(dtype=transl.dtype) * normal_weights[..., None]).sum(dim=-2)
    support_normal = support_normal / normal_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    support_normal = torch.nn.functional.normalize(support_normal, dim=-1, eps=1e-6)

    clean_prob = min(max(float(prior_cfg.get("smpl_contact_noise_clean_prob", 0.20)), 0.0), 1.0)
    clip_shape = (transl.shape[0], 1, transl.shape[2], 1)
    noisy_slot = (torch.rand(*clip_shape, device=transl.device) >= clean_prob).expand(*transl.shape[:-1], 1)
    noisy_slot = noisy_slot & active_person.unsqueeze(-1)
    requested_noisy_slot = noisy_slot.clone()
    noisy_pose = pose6d.clone()
    noisy_transl = transl.clone()
    signed_noise = transl.new_zeros(*transl.shape[:-1], 1)

    use_root = mode == "contact_root" or bool(prior_cfg.get("smpl_contact_pose_include_root_noise", True))
    if use_root:
        float_levels = parse_float_schedule(prior_cfg.get("smpl_contact_float_levels_m", "0.02,0.05,0.08,0.12"))
        penetration_levels = parse_float_schedule(prior_cfg.get("smpl_contact_penetration_levels_m", "0.01,0.02,0.04,0.06"))
        levels = transl.new_tensor(float_levels + [-value for value in penetration_levels])
        selection = torch.randint(0, int(levels.numel()), clip_shape[:-1], device=transl.device)
        offset = levels[selection].unsqueeze(-1).expand(*transl.shape[:-1], 1)
        offset = offset * noisy_slot.to(dtype=transl.dtype)
        noisy_transl = noisy_transl + support_normal * offset
        signed_noise = offset

    if mode == "contact_pose":
        noisy_pose, pose_changed = _sample_contact_pose_noise(
            pose6d,
            betas,
            noisy_transl,
            transl,
            active_person & noisy_slot[..., 0],
            batch,
            config,
        )
        if not use_root:
            noisy_slot = noisy_slot & pose_changed.unsqueeze(-1)
        else:
            noisy_slot = requested_noisy_slot
    clean_mask = (~noisy_slot | ~valid.unsqueeze(-1)).to(dtype=transl.dtype)
    return noisy_pose, noisy_transl, signed_noise, clean_mask


def _sample_contact_pose_noise(
    clean_pose6d: torch.Tensor,
    betas: torch.Tensor,
    noisy_transl: torch.Tensor,
    clean_transl: torch.Tensor,
    active_person: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_cfg = config.get("training_prior", {})
    max_deg = clean_pose6d.new_tensor(
        [
            float(prior_cfg.get("smpl_contact_hip_noise_deg", 4.0)),
            float(prior_cfg.get("smpl_contact_hip_noise_deg", 4.0)),
            float(prior_cfg.get("smpl_contact_knee_noise_deg", 8.0)),
            float(prior_cfg.get("smpl_contact_knee_noise_deg", 8.0)),
            float(prior_cfg.get("smpl_contact_ankle_noise_deg", 10.0)),
            float(prior_cfg.get("smpl_contact_ankle_noise_deg", 10.0)),
        ]
    ) * (math.pi / 180.0)
    lower = torch.tensor([1, 2, 4, 5, 7, 8], device=clean_pose6d.device, dtype=torch.long)
    output = clean_pose6d.clone()
    accepted = torch.zeros_like(active_person)
    attempts = max(int(prior_cfg.get("smpl_contact_pose_rejection_attempts", 4)), 1)
    min_delta = float(prior_cfg.get("smpl_contact_pose_min_effect_m", 0.01))
    max_delta = float(prior_cfg.get("smpl_contact_pose_max_effect_m", 0.10))
    clean_foot = _decode_sole_centers(clean_pose6d, betas, clean_transl, config, person_mask=active_person)
    plane_normal = batch["contact_plane_normal_cam"].to(dtype=clean_pose6d.dtype)
    contact_active = batch["contact_teacher_valid"].bool() & batch["contact_label"].bool()
    for _ in range(attempts):
        base_rot = rot6d_to_rotmat(clean_pose6d.reshape(-1, 24, 6)).reshape(*clean_pose6d.shape[:-1], 24, 3, 3)
        clip_noise = clean_pose6d.new_empty(clean_pose6d.shape[0], 1, clean_pose6d.shape[2], 6, 3).uniform_(-1.0, 1.0)
        clip_noise = clip_noise * max_deg.reshape(1, 1, 1, 6, 1)
        delta_rot = axis_angle_to_rotmat(clip_noise.expand(*clean_pose6d.shape[:3], 6, 3))
        candidate_rot = base_rot.clone()
        candidate_rot[..., lower, :, :] = delta_rot @ base_rot[..., lower, :, :]
        candidate = candidate_rot[..., :2, :].reshape_as(clean_pose6d)
        unresolved = active_person & ~accepted
        candidate_foot = _decode_sole_centers(candidate, betas, noisy_transl, config, person_mask=unresolved)
        displacement = ((candidate_foot - clean_foot) * plane_normal).sum(dim=-1).abs()
        effect = torch.where(contact_active, displacement, torch.zeros_like(displacement)).amax(dim=-1)
        take = active_person & ~accepted & (effect >= min_delta) & (effect <= max_delta)
        output = torch.where(take[..., None], candidate, output)
        accepted = accepted | take
    return output, accepted


def _decode_sole_centers(
    pose6d: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    config: dict[str, Any],
    person_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    model_dir = str(config.get("assets", {}).get("smpl_model_dir", ""))
    key = (model_dir, str(pose6d.device))
    smpl = _SMPL_NOISE_CACHE.get(key)
    if smpl is None:
        smpl = SMPLLayer(model_dir).to(pose6d.device).eval()
        for parameter in smpl.parameters():
            parameter.requires_grad = False
        _SMPL_NOISE_CACHE[key] = smpl
    flat_pose = pose6d.reshape(-1, 144)
    flat_betas = betas.reshape(-1, betas.shape[-1])
    flat_transl = transl.reshape(-1, 3)
    select = (
        person_mask.reshape(-1).bool()
        if person_mask is not None
        else torch.ones(flat_pose.shape[0], dtype=torch.bool, device=pose6d.device)
    )
    output = pose6d.new_zeros(flat_pose.shape[0], 2, 3)
    if not select.any():
        return output.reshape(*pose6d.shape[:3], 2, 3)
    aa = rot6d_to_axis_angle(flat_pose[select].reshape(-1, 24, 6)).reshape(-1, 72)
    with torch.no_grad():
        vertices, _ = smpl(aa.float(), flat_betas[select].float())
    sole_idx = build_sole_vertex_indices(smpl.layer.v_template.detach(), 48).to(pose6d.device)
    centers = vertices[:, sole_idx].mean(dim=-2).to(dtype=pose6d.dtype) + flat_transl[select, None, :]
    output[select] = centers
    return output.reshape(*pose6d.shape[:3], 2, 3)


def parse_float_schedule(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    text = str(value or "0.0").strip()
    if not text:
        return [0.0]
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have key=value format, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = parse_override_value(raw_value)
    return updated


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def format_log(prefix: str, epoch: int, step: int, steps: int, global_step: int, losses: dict[str, Any]) -> str:
    loss_items = []
    for key, value in losses.items():
        scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        if math.isfinite(scalar):
            loss_items.append(f"{key}={scalar:.6f}")
    return f"[{prefix}] epoch={epoch + 1} step={step + 1}/{steps} global_step={global_step} " + " ".join(loss_items)


def format_epoch_summary(prefix: str, epoch: int, losses: dict[str, float], config: dict[str, Any]) -> str:
    keys = get_val_log_keys(config) if prefix.startswith("val") else get_progress_log_keys(config)
    max_items = int(config.get("optim", {}).get("progress_max_loss_items", 8))
    items: list[str] = []
    for key in keys:
        if key in losses and math.isfinite(float(losses[key])):
            items.append(f"{compact_loss_name(key)}={float(losses[key]):.6g}")
        if len(items) >= max_items:
            break
    if "loss_total" not in keys and "loss_total" in losses and len(items) < max_items:
        items.insert(0, f"total={float(losses['loss_total']):.6g}")
    return f"[{prefix}] epoch={epoch + 1} " + " ".join(items)


def format_progress_log(
    prefix: str,
    epoch: int,
    total_epochs: int,
    step: int,
    steps: int,
    global_step: int,
    losses: dict[str, Any],
    started_at: float | None,
    total_steps: int | None,
    done_steps: int,
    config: dict[str, Any],
) -> str:
    total = max(int(total_steps or steps), 1)
    done = min(max(int(done_steps), 0), total)
    ratio = done / total
    elapsed = max(time.monotonic() - started_at, 1e-6) if started_at is not None else 0.0
    speed = done / elapsed if elapsed > 0 and done > 0 else 0.0
    remaining = (total - done) / speed if speed > 0 else math.inf
    bar = make_progress_bar(ratio, width=int(config.get("optim", {}).get("progress_bar_width", 24)))
    scalars = scalarize_losses(losses)
    loss_text = format_compact_loss_items(scalars, config)
    eta_text = format_duration(remaining) if math.isfinite(remaining) else "--:--:--"
    elapsed_text = format_duration(elapsed)
    return (
        f"[{prefix}] ep {epoch + 1}/{total_epochs} step {step + 1}/{steps} gs={global_step} "
        f"{bar} {ratio * 100.0:5.1f}% {speed:.2f}it/s elapsed={elapsed_text} eta={eta_text} "
        f"{loss_text}"
    )


def scalarize_losses(losses: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in losses.items():
        try:
            scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            scalars[key] = scalar
    return scalars


def format_compact_loss_items(scalars: dict[str, float], config: dict[str, Any]) -> str:
    keys = get_progress_log_keys(config)
    max_items = int(config.get("optim", {}).get("progress_max_loss_items", 8))
    items: list[str] = []
    for key in keys:
        if key not in scalars:
            continue
        value = scalars[key]
        short_key = compact_loss_name(key)
        items.append(f"{short_key}={value:.4g}")
        if len(items) >= max_items:
            break
    if "loss_total" not in keys and "loss_total" in scalars and len(items) < max_items:
        items.insert(0, f"total={scalars['loss_total']:.4g}")
    return " ".join(items)


def get_progress_log_keys(config: dict[str, Any]) -> list[str]:
    raw = config.get("optim", {}).get("progress_log_keys")
    if isinstance(raw, str) and raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [
        "loss_total",
        "loss_hsi_smpl_scale_teacher",
        "loss_hsi_depth_teacher",
        "loss_hsi_teacher_scene_affine",
        "loss_hsi_transl_cam",
        "loss_hsi_joints3d",
        "loss_hsi_vertices",
        "loss_hsi_teacher_transl",
        "loss_hsi_teacher_joints",
        "loss_hsi_transl_velocity",
        "loss_hsi_joints_velocity",
        "loss_hsi_joints_acceleration",
        "loss_hsi_temporal_no_worse",
        "loss_hsi_foot_sliding",
        "loss_hsi_scene_scale_temporal",
        "loss_hsi_scene_bias_temporal",
        "metric_hsi_scene_log_scale_delta",
        "metric_hsi_scene_bias_delta",
        "metric_hsi_smpl_scale_teacher_valid_points",
        "metric_hsi_smpl_scale_teacher_scale",
        "metric_hsi_smpl_scale_teacher_pred_scale",
        "metric_hsi_smpl_scale_teacher_l1",
        "metric_hsi_smpl_scale_teacher_rel_l1",
        "metric_hsi_base_transl_l1",
        "metric_hsi_refined_transl_l1",
        "metric_hsi_transl_l1_delta",
        "loss_hsi_ray_delta",
        "loss_hsi_align_point",
        "loss_hsi_align_delta_reg",
        "loss_hsi_align_no_worse",
        "metric_hsi_ray_delta_base_l1",
        "metric_hsi_ray_delta_refined_l1",
        "metric_hsi_ray_delta_l1_delta",
        "metric_hsi_ray_delta_sign_acc",
        "metric_hsi_align_base_point_l1",
        "metric_hsi_align_refined_point_l1",
        "metric_hsi_align_point_l1_delta",
        "metric_hsi_align_delta_l1",
        "metric_hsi_align_gate_mean",
        "metric_hsi_align_valid_ratio",
    ]


def get_val_log_keys(config: dict[str, Any]) -> list[str]:
    raw = config.get("optim", {}).get("val_log_keys")
    if isinstance(raw, str) and raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return get_progress_log_keys(config)


def compact_loss_name(key: str) -> str:
    mapping = {
        "loss_total": "total",
        "loss_hsi_depth_teacher": "depth",
        "loss_hsi_smpl_scale_teacher": "smplScale",
        "loss_hsi_anchor_depth": "anchorD",
        "loss_hsi_teacher_scene_affine": "affineT",
        "loss_hsi_transl_cam": "transl",
        "loss_hsi_joints3d": "j3d",
        "loss_hsi_vertices": "verts",
        "loss_hsi_no_worse": "noWorse",
        "loss_hsi_teacher_transl": "translT",
        "loss_hsi_teacher_joints": "jointsT",
        "loss_hsi_teacher_vertices": "vertsT",
        "loss_hsi_transl_velocity": "velT",
        "loss_hsi_joints_velocity": "velJ",
        "loss_hsi_joints_acceleration": "accJ",
        "loss_hsi_temporal_no_worse": "tmpNoWorse",
        "metric_hsi_temporal_no_worse_ratio": "tmpWorse",
        "metric_hsi_temporal_no_worse_l1": "tmpExcess",
        "loss_hsi_foot_sliding": "footSlide",
        "loss_hsi_foot_sole_contact": "sole",
        "loss_hsi_support_plane_contact": "plane",
        "loss_hsi_contact": "contact",
        "loss_hsi_scene_scale_temporal": "scaleTmp",
        "loss_hsi_scene_scale_sequence": "scaleSeq",
        "loss_hsi_scene_bias_temporal": "biasTmp",
        "loss_hsi_scene_bias_sequence": "biasSeq",
        "metric_hsi_scene_log_scale_delta": "dLogS",
        "metric_hsi_scene_bias_delta": "dBias",
        "metric_hsi_smpl_scale_teacher_valid_points": "smplPts",
        "metric_hsi_smpl_scale_teacher_scale": "scaleT",
        "metric_hsi_smpl_scale_teacher_pred_scale": "scaleP",
        "metric_hsi_smpl_scale_teacher_l1": "scaleL1",
        "metric_hsi_smpl_scale_teacher_log_l1": "scaleLog",
        "metric_hsi_smpl_scale_teacher_rel_l1": "scaleRel",
        "metric_hsi_base_transl_l1": "hsiBaseT",
        "metric_hsi_refined_transl_l1": "hsiRefT",
        "metric_hsi_transl_l1_delta": "hsiDT",
        "loss_hsi_ray_delta": "ray",
        "metric_hsi_ray_delta_l1": "rayL1",
        "metric_hsi_ray_delta_base_l1": "rayBase",
        "metric_hsi_ray_delta_refined_l1": "rayRef",
        "metric_hsi_ray_delta_l1_delta": "rayDT",
        "metric_hsi_ray_delta_expected_abs": "rayExp",
        "metric_hsi_ray_delta_pred_abs": "rayPred",
        "metric_hsi_ray_delta_sign_acc": "raySign",
        "loss_hsi_align_point": "align",
        "loss_hsi_align_delta_reg": "alignReg",
        "loss_hsi_align_no_worse": "alignNoW",
        "metric_hsi_align_base_point_l1": "alignBase",
        "metric_hsi_align_refined_point_l1": "alignRef",
        "metric_hsi_align_point_l1_delta": "alignDT",
        "metric_hsi_align_delta_l1": "alignDelta",
        "metric_hsi_align_gate_mean": "alignGate",
        "metric_hsi_align_valid_ratio": "alignValid",
        "loss_hsi_contact_refine_plane": "contactPlane",
        "loss_hsi_contact_refine_class": "contactCls",
        "loss_hsi_contact_refine_no_worse": "contactNW",
        "loss_hsi_contact_refine_swing_no_pull": "swingPull",
        "metric_hsi_contact_float_p95_m": "floatP95",
        "metric_hsi_contact_penetration_p95_m": "penetrP95",
        "metric_hsi_contact_false_pull_rate": "falsePull",
        "metric_hsi_contact_contact_gate_mean": "contactGate",
        "metric_hsi_contact_swing_gate_mean": "swingGate",
        "metric_hsi_contact_base_abs_p95_m": "baseP95",
        "metric_hsi_contact_refined_abs_p95_m": "refinedP95",
        "metric_hsi_contact_swing_displacement_mean_m": "swingDisp",
        "metric_hsi_contact_temporal_velocity_valid_rate": "velValid",
        "metric_hsi_contact_temporal_velocity_median": "velMedian",
        "loss_hsi_grounding_gate": "groundGate",
        "metric_hsi_grounding_valid_coverage": "groundValid",
        "metric_hsi_grounding_apply_target_rate": "groundTarget",
        "metric_hsi_grounding_gate_mean": "groundProb",
        "metric_hsi_grounding_gate_accuracy": "groundAcc",
        "metric_hsi_grounding_base_l2_p95_m": "groundBaseP95",
        "metric_hsi_grounding_candidate_l2_p95_m": "groundCandP95",
        "metric_hsi_grounding_refined_l2_p95_m": "groundRefP95",
        "metric_hsi_grounding_improvement_rate": "groundImprove",
        "metric_hsi_grounding_float_p95_m": "groundFloatP95",
        "metric_hsi_grounding_penetration_p95_m": "groundPenP95",
        "metric_hsi_grounding_clean_displacement_p95_m": "groundCleanP95",
        "loss_transl_refine_delta_reg": "tDeltaReg",
        "loss_local_joints3d": "localJ",
        "loss_local_vertices": "localV",
        "loss_transl_refine_ray_depth": "rayD",
        "loss_transl_refine_tangent": "tan",
        "loss_transl_temporal_velocity": "tVel",
        "loss_transl_temporal_acceleration": "tAcc",
        "loss_transl_temporal_no_worse": "tNoWorse",
        "metric_base_transl_l1": "baseT",
        "metric_refined_transl_l1": "refT",
        "metric_transl_refine_l1_delta": "dT",
        "metric_transl_box_prior_weight_abs": "boxPriorW",
        "metric_transl_ray_depth_l1_delta": "dRay",
        "metric_transl_tangent_l1_delta": "dTan",
        "metric_transl_temporal_no_worse_ratio": "tWorse",
        "metric_transl_temporal_no_worse_l1": "tExcess",
    }
    return mapping.get(key, key.removeprefix("loss_").removeprefix("metric_"))


def make_progress_bar(ratio: float, width: int = 24) -> str:
    width = max(width, 8)
    filled = int(round(max(0.0, min(1.0, ratio)) * width))
    return "[" + "=" * filled + "." * (width - filled) + "]"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds_i = int(seconds)
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = config.get("checkpoint", {})
    state_dict = build_checkpoint_state_dict(model, config)
    payload: dict[str, Any] = {
        "model": state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "checkpoint_format": {
            "save_scope": str(ckpt_cfg.get("save_scope", "full")),
            "save_prefixes": normalize_string_list(ckpt_cfg.get("save_prefixes", [])),
            "num_model_tensors": len(state_dict),
        },
    }
    if bool(ckpt_cfg.get("save_optimizer", False)):
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"[ckpt] saved: {path} tensors={len(state_dict)} scope={payload['checkpoint_format']['save_scope']}")


def build_checkpoint_state_dict(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    ckpt_cfg = config.get("checkpoint", {})
    scope = str(ckpt_cfg.get("save_scope", "full") or "full").lower()
    full_state = model.state_dict()
    if scope == "full":
        return full_state
    if scope == "hsi":
        prefixes = normalize_string_list(ckpt_cfg.get("save_prefixes", ["hsi_refinement_head."]))
        if not prefixes:
            prefixes = ["hsi_refinement_head."]
        return {
            key: value
            for key, value in full_state.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        }
    if scope == "trainable":
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        return {key: value for key, value in full_state.items() if key in trainable_names}
    raise ValueError(f"Unsupported checkpoint.save_scope: {scope!r}. Expected full, hsi, or trainable.")


def init_topk_state(output_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if int(config.get("checkpoint", {}).get("save_top_k", 0) or 0) <= 0:
        return []
    index_path = output_dir / "checkpoint_topk_index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entries = data.get("entries", []) if isinstance(data, dict) else []
    state: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        if path.exists():
            state.append(
                {
                    "metric": float(item.get("metric", math.inf)),
                    "epoch": int(item.get("epoch", 0)),
                    "path": path,
                    "source": str(item.get("source", "unknown")),
                }
            )
    return state


def maybe_save_topk_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    output_dir: Path,
    topk_state: list[dict[str, Any]],
    losses: dict[str, float],
    source: str,
) -> None:
    ckpt_cfg = config.get("checkpoint", {})
    top_k = int(ckpt_cfg.get("save_top_k", 0) or 0)
    if top_k <= 0:
        return
    monitor = str(ckpt_cfg.get("monitor", "loss_total"))
    mode = str(ckpt_cfg.get("monitor_mode", "min")).lower()
    if monitor not in losses:
        print(f"[ckpt] top-k skipped: monitor={monitor!r} not found in {source} losses")
        return
    metric = float(losses[monitor])
    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    if len(topk_state) >= top_k and not better(metric, float(topk_state[-1]["metric"])):
        return
    safe_monitor = monitor.replace("/", "_").replace("\\", "_")
    raw_path = output_dir / f"checkpoint_top_{source}_epoch_{epoch:04d}_{safe_monitor}_{metric:.6f}.pt"
    save_checkpoint(model, optimizer, epoch, global_step, config, raw_path)
    topk_state.append({"metric": metric, "epoch": int(epoch), "path": raw_path, "source": source})
    topk_state.sort(key=lambda item: float(item["metric"]), reverse=(mode == "max"))
    while len(topk_state) > top_k:
        removed = topk_state.pop()
        path = Path(removed["path"])
        if path.exists():
            path.unlink()
    save_topk_index(output_dir, monitor, mode, topk_state)
    if not bool(ckpt_cfg.get("topk_create_stable_copies", False)):
        for rank, item in enumerate(topk_state, start=1):
            print(
                f"[ckpt] top{rank:02d} {source} {monitor}={float(item['metric']):.6f}: {item['path']}"
            )
        return
    for rank, item in enumerate(topk_state, start=1):
        stable_path = output_dir / f"checkpoint_top{rank:02d}.pt"
        shutil.copyfile(item["path"], stable_path)
        print(f"[ckpt] top{rank:02d} {monitor}={float(item['metric']):.6f}: {stable_path}")


def save_topk_index(output_dir: Path, monitor: str, mode: str, topk_state: list[dict[str, Any]]) -> None:
    payload = {
        "monitor": monitor,
        "monitor_mode": mode,
        "entries": [
            {
                "rank": rank,
                "metric": float(item["metric"]),
                "epoch": int(item["epoch"]),
                "source": str(item.get("source", "unknown")),
                "path": str(item["path"]),
            }
            for rank, item in enumerate(topk_state, start=1)
        ],
    }
    save_json(payload, output_dir / "checkpoint_topk_index.json")


def should_save_checkpoint(config: dict[str, Any], epoch: int, total_epochs: int) -> bool:
    save_interval = int(config.get("optim", {}).get("save_interval", 1) or 0)
    interval_due = save_interval > 0 and epoch % save_interval == 0
    final_due = bool(config.get("checkpoint", {}).get("save_final", False)) and epoch == int(total_epochs)
    return interval_due or final_due


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
