import argparse
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
from vggt_omega.models import VGGTOmega
from vggt_omega.training import HungarianSMPLLoss, HungarianSMPLMatcher, SMPLSlotLoss
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


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

    train_loader = build_loader(config, split=config["data"]["train_split"], shuffle=True)
    val_loader = None
    if config["data"].get("val_split"):
        try:
            val_loader = build_loader(config, split=config["data"]["val_split"], shuffle=False)
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
        global_step = train_one_epoch(
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
        if val_loader is not None and (epoch + 1) % int(config["optim"].get("val_interval", 1)) == 0:
            val_losses = validate(model, criterion, val_loader, device, epoch, config, teacher_model=teacher_model)
            maybe_save_topk_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir, topk_state, val_losses)
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


def build_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "bedlam")).lower()
    if dataset_name in {"3dpw", "threedpw"}:
        return build_3dpw_loader(config, split=split, shuffle=shuffle)
    if dataset_name in {"hf_bedlam", "hfb_edlam", "bedlam_hf"}:
        return build_hf_bedlam_loader(config, split=split, shuffle=shuffle)
    if dataset_name != "bedlam":
        raise ValueError(f"Unsupported data.dataset: {data_cfg.get('dataset')!r}")
    root = require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"))
    boxes_root = None
    if data_cfg.get("boxes_root_key"):
        boxes_root = require_path(config, data_cfg["boxes_root_key"], allow_empty=not bool(data_cfg.get("require_boxes", False)))
    dataset = BedlamDataset(
        root=root,
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg["image_size"]),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_depth=bool(data_cfg.get("require_depth", False)),
        boxes_root=boxes_root,
        require_boxes=bool(data_cfg.get("require_boxes", False)),
        query_source=str(data_cfg.get("query_source", "persons")),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=shuffle,
    )


def build_3dpw_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = ThreeDPWDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.threedpw_root")),
        annotation_root=require_path(config, data_cfg.get("annotation_root_key", "datasets.threedpw_smpl_base_root")),
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg["image_size"]),
        max_humans=int(data_cfg.get("max_humans", 2)),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=threedpw_collate_fn,
        drop_last=shuffle,
    )


def build_hf_bedlam_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = HFBedlamDataset(
        images_root=require_path(config, data_cfg.get("images_root_key", "datasets.hf_bedlam_images_root")),
        npz_root=require_path(config, data_cfg.get("npz_root_key", "datasets.hf_bedlam_npz_root")),
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg["image_size"]),
        max_humans=int(data_cfg.get("max_humans", 20)),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        bbox_expand=float(data_cfg.get("bbox_expand", 0.15)),
        transl_add_cam_ext=bool(data_cfg.get("transl_add_cam_ext", True)),
        skip_missing_images=bool(data_cfg.get("skip_missing_images", True)),
        max_npz_files=int(data_cfg.get("max_npz_files", 0) or 0),
        max_frames=int(data_cfg.get("max_frames", 0) or 0),
    )
    dataset = maybe_subset_dataset(dataset, data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=hf_bedlam_collate_fn,
        drop_last=shuffle,
    )


def maybe_subset_dataset(dataset: Any, data_cfg: dict[str, Any], split: str) -> Any | Subset:
    train_split = str(data_cfg.get("train_split", split))
    if split != train_split and not bool(data_cfg.get("subset_apply_to_val", False)):
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
        smpl_model_dir=str(config.get("assets", {}).get("smpl_model_dir", "")),
        image_size=int(config.get("data", {}).get("image_size", 518)),
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
        if "projection_image_size" not in loss_cfg:
            loss_cfg["projection_image_size"] = int(config.get("data", {}).get("image_size", 518))
        match_cfg = config.get("matching", {})
        matcher = HungarianSMPLMatcher(
            cost_conf=float(match_cfg.get("cost_conf", 1.0)),
            cost_bbox=float(match_cfg.get("cost_bbox", 5.0)),
            cost_giou=float(match_cfg.get("cost_giou", 2.0)),
            cost_kpts=float(match_cfg.get("cost_kpts", 0.0)),
            j2ds_norm_scale=float(config.get("data", {}).get("image_size", 518)),
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
            raise ValueError("train_smpl_translation_decode_heads=true requires model.smpl_translation_output_mode=ray_offset_depth")
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
    strict = bool(ckpt_cfg.get("resume_strict", True))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if bool(ckpt_cfg.get("resume_optimizer", True)):
        if "optimizer" not in checkpoint:
            raise ValueError(f"Checkpoint has no optimizer state: {checkpoint_path}")
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[ckpt] resumed model: {checkpoint_path}")
    print(f"[ckpt] resume_missing={len(missing)} resume_unexpected={len(unexpected)} strict={strict}")
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


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
) -> int:
    model.train()
    apply_freeze_policy(model, config)
    log_interval = int(config["optim"].get("log_interval", 10))
    grad_clip_norm = float(config["optim"].get("grad_clip_norm", 0.0))
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_model(model, batch, config)
        if teacher_model is not None:
            attach_teacher_predictions(predictions, forward_teacher_model(teacher_model, batch, config))
        losses = criterion(predictions, batch)
        loss = losses["loss_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={step}: {loss.item()}")
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        global_step += 1
        if global_step % log_interval == 0:
            log_style = str(config.get("optim", {}).get("log_style", "full")).lower()
            if log_style in {"progress", "compact"}:
                print(
                    format_progress_log(
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
                    ),
                    flush=True,
                )
            else:
                print(format_log("train", epoch, step, len(loader), global_step, losses), flush=True)
    return global_step


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
    for batch in loader:
        batch = move_to_device(batch, device)
        predictions = forward_model(model, batch, config)
        if teacher_model is not None:
            attach_teacher_predictions(predictions, forward_teacher_model(teacher_model, batch, config))
        losses = criterion(predictions, batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    print(format_log("val", epoch, 0, len(loader), 0, averaged))
    return averaged


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not bool(config.get("model", {}).get("smpl_query_box_prior", False)):
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
    return model(
        batch["images"],
        smpl_query_boxes=boxes,
        smpl_query_boxes_mask=mask,
        smpl_query_patch_masks=batch.get("smpl_query_patch_masks"),
        smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
        smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
        external_track_ids=batch.get("external_track_ids"),
        external_track_mask=batch.get("external_track_mask"),
        external_track_confidence=batch.get("external_track_confidence"),
    )


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
    ]


def compact_loss_name(key: str) -> str:
    mapping = {
        "loss_total": "total",
        "loss_hsi_depth_teacher": "depth",
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
        "loss_transl_refine_delta_reg": "tDeltaReg",
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
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        path,
    )
    print(f"[ckpt] saved: {path}")


def init_topk_state(output_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    _ = output_dir
    _ = config
    return []


def maybe_save_topk_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    output_dir: Path,
    topk_state: list[dict[str, Any]],
    val_losses: dict[str, float],
) -> None:
    ckpt_cfg = config.get("checkpoint", {})
    top_k = int(ckpt_cfg.get("save_top_k", 0) or 0)
    if top_k <= 0:
        return
    monitor = str(ckpt_cfg.get("monitor", "loss_total"))
    mode = str(ckpt_cfg.get("monitor_mode", "min")).lower()
    if monitor not in val_losses:
        print(f"[ckpt] top-k skipped: monitor={monitor!r} not found in validation losses")
        return
    metric = float(val_losses[monitor])
    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    if len(topk_state) >= top_k and not better(metric, float(topk_state[-1]["metric"])):
        return
    raw_path = output_dir / f"checkpoint_top_epoch_{epoch:04d}_{monitor}_{metric:.6f}.pt"
    save_checkpoint(model, optimizer, epoch, global_step, config, raw_path)
    topk_state.append({"metric": metric, "epoch": int(epoch), "path": raw_path})
    topk_state.sort(key=lambda item: float(item["metric"]), reverse=(mode == "max"))
    while len(topk_state) > top_k:
        removed = topk_state.pop()
        path = Path(removed["path"])
        if path.exists():
            path.unlink()
    for rank, item in enumerate(topk_state, start=1):
        stable_path = output_dir / f"checkpoint_top{rank:02d}.pt"
        shutil.copyfile(item["path"], stable_path)
        print(f"[ckpt] top{rank:02d} {monitor}={float(item['metric']):.6f}: {stable_path}")


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
