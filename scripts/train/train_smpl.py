import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
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

from vggt_omega.data import BedlamDataset, bedlam_collate_fn
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

    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    apply_freeze_policy(model, config)
    criterion = build_criterion(config).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters after applying freeze policy")
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

    epochs = int(config["optim"]["epochs"])
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
        )
        if val_loader is not None and (epoch + 1) % int(config["optim"].get("val_interval", 1)) == 0:
            validate(model, criterion, val_loader, device, epoch, config)
        if (epoch + 1) % int(config["optim"].get("save_interval", 1)) == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
            save_checkpoint(model, optimizer, epoch + 1, global_step, config, output_dir / "checkpoint_latest.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGGT-Omega SMPL query head on BEDLAM-style data")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.key=value")
    return parser.parse_args()


def build_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
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
    )
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=shuffle,
    )


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
        smpl_query_box_prior=bool(model_cfg.get("smpl_query_box_prior", False)),
        smpl_query_patch_pool=bool(model_cfg.get("smpl_query_patch_pool", False)),
        smpl_query_patch_pool_expand=float(model_cfg.get("smpl_query_patch_pool_expand", 0.10)),
        enable_hsi_refine=bool(model_cfg.get("enable_hsi_refine", False)),
        hsi_hidden_dim=int(model_cfg.get("hsi_hidden_dim", 512)),
        hsi_num_layers=int(model_cfg.get("hsi_num_layers", 5)),
        hsi_num_heads=int(model_cfg.get("hsi_num_heads", 8)),
        hsi_num_iters=int(model_cfg.get("hsi_num_iters", 3)),
        hsi_scene_window=int(model_cfg.get("hsi_scene_window", 3)),
        hsi_use_delta_gate=bool(model_cfg.get("hsi_use_delta_gate", False)),
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
        if bool(model_cfg.get("train_hsi_transl_only", False)):
            for name in ("pose_delta", "betas_delta", "contact_head"):
                module = getattr(hsi_head, name, None)
                if module is not None:
                    freeze_module(module)
        if bool(model_cfg.get("train_hsi_scene_affine_only", False)):
            for name in ("pose_delta", "betas_delta", "transl_delta", "contact_head"):
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
) -> int:
    model.train()
    apply_freeze_policy(model, config)
    log_interval = int(config["optim"].get("log_interval", 10))
    grad_clip_norm = float(config["optim"].get("grad_clip_norm", 0.0))
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_model(model, batch, config)
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
            print(format_log("train", epoch, step, len(loader), global_step, losses))
    return global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
) -> None:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        losses = criterion(forward_model(model, batch, config), batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    print(format_log("val", epoch, 0, len(loader), 0, averaged))


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not bool(config.get("model", {}).get("smpl_query_box_prior", False)):
        return model(batch["images"])
    boxes = batch.get("gt_boxes")
    mask = batch.get("boxes_mask")
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
    )


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
