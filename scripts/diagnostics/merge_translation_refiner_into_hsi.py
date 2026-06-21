#!/usr/bin/env python
"""Merge a depth-free SMPL translation checkpoint back into a full HSI checkpoint.

The translation repair stage is usually trained with ``enable_hsi_refine=false``
to avoid raw-depth/HSI coupling. Its checkpoint therefore does not contain HSI
head weights. This utility starts from the stable full HSI checkpoint and
overlays only the selected SMPL translation repair keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


TRANSLATION_REFINER_PREFIXES = (
    "smpl_head.regression_head.translation_refiner.",
)
TRANSLATION_HEAD_PREFIXES = (
    "smpl_head.regression_head.transl_cam_heads.",
)
BOX_HEAD_PREFIXES = (
    "smpl_head.regression_head.box_heads.",
    "smpl_head.regression_head.box_delta_heads.",
)
QUERY_PRIOR_PREFIXES = (
    "aggregator.smpl_query_token",
    "aggregator.smpl_box_prior_embed.",
    "aggregator.smpl_patch_pool_embed.",
)


def main() -> None:
    args = parse_args()
    hsi_ckpt = torch.load(args.hsi_checkpoint, map_location="cpu")
    translation_ckpt = torch.load(args.translation_checkpoint, map_location="cpu")
    hsi_state = extract_state_dict(hsi_ckpt)
    translation_state = extract_state_dict(translation_ckpt)

    prefixes = list(TRANSLATION_REFINER_PREFIXES)
    if args.include_translation_heads:
        prefixes.extend(TRANSLATION_HEAD_PREFIXES)
    if args.include_box_heads:
        prefixes.extend(BOX_HEAD_PREFIXES)
    if args.include_query_priors:
        prefixes.extend(QUERY_PRIOR_PREFIXES)

    merged = dict(hsi_state)
    copied: list[str] = []
    skipped_shape: list[str] = []
    for key, value in translation_state.items():
        if not key.startswith(tuple(prefixes)):
            continue
        if key in merged and tuple(merged[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        merged[key] = value.detach().cpu()
        copied.append(key)

    if not copied:
        raise RuntimeError(
            "No translation repair keys were copied. Check that the translation "
            "checkpoint was trained with model.smpl_enable_translation_refine=true."
        )

    output: dict[str, Any] = {}
    if isinstance(hsi_ckpt, dict):
        for key, value in hsi_ckpt.items():
            if key not in {"model", "state_dict", "model_state_dict", "optimizer"}:
                output[key] = value
    output["model"] = merged
    output["epoch"] = choose_epoch(args.epoch_source, hsi_ckpt, translation_ckpt)
    output["global_step"] = choose_global_step(args.epoch_source, hsi_ckpt, translation_ckpt)
    output["merged_from"] = {
        "hsi_checkpoint": str(args.hsi_checkpoint),
        "translation_checkpoint": str(args.translation_checkpoint),
        "copied_key_count": len(copied),
        "include_translation_heads": bool(args.include_translation_heads),
        "include_box_heads": bool(args.include_box_heads),
        "include_query_priors": bool(args.include_query_priors),
        "skipped_shape_count": len(skipped_shape),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"[merge] hsi checkpoint         : {args.hsi_checkpoint}")
    print(f"[merge] translation checkpoint : {args.translation_checkpoint}")
    print(f"[merge] output                 : {args.output}")
    print(f"[merge] copied keys            : {len(copied)}")
    print(f"[merge] skipped shape mismatch : {len(skipped_shape)}")
    if args.print_keys:
        for key in copied:
            print(f"[merge:key] {key}")
        for key in skipped_shape:
            print(f"[merge:skip-shape] {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SMPL translation repair weights into a full HSI checkpoint")
    parser.add_argument("--hsi-checkpoint", type=Path, required=True)
    parser.add_argument("--translation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-translation-heads", action="store_true")
    parser.add_argument("--include-box-heads", action="store_true")
    parser.add_argument("--include-query-priors", action="store_true")
    parser.add_argument("--epoch-source", choices=("hsi", "translation", "max"), default="translation")
    parser.add_argument("--print-keys", action="store_true")
    return parser.parse_args()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {name.removeprefix("module."): tensor for name, tensor in value.items() if torch.is_tensor(tensor)}
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return {name.removeprefix("module."): tensor for name, tensor in checkpoint.items()}
    raise ValueError("Could not find a model state_dict in checkpoint")


def choose_epoch(source: str, hsi_ckpt: Any, translation_ckpt: Any) -> int:
    hsi_epoch = checkpoint_int(hsi_ckpt, "epoch")
    translation_epoch = checkpoint_int(translation_ckpt, "epoch")
    if source == "hsi":
        return hsi_epoch
    if source == "max":
        return max(hsi_epoch, translation_epoch)
    return translation_epoch


def choose_global_step(source: str, hsi_ckpt: Any, translation_ckpt: Any) -> int:
    hsi_step = checkpoint_int(hsi_ckpt, "global_step")
    translation_step = checkpoint_int(translation_ckpt, "global_step")
    if source == "hsi":
        return hsi_step
    if source == "max":
        return max(hsi_step, translation_step)
    return translation_step


def checkpoint_int(checkpoint: Any, key: str) -> int:
    if isinstance(checkpoint, dict):
        try:
            return int(checkpoint.get(key, 0))
        except (TypeError, ValueError):
            return 0
    return 0


if __name__ == "__main__":
    main()
