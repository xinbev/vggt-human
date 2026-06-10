#!/usr/bin/env python
"""Compare two VGGT-Omega SMPL/HSI checkpoints.

This is a lightweight checkpoint diagnostic that does not import project modules.
It is meant for server-side debugging when a later checkpoint suddenly produces
bad visualization results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    args = parse_args()
    before = load_checkpoint(args.before)
    after = load_checkpoint(args.after)
    before_state = extract_state_dict(before)
    after_state = extract_state_dict(after)

    print("========== checkpoint meta ==========")
    print_meta("before", args.before, before)
    print_meta("after ", args.after, after)

    common_keys = sorted(set(before_state) & set(after_state))
    missing_after = sorted(set(before_state) - set(after_state))
    new_after = sorted(set(after_state) - set(before_state))
    print("\n========== key summary ==========")
    print(f"common_keys={len(common_keys)} missing_after={len(missing_after)} new_after={len(new_after)}")
    if missing_after:
        print("missing_after sample:", missing_after[:10])
    if new_after:
        print("new_after sample:", new_after[:10])

    tensor_rows = []
    prefix_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"delta_sq": 0.0, "before_sq": 0.0, "after_sq": 0.0, "numel": 0.0, "max_abs_delta": 0.0})
    for key in common_keys:
        a = before_state[key]
        b = after_state[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape != b.shape or not a.is_floating_point():
            continue
        af = a.detach().float().cpu()
        bf = b.detach().float().cpu()
        delta = bf - af
        delta_norm = float(torch.linalg.norm(delta))
        before_norm = float(torch.linalg.norm(af))
        after_norm = float(torch.linalg.norm(bf))
        max_abs_delta = float(delta.abs().max()) if delta.numel() else 0.0
        rel = delta_norm / max(before_norm, 1e-12)
        tensor_rows.append((delta_norm, rel, max_abs_delta, before_norm, after_norm, key))

        prefix = classify_prefix(key)
        stats = prefix_stats[prefix]
        stats["delta_sq"] += delta_norm * delta_norm
        stats["before_sq"] += before_norm * before_norm
        stats["after_sq"] += after_norm * after_norm
        stats["numel"] += float(delta.numel())
        stats["max_abs_delta"] = max(stats["max_abs_delta"], max_abs_delta)

    print("\n========== prefix drift ==========")
    rows = []
    for prefix, stats in prefix_stats.items():
        delta_norm = stats["delta_sq"] ** 0.5
        before_norm = stats["before_sq"] ** 0.5
        after_norm = stats["after_sq"] ** 0.5
        rel = delta_norm / max(before_norm, 1e-12)
        rows.append((delta_norm, rel, stats["max_abs_delta"], before_norm, after_norm, int(stats["numel"]), prefix))
    for delta_norm, rel, max_abs_delta, before_norm, after_norm, numel, prefix in sorted(rows, reverse=True):
        print(
            f"{prefix:42s} delta_norm={delta_norm:.6g} rel={rel:.6g} "
            f"max_abs_delta={max_abs_delta:.6g} before_norm={before_norm:.6g} after_norm={after_norm:.6g} numel={numel}"
        )

    print(f"\n========== top {args.top_k} changed tensors ==========")
    for delta_norm, rel, max_abs_delta, before_norm, after_norm, key in sorted(tensor_rows, reverse=True)[: args.top_k]:
        print(
            f"{key:90s} delta_norm={delta_norm:.6g} rel={rel:.6g} "
            f"max_abs_delta={max_abs_delta:.6g} before_norm={before_norm:.6g} after_norm={after_norm:.6g}"
        )

    print("\n========== HSI affine head tensors ==========")
    for key in common_keys:
        if "hsi_refinement_head.scale_delta" in key or "hsi_refinement_head.bias_delta" in key:
            describe_pair(key, before_state[key], after_state[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two VGGT-Omega checkpoints")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {str(k).removeprefix("module."): v for k, v in value.items()}
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}
    raise ValueError("Could not find model state_dict in checkpoint")


def print_meta(label: str, path: Path, checkpoint: Any) -> None:
    if isinstance(checkpoint, dict):
        print(f"{label}: path={path} epoch={checkpoint.get('epoch')} global_step={checkpoint.get('global_step')}")
    else:
        print(f"{label}: path={path} non-dict checkpoint")


def classify_prefix(key: str) -> str:
    if key.startswith("hsi_refinement_head.scale_delta"):
        return "hsi_refinement_head.scale_delta"
    if key.startswith("hsi_refinement_head.bias_delta"):
        return "hsi_refinement_head.bias_delta"
    if key.startswith("hsi_refinement_head.transl_delta"):
        return "hsi_refinement_head.transl_delta"
    if key.startswith("hsi_refinement_head.pose_delta"):
        return "hsi_refinement_head.pose_delta"
    if key.startswith("hsi_refinement_head.betas_delta"):
        return "hsi_refinement_head.betas_delta"
    if key.startswith("hsi_refinement_head.contact_head"):
        return "hsi_refinement_head.contact_head"
    if key.startswith("hsi_refinement_head.blocks"):
        return "hsi_refinement_head.blocks"
    if key.startswith("hsi_refinement_head.scene_projs"):
        return "hsi_refinement_head.scene_projs"
    if key.startswith("hsi_refinement_head.token_mlp"):
        return "hsi_refinement_head.token_mlp"
    if key.startswith("smpl_head"):
        return "smpl_head"
    if key.startswith("aggregator.smpl_"):
        return "aggregator.smpl_prior_embeddings"
    if key.startswith("aggregator"):
        return "aggregator.frozen"
    if key.startswith("camera_head"):
        return "camera_head.frozen"
    if key.startswith("dense_head"):
        return "dense_head.frozen"
    return key.split(".", 1)[0]


def describe_pair(key: str, before: torch.Tensor, after: torch.Tensor) -> None:
    if not before.is_floating_point() or before.shape != after.shape:
        return
    a = before.detach().float().cpu()
    b = after.detach().float().cpu()
    d = b - a
    print(key)
    print(
        "  before "
        f"mean={float(a.mean()):.6g} std={float(a.std(unbiased=False)):.6g} "
        f"min={float(a.min()):.6g} max={float(a.max()):.6g} norm={float(torch.linalg.norm(a)):.6g}"
    )
    print(
        "  after  "
        f"mean={float(b.mean()):.6g} std={float(b.std(unbiased=False)):.6g} "
        f"min={float(b.min()):.6g} max={float(b.max()):.6g} norm={float(torch.linalg.norm(b)):.6g}"
    )
    print(
        "  delta  "
        f"mean={float(d.mean()):.6g} std={float(d.std(unbiased=False)):.6g} "
        f"min={float(d.min()):.6g} max={float(d.max()):.6g} norm={float(torch.linalg.norm(d)):.6g}"
    )


if __name__ == "__main__":
    main()
