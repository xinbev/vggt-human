from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import (  # noqa: E402
    apply_freeze_policy,
    apply_overrides,
    build_loader,
    build_model,
    forward_model,
    load_initial_checkpoint,
    move_to_device,
)
from vggt_omega.training.config import deep_update, load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real VGGT+NLF+HSI forward pass and check output contract.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/debug/nlf_hsi_forward_smoke")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    config.setdefault("optim", {})["batch_size"] = int(config.get("optim", {}).get("batch_size", 1))
    config.setdefault("data", {})["num_workers"] = int(config.get("data", {}).get("num_workers", 0))
    config.setdefault("data", {})["pin_memory"] = False
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    apply_freeze_policy(model, config)
    model.eval()
    loader = build_loader(config, split=config["data"]["train_split"], shuffle=False)

    summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(args.max_batches):
                break
            batch = move_to_device(batch, device)
            predictions = forward_model(model, batch, config)
            summary = summarize_forward(batch_idx, batch, predictions)
            summaries.append(summary)
            assert_contract(predictions)

    if not summaries:
        raise RuntimeError("No batches were processed by the NLF+HSI smoke test")
    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps({"batches": summaries}, indent=2), encoding="utf-8")
    print("[ok] NLF HSI forward smoke passed")
    print(json.dumps({"summary": str(out_json)}, indent=2))


def summarize_forward(batch_idx: int, batch: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
    images = batch["images"]
    pred_transl = predictions["pred_transl_cam"]
    refined_transl = predictions.get("hsi_refined_pred_transl_cam")
    scene_scale = predictions.get("hsi_scene_scale")
    summary: dict[str, Any] = {
        "batch_idx": int(batch_idx),
        "image_shape": list(images.shape),
        "image_hw": list(images.shape[-2:]),
        "pred_transl_cam_shape": list(pred_transl.shape),
        "pred_transl_cam_finite": bool(torch.isfinite(pred_transl).all().detach().cpu()),
        "pred_confs_shape": list(predictions["pred_confs"].shape),
        "nlf_image_hw": predictions.get("nlf_image_hw", torch.empty(0, device=images.device)).detach().cpu().tolist(),
    }
    if refined_transl is not None:
        summary["hsi_refined_pred_transl_cam_shape"] = list(refined_transl.shape)
        summary["hsi_refined_pred_transl_cam_finite"] = bool(torch.isfinite(refined_transl).all().detach().cpu())
    if scene_scale is not None:
        summary["hsi_scene_scale_shape"] = list(scene_scale.shape)
        summary["hsi_scene_scale_finite"] = bool(torch.isfinite(scene_scale).all().detach().cpu())
    return summary


def assert_contract(predictions: dict[str, torch.Tensor]) -> None:
    required = (
        "pred_poses",
        "pred_pose_6d",
        "pred_betas",
        "pred_transl_cam",
        "pred_confs",
        "pred_boxes",
        "base_pred_transl_cam",
        "nlf_intrinsics",
        "nlf_image_hw",
        "hsi_refined_pred_poses",
        "hsi_refined_pred_pose_6d",
        "hsi_refined_pred_betas",
        "hsi_refined_pred_transl_cam",
        "hsi_scene_scale",
        "hsi_scene_depth_bias",
    )
    missing = [key for key in required if key not in predictions]
    if missing:
        raise AssertionError(f"Missing required prediction keys: {missing}")
    for key in required:
        value = predictions[key]
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"Prediction {key} contains non-finite values")
    if predictions["pred_transl_cam"].shape[-1] != 3:
        raise AssertionError(f"pred_transl_cam must end with 3, got {tuple(predictions['pred_transl_cam'].shape)}")
    if predictions["pred_pose_6d"].shape[-1] != 144:
        raise AssertionError(f"pred_pose_6d must end with 144, got {tuple(predictions['pred_pose_6d'].shape)}")


if __name__ == "__main__":
    main()
