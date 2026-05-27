import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, require_path
from vggt_omega.utils.pose_enc import encoding_to_camera


COLORS = [
    (255, 64, 64),
    (64, 192, 255),
    (64, 255, 128),
    (255, 192, 64),
    (192, 64, 255),
    (255, 64, 192),
    (128, 255, 255),
    (255, 128, 128),
]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config["model"]["enable_camera"] = True
    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    image_path = Path(args.image).expanduser()
    input_size = int(config["data"].get("image_size", args.image_size))
    image_tensor, orig_image = load_image(image_path, input_size)
    with torch.no_grad():
        predictions = model(image_tensor.to(device))
    results = collect_predictions(predictions, orig_image.size, args.conf_threshold, args.top_k)
    if args.draw_smpl_joints:
        add_projected_smpl_joints(results, predictions, config, args, orig_image.size, input_size, device)

    out_image = output_dir / f"{image_path.stem}_smpl_predictions.jpg"
    out_json = output_dir / f"{image_path.stem}_smpl_predictions.json"
    draw_predictions(orig_image, results, out_image)
    out_json.write_text(json.dumps({"image": str(image_path), "checkpoint": str(args.checkpoint), "predictions": results}, indent=2), encoding="utf-8")
    print(json.dumps({"output_image": str(out_image), "output_json": str(out_json), "num_predictions": len(results)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one-image SMPL query inference results")
    parser.add_argument("--image", required=True, help="Input RGB image path")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint, e.g. outputs/train/.../checkpoint_latest.pt")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/smpl_inference")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--draw-smpl-joints", action="store_true", help="Decode SMPL and draw projected joints")
    parser.add_argument("--smpl-model-dir", default="", help="Override assets.smpl_model_dir")
    parser.add_argument("--baseline-checkpoint", default="", help="Override checkpoints.vggt_baseline for loading VGGT camera head")
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.key=value")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    path_config = load_yaml_config(args.path_config)
    train_config = load_yaml_config(args.train_config)
    config = deep_update(path_config, train_config)
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    config.setdefault("data", {})["image_size"] = int(config.get("data", {}).get("image_size", args.image_size))
    config.setdefault("model", {})["enable_smpl"] = True
    return config


def load_vggt_baseline_for_camera(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_path = require_path(config, "checkpoints.vggt_baseline", allow_empty=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded VGGT baseline for camera: {checkpoint_path}")
    print(f"[ckpt] baseline missing={len(missing)} unexpected={len(unexpected)}")


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        raise ValueError(f"Training checkpoint missing 'model' state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()}, strict=False)
    print(f"[ckpt] loaded training checkpoint: {checkpoint_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return {name.removeprefix("module."): value for name, value in checkpoint[key].items()}
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return {name.removeprefix("module."): value for name, value in checkpoint.items()}
    raise ValueError("Could not find a model state_dict in checkpoint")


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
    return tensor, image


def collect_predictions(
    predictions: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    conf_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if "pred_boxes" not in predictions:
        raise ValueError("Model predictions missing pred_boxes; set model.predict_boxes=true")
    width, height = image_size
    confs = predictions["pred_confs"][0, 0, :, 0].detach().float().cpu()
    boxes = predictions["pred_boxes"][0, 0].detach().float().cpu().clamp(0.0, 1.0)
    transl_cam = predictions.get("pred_transl_cam")
    transl = transl_cam[0, 0].detach().float().cpu() if transl_cam is not None else None
    order = torch.argsort(confs, descending=True)
    results = []
    for rank, query_idx in enumerate(order[: max(top_k, 0)].tolist()):
        conf = float(confs[query_idx])
        if conf < conf_threshold:
            continue
        cx, cy, bw, bh = [float(v) for v in boxes[query_idx]]
        xyxy = [
            (cx - 0.5 * bw) * width,
            (cy - 0.5 * bh) * height,
            (cx + 0.5 * bw) * width,
            (cy + 0.5 * bh) * height,
        ]
        item: dict[str, Any] = {
            "rank": rank,
            "query_index": query_idx,
            "confidence": conf,
            "bbox_cxcywh_norm": [cx, cy, bw, bh],
            "bbox_xyxy_pixels": xyxy,
        }
        if transl is not None:
            item["pred_transl_cam"] = [float(v) for v in transl[query_idx]]
        results.append(item)
    return results


def add_projected_smpl_joints(
    results: list[dict[str, Any]],
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    image_size: tuple[int, int],
    input_size: int,
    device: torch.device,
) -> None:
    if not results:
        return
    smpl_model_dir = args.smpl_model_dir or str(config.get("assets", {}).get("smpl_model_dir", ""))
    if not smpl_model_dir:
        raise ValueError("SMPL model dir is required. Set assets.smpl_model_dir or pass --smpl-model-dir")
    for key in ("pose_enc", "pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in predictions:
            raise ValueError(f"SMPL projection requires model output {key}")

    query_indices = [int(item["query_index"]) for item in results]
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    poses = predictions["pred_poses"][0, 0, index_tensor].detach()
    betas = predictions["pred_betas"][0, 0, index_tensor].detach()
    transl_cam = predictions["pred_transl_cam"][0, 0, index_tensor].detach()

    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    with torch.no_grad():
        _, joints = smpl(poses.reshape(-1, 72), betas)
    joints_cam = joints + transl_cam[:, None, :]
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], image_size_hw=(input_size, input_size), build_intrinsics=True)
    intrinsics_0 = intrinsics[0, 0].to(device=device, dtype=joints_cam.dtype)
    joints_2d_input = project_points(joints_cam, intrinsics_0)
    in_front = joints_cam[..., 2] > 1e-4
    width, height = image_size
    scale = joints_2d_input.new_tensor([float(width) / float(input_size), float(height) / float(input_size)])
    joints_2d = joints_2d_input * scale
    in_image = (
        in_front
        & (joints_2d[..., 0] >= 0)
        & (joints_2d[..., 0] < float(width))
        & (joints_2d[..., 1] >= 0)
        & (joints_2d[..., 1] < float(height))
    )
    for item, points, mask in zip(results, joints_2d.detach().cpu(), in_image.detach().cpu(), strict=True):
        item["projected_joints_2d"] = points.tolist()
        item["projected_joints_mask"] = mask.tolist()
        item["vggt_intrinsics_input"] = intrinsics_0.detach().cpu().tolist()
        item["vggt_input_size"] = input_size
        item["vggt_camera_from_world"] = extrinsics[0, 0].detach().cpu().tolist()


def project_points(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    z = points_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[0, 0] * points_cam[..., 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points_cam[..., 1] / z + intrinsics[1, 2]
    return torch.stack([x, y], dim=-1)


def draw_predictions(image: Image.Image, predictions: list[dict[str, Any]], output_path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, pred in enumerate(predictions):
        color = COLORS[idx % len(COLORS)]
        box = [float(v) for v in pred["bbox_xyxy_pixels"]]
        draw.rectangle(box, outline=color, width=4)
        transl = pred.get("pred_transl_cam")
        transl_text = ""
        if isinstance(transl, list) and len(transl) == 3:
            transl_text = f" t=({transl[0]:.2f},{transl[1]:.2f},{transl[2]:.2f})"
        label = f"q{pred['query_index']} conf={pred['confidence']:.2f}{transl_text}"
        draw_label(draw, font, (box[0], max(0.0, box[1] - 18.0)), label, color)
        draw_projected_joints(draw, pred, color)
    draw_label(draw, font, (8, 8), f"predictions={len(predictions)}", (255, 255, 255), fill=(0, 0, 0))
    canvas.save(output_path, quality=95)


def draw_projected_joints(draw: ImageDraw.ImageDraw, pred: dict[str, Any], color: tuple[int, int, int]) -> None:
    joints = pred.get("projected_joints_2d")
    masks = pred.get("projected_joints_mask")
    if not isinstance(joints, list) or not isinstance(masks, list):
        return
    for joint_idx, xy in enumerate(joints[:24]):
        if joint_idx >= len(masks) or not masks[joint_idx]:
            continue
        x, y = float(xy[0]), float(xy[1])
        radius = 3
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0))


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    xy: tuple[float, float],
    text: str,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] = (32, 32, 32),
) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=fill, outline=outline, width=1)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


if __name__ == "__main__":
    main()
