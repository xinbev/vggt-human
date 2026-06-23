import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
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

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, require_path
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import axis_angle_to_rot6d, rot6d_to_axis_angle


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
    if args.export_scene_ply:
        config["model"]["enable_depth"] = True
    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    image_path = Path(args.image).expanduser()
    input_size = int(config["data"].get("image_size", args.image_size))
    image_tensor, orig_image = load_image(image_path, input_size)
    prior_boxes, prior_mask = load_gt_box_prior(image_path, config, args, device) if args.use_gt_box_prior else (None, None)
    if prior_boxes is not None and prior_mask is not None:
        prior_boxes, prior_mask = make_noisy_gt_box_prior(
            prior_boxes,
            prior_mask,
            args.gt_box_prior_center_noise,
            args.gt_box_prior_size_noise,
            args.gt_box_prior_drop_prob,
        )
    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=prior_boxes,
            smpl_query_boxes_mask=prior_mask,
        )
    results = collect_predictions(predictions, orig_image.size, args.conf_threshold, args.top_k)
    if args.draw_smpl_joints:
        add_projected_smpl_joints(results, predictions, config, args, orig_image.size, input_size, device)
    if args.draw_gt_smpl_joints:
        add_projected_gt_smpl_joints(results, image_path, predictions, config, args, orig_image.size, input_size, device)
    scene_alignment: dict[str, Any] | None = None
    ply_files = []
    if args.export_ply:
        ply_files = export_prediction_gt_ply(results, image_path, predictions, config, args, output_dir, device)
    if args.export_scene_ply:
        scene_export = export_scene_ply(results, image_tensor, image_path, predictions, config, args, output_dir, input_size, device)
        ply_files.extend(scene_export["ply_files"])
        scene_alignment = scene_export.get("scene_alignment")
        hsi_scene_alignment = scene_export.get("hsi_scene_alignment")
    else:
        hsi_scene_alignment = None

    out_image = output_dir / f"{image_path.stem}_smpl_predictions.jpg"
    out_json = output_dir / f"{image_path.stem}_smpl_predictions.json"
    draw_predictions(orig_image, results, out_image)
    out_json.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "checkpoint": str(args.checkpoint),
                "use_gt_box_prior": bool(args.use_gt_box_prior),
                "gt_box_prior_center_noise": float(args.gt_box_prior_center_noise),
                "gt_box_prior_size_noise": float(args.gt_box_prior_size_noise),
                "gt_box_prior_drop_prob": float(args.gt_box_prior_drop_prob),
                "ply_files": ply_files,
                "scene_alignment": scene_alignment,
                "hsi_scene_alignment": hsi_scene_alignment,
                "predictions": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
    parser.add_argument("--draw-gt-smpl-joints", action="store_true", help="Draw GT SMPL joints for BEDLAM images")
    parser.add_argument("--use-gt-box-prior", action="store_true", help="Load BEDLAM GT/preprocessed boxes for oracle query priors")
    parser.add_argument("--gt-box-prior-center-noise", type=float, default=0.0, help="Uniform cx/cy noise range for GT box priors, normalized units")
    parser.add_argument("--gt-box-prior-size-noise", type=float, default=0.0, help="Uniform relative w/h noise range for GT box priors")
    parser.add_argument("--gt-box-prior-drop-prob", type=float, default=0.0, help="Probability of dropping a valid GT box prior")
    parser.add_argument("--smpl-model-dir", default="", help="Override assets.smpl_model_dir")
    parser.add_argument("--baseline-checkpoint", default="", help="Override checkpoints.vggt_baseline for loading VGGT camera head")
    parser.add_argument("--export-ply", action="store_true", help="Export one multi-person SMPL mesh PLY per image")
    parser.add_argument("--export-scene-ply", action="store_true", help="Export VGGT dense environment point cloud with predicted SMPL meshes")
    parser.add_argument("--align-scene-to-smpl", action="store_true", help="Scale VGGT scene depth/points to SMPL metric depth using visible SMPL anchors")
    parser.add_argument("--align-min-anchor-pixels", type=int, default=64, help="Minimum valid SMPL/depth anchor pixels required for scene alignment")
    parser.add_argument("--align-scale-min", type=float, default=0.25, help="Minimum allowed scene-to-SMPL scale")
    parser.add_argument("--align-scale-max", type=float, default=20.0, help="Maximum allowed scene-to-SMPL scale")
    parser.add_argument("--align-anchor-stride", type=int, default=8, help="Use every Nth SMPL vertex as a depth anchor candidate")
    parser.add_argument(
        "--align-use-gt-smpl-anchors",
        action="store_true",
        help="Use BEDLAM GT SMPL vertices as oracle scale anchors when available",
    )
    parser.add_argument("--ply-top-k", type=int, default=3, help="Maximum ranked predictions to include in mesh PLY exports")
    parser.add_argument("--use-hsi-refined", action="store_true", help="Use HSI refined SMPL outputs when available")
    parser.add_argument("--export-hsi-comparison", action="store_true", help="Export base and HSI refined mesh comparison PLYs")
    parser.add_argument("--export-pre-refine-comparison", action="store_true", help="Export base_pred_transl_cam mesh before the camera-ray translation refiner")
    parser.add_argument("--export-translation-debug-json", action="store_true", help="Export per-query pre/post/HSI/GT translation values and L2 errors")
    parser.add_argument("--export-translation-only-comparison", action="store_true", help="Export same-pose meshes at pre/post/HSI/GT translations to isolate root translation")
    parser.add_argument("--hsi-align-scene", action="store_true", help="Export scene mesh from HSI affine depth s*depth+b")
    parser.add_argument("--ply-use-vertices", action="store_true", help="Deprecated; mesh export is always enabled when --export-ply is set")
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


def load_gt_box_prior(
    image_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_cfg = config.get("data", {})
    max_humans = int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20)))
    boxes_root = require_path(config, data_cfg.get("boxes_root_key", "datasets.bedlam_boxes_root"), allow_empty=False)
    dataset_root = require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)
    split = str(data_cfg.get("train_split", "Training"))
    box_path = _bedlam_box_path(image_path, Path(dataset_root), Path(boxes_root), split)
    boxes = torch.zeros(1, 1, max_humans, 4, dtype=torch.float32, device=device)
    mask = torch.zeros(1, 1, max_humans, dtype=torch.bool, device=device)
    with box_path.open("rb") as file:
        data = pickle.load(file)
    persons = data.get("persons") if isinstance(data, dict) else None
    if not isinstance(persons, list):
        raise TypeError(f"Preprocessed bbox annotation must contain a persons list: {box_path}")
    for person_idx, person in enumerate(persons[:max_humans]):
        if bool(person.get("bbox_valid", False)):
            boxes[0, 0, person_idx] = torch.as_tensor(person["bbox_cxcywh_norm"], dtype=torch.float32, device=device).reshape(4).clamp(0.0, 1.0)
            mask[0, 0, person_idx] = True
    return boxes, mask


def make_noisy_gt_box_prior(
    boxes: torch.Tensor,
    mask: torch.Tensor,
    center_noise: float,
    size_noise: float,
    drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _bedlam_box_path(image_path: Path, dataset_root: Path, boxes_root: Path, split: str) -> Path:
    image_path = image_path.resolve()
    rgb_dir = image_path.parent
    if rgb_dir.name != "rgb":
        raise ValueError(f"--use-gt-box-prior expects a BEDLAM rgb image path, got: {image_path}")
    sequence_dir = rgb_dir.parent
    split_root = (dataset_root / split).resolve()
    try:
        sequence_rel = sequence_dir.resolve().relative_to(split_root)
    except ValueError as exc:
        raise ValueError(f"Image path is not under BEDLAM split root {split_root}: {image_path}") from exc
    box_path = boxes_root / split / sequence_rel / "smpl_boxes" / f"{image_path.stem}.pkl"
    if not box_path.is_file():
        raise FileNotFoundError(f"GT box prior file not found: {box_path}")
    return box_path


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
    transl_by_key = {}
    for key in ("base_pred_transl_cam", "pred_transl_cam", "hsi_refined_pred_transl_cam"):
        value = predictions.get(key)
        if isinstance(value, torch.Tensor):
            transl_by_key[key] = value[0, 0].detach().float().cpu()
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
        for key, transl in transl_by_key.items():
            item[key] = [float(v) for v in transl[query_idx]]
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
    for key in ("pose_enc", "pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in predictions:
            raise ValueError(f"SMPL projection requires model output {key}")

    smpl_model_dir = require_smpl_model_dir(config, args)
    pred_points = decode_pred_smpl_points(results, predictions, smpl_model_dir, device)
    joints_cam = pred_points["joints"]
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


def add_projected_gt_smpl_joints(
    results: list[dict[str, Any]],
    image_path: Path,
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    image_size: tuple[int, int],
    input_size: int,
    device: torch.device,
) -> None:
    if not results:
        return
    smpl_model_dir = require_smpl_model_dir(config, args)
    try:
        gt_points = decode_gt_smpl_points(image_path, config, args, smpl_model_dir, device)
    except (FileNotFoundError, ValueError):
        gt_points = None
    if gt_points is None:
        return
    joints_cam = gt_points["joints"]
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
    for item in results:
        query_idx = int(item["query_index"])
        if query_idx >= joints_2d.shape[0]:
            continue
        item["gt_projected_joints_2d"] = joints_2d[query_idx].detach().cpu().tolist()
        item["gt_projected_joints_mask"] = in_image[query_idx].detach().cpu().tolist()


def require_smpl_model_dir(config: dict[str, Any], args: argparse.Namespace) -> str:
    smpl_model_dir = args.smpl_model_dir or str(config.get("assets", {}).get("smpl_model_dir", ""))
    if not smpl_model_dir:
        raise ValueError("SMPL model dir is required. Set assets.smpl_model_dir or pass --smpl-model-dir")
    return smpl_model_dir


def decode_pred_smpl_points(
    results: list[dict[str, Any]],
    predictions: dict[str, torch.Tensor],
    smpl_model_dir: str,
    device: torch.device,
    use_hsi_refined: bool = False,
    pose_key: str | None = None,
    betas_key: str | None = None,
    transl_key: str | None = None,
) -> dict[str, torch.Tensor]:
    query_indices = [int(item["query_index"]) for item in results]
    if not query_indices:
        empty = torch.empty(0, 0, 3, device=device)
        return {"vertices": empty, "joints": empty, "query_indices": torch.empty(0, dtype=torch.long, device=device)}
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    pose_key = pose_key or ("hsi_refined_pred_poses" if use_hsi_refined and "hsi_refined_pred_poses" in predictions else "pred_poses")
    betas_key = betas_key or ("hsi_refined_pred_betas" if use_hsi_refined and "hsi_refined_pred_betas" in predictions else "pred_betas")
    transl_key = transl_key or ("hsi_refined_pred_transl_cam" if use_hsi_refined and "hsi_refined_pred_transl_cam" in predictions else "pred_transl_cam")
    for key in (pose_key, betas_key, transl_key):
        if key not in predictions:
            raise ValueError(f"SMPL decode requires model output {key}")
    poses = predictions[pose_key][0, 0, index_tensor].detach()
    betas = predictions[betas_key][0, 0, index_tensor].detach()
    transl_cam = predictions[transl_key][0, 0, index_tensor].detach()

    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    with torch.no_grad():
        vertices, joints = smpl(poses.reshape(-1, 72), betas)
    return {
        "vertices": vertices + transl_cam[:, None, :],
        "joints": joints + transl_cam[:, None, :],
        "faces": torch.as_tensor(smpl.faces, dtype=torch.long, device=device),
        "query_indices": index_tensor,
    }


def decode_gt_smpl_points(
    image_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    smpl_model_dir: str,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    gt = load_gt_smpl_for_image(image_path, config, args, device)
    if gt["poses_6d"].numel() == 0:
        return None
    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    poses = rot6d_to_axis_angle(gt["poses_6d"]).reshape(-1, 72)
    with torch.no_grad():
        vertices, joints = smpl(poses, gt["betas"])
    return {
        "vertices": vertices + gt["transl_cam"][:, None, :],
        "joints": joints + gt["transl_cam"][:, None, :],
        "faces": torch.as_tensor(smpl.faces, dtype=torch.long, device=device),
        "transl_cam": gt["transl_cam"],
    }


def prediction_query_vector(predictions: dict[str, torch.Tensor], key: str, query_idx: int) -> list[float] | None:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().float().cpu()
    while tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.ndim != 2 or query_idx < 0 or query_idx >= tensor.shape[0]:
        return None
    return [float(v) for v in tensor[query_idx].reshape(-1).tolist()]


def export_translation_debug_json(
    selected: list[dict[str, Any]],
    image_path: Path,
    predictions: dict[str, torch.Tensor],
    gt_points: dict[str, torch.Tensor] | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    rows = []
    gt_transl = gt_points.get("transl_cam") if gt_points is not None else None
    if isinstance(gt_transl, torch.Tensor):
        gt_transl_cpu = gt_transl.detach().float().cpu()
    else:
        gt_transl_cpu = None

    transl_sources = {
        "pre_refine_base": "base_pred_transl_cam",
        "post_refine_pred": "pred_transl_cam",
        "hsi_refined": "hsi_refined_pred_transl_cam",
    }
    component_sources = (
        "base_pred_transl_ray_depth",
        "pred_transl_ray_depth",
        "base_pred_transl_tangent",
        "pred_transl_tangent",
        "pred_transl_ray_dir",
        "pred_transl_tangent_x",
        "pred_transl_tangent_y",
        "pred_transl_box_prior_weight",
    )

    for item in selected:
        query_idx = int(item["query_index"])
        record: dict[str, Any] = {
            "rank": int(item.get("rank", len(rows))),
            "query_index": query_idx,
            "confidence": float(item.get("confidence", 0.0)),
            "bbox_cxcywh_norm": item.get("bbox_cxcywh_norm"),
            "bbox_xyxy_pixels": item.get("bbox_xyxy_pixels"),
            "translations": {},
            "errors_l2_m": {},
            "components": {},
        }
        gt_vec = None
        if gt_transl_cpu is not None and query_idx < gt_transl_cpu.shape[0]:
            gt_vec = [float(v) for v in gt_transl_cpu[query_idx].reshape(-1).tolist()]
            record["translations"]["gt_query_slot"] = gt_vec

        for label, key in transl_sources.items():
            vec = prediction_query_vector(predictions, key, query_idx)
            if vec is None:
                continue
            record["translations"][label] = vec
            if gt_vec is not None:
                record["errors_l2_m"][label] = float(np.linalg.norm(np.asarray(vec, dtype=np.float32) - np.asarray(gt_vec, dtype=np.float32)))

        base_vec = record["translations"].get("pre_refine_base")
        pred_vec = record["translations"].get("post_refine_pred")
        hsi_vec = record["translations"].get("hsi_refined")
        if base_vec is not None and pred_vec is not None:
            record["refine_delta_l2_m"] = float(np.linalg.norm(np.asarray(pred_vec, dtype=np.float32) - np.asarray(base_vec, dtype=np.float32)))
        if pred_vec is not None and hsi_vec is not None:
            record["hsi_delta_l2_m"] = float(np.linalg.norm(np.asarray(hsi_vec, dtype=np.float32) - np.asarray(pred_vec, dtype=np.float32)))

        for key in component_sources:
            vec = prediction_query_vector(predictions, key, query_idx)
            if vec is not None:
                record["components"][key] = vec
        rows.append(record)

    debug = {
        "image": str(image_path),
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "gt_association": "BEDLAM GT slot is read by query_index; this is intended for GT box prior diagnostics.",
        "translation_units": "meters",
        "sources": {
            "pre_refine_base": "predictions['base_pred_transl_cam'] before camera-ray translation refiner",
            "post_refine_pred": "predictions['pred_transl_cam'] after camera-ray translation refiner when enabled",
            "hsi_refined": "predictions['hsi_refined_pred_transl_cam'] after HSI when available",
            "gt_query_slot": "BEDLAM smplx_transl at the same query/GT slot",
        },
        "records": rows,
    }
    path = output_dir / f"{image_path.stem}_translation_debug_top{len(selected):02d}.json"
    path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return str(path)


def export_translation_only_comparison_ply(
    selected: list[dict[str, Any]],
    image_path: Path,
    predictions: dict[str, torch.Tensor],
    gt_points: dict[str, torch.Tensor] | None,
    smpl_model_dir: str,
    output_dir: Path,
    device: torch.device,
) -> list[str]:
    if gt_points is None or "transl_cam" not in gt_points:
        return []
    if "pred_poses" not in predictions or "pred_betas" not in predictions or "pred_transl_cam" not in predictions:
        return []
    query_indices = [int(item["query_index"]) for item in selected]
    if not query_indices:
        return []
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    with torch.no_grad():
        local_vertices, _ = smpl(
            predictions["pred_poses"][0, 0, index_tensor].detach().reshape(-1, 72),
            predictions["pred_betas"][0, 0, index_tensor].detach(),
        )
    faces = torch.as_tensor(smpl.faces, dtype=torch.long, device=device).detach().cpu().numpy()
    gt_transl = gt_points["transl_cam"].detach().to(device=device, dtype=local_vertices.dtype)

    stage_specs: list[tuple[str, torch.Tensor, tuple[int, int, int]]] = []
    if "base_pred_transl_cam" in predictions:
        stage_specs.append(("pre_refine", predictions["base_pred_transl_cam"][0, 0, index_tensor].detach(), (255, 160, 32)))
    stage_specs.append(("post_refine", predictions["pred_transl_cam"][0, 0, index_tensor].detach(), (255, 64, 64)))
    if "hsi_refined_pred_transl_cam" in predictions:
        stage_specs.append(("hsi_refined", predictions["hsi_refined_pred_transl_cam"][0, 0, index_tensor].detach(), (64, 192, 255)))
    gt_query_transl = []
    valid_gt = []
    for local_idx, query_idx in enumerate(query_indices):
        if 0 <= query_idx < gt_transl.shape[0]:
            gt_query_transl.append(gt_transl[query_idx])
            valid_gt.append(local_idx)
    if len(gt_query_transl) == len(query_indices):
        stage_specs.append(("gt_translation", torch.stack(gt_query_transl, dim=0), (64, 255, 128)))

    written: list[str] = []
    all_meshes: list[np.ndarray] = []
    all_colors: list[tuple[int, int, int]] = []
    for local_idx, item in enumerate(selected):
        person_meshes: list[np.ndarray] = []
        person_colors: list[tuple[int, int, int]] = []
        for _, transl, color in stage_specs:
            mesh = (local_vertices[local_idx] + transl[local_idx][None, :]).detach().cpu().numpy()
            person_meshes.append(mesh)
            person_colors.append(color)
            all_meshes.append(mesh)
            all_colors.append(color)
        query_idx = int(item["query_index"])
        rank = int(item.get("rank", local_idx))
        person_path = output_dir / f"{image_path.stem}_rank{rank:02d}_q{query_idx:02d}_translation_only_compare.ply"
        write_ply_meshes(person_path, person_meshes, faces, person_colors)
        written.append(str(person_path))

    if all_meshes:
        top_path = output_dir / f"{image_path.stem}_translation_only_compare_top{len(selected):02d}.ply"
        write_ply_meshes(top_path, all_meshes, faces, all_colors)
        written.append(str(top_path))
    return written


def export_prediction_gt_ply(
    results: list[dict[str, Any]],
    image_path: Path,
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> list[str]:
    selected = results[: max(int(args.ply_top_k), 0)]
    if not selected:
        return []
    for key in ("pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in predictions:
            raise ValueError(f"PLY export requires model output {key}")

    smpl_model_dir = require_smpl_model_dir(config, args)
    pred_points = decode_pred_smpl_points(selected, predictions, smpl_model_dir, device)
    pre_points = None
    if getattr(args, "export_pre_refine_comparison", False) and "base_pred_transl_cam" in predictions:
        pre_points = decode_pred_smpl_points(selected, predictions, smpl_model_dir, device, transl_key="base_pred_transl_cam")
    hsi_points = None
    if args.export_hsi_comparison and "hsi_refined_pred_poses" in predictions:
        hsi_points = decode_pred_smpl_points(selected, predictions, smpl_model_dir, device, use_hsi_refined=True)
    try:
        gt_points = decode_gt_smpl_points(image_path, config, args, smpl_model_dir, device)
    except (FileNotFoundError, ValueError):
        gt_points = None
    written: list[str] = []

    pred_meshes = [pred_points["vertices"][idx].detach().cpu().numpy() for idx in range(len(selected))]
    pred_faces = pred_points["faces"].detach().cpu().numpy()
    pred_colors = [COLORS[idx % len(COLORS)] for idx in range(len(pred_meshes))]
    pre_meshes: list[np.ndarray] = []
    if pre_points is not None:
        pre_meshes = [pre_points["vertices"][idx].detach().cpu().numpy() for idx in range(len(selected))]
        pre_path = output_dir / f"{image_path.stem}_pre_refine_mesh_top{len(pre_meshes):02d}.ply"
        write_ply_meshes(pre_path, pre_meshes, pre_points["faces"].detach().cpu().numpy(), pred_colors)
        written.append(str(pre_path))
    pred_path = output_dir / f"{image_path.stem}_pred_mesh_top{len(pred_meshes):02d}.ply"
    write_ply_meshes(pred_path, pred_meshes, pred_faces, pred_colors)
    written.append(str(pred_path))
    hsi_meshes: list[np.ndarray] = []
    if hsi_points is not None:
        hsi_meshes = [hsi_points["vertices"][idx].detach().cpu().numpy() for idx in range(len(selected))]
        hsi_path = output_dir / f"{image_path.stem}_hsi_refined_mesh_top{len(hsi_meshes):02d}.ply"
        write_ply_meshes(hsi_path, hsi_meshes, hsi_points["faces"].detach().cpu().numpy(), pred_colors)
        written.append(str(hsi_path))

    gt_meshes: list[np.ndarray] = []
    gt_colors: list[tuple[int, int, int]] = []
    if gt_points is not None:
        for idx, item in enumerate(selected):
            query_idx = int(item["query_index"])
            if query_idx >= gt_points["vertices"].shape[0]:
                continue
            gt_meshes.append(gt_points["vertices"][query_idx].detach().cpu().numpy())
            gt_colors.append(COLORS[idx % len(COLORS)])
        if gt_meshes:
            gt_faces = gt_points["faces"].detach().cpu().numpy()
            gt_path = output_dir / f"{image_path.stem}_gt_mesh_top{len(gt_meshes):02d}.ply"
            write_ply_meshes(gt_path, gt_meshes, gt_faces, gt_colors)
            written.append(str(gt_path))

            combined_path = output_dir / f"{image_path.stem}_pred_gt_mesh_top{len(pred_meshes):02d}.ply"
            write_ply_meshes(
                combined_path,
                pred_meshes + gt_meshes,
                pred_faces,
                [(255, 64, 64)] * len(pred_meshes) + [(64, 255, 128)] * len(gt_meshes),
            )
            written.append(str(combined_path))

    if pre_meshes or hsi_meshes:
        comparison_meshes: list[np.ndarray] = []
        comparison_colors: list[tuple[int, int, int]] = []
        if pre_meshes:
            comparison_meshes.extend(pre_meshes)
            comparison_colors.extend([(255, 160, 32)] * len(pre_meshes))
        comparison_meshes.extend(pred_meshes)
        comparison_colors.extend([(255, 64, 64)] * len(pred_meshes))
        if hsi_meshes:
            comparison_meshes.extend(hsi_meshes)
            comparison_colors.extend([(64, 192, 255)] * len(hsi_meshes))
        if gt_meshes:
            comparison_meshes.extend(gt_meshes)
            comparison_colors.extend([(64, 255, 128)] * len(gt_meshes))
        comparison_path = output_dir / f"{image_path.stem}_translation_stage_compare_top{len(selected):02d}.ply"
        write_ply_meshes(comparison_path, comparison_meshes, pred_faces, comparison_colors)
        written.append(str(comparison_path))

    if getattr(args, "export_translation_debug_json", False) or getattr(args, "export_pre_refine_comparison", False):
        written.append(export_translation_debug_json(selected, image_path, predictions, gt_points, args, output_dir))

    if getattr(args, "export_translation_only_comparison", False):
        written.extend(export_translation_only_comparison_ply(selected, image_path, predictions, gt_points, smpl_model_dir, output_dir, device))

    return written


def export_scene_ply(
    results: list[dict[str, Any]],
    image_tensor: torch.Tensor,
    image_path: Path,
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    input_size: int,
    device: torch.device,
) -> dict[str, Any]:
    selected = results[: max(int(args.ply_top_k), 0)]
    if not selected:
        return {"ply_files": [], "scene_alignment": None, "hsi_scene_alignment": None}
    for key in ("depth", "pose_enc", "pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in predictions:
            raise ValueError(f"Scene PLY export requires model output {key}")

    smpl_model_dir = require_smpl_model_dir(config, args)
    pred_points = decode_pred_smpl_points(selected, predictions, smpl_model_dir, device, use_hsi_refined=bool(args.use_hsi_refined))
    depth = predictions["depth"][0, 0].detach()
    env_vertices_raw, env_colors, env_faces = dense_depth_to_camera_mesh(
        depth,
        predictions["pose_enc"],
        image_tensor[0, 0].detach(),
        input_size,
    )
    scene_alignment = None
    hsi_scene_alignment = None
    env_vertices = env_vertices_raw
    if args.align_scene_to_smpl:
        anchor_vertices = pred_points["vertices"].detach()
        anchor_source = "pred_smpl"
        if args.align_use_gt_smpl_anchors:
            try:
                gt_points = decode_gt_smpl_points(image_path, config, args, smpl_model_dir, device)
            except (FileNotFoundError, ValueError):
                gt_points = None
            if gt_points is not None:
                gt_meshes = []
                for item in selected:
                    query_idx = int(item["query_index"])
                    if query_idx < gt_points["vertices"].shape[0]:
                        gt_meshes.append(gt_points["vertices"][query_idx])
                if gt_meshes:
                    anchor_vertices = torch.stack(gt_meshes, dim=0).detach()
                    anchor_source = "gt_smpl"
        scene_alignment = estimate_scene_to_smpl_scale(
            smpl_vertices=anchor_vertices,
            depth=depth,
            pose_enc=predictions["pose_enc"],
            input_size=input_size,
            min_anchor_pixels=int(args.align_min_anchor_pixels),
            scale_min=float(args.align_scale_min),
            scale_max=float(args.align_scale_max),
            anchor_stride=int(args.align_anchor_stride),
        )
        scene_alignment["anchor_source"] = anchor_source
        scale = float(scene_alignment["scale"])
        if bool(scene_alignment["applied"]):
            env_vertices = np.asarray(env_vertices_raw, dtype=np.float32) * scale
    hsi_env_vertices = None
    hsi_env_colors = None
    hsi_env_faces = None
    if args.hsi_align_scene and "hsi_scene_scale" in predictions and "hsi_scene_depth_bias" in predictions:
        hsi_scale = float(predictions["hsi_scene_scale"][0, 0].detach().float().reshape(-1)[0].cpu())
        hsi_bias = float(predictions["hsi_scene_depth_bias"][0, 0].detach().float().reshape(-1)[0].cpu())
        hsi_depth = depth * depth.new_tensor(hsi_scale) + depth.new_tensor(hsi_bias)
        hsi_env_vertices, hsi_env_colors, hsi_env_faces = dense_depth_to_camera_mesh(
            hsi_depth,
            predictions["pose_enc"],
            image_tensor[0, 0].detach(),
            input_size,
        )
        hsi_scene_alignment = {
            "method": "hsi_affine_depth",
            "applied": True,
            "scale": hsi_scale,
            "depth_bias": hsi_bias,
        }
    meshes = [pred_points["vertices"][idx].detach().cpu().numpy() for idx in range(len(selected))]
    mesh_colors = [COLORS[idx % len(COLORS)] for idx in range(len(meshes))]
    faces = pred_points["faces"].detach().cpu().numpy()
    env_path = output_dir / f"{image_path.stem}_vggt_depth_mesh.ply"
    write_ply_vertices_faces(env_path, env_vertices_raw, env_colors, env_faces)
    written = [str(env_path)]
    if args.align_scene_to_smpl:
        aligned_env_path = output_dir / f"{image_path.stem}_vggt_depth_mesh_smpl_aligned.ply"
        write_ply_vertices_faces(aligned_env_path, env_vertices, env_colors, env_faces)
        written.append(str(aligned_env_path))
    if hsi_env_vertices is not None and hsi_env_colors is not None and hsi_env_faces is not None:
        hsi_env_path = output_dir / f"{image_path.stem}_vggt_depth_mesh_hsi_aligned.ply"
        write_ply_vertices_faces(hsi_env_path, hsi_env_vertices, hsi_env_colors, hsi_env_faces)
        written.append(str(hsi_env_path))
    suffix = "scene_vggt_points_smpl_aligned_mesh" if args.align_scene_to_smpl else "scene_vggt_points_smpl_mesh"
    path = output_dir / f"{image_path.stem}_{suffix}.ply"
    write_ply_scene(path, env_vertices, env_colors, env_faces, meshes, faces, mesh_colors)
    written.append(str(path))
    if hsi_env_vertices is not None and hsi_env_colors is not None and hsi_env_faces is not None:
        hsi_scene_path = output_dir / f"{image_path.stem}_scene_vggt_points_hsi_aligned_mesh.ply"
        write_ply_scene(hsi_scene_path, hsi_env_vertices, hsi_env_colors, hsi_env_faces, meshes, faces, mesh_colors)
        written.append(str(hsi_scene_path))
    return {"ply_files": written, "scene_alignment": scene_alignment, "hsi_scene_alignment": hsi_scene_alignment}


def estimate_scene_to_smpl_scale(
    smpl_vertices: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    input_size: int,
    min_anchor_pixels: int,
    scale_min: float,
    scale_max: float,
    anchor_stride: int,
) -> dict[str, Any]:
    if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
        raise ValueError(f"Invalid alignment scale range: [{scale_min}, {scale_max}]")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W) or (H, W, 1), got {tuple(depth.shape)}")
    if smpl_vertices.ndim != 3 or smpl_vertices.shape[-1] != 3:
        raise ValueError(f"Expected smpl_vertices shape (N, V, 3), got {tuple(smpl_vertices.shape)}")

    anchor_stride = max(int(anchor_stride), 1)
    vertices = smpl_vertices[:, ::anchor_stride].reshape(-1, 3).to(device=depth.device, dtype=depth.dtype)
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(input_size, input_size), build_intrinsics=True)
    intrinsics_0 = intrinsics[0, 0].to(device=depth.device, dtype=depth.dtype)
    projected = project_points(vertices, intrinsics_0)
    height, width = depth.shape
    px = projected[:, 0].round().long()
    py = projected[:, 1].round().long()
    valid = (
        torch.isfinite(vertices).all(dim=-1)
        & torch.isfinite(projected).all(dim=-1)
        & (vertices[:, 2] > 1e-6)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    if not valid.any():
        return _scene_alignment_result(1.0, False, "no_projected_smpl_anchors", 0, 0, None)

    px = px[valid]
    py = py[valid]
    z_smpl = vertices[valid, 2]
    z_vggt = depth[py, px]
    valid_depth = torch.isfinite(z_vggt) & (z_vggt > 1e-6)
    if not valid_depth.any():
        return _scene_alignment_result(1.0, False, "no_valid_vggt_depth_at_anchors", int(valid.sum().item()), 0, None)

    px = px[valid_depth]
    py = py[valid_depth]
    z_smpl = z_smpl[valid_depth]
    z_vggt = z_vggt[valid_depth]
    px, py, z_smpl, z_vggt = _nearest_smpl_anchor_per_pixel(px, py, z_smpl, z_vggt, width)
    ratios = (z_smpl / z_vggt).flatten()
    ratios = ratios[torch.isfinite(ratios) & (ratios > 0)]
    if ratios.numel() == 0:
        return _scene_alignment_result(1.0, False, "no_valid_depth_ratios", int(valid.sum().item()), 0, None)

    raw_stats = _ratio_stats(ratios)
    in_range = (ratios >= scale_min) & (ratios <= scale_max)
    ratios_in_range = ratios[in_range]
    num_anchor_pixels = int(ratios_in_range.numel())
    if num_anchor_pixels < int(min_anchor_pixels):
        return _scene_alignment_result(
            float(raw_stats["median"]),
            False,
            "insufficient_anchor_pixels",
            int(valid.sum().item()),
            num_anchor_pixels,
            raw_stats,
            raw_stats=raw_stats,
            num_raw_anchor_pixels=int(ratios.numel()),
            num_low_filtered=int((ratios < scale_min).sum().item()),
            num_high_filtered=int((ratios > scale_max).sum().item()),
        )

    stats = _ratio_stats(ratios_in_range)
    scale = float(stats["median"])
    before = torch.abs(z_vggt[in_range] - z_smpl[in_range]).median()
    after = torch.abs(z_vggt[in_range] * z_vggt.new_tensor(scale) - z_smpl[in_range]).median()
    return {
        "method": "smpl_anchor_median_depth_ratio",
        "applied": True,
        "reason": "ok",
        "scale": scale,
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "anchor_stride": int(anchor_stride),
        "num_projected_anchors": int(valid.sum().item()),
        "num_raw_anchor_pixels": int(ratios.numel()),
        "num_anchor_pixels": num_anchor_pixels,
        "num_low_filtered": int((ratios < scale_min).sum().item()),
        "num_high_filtered": int((ratios > scale_max).sum().item()),
        "ratio_all_median": float(raw_stats["median"]),
        "ratio_all_mad": float(raw_stats["mad"]),
        "ratio_all_mean": float(raw_stats["mean"]),
        "ratio_all_std": float(raw_stats["std"]),
        "ratio_all_p05": float(raw_stats["p05"]),
        "ratio_all_p25": float(raw_stats["p25"]),
        "ratio_all_p50": float(raw_stats["p50"]),
        "ratio_all_p75": float(raw_stats["p75"]),
        "ratio_all_p90": float(raw_stats["p90"]),
        "ratio_all_p95": float(raw_stats["p95"]),
        "ratio_median": scale,
        "ratio_mad": float(stats["mad"]),
        "ratio_mean": float(stats["mean"]),
        "ratio_std": float(stats["std"]),
        "ratio_p05": float(stats["p05"]),
        "ratio_p25": float(stats["p25"]),
        "ratio_p50": float(stats["p50"]),
        "ratio_p75": float(stats["p75"]),
        "ratio_p90": float(stats["p90"]),
        "ratio_p95": float(stats["p95"]),
        "anchor_depth_l1_median_before": float(before.detach().cpu()),
        "anchor_depth_l1_median_after": float(after.detach().cpu()),
    }


def _nearest_smpl_anchor_per_pixel(
    px: torch.Tensor,
    py: torch.Tensor,
    z_smpl: torch.Tensor,
    z_vggt: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel_idx = (py * int(width) + px).detach().cpu().numpy()
    z_np = z_smpl.detach().float().cpu().numpy()
    order = np.lexsort((z_np, pixel_idx))
    sorted_pixels = pixel_idx[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
    keep_idx = torch.as_tensor(order[keep], dtype=torch.long, device=px.device)
    return px[keep_idx], py[keep_idx], z_smpl[keep_idx], z_vggt[keep_idx]


def _ratio_stats(ratios: torch.Tensor) -> dict[str, float]:
    ratios_f = ratios.detach().float()
    median = ratios_f.median()
    mad = torch.abs(ratios_f - median).median()
    std = ratios_f.std(unbiased=False) if ratios_f.numel() > 1 else ratios_f.new_zeros(())
    quantiles = torch.quantile(
        ratios_f,
        ratios_f.new_tensor([0.05, 0.25, 0.5, 0.75, 0.9, 0.95]),
    )
    return {
        "median": float(median.cpu()),
        "mad": float(mad.cpu()),
        "mean": float(ratios_f.mean().cpu()),
        "std": float(std.cpu()),
        "p05": float(quantiles[0].cpu()),
        "p25": float(quantiles[1].cpu()),
        "p50": float(quantiles[2].cpu()),
        "p75": float(quantiles[3].cpu()),
        "p90": float(quantiles[4].cpu()),
        "p95": float(quantiles[5].cpu()),
    }


def _scene_alignment_result(
    scale: float,
    applied: bool,
    reason: str,
    num_projected_anchors: int,
    num_anchor_pixels: int,
    stats: dict[str, float] | None,
    raw_stats: dict[str, float] | None = None,
    num_raw_anchor_pixels: int = 0,
    num_low_filtered: int = 0,
    num_high_filtered: int = 0,
) -> dict[str, Any]:
    return {
        "method": "smpl_anchor_median_depth_ratio",
        "applied": bool(applied),
        "reason": reason,
        "scale": float(scale),
        "num_projected_anchors": int(num_projected_anchors),
        "num_raw_anchor_pixels": int(num_raw_anchor_pixels),
        "num_anchor_pixels": int(num_anchor_pixels),
        "num_low_filtered": int(num_low_filtered),
        "num_high_filtered": int(num_high_filtered),
        "ratio_all_median": None if raw_stats is None else float(raw_stats["median"]),
        "ratio_all_mad": None if raw_stats is None else float(raw_stats["mad"]),
        "ratio_all_mean": None if raw_stats is None else float(raw_stats["mean"]),
        "ratio_all_std": None if raw_stats is None else float(raw_stats["std"]),
        "ratio_all_p05": None if raw_stats is None else float(raw_stats["p05"]),
        "ratio_all_p25": None if raw_stats is None else float(raw_stats["p25"]),
        "ratio_all_p50": None if raw_stats is None else float(raw_stats["p50"]),
        "ratio_all_p75": None if raw_stats is None else float(raw_stats["p75"]),
        "ratio_all_p90": None if raw_stats is None else float(raw_stats["p90"]),
        "ratio_all_p95": None if raw_stats is None else float(raw_stats["p95"]),
        "ratio_median": None if stats is None else float(stats["median"]),
        "ratio_mad": None if stats is None else float(stats["mad"]),
        "ratio_mean": None if stats is None else float(stats["mean"]),
        "ratio_std": None if stats is None else float(stats["std"]),
        "ratio_p05": None if stats is None else float(stats["p05"]),
        "ratio_p25": None if stats is None else float(stats["p25"]),
        "ratio_p50": None if stats is None else float(stats["p50"]),
        "ratio_p75": None if stats is None else float(stats["p75"]),
        "ratio_p90": None if stats is None else float(stats["p90"]),
        "ratio_p95": None if stats is None else float(stats["p95"]),
    }


def write_ply_points(path: Path, points: np.ndarray, colors: np.ndarray | tuple[int, int, int]) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if isinstance(colors, tuple):
        color_arr = np.tile(np.asarray(colors, dtype=np.uint8).reshape(1, 3), (points.shape[0], 1))
    else:
        color_arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[finite]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, color_arr, strict=True):
            file.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def dense_depth_to_camera_mesh(
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    image_tensor: torch.Tensor,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W) or (H, W, 1), got {tuple(depth.shape)}")
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(input_size, input_size), build_intrinsics=True)
    intrinsics_0 = intrinsics[0, 0].to(device=depth.device, dtype=depth.dtype)
    height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth.clamp(min=1e-6)
    x = (xs - intrinsics_0[0, 2]) / intrinsics_0[0, 0] * z
    y = (ys - intrinsics_0[1, 2]) / intrinsics_0[1, 1] * z
    points = torch.stack([x, y, z], dim=-1)
    image_tensor = image_tensor.to(device=depth.device)
    if image_tensor.shape[-2:] != (height, width):
        image_tensor = F.interpolate(
            image_tensor[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
    colors = (image_tensor.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    mask = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    depth_np = z.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    return depth_grid_to_surface_mesh(points_np, colors_np, depth_np, mask_np)


def depth_grid_to_surface_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    depth_edge_rtol: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    index_map = -np.ones((height, width), dtype=np.int64)
    index_map[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    vertices = np.asarray(points[mask], dtype=np.float32).reshape(-1, 3)
    vertex_colors = np.asarray(colors[mask], dtype=np.uint8).reshape(-1, 3)

    cell_mask = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    d00 = depth[:-1, :-1]
    d01 = depth[:-1, 1:]
    d10 = depth[1:, :-1]
    d11 = depth[1:, 1:]
    dmax = np.maximum.reduce([d00, d01, d10, d11])
    dmin = np.minimum.reduce([d00, d01, d10, d11])
    dmean = (d00 + d01 + d10 + d11) * 0.25
    cell_mask &= ((dmax - dmin) / np.maximum(np.abs(dmean), 1e-6)) <= depth_edge_rtol

    ys, xs = np.nonzero(cell_mask)
    if ys.size == 0:
        return vertices, vertex_colors, np.empty((0, 3), dtype=np.int64)
    i00 = index_map[ys, xs]
    i01 = index_map[ys, xs + 1]
    i10 = index_map[ys + 1, xs]
    i11 = index_map[ys + 1, xs + 1]
    faces = np.concatenate(
        [
            np.stack([i00, i10, i01], axis=1),
            np.stack([i10, i11, i01], axis=1),
        ],
        axis=0,
    )
    return vertices, vertex_colors, faces.astype(np.int64, copy=False)


def write_ply_meshes(
    path: Path,
    meshes: list[np.ndarray],
    faces: np.ndarray,
    colors: list[tuple[int, int, int]],
) -> None:
    vertices_parts = []
    colors_parts = []
    faces_parts = []
    face_template = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    offset = 0
    for mesh_idx, vertices in enumerate(meshes):
        clean_vertices, clean_faces, clean_colors = prepare_mesh_arrays(
            vertices,
            face_template,
            colors[mesh_idx % len(colors)],
        )
        vertices_parts.append(clean_vertices)
        colors_parts.append(clean_colors)
        faces_parts.append(clean_faces + offset)
        offset += clean_vertices.shape[0]
    vertices_all = np.concatenate(vertices_parts, axis=0) if vertices_parts else np.empty((0, 3), dtype=np.float32)
    colors_all = np.concatenate(colors_parts, axis=0) if colors_parts else np.empty((0, 3), dtype=np.uint8)
    faces_all = np.concatenate(faces_parts, axis=0) if faces_parts else np.empty((0, 3), dtype=np.int64)
    write_ply_vertices_faces(path, vertices_all, colors_all, faces_all)


def write_ply_scene(
    path: Path,
    env_vertices: np.ndarray,
    env_colors: np.ndarray,
    env_faces: np.ndarray,
    meshes: list[np.ndarray],
    faces: np.ndarray,
    mesh_colors: list[tuple[int, int, int]],
) -> None:
    vertices_parts = [np.asarray(env_vertices, dtype=np.float32).reshape(-1, 3)]
    colors_parts = [np.asarray(env_colors, dtype=np.uint8).reshape(-1, 3)]
    faces_parts = [np.asarray(env_faces, dtype=np.int64).reshape(-1, 3)]
    face_template = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    offset = vertices_parts[0].shape[0]
    for mesh_idx, mesh in enumerate(meshes):
        clean_vertices, clean_faces, clean_colors = prepare_mesh_arrays(
            mesh,
            face_template,
            mesh_colors[mesh_idx % len(mesh_colors)],
        )
        vertices_parts.append(clean_vertices)
        colors_parts.append(clean_colors)
        faces_parts.append(clean_faces + offset)
        offset += clean_vertices.shape[0]
    vertices_all = np.concatenate(vertices_parts, axis=0)
    colors_all = np.concatenate(colors_parts, axis=0)
    faces_all = np.concatenate(faces_parts, axis=0) if faces_parts else np.empty((0, 3), dtype=np.int64)
    write_ply_vertices_faces(path, vertices_all, colors_all, faces_all)


def prepare_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    finite = np.isfinite(vertices).all(axis=1)
    if not finite.all():
        index_map = -np.ones(vertices.shape[0], dtype=np.int64)
        index_map[finite] = np.arange(int(finite.sum()), dtype=np.int64)
        valid_faces = finite[faces].all(axis=1)
        faces = index_map[faces[valid_faces]]
        vertices = vertices[finite]
    color_arr = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 3), (vertices.shape[0], 1))
    return vertices, faces, color_arr


def write_ply_vertices_faces(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {vertices.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {faces.shape[0]}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            file.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | tuple[int, int, int]) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    finite = np.isfinite(vertices).all(axis=1)
    if not finite.all():
        index_map = -np.ones(vertices.shape[0], dtype=np.int64)
        index_map[finite] = np.arange(int(finite.sum()), dtype=np.int64)
        valid_faces = finite[faces].all(axis=1)
        faces = index_map[faces[valid_faces]]
        vertices = vertices[finite]
    if isinstance(colors, tuple):
        color_arr = np.tile(np.asarray(colors, dtype=np.uint8).reshape(1, 3), (vertices.shape[0], 1))
    else:
        color_arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        color_arr = color_arr[finite] if color_arr.shape[0] == finite.shape[0] else color_arr
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {vertices.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {faces.shape[0]}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, color_arr, strict=True):
            file.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def load_gt_smpl_for_image(
    image_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    data_cfg = config.get("data", {})
    max_humans = int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20)))
    dataset_root = Path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False))
    split = str(data_cfg.get("train_split", "Training"))
    image_path = image_path.resolve()
    rgb_dir = image_path.parent
    if rgb_dir.name != "rgb":
        raise ValueError(f"--draw-gt-smpl-joints expects a BEDLAM rgb image path, got: {image_path}")
    smpl_path = rgb_dir.parent / "smpl" / f"{image_path.stem}.pkl"
    if not smpl_path.is_file():
        raise FileNotFoundError(f"GT SMPL annotation not found: {smpl_path}")
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL annotation must be a list: {smpl_path}")
    poses = torch.zeros(max_humans, 144, dtype=torch.float32, device=device)
    betas = torch.zeros(max_humans, 10, dtype=torch.float32, device=device)
    transl_cam = torch.zeros(max_humans, 3, dtype=torch.float32, device=device)
    for person_idx, person in enumerate(persons[:max_humans]):
        root_pose = torch.as_tensor(person["smplx_root_pose"], dtype=torch.float32, device=device).reshape(1, 3)
        body_pose = torch.as_tensor(person["smplx_body_pose"], dtype=torch.float32, device=device).reshape(21, 3)
        aa_22 = torch.cat([root_pose, body_pose], dim=0)
        pose_6d_22 = axis_angle_to_rot6d(aa_22).reshape(22, 6)
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device).expand(2, -1)
        poses[person_idx] = torch.cat([pose_6d_22, identity_6d], dim=0).reshape(144)
        betas[person_idx] = torch.as_tensor(person["smplx_shape"], dtype=torch.float32, device=device).reshape(-1)[:10]
        transl_cam[person_idx] = torch.as_tensor(person["smplx_transl"], dtype=torch.float32, device=device).reshape(3)
    return {"poses_6d": poses, "betas": betas, "transl_cam": transl_cam}


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
        joints = []
        masks = []
    _draw_joint_points(draw, joints, masks, color, radius=3)
    gt_joints = pred.get("gt_projected_joints_2d")
    gt_masks = pred.get("gt_projected_joints_mask")
    if isinstance(gt_joints, list) and isinstance(gt_masks, list):
        _draw_joint_points(draw, gt_joints, gt_masks, (255, 255, 255), radius=2)


def _draw_joint_points(
    draw: ImageDraw.ImageDraw,
    joints: list[Any],
    masks: list[Any],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for joint_idx, xy in enumerate(joints[:24]):
        if joint_idx >= len(masks) or not masks[joint_idx]:
            continue
        x, y = float(xy[0]), float(xy[1])
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
