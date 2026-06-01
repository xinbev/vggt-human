import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    ply_files = []
    if args.export_ply:
        ply_files = export_prediction_gt_ply(results, image_path, predictions, config, args, output_dir, device)

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
    parser.add_argument("--export-ply", action="store_true", help="Export selected predicted and GT SMPL 3D point clouds")
    parser.add_argument("--ply-top-k", type=int, default=3, help="Maximum ranked predictions to export as PLY")
    parser.add_argument("--ply-use-vertices", action="store_true", help="Also export SMPL mesh PLY files; joints are always exported")
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
) -> dict[str, torch.Tensor]:
    query_indices = [int(item["query_index"]) for item in results]
    if not query_indices:
        empty = torch.empty(0, 0, 3, device=device)
        return {"vertices": empty, "joints": empty, "query_indices": torch.empty(0, dtype=torch.long, device=device)}
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    poses = predictions["pred_poses"][0, 0, index_tensor].detach()
    betas = predictions["pred_betas"][0, 0, index_tensor].detach()
    transl_cam = predictions["pred_transl_cam"][0, 0, index_tensor].detach()

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
    }


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
    try:
        gt_points = decode_gt_smpl_points(image_path, config, args, smpl_model_dir, device)
    except (FileNotFoundError, ValueError):
        gt_points = None
    written: list[str] = []
    pred_color = (255, 64, 64)
    gt_color = (64, 255, 128)

    for local_idx, item in enumerate(selected):
        rank = int(item["rank"])
        query_idx = int(item["query_index"])
        prefix = output_dir / f"{image_path.stem}_rank{rank:02d}_q{query_idx}"

        pred_joints = pred_points["joints"][local_idx].detach().cpu().numpy()
        pred_joints_path = prefix.with_name(f"{prefix.name}_pred_joints.ply")
        write_ply_points(pred_joints_path, pred_joints, pred_color)
        written.append(str(pred_joints_path))

        if gt_points is not None and query_idx < gt_points["joints"].shape[0]:
            gt_joints = gt_points["joints"][query_idx].detach().cpu().numpy()
            gt_joints_path = prefix.with_name(f"{prefix.name}_gt_joints.ply")
            write_ply_points(gt_joints_path, gt_joints, gt_color)
            written.append(str(gt_joints_path))

            combined_path = prefix.with_name(f"{prefix.name}_pred_gt_joints.ply")
            combined_points = np.concatenate([pred_joints, gt_joints], axis=0)
            combined_colors = np.concatenate(
                [
                    np.tile(np.asarray(pred_color, dtype=np.uint8), (pred_joints.shape[0], 1)),
                    np.tile(np.asarray(gt_color, dtype=np.uint8), (gt_joints.shape[0], 1)),
                ],
                axis=0,
            )
            write_ply_points(combined_path, combined_points, combined_colors)
            written.append(str(combined_path))

        if bool(args.ply_use_vertices):
            pred_vertices = pred_points["vertices"][local_idx].detach().cpu().numpy()
            pred_faces = pred_points["faces"].detach().cpu().numpy()
            pred_vertices_path = prefix.with_name(f"{prefix.name}_pred_mesh.ply")
            write_ply_mesh(pred_vertices_path, pred_vertices, pred_faces, pred_color)
            written.append(str(pred_vertices_path))
            if gt_points is not None and query_idx < gt_points["vertices"].shape[0]:
                gt_vertices = gt_points["vertices"][query_idx].detach().cpu().numpy()
                gt_faces = gt_points["faces"].detach().cpu().numpy()
                gt_vertices_path = prefix.with_name(f"{prefix.name}_gt_mesh.ply")
                write_ply_mesh(gt_vertices_path, gt_vertices, gt_faces, gt_color)
                written.append(str(gt_vertices_path))

    return written


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
