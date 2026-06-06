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
    ply_outputs = {}
    if args.export_ply:
        ply_outputs = export_ply_outputs(
            results,
            predictions,
            config,
            args,
            image_tensor,
            input_size,
            output_dir,
            image_path.stem,
            device,
        )

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
                "ply_outputs": ply_outputs,
                "predictions": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_image": str(out_image), "output_json": str(out_json), "ply_outputs": ply_outputs, "num_predictions": len(results)}, indent=2))


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
    parser.add_argument("--use-gt-box-prior", action="store_true", help="Load BEDLAM GT/preprocessed boxes for oracle query priors")
    parser.add_argument("--gt-box-prior-center-noise", type=float, default=0.0, help="Uniform cx/cy noise range for GT box priors, normalized units")
    parser.add_argument("--gt-box-prior-size-noise", type=float, default=0.0, help="Uniform relative w/h noise range for GT box priors")
    parser.add_argument("--gt-box-prior-drop-prob", type=float, default=0.0, help="Probability of dropping a valid GT box prior")
    parser.add_argument("--smpl-model-dir", default="", help="Override assets.smpl_model_dir")
    parser.add_argument("--baseline-checkpoint", default="", help="Override checkpoints.vggt_baseline for loading VGGT camera head")
    parser.add_argument("--export-ply", action="store_true", help="Export predicted SMPL meshes and VGGT depth point cloud as PLY files")
    parser.add_argument("--ply-coordinate-frame", choices=("camera", "world"), default="camera", help="Coordinate frame for exported PLY geometry")
    parser.add_argument("--ply-max-depth-points", type=int, default=0, help="Maximum VGGT depth points to export; 0 keeps all valid points")
    parser.add_argument("--ply-depth-conf-percentile", type=float, default=0.0, help="Keep depth points above this confidence percentile; 0 disables percentile filtering")
    parser.add_argument("--ply-depth-conf-min", type=float, default=1e-5, help="Minimum absolute depth confidence for exported points")
    parser.add_argument("--ply-filter-depth-edges", action="store_true", help="Drop depth pixels near local depth discontinuities before PLY export")
    parser.add_argument("--ply-depth-edge-rtol", type=float, default=0.03, help="Relative depth jump threshold for --ply-filter-depth-edges")
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


def export_ply_outputs(
    results: list[dict[str, Any]],
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    image_tensor: torch.Tensor,
    input_size: int,
    output_dir: Path,
    output_stem: str,
    device: torch.device,
) -> dict[str, Any]:
    if "pose_enc" not in predictions:
        raise ValueError("PLY export requires model output pose_enc; set model.enable_camera=true")
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], image_size_hw=(input_size, input_size), build_intrinsics=True)
    camera_from_world = extrinsics[0, 0].detach().cpu().numpy()

    outputs: dict[str, Any] = {
        "coordinate_frame": args.ply_coordinate_frame,
        "camera_from_world": camera_from_world.tolist(),
    }

    depth_vertices = np.empty((0, 3), dtype=np.float32)
    depth_colors = np.empty((0, 3), dtype=np.uint8)
    if "depth" in predictions:
        depth_vertices, depth_colors = depth_to_point_cloud(
            predictions["depth"][0, 0].detach().cpu().numpy(),
            predictions.get("depth_conf")[0, 0].detach().cpu().numpy() if predictions.get("depth_conf") is not None else None,
            intrinsics[0, 0].detach().cpu().numpy(),
            image_tensor[0, 0].detach().cpu().numpy(),
            max_points=args.ply_max_depth_points,
            conf_percentile=args.ply_depth_conf_percentile,
            conf_min=args.ply_depth_conf_min,
            filter_depth_edges=args.ply_filter_depth_edges,
            depth_edge_rtol=args.ply_depth_edge_rtol,
        )
        if args.ply_coordinate_frame == "world":
            depth_vertices = camera_to_world_points(depth_vertices, camera_from_world)
        depth_path = output_dir / f"{output_stem}_environment_points_{args.ply_coordinate_frame}.ply"
        write_point_cloud_ply(depth_path, depth_vertices, depth_colors)
        outputs["environment_points_ply"] = str(depth_path)
        outputs["num_environment_points"] = int(depth_vertices.shape[0])

    smpl_vertices = np.empty((0, 3), dtype=np.float32)
    smpl_colors = np.empty((0, 3), dtype=np.uint8)
    smpl_faces = np.empty((0, 3), dtype=np.int64)
    if results:
        smpl_vertices, smpl_colors, smpl_faces = decode_smpl_meshes_for_results(results, predictions, config, args, device)
        if args.ply_coordinate_frame == "world":
            smpl_vertices = camera_to_world_points(smpl_vertices, camera_from_world)
        smpl_path = output_dir / f"{output_stem}_smpl_meshes_{args.ply_coordinate_frame}.ply"
        write_mesh_ply(smpl_path, smpl_vertices, smpl_colors, smpl_faces)
        outputs["smpl_meshes_ply"] = str(smpl_path)
        outputs["num_smpl_vertices"] = int(smpl_vertices.shape[0])
        outputs["num_smpl_faces"] = int(smpl_faces.shape[0])

    return outputs


def depth_to_point_cloud(
    depth: np.ndarray,
    depth_conf: np.ndarray | None,
    intrinsics: np.ndarray,
    image_chw: np.ndarray,
    max_points: int,
    conf_percentile: float,
    conf_min: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3 and depth_2d.shape[-1] == 1:
        depth_2d = depth_2d[..., 0]
    if depth_2d.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W] or [H,W,1], got {depth.shape}")

    height, width = depth_2d.shape
    source_height, source_width = int(image_chw.shape[1]), int(image_chw.shape[2])
    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    if (source_height, source_width) != (height, width):
        intrinsics[0, 0] *= float(width) / float(source_width)
        intrinsics[0, 2] *= float(width) / float(source_width)
        intrinsics[1, 1] *= float(height) / float(source_height)
        intrinsics[1, 2] *= float(height) / float(source_height)

    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    z = depth_2d
    valid = np.isfinite(z) & (z > 1e-6)
    if depth_conf is not None:
        conf = np.asarray(depth_conf, dtype=np.float32)
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        valid &= np.isfinite(conf) & (conf > float(conf_min))
        if np.any(valid) and conf_percentile > 0:
            valid &= conf >= np.percentile(conf[valid], float(conf_percentile))
    if filter_depth_edges:
        valid &= ~depth_edge_mask(depth_2d, rtol=depth_edge_rtol)

    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    vertices = np.stack([(xx - cx) * z / fx, (yy - cy) * z / fy, z], axis=-1)[valid]

    image_hwc = resize_image_chw_to_hwc(image_chw, height, width)
    colors = (image_hwc.clip(0.0, 1.0) * 255.0).astype(np.uint8)[valid]
    if max_points > 0 and vertices.shape[0] > max_points:
        indices = np.linspace(0, vertices.shape[0] - 1, int(max_points)).astype(np.int64)
        vertices = vertices[indices]
        colors = colors[indices]
    return vertices.astype(np.float32), colors


def resize_image_chw_to_hwc(image_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    image_hwc = np.transpose(np.asarray(image_chw, dtype=np.float32), (1, 2, 0))
    if image_hwc.shape[:2] == (height, width):
        return image_hwc
    image_u8 = (image_hwc.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(image_u8, mode="RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def depth_edge_mask(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(depth, ((pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[y : y + depth.shape[0], x : x + depth.shape[1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    return (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6) > float(rtol)


def decode_smpl_meshes_for_results(
    results: list[dict[str, Any]],
    predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smpl_model_dir = args.smpl_model_dir or str(config.get("assets", {}).get("smpl_model_dir", ""))
    if not smpl_model_dir:
        raise ValueError("SMPL model dir is required for PLY export. Set assets.smpl_model_dir or pass --smpl-model-dir")
    for key in ("pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in predictions:
            raise ValueError(f"SMPL PLY export requires model output {key}")

    query_indices = [int(item["query_index"]) for item in results]
    index_tensor = torch.as_tensor(query_indices, dtype=torch.long, device=device)
    poses = predictions["pred_poses"][0, 0, index_tensor].detach()
    betas = predictions["pred_betas"][0, 0, index_tensor].detach()
    transl_cam = predictions["pred_transl_cam"][0, 0, index_tensor].detach()

    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    with torch.no_grad():
        vertices, _ = smpl(poses.reshape(-1, 72), betas)
    vertices = vertices + transl_cam[:, None, :]

    faces = np.asarray(smpl.faces, dtype=np.int64)
    mesh_vertices = []
    mesh_colors = []
    mesh_faces = []
    vertex_offset = 0
    for result_idx, item in enumerate(results):
        color = np.asarray(COLORS[result_idx % len(COLORS)], dtype=np.uint8)
        person_vertices = vertices[result_idx].detach().cpu().numpy().astype(np.float32)
        mesh_vertices.append(person_vertices)
        mesh_colors.append(np.tile(color[None], (person_vertices.shape[0], 1)))
        mesh_faces.append(faces + vertex_offset)
        item["smpl_mesh_vertex_count"] = int(person_vertices.shape[0])
        item["smpl_mesh_face_count"] = int(faces.shape[0])
        vertex_offset += int(person_vertices.shape[0])
    return np.concatenate(mesh_vertices, axis=0), np.concatenate(mesh_colors, axis=0), np.concatenate(mesh_faces, axis=0)


def camera_to_world_points(points_cam: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    if points_cam.size == 0:
        return points_cam.astype(np.float32)
    rotation = camera_from_world[:3, :3]
    translation = camera_from_world[:3, 3]
    return ((points_cam - translation[None]) @ rotation).astype(np.float32)


def write_point_cloud_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    write_mesh_ply(path, vertices, colors, np.empty((0, 3), dtype=np.int64))


def write_mesh_ply(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if colors.shape[0] != vertices.shape[0]:
        raise ValueError(f"PLY colors must match vertices, got vertices={vertices.shape} colors={colors.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {vertices.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write(f"element face {faces.shape[0]}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            file.write(f"{vertex[0]:.7g} {vertex[1]:.7g} {vertex[2]:.7g} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


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
