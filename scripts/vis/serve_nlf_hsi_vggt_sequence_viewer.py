#!/usr/bin/env python
"""Serve a Viser viewer for VGGT-Omega + NLF + HSI sequence inference.

The important invariant is that the full selected frame sequence is processed in
one VGGT forward pass.  This preserves the VGGT camera/world frame shared by all
frames in the sequence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
)
from vggt_omega.data.geometry import (  # noqa: E402
    ResizeGeometry,
    compute_resize_geometry,
    pad_image_batch,
    resolve_image_size_config,
    resize_image_with_geometry,
    transform_xyxy_to_normalized_cxcywh,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.tracking.io import IMAGE_EXTENSIONS, iter_image_files  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


PALETTE: list[tuple[int, int, int]] = [
    (41, 98, 255),
    (239, 71, 111),
    (6, 180, 162),
    (255, 176, 0),
    (131, 90, 241),
    (46, 204, 113),
    (236, 72, 153),
    (14, 165, 233),
    (217, 119, 6),
    (99, 102, 241),
]


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames_dir = resolve_project_path(args.frames_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = select_frames(frames_dir, args)
    if not frame_paths:
        raise RuntimeError(f"No RGB frames found under {frames_dir}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")

    config = load_config(args)
    patch_size = int(config.get("model", {}).get("patch_size", 16))
    _, image_resolution = resolve_image_size_config(config.get("data", {}), args.image_size)
    max_humans = int(args.max_humans or config.get("model", {}).get("num_smpl_queries", 20))

    images, geometries = load_sequence_images(frame_paths, image_resolution, patch_size, str(config["data"].get("resize_mode", "balanced")))
    priors = build_query_priors(frame_paths, geometries, args, max_humans, device) if args.query_source == "bedlam_sidecar" else None

    model = build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model, config, device)
    checkpoint = resolve_stage_checkpoint(args)
    load_training_checkpoint(model, checkpoint, device)
    smpl = SMPLLayer(require_smpl_model_dir(config, args)).to(device).eval()

    image_sequence = images.unsqueeze(0).to(device)
    with torch.inference_mode():
        predictions = run_model(model, image_sequence, priors)

    scene = build_scene_data(
        frame_paths=frame_paths,
        images=image_sequence,
        predictions=predictions,
        priors=priors,
        smpl=smpl,
        args=args,
        device=device,
    )
    validate_scene(scene, predictions, image_sequence)

    summary = build_summary(args, frame_paths, checkpoint, image_sequence, predictions, scene, output_dir)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"viewer": f"http://127.0.0.1:{int(args.port)}", "summary": str(summary_path)}, indent=2), flush=True)
    if bool(args.smoke_only):
        print("[ok] NLF-HSI VGGT sequence viewer smoke passed", flush=True)
        return

    server = viser.ViserServer(port=int(args.port))
    viewer = SequenceViewer(server=server, transforms=vtf, scene=scene, args=args)
    viewer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--query-source", choices=["bedlam_sidecar", "nlf_detector"], default="bedlam_sidecar")
    parser.add_argument("--preprocessed-root", default="outputs/preprocess/bedlam_boxes")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--stage2-dir", default="outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl")
    parser.add_argument("--checkpoint", default="", help="Explicit HSI checkpoint. If omitted, rank1 from stage2 checkpoint_topk_index.json is used.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/nlf_hsi_vggt_sequence_viewer")
    parser.add_argument("--device", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--depth-point-stride", type=int, default=4)
    parser.add_argument("--max-scene-depth", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--camera-frustum-scale", type=float, default=0.20)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--smoke-only", action="store_true", help="Run inference, validation, and summary export, then exit without serving Viser.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def ensure_viser_available() -> None:
    try:
        import viser  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "The Viser viewer requires the optional demo dependency 'viser'. "
            "Install it in the server environment with `pip install viser` or install the project demo extra."
        ) from exc


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def select_frames(frames_dir: Path, args: argparse.Namespace) -> list[Path]:
    paths = iter_image_files(frames_dir)
    start = max(0, int(args.start_index))
    stride = max(1, int(args.frame_stride))
    selected = paths[start::stride]
    if int(args.max_frames) > 0:
        selected = selected[: int(args.max_frames)]
    return selected


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(resolve_project_path(args.path_config)), load_yaml_config(resolve_project_path(args.train_config)))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = str(resolve_project_path(args.baseline_checkpoint))
    data_cfg = config.setdefault("data", {})
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    data_cfg["image_size"] = int(image_size)
    data_cfg["image_resolution"] = int(image_resolution)
    data_cfg.setdefault("resize_mode", "balanced")

    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["enable_hsi_refine"] = True
    model_cfg["smpl_provider"] = "nlf"
    model_cfg["num_smpl_queries"] = int(args.max_humans)
    model_cfg["smpl_query_box_prior"] = args.query_source == "bedlam_sidecar"
    model_cfg["smpl_query_patch_pool"] = False
    model_cfg["nlf_use_detector"] = args.query_source == "nlf_detector"
    model_cfg["nlf_require_boxes"] = args.query_source == "bedlam_sidecar"
    model_cfg["smpl_track_assignment_mode"] = "gt" if args.query_source == "bedlam_sidecar" else "none"
    model_cfg["smpl_use_external_track_prior"] = False
    if args.smpl_model_dir:
        config.setdefault("assets", {})["smpl_model_dir"] = str(resolve_project_path(args.smpl_model_dir))
    return config


def resolve_stage_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return resolve_project_path(args.checkpoint)
    stage_dir = resolve_project_path(args.stage2_dir)
    index_path = stage_dir / "checkpoint_topk_index.json"
    if not index_path.is_file():
        latest = stage_dir / "checkpoint_latest.pt"
        if latest.is_file():
            return latest
        raise FileNotFoundError(f"Missing stage2 checkpoint index and latest checkpoint: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"No top-k checkpoint entries in {index_path}")
    return resolve_project_path(entries[0]["path"])


def require_smpl_model_dir(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.smpl_model_dir:
        return str(resolve_project_path(args.smpl_model_dir))
    return require_path(config, "assets.smpl_model_dir", allow_empty=False)


def load_sequence_images(
    frame_paths: list[Path],
    image_resolution: int,
    patch_size: int,
    resize_mode: str,
) -> tuple[torch.Tensor, list[ResizeGeometry]]:
    tensors: list[torch.Tensor] = []
    geometries: list[ResizeGeometry] = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        geometry = compute_resize_geometry((image.height, image.width), image_resolution=image_resolution, patch_size=patch_size, mode=resize_mode)
        resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        geometries.append(geometry)
    batch, pads = pad_image_batch(tensors, patch_size=patch_size, value=1.0)
    input_hw = (int(batch.shape[-2]), int(batch.shape[-1]))
    padded_geometries = [
        replace(geometry, input_hw=input_hw, pad_xyxy=tuple(int(v) for v in pads[idx]))
        for idx, geometry in enumerate(geometries)
    ]
    return batch, padded_geometries


def build_query_priors(
    frame_paths: list[Path],
    geometries: list[ResizeGeometry],
    args: argparse.Namespace,
    max_humans: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boxes = torch.zeros(1, len(frame_paths), max_humans, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(1, len(frame_paths), max_humans, dtype=torch.bool, device=device)
    track_ids = torch.full((1, len(frame_paths), max_humans), -1, dtype=torch.long, device=device)
    track_mask = torch.zeros(1, len(frame_paths), max_humans, dtype=torch.bool, device=device)
    preprocessed_root = resolve_project_path(args.preprocessed_root)
    bedlam_root = resolve_project_path(args.bedlam_root) if args.bedlam_root else None

    for frame_idx, image_path in enumerate(frame_paths):
        frame = load_sidecar_frame(preprocessed_root, bedlam_root, image_path)
        persons = frame.get("persons", [])
        if not isinstance(persons, list):
            continue
        image_h, image_w = frame_hw(frame, geometries[frame_idx].orig_hw)
        slot = 0
        for person_idx, person in enumerate(persons):
            if slot >= max_humans:
                break
            if not person_train_valid(person) or not bool(person.get("bbox_valid", False)):
                continue
            xyxy = person_xyxy(person, image_w=image_w, image_h=image_h)
            if xyxy is None:
                continue
            box, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometries[frame_idx])
            if not valid:
                continue
            boxes[0, frame_idx, slot] = torch.as_tensor(box, dtype=torch.float32, device=device)
            box_mask[0, frame_idx, slot] = True
            track_ids[0, frame_idx, slot] = int(person_track_id(person, person_idx))
            track_mask[0, frame_idx, slot] = True
            slot += 1
    if not bool(box_mask.any()):
        raise RuntimeError("No valid sidecar boxes were loaded. Check FRAMES_DIR, BEDLAM_ROOT, and PREPROCESSED_ROOT.")
    return {"smpl_query_boxes": boxes, "smpl_query_boxes_mask": box_mask, "smpl_track_ids": track_ids, "smpl_track_mask": track_mask}


def load_sidecar_frame(preprocessed_root: Path, bedlam_root: Path | None, image_path: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if bedlam_root is not None:
        try:
            rel = image_path.resolve().relative_to(bedlam_root.resolve())
            parts = rel.parts
            if len(parts) >= 4 and parts[2] == "rgb":
                candidates.append(preprocessed_root / parts[0] / parts[1] / "smpl_boxes" / f"{image_path.stem}.pkl")
        except ValueError:
            pass
    candidates.extend(
        [
            preprocessed_root / "smpl_boxes" / f"{image_path.stem}.pkl",
            preprocessed_root / f"{image_path.stem}.pkl",
        ]
    )
    for path in candidates:
        if path.is_file():
            with path.open("rb") as file:
                data = pickle.load(file)
            if not isinstance(data, dict):
                raise TypeError(f"Sidecar must contain a frame dict: {path}")
            return data
    raise FileNotFoundError(f"Missing sidecar for frame {image_path.name}. Tried: {[str(path) for path in candidates]}")


def frame_hw(frame: dict[str, Any], fallback_hw: tuple[int, int]) -> tuple[int, int]:
    if "image_hw" in frame:
        h, w = frame["image_hw"]
        return int(h), int(w)
    return int(fallback_hw[0]), int(fallback_hw[1])


def person_train_valid(person: dict[str, Any]) -> bool:
    if "train_valid" in person:
        return bool(person["train_valid"])
    if "valid" in person:
        return bool(person["valid"])
    return bool(person.get("bbox_valid", False))


def person_xyxy(person: dict[str, Any], image_w: int, image_h: int) -> np.ndarray | None:
    if "bbox_xyxy_pixels" in person:
        return np.asarray(person["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
    if "bbox_cxcywh_norm" in person:
        cx, cy, bw, bh = np.asarray(person["bbox_cxcywh_norm"], dtype=np.float32).reshape(4)
        return np.asarray(
            [
                (cx - 0.5 * bw) * float(image_w),
                (cy - 0.5 * bh) * float(image_h),
                (cx + 0.5 * bw) * float(image_w),
                (cy + 0.5 * bh) * float(image_h),
            ],
            dtype=np.float32,
        )
    return None


def person_track_id(person: dict[str, Any], fallback_index: int) -> int:
    for key in ("person_id", "track_id_prior", "track_id", "person_index"):
        if key not in person:
            continue
        try:
            value = int(person[key])
            if value >= 0:
                return value
        except (TypeError, ValueError):
            continue
    return int(fallback_index)


def run_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    priors: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if priors is not None:
        kwargs.update(
            {
                "smpl_query_boxes": priors["smpl_query_boxes"],
                "smpl_query_boxes_mask": priors["smpl_query_boxes_mask"],
                "smpl_track_ids": priors["smpl_track_ids"],
                "smpl_track_mask": priors["smpl_track_mask"],
            }
        )
    predictions = model(images, **kwargs)
    predictions["images"] = images
    return predictions


def build_scene_data(
    frame_paths: list[Path],
    images: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    priors: dict[str, torch.Tensor] | None,
    smpl: SMPLLayer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    image_hw = tuple(int(v) for v in images.shape[-2:])
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"].detach().float(), image_size_hw=image_hw, build_intrinsics=True)
    raw_depth = canonical_depth(predictions["depth"]).detach().float()
    hsi_depth = raw_depth
    if "hsi_scene_scale" in predictions and "hsi_scene_depth_bias" in predictions:
        scale = predictions["hsi_scene_scale"].detach().float().reshape(raw_depth.shape[:2] + (1, 1)).to(raw_depth.device)
        bias = predictions["hsi_scene_depth_bias"].detach().float().reshape(raw_depth.shape[:2] + (1, 1)).to(raw_depth.device)
        hsi_depth = raw_depth * scale + bias

    people = decode_people(predictions, smpl, args, device)
    faces = np.asarray(smpl.faces, dtype=np.int64).reshape(-1, 3)
    track_palette: dict[int, int] = {}
    frames = []
    for idx, image_path in enumerate(frame_paths):
        extrinsic = extrinsics[0, idx].detach().float().cpu().numpy()
        intrinsic = intrinsics[0, idx].detach().float().cpu().numpy()
        rgb = images[0, idx].detach().float().cpu()
        raw_points, raw_colors = depth_to_world_points(raw_depth[0, idx], rgb, intrinsic, extrinsic, args)
        hsi_points, hsi_colors = depth_to_world_points(hsi_depth[0, idx], rgb, intrinsic, extrinsic, args)
        frame_people = select_frame_people(predictions, people, priors, idx, extrinsic, faces, track_palette, args)
        frames.append(
            {
                "frame_index": int(idx),
                "frame_id": image_path.stem,
                "image": str(image_path),
                "raw_points": raw_points,
                "raw_colors": raw_colors,
                "hsi_points": hsi_points,
                "hsi_colors": hsi_colors,
                "people": frame_people,
                "camera": camera_pose_from_extrinsic(extrinsic, intrinsic),
                "hsi_scene_scale": prediction_scalar(predictions, "hsi_scene_scale", idx),
                "hsi_scene_depth_bias": prediction_scalar(predictions, "hsi_scene_depth_bias", idx),
            }
        )
    return {"frames": frames, "image_hw": list(image_hw), "track_palette": track_palette}


def canonical_depth(tensor: torch.Tensor) -> torch.Tensor:
    depth = tensor
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W] or [B,S,H,W,1], got {tuple(tensor.shape)}")
    return depth


def depth_to_world_points(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    depth = depth.detach().float()
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    step = max(1, int(args.depth_point_stride))
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=torch.float32),
        torch.arange(0, width, step, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()]
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points = torch.stack([x, y, z], dim=-1)
    rgb_use = rgb.to(device=depth.device, dtype=torch.float32)
    if tuple(rgb_use.shape[-2:]) != (height, width):
        rgb_use = F.interpolate(rgb_use[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    colors = (rgb_use[:, ys.long(), xs.long()].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    mask = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    if float(args.max_scene_depth) > 0:
        mask = mask & (z <= float(args.max_scene_depth))
    points_np = points[mask].detach().cpu().numpy().astype(np.float32, copy=False)
    colors_np = colors[mask].detach().cpu().numpy().astype(np.uint8, copy=False)
    return camera_points_to_world_np(points_np, extrinsic), colors_np


def camera_points_to_world_np(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation[None, :]) @ rotation).astype(np.float32)


def decode_people(
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix, pose_key, beta_key, transl_key in [
        ("base", "pred_poses", "pred_betas", "pred_transl_cam"),
        ("hsi", "hsi_refined_pred_poses", "hsi_refined_pred_betas", "hsi_refined_pred_transl_cam"),
    ]:
        if pose_key not in predictions or beta_key not in predictions or transl_key not in predictions:
            continue
        poses = predictions[pose_key].detach()
        betas = predictions[beta_key].detach()
        transl = predictions[transl_key].detach()
        shape = poses.shape[:3]
        with torch.no_grad():
            vertices, _ = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
        vertices = vertices.reshape(*shape, vertices.shape[-2], 3).to(device=device, dtype=transl.dtype) + transl[..., None, :]
        out[f"{prefix}_vertices_cam"] = vertices.detach()
    return out


def select_frame_people(
    predictions: dict[str, torch.Tensor],
    decoded: dict[str, torch.Tensor],
    priors: dict[str, torch.Tensor] | None,
    frame_index: int,
    extrinsic: np.ndarray,
    faces: np.ndarray,
    track_palette: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    confs = predictions["pred_confs"][0, frame_index, :, 0].detach().float().cpu()
    if priors is not None:
        valid = priors["smpl_query_boxes_mask"][0, frame_index].detach().cpu().bool()
        track_ids = priors["smpl_track_ids"][0, frame_index].detach().cpu().long()
    else:
        valid = confs >= float(args.conf_threshold)
        track_ids = torch.arange(confs.numel(), dtype=torch.long)
    order = torch.argsort(confs, descending=True).tolist()
    people: list[dict[str, Any]] = []
    for query_idx in order:
        if not bool(valid[query_idx]) or float(confs[query_idx]) < float(args.conf_threshold):
            continue
        track_id = int(track_ids[query_idx].item()) if int(track_ids[query_idx].item()) >= 0 else int(query_idx)
        color = PALETTE[palette_index_for_track(track_id, track_palette)]
        item: dict[str, Any] = {
            "query_index": int(query_idx),
            "track_id": int(track_id),
            "confidence": float(confs[query_idx].item()),
            "color": color,
            "faces": faces,
        }
        for prefix in ("base", "hsi"):
            key = f"{prefix}_vertices_cam"
            if key in decoded:
                mesh_cam = decoded[key][0, frame_index, query_idx].detach().float().cpu().numpy()
                item[f"{prefix}_vertices"] = camera_points_to_world_np(mesh_cam, extrinsic)
        people.append(item)
    return people


def palette_index_for_track(track_id: int, state: dict[int, int]) -> int:
    if track_id not in state:
        state[track_id] = len(state) % len(PALETTE)
    return state[track_id]


def camera_pose_from_extrinsic(extrinsic: np.ndarray, intrinsic: np.ndarray) -> dict[str, Any]:
    rotation_w2c = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    rotation_c2w = rotation_w2c.T
    position = -rotation_c2w @ translation
    fy = max(float(intrinsic[1, 1]), 1e-6)
    height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
    width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
    return {
        "rotation_c2w": rotation_c2w.astype(np.float32),
        "position": position.astype(np.float32),
        "fov": float(2.0 * np.arctan((height * 0.5) / fy)),
        "aspect": float(width / height),
    }


def prediction_scalar(predictions: dict[str, torch.Tensor], key: str, frame_index: int) -> float | None:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return float(value[0, frame_index].detach().float().reshape(-1)[0].cpu())


def validate_scene(scene: dict[str, Any], predictions: dict[str, torch.Tensor], images: torch.Tensor) -> None:
    required = ["pose_enc", "depth", "hsi_scene_scale", "hsi_scene_depth_bias"]
    missing = [key for key in required if key not in predictions]
    if missing:
        raise RuntimeError(f"Missing required prediction fields for HSI viewer: {missing}")
    if "nlf_image_hw" in predictions:
        nlf_hw = [int(v) for v in predictions["nlf_image_hw"].detach().cpu().reshape(-1).tolist()]
        if nlf_hw != [int(images.shape[-2]), int(images.shape[-1])]:
            raise RuntimeError(f"NLF image HW mismatch: nlf={nlf_hw} images={list(images.shape[-2:])}")
    if not scene["frames"]:
        raise RuntimeError("Viewer scene has no frames")
    if max(frame["hsi_points"].shape[0] for frame in scene["frames"]) <= 0:
        raise RuntimeError("HSI/world point cloud has no valid points")
    if not any("hsi_vertices" in person for frame in scene["frames"] for person in frame["people"]):
        raise RuntimeError("No finite HSI SMPL meshes were decoded")


def build_summary(
    args: argparse.Namespace,
    frame_paths: list[Path],
    checkpoint: Path,
    images: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    scene: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "frames_dir": str(resolve_project_path(args.frames_dir)),
        "num_frames": len(frame_paths),
        "checkpoint": str(checkpoint),
        "query_source": str(args.query_source),
        "image_shape": list(images.shape),
        "nlf_image_hw": [int(v) for v in predictions.get("nlf_image_hw", torch.tensor([], device=images.device)).detach().cpu().reshape(-1).tolist()],
        "point_counts_hsi": [int(frame["hsi_points"].shape[0]) for frame in scene["frames"]],
        "people_counts": [int(len(frame["people"])) for frame in scene["frames"]],
        "hsi_scene_scale": [frame["hsi_scene_scale"] for frame in scene["frames"]],
        "hsi_scene_depth_bias": [frame["hsi_scene_depth_bias"] for frame in scene["frames"]],
        "output_dir": str(output_dir),
    }


class SequenceViewer:
    def __init__(self, server: Any, transforms: Any, scene: dict[str, Any], args: argparse.Namespace) -> None:
        self.server = server
        self.transforms = transforms
        self.scene = scene
        self.args = args
        self.handles: list[dict[str, Any]] = []
        self.current_step = 0
        self.clients: dict[int, Any] = {}
        self._build_scene()
        self._build_gui()
        self._register_clients()
        self._update_visibility()

    def run(self) -> None:
        try:
            while True:
                if bool(self.play.value):
                    self.current_step = (int(self.timestep.value) + 1) % len(self.scene["frames"])
                    self.timestep.value = self.current_step
                    self._update_visibility()
                    if bool(self.follow_camera.value):
                        self._follow_pred_camera(self.current_step)
                time.sleep(1.0 / max(float(self.fps.value), 1.0))
        except KeyboardInterrupt:
            print("[viewer] stopped", flush=True)

    def _build_scene(self) -> None:
        for frame in self.scene["frames"]:
            idx = int(frame["frame_index"])
            frame_handles: dict[str, Any] = {"raw": [], "hsi": [], "base_humans": [], "hsi_humans": [], "cameras": []}
            frame_handles["raw"].append(
                add_point_cloud(self.server, f"/frames/{idx:04d}/points_raw_depth", frame["raw_points"], frame["raw_colors"], float(self.args.point_size))
            )
            frame_handles["hsi"].append(
                add_point_cloud(self.server, f"/frames/{idx:04d}/points_hsi_depth", frame["hsi_points"], frame["hsi_colors"], float(self.args.point_size))
            )
            for person in frame["people"]:
                color = tuple(int(v) for v in person["color"])
                track_id = int(person["track_id"])
                query_idx = int(person["query_index"])
                if "base_vertices" in person:
                    frame_handles["base_humans"].append(
                        add_mesh(self.server, f"/frames/{idx:04d}/human_base_t{track_id}_q{query_idx}", person["base_vertices"], person["faces"], color)
                    )
                if "hsi_vertices" in person:
                    frame_handles["hsi_humans"].append(
                        add_mesh(self.server, f"/frames/{idx:04d}/human_hsi_t{track_id}_q{query_idx}", person["hsi_vertices"], person["faces"], color)
                    )
            frame_handles["cameras"].append(add_camera(self.server, self.transforms, f"/frames/{idx:04d}/camera", frame["camera"], self.args))
            self.handles.append(frame_handles)

    def _build_gui(self) -> None:
        self.timestep = add_slider(self.server, "Timestep", 0, len(self.scene["frames"]) - 1, 1, 0)
        self.play = add_checkbox(self.server, "Playing", False)
        self.fps = add_slider(self.server, "FPS", 1, 30, 1, 6)
        self.mode = add_dropdown(self.server, "Mode", ["4D current frame", "3D accumulate", "Hybrid"], "4D current frame")
        self.depth_source = add_dropdown(self.server, "Depth Source", ["hsi_depth", "raw_depth"], "hsi_depth")
        self.show_hsi = add_checkbox(self.server, "Show HSI SMPL", True)
        self.show_base = add_checkbox(self.server, "Show Base SMPL", False)
        self.show_cameras = add_checkbox(self.server, "Show Cameras", True)
        self.follow_camera = add_checkbox(self.server, "Follow Pred Camera", False)
        for handle in [self.timestep, self.mode, self.depth_source, self.show_hsi, self.show_base, self.show_cameras, self.follow_camera]:
            bind_update(handle, self._on_gui_update)

    def _register_clients(self) -> None:
        if hasattr(self.server, "on_client_connect"):
            @self.server.on_client_connect
            def _on_connect(client: Any) -> None:
                self.clients[int(getattr(client, "client_id", len(self.clients)))] = client

    def _on_gui_update(self, _: Any = None) -> None:
        self.current_step = int(self.timestep.value)
        self._update_visibility()
        if bool(self.follow_camera.value):
            self._follow_pred_camera(self.current_step)

    def _update_visibility(self) -> None:
        current = int(self.timestep.value)
        mode = str(self.mode.value)
        depth_source = str(self.depth_source.value)
        for idx, frame_handles in enumerate(self.handles):
            if mode == "3D accumulate":
                show_points = idx <= current
                show_humans = idx <= current
            elif mode == "Hybrid":
                show_points = idx <= current
                show_humans = idx == current
            else:
                show_points = idx == current
                show_humans = idx == current
            set_group_visible(frame_handles["raw"], show_points and depth_source == "raw_depth")
            set_group_visible(frame_handles["hsi"], show_points and depth_source == "hsi_depth")
            set_group_visible(frame_handles["base_humans"], show_humans and bool(self.show_base.value))
            set_group_visible(frame_handles["hsi_humans"], show_humans and bool(self.show_hsi.value))
            set_group_visible(frame_handles["cameras"], bool(self.show_cameras.value) and (idx <= current if mode != "4D current frame" else idx == current))

    def _follow_pred_camera(self, step: int) -> None:
        camera = self.scene["frames"][int(step)]["camera"]
        rotation = camera["rotation_c2w"]
        position = camera["position"]
        wxyz = self.transforms.SO3.from_matrix(rotation).wxyz
        clients = list(self.clients.values())
        if hasattr(self.server, "get_clients"):
            try:
                clients = list(self.server.get_clients().values())
            except Exception:
                pass
        for client in clients:
            try:
                client.camera.wxyz = wxyz
                client.camera.position = position
                client.camera.fov = float(camera["fov"])
            except Exception:
                continue


def scene_api(server: Any) -> Any:
    return getattr(server, "scene", server)


def gui_api(server: Any) -> Any:
    return getattr(server, "gui", server)


def add_point_cloud(server: Any, name: str, points: np.ndarray, colors: np.ndarray, point_size: float) -> Any:
    api = scene_api(server)
    try:
        return api.add_point_cloud(name=name, points=points, colors=colors, point_size=point_size)
    except TypeError:
        return api.add_point_cloud(name, points, colors, point_size=point_size)


def add_mesh(server: Any, name: str, vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]) -> Any:
    api = scene_api(server)
    color_float = tuple(float(v) / 255.0 for v in color)
    try:
        return api.add_mesh_simple(name=name, vertices=vertices, faces=faces, color=color_float)
    except TypeError:
        return api.add_mesh_simple(name, vertices, faces, color=color_float)


def add_camera(server: Any, transforms: Any, name: str, camera: dict[str, Any], args: argparse.Namespace) -> Any:
    api = scene_api(server)
    wxyz = transforms.SO3.from_matrix(camera["rotation_c2w"]).wxyz
    try:
        return api.add_camera_frustum(
            name=name,
            fov=float(camera["fov"]),
            aspect=float(camera["aspect"]),
            scale=float(args.camera_frustum_scale),
            wxyz=wxyz,
            position=camera["position"],
            color=(255, 255, 255),
        )
    except TypeError:
        return api.add_camera_frustum(name, float(camera["fov"]), float(camera["aspect"]), float(args.camera_frustum_scale), wxyz, camera["position"])


def add_slider(server: Any, name: str, min_value: int, max_value: int, step: int, initial: int) -> Any:
    api = gui_api(server)
    try:
        return api.add_slider(name, min=min_value, max=max_value, step=step, initial_value=initial)
    except AttributeError:
        return server.add_gui_slider(name, min=min_value, max=max_value, step=step, initial_value=initial)


def add_checkbox(server: Any, name: str, initial: bool) -> Any:
    api = gui_api(server)
    try:
        return api.add_checkbox(name, initial_value=initial)
    except AttributeError:
        return server.add_gui_checkbox(name, initial)


def add_dropdown(server: Any, name: str, options: list[str], initial: str) -> Any:
    api = gui_api(server)
    try:
        return api.add_dropdown(name, options=options, initial_value=initial)
    except AttributeError:
        return server.add_gui_dropdown(name, options, initial)


def bind_update(handle: Any, callback: Any) -> None:
    if hasattr(handle, "on_update"):
        handle.on_update(callback)


def set_group_visible(handles: list[Any], visible: bool) -> None:
    for handle in handles:
        try:
            handle.visible = bool(visible)
        except Exception:
            pass


if __name__ == "__main__":
    main()
