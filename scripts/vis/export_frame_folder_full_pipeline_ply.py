#!/usr/bin/env python
"""Run folder-frame VGGT-Omega inference and export per-frame PLY assets."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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
    extract_state_dict,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    write_ply_meshes,
    write_ply_vertices_faces,
)
from vggt_omega.data.geometry import (  # noqa: E402
    ResizeGeometry,
    compute_resize_geometry,
    pixel_mask_to_patch_mask_hw,
    resize_image_with_geometry,
    resize_mask_with_geometry,
    resolve_image_size_config,
    transform_xyxy_to_normalized_cxcywh,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.tracking.io import iter_image_files  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


PAPER_PALETTE_10: list[tuple[int, int, int]] = [
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
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames_dir = resolve_project_path(args.frames_dir)
    sidecar_root = resolve_project_path(args.sidecar_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    patch_size = int(config.get("model", {}).get("patch_size", 16))
    _, image_resolution = resolve_image_size_config(config.get("data", {}), args.image_size)
    num_queries = int(config.get("model", {}).get("num_smpl_queries", 20))

    model = build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, resolve_project_path(args.checkpoint), device)
    smpl = SMPLLayer(require_smpl_model_dir(config, args)).to(device).eval()
    faces = np.asarray(smpl.faces, dtype=np.int64).reshape(-1, 3)

    frame_paths = iter_selected_frames(frames_dir, args)
    if not frame_paths:
        raise RuntimeError(f"No image frames found under {frames_dir}")

    manifest: dict[str, Any] = {
        "purpose": "Full project pipeline export for paper/inspection assets.",
        "frames_dir": str(frames_dir),
        "sidecar_root": str(sidecar_root),
        "checkpoint": str(resolve_project_path(args.checkpoint)),
        "train_config": str(args.train_config),
        "path_config": str(args.path_config),
        "image_resolution": int(image_resolution),
        "resize_mode": str(config.get("data", {}).get("resize_mode", "balanced")),
        "patch_size": int(patch_size),
        "num_model_queries": int(num_queries),
        "max_export_people": int(args.max_export_people),
        "palette_rgb": [list(color) for color in PAPER_PALETTE_10],
        "frames": [],
    }
    track_palette: dict[int, int] = {}

    for out_index, image_path in enumerate(frame_paths):
        frame_record = process_frame(
            image_path=image_path,
            sidecar_root=sidecar_root,
            output_dir=output_dir,
            frame_index=out_index,
            model=model,
            smpl=smpl,
            faces=faces,
            config=config,
            image_resolution=image_resolution,
            patch_size=patch_size,
            num_queries=num_queries,
            track_palette=track_palette,
            args=args,
            device=device,
        )
        manifest["frames"].append(frame_record)
        if args.log_interval > 0 and (out_index + 1) % int(args.log_interval) == 0:
            print(f"[export] {out_index + 1}/{len(frame_paths)} frame={image_path.name} people={len(frame_record['people'])}", flush=True)

    manifest["track_palette"] = {str(k): {"palette_index": v, "rgb": list(PAPER_PALETTE_10[v])} for k, v in sorted(track_palette.items())}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "num_frames": len(manifest["frames"])}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sidecar-root", required=True, help="Output folder from prepare_video_person_tracks.py for this frame folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/full_pipeline_frame_folder_ply")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-export-people", type=int, default=10)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--track-iou-threshold", type=float, default=0.20)
    parser.add_argument("--mask-patch-threshold", type=float, default=0.10)
    parser.add_argument("--min-mask-patches", type=int, default=4)
    parser.add_argument("--depth-point-stride", type=int, default=2)
    parser.add_argument("--max-scene-depth", type=float, default=30.0)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--use-hsi-refined", action="store_true")
    parser.add_argument("--export-combined-frame", action="store_true", help="Also export one PLY containing environment and all people per frame.")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    data_cfg = config.setdefault("data", {})
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    data_cfg["image_size"] = int(image_size)
    data_cfg["image_resolution"] = int(image_resolution)
    data_cfg.setdefault("resize_mode", "balanced")
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["smpl_query_box_prior"] = True
    model_cfg["smpl_query_patch_pool"] = True
    model_cfg["smpl_query_patch_pool_mode"] = "mask_intersection"
    model_cfg.setdefault("smpl_query_mask_min_patch_count", 4)
    model_cfg.setdefault("smpl_query_mask_fallback_to_box", True)
    model_cfg.setdefault("smpl_track_assignment_mode", "external_prior")
    model_cfg.setdefault("smpl_use_external_track_prior", True)
    return config


def process_frame(
    image_path: Path,
    sidecar_root: Path,
    output_dir: Path,
    frame_index: int,
    model: torch.nn.Module,
    smpl: SMPLLayer,
    faces: np.ndarray,
    config: dict[str, Any],
    image_resolution: int,
    patch_size: int,
    num_queries: int,
    track_palette: dict[int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    image_tensor, original, geometry = load_frame_tensor(image_path, image_resolution, patch_size, str(config["data"].get("resize_mode", "balanced")))
    priors = build_frame_priors(
        sidecar_root=sidecar_root,
        frame_id=image_path.stem,
        geometry=geometry,
        num_queries=num_queries,
        patch_size=patch_size,
        mask_patch_threshold=float(args.mask_patch_threshold),
        min_mask_patches=int(args.min_mask_patches),
        track_iou_threshold=float(args.track_iou_threshold),
        device=device,
    )
    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=priors["smpl_query_boxes"],
            smpl_query_boxes_mask=priors["smpl_query_boxes_mask"],
            smpl_query_patch_masks=priors["smpl_query_patch_masks"],
            external_track_ids=priors["external_track_ids"],
            external_track_mask=priors["external_track_mask"],
            external_track_confidence=priors["external_track_confidence"],
        )
    frame_dir = output_dir / "frames" / f"{frame_index:06d}_{image_path.stem}"
    env_dir = output_dir / "environment"
    people_dir = output_dir / "smpl_people" / f"{frame_index:06d}_{image_path.stem}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    people_dir.mkdir(parents=True, exist_ok=True)

    env_vertices, env_colors = depth_to_camera_points(
        depth=predictions["depth"][0, 0].detach(),
        pose_enc=predictions["pose_enc"],
        image_tensor=image_tensor[0, 0].detach(),
        stride=int(args.depth_point_stride),
        max_depth=float(args.max_scene_depth),
    )
    env_path = env_dir / f"{frame_index:06d}_{image_path.stem}_environment_rgb_depth_points.ply"
    write_ply_vertices_faces(env_path, env_vertices, env_colors, np.empty((0, 3), dtype=np.int64))

    people = select_people(predictions, priors, max_people=int(args.max_export_people), conf_threshold=float(args.conf_threshold))
    meshes, mesh_colors = decode_people_meshes(people, predictions, smpl, track_palette, bool(args.use_hsi_refined), device)
    people_records = []
    for person, mesh, color in zip(people, meshes, mesh_colors, strict=True):
        track_id = int(person["track_id"])
        palette_index = int(person["palette_index"])
        person_path = people_dir / f"person_track{track_id:03d}_q{int(person['query_index']):02d}.ply"
        write_ply_meshes(person_path, [mesh], faces, [color])
        people_records.append(
            {
                **person,
                "color_rgb": list(color),
                "palette_index": palette_index,
                "ply": str(person_path),
            }
        )

    combined_path = None
    if args.export_combined_frame:
        combined_path = frame_dir / f"{frame_index:06d}_{image_path.stem}_environment_plus_people.ply"
        write_combined_scene_ply(combined_path, env_vertices, env_colors, meshes, faces, mesh_colors)

    return {
        "frame_index": int(frame_index),
        "frame_id": image_path.stem,
        "image": str(image_path),
        "orig_hw": [int(geometry.orig_hw[0]), int(geometry.orig_hw[1])],
        "model_hw": [int(geometry.input_hw[0]), int(geometry.input_hw[1])],
        "num_query_boxes": int(priors["smpl_query_boxes_mask"].sum().detach().cpu().item()),
        "num_query_patch_masks": int(priors["smpl_query_patch_masks_valid"].sum().detach().cpu().item()),
        "environment_ply": str(env_path),
        "combined_ply": None if combined_path is None else str(combined_path),
        "people": people_records,
    }


def load_frame_tensor(image_path: Path, image_resolution: int, patch_size: int, resize_mode: str) -> tuple[torch.Tensor, Image.Image, ResizeGeometry]:
    image = Image.open(image_path).convert("RGB")
    geometry = compute_resize_geometry((image.height, image.width), image_resolution=image_resolution, patch_size=patch_size, mode=resize_mode)
    resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)
    return tensor, image, geometry


def build_frame_priors(
    sidecar_root: Path,
    frame_id: str,
    geometry: ResizeGeometry,
    num_queries: int,
    patch_size: int,
    mask_patch_threshold: float,
    min_mask_patches: int,
    track_iou_threshold: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    frame = load_sidecar_frame(sidecar_root, frame_id)
    detections = sorted(frame_detections(frame), key=lambda det: (float(det.get("det_score", det.get("score", 0.0))), det_area(det)), reverse=True)
    detections = detections[: int(num_queries)]
    boxes = torch.zeros(1, 1, num_queries, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(1, 1, num_queries, dtype=torch.bool, device=device)
    patch_grid = (int(geometry.input_hw[0]) // int(patch_size), int(geometry.input_hw[1]) // int(patch_size))
    patch_masks = torch.zeros(1, 1, num_queries, patch_grid[0] * patch_grid[1], dtype=torch.bool, device=device)
    patch_mask_valid = torch.zeros(1, 1, num_queries, dtype=torch.bool, device=device)
    external_ids = torch.full((1, 1, num_queries), -1, dtype=torch.long, device=device)
    external_mask = torch.zeros(1, 1, num_queries, dtype=torch.bool, device=device)
    external_conf = torch.zeros(1, 1, num_queries, dtype=torch.float32, device=device)

    image_h, image_w = frame_hw(frame, geometry.orig_hw)
    for slot, det in enumerate(detections):
        xyxy = det_xyxy(det, image_w=image_w, image_h=image_h)
        box, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometry)
        if not valid:
            continue
        boxes[0, 0, slot] = torch.as_tensor(box, dtype=torch.float32, device=device)
        box_mask[0, 0, slot] = True
        mask = load_det_patch_mask(sidecar_root, det, geometry, patch_size, mask_patch_threshold, min_mask_patches)
        if mask is not None:
            patch_masks[0, 0, slot] = torch.as_tensor(mask.reshape(-1), dtype=torch.bool, device=device)
            patch_mask_valid[0, 0, slot] = True
        track_id, track_conf = match_track_id(det, xyxy, frame, image_w, image_h, track_iou_threshold)
        if track_id >= 0:
            external_ids[0, 0, slot] = int(track_id)
            external_mask[0, 0, slot] = True
            external_conf[0, 0, slot] = float(track_conf)

    return {
        "smpl_query_boxes": boxes,
        "smpl_query_boxes_mask": box_mask,
        "smpl_query_patch_masks": patch_masks,
        "smpl_query_patch_masks_valid": patch_mask_valid,
        "external_track_ids": external_ids,
        "external_track_mask": external_mask,
        "external_track_confidence": external_conf,
    }


def select_people(predictions: dict[str, torch.Tensor], priors: dict[str, torch.Tensor], max_people: int, conf_threshold: float) -> list[dict[str, Any]]:
    confs = predictions["pred_confs"][0, 0, :, 0].detach().float().cpu()
    valid = priors["smpl_query_boxes_mask"][0, 0].detach().cpu().bool()
    order = torch.argsort(confs, descending=True).tolist()
    assigned = predictions.get("assigned_track_ids", priors["external_track_ids"])
    assigned_cpu = assigned[0, 0].detach().cpu().long() if isinstance(assigned, torch.Tensor) else priors["external_track_ids"][0, 0].detach().cpu().long()
    out = []
    for rank, query_idx in enumerate(order):
        if len(out) >= int(max_people):
            break
        if not bool(valid[query_idx]) or float(confs[query_idx]) < float(conf_threshold):
            continue
        track_id = int(assigned_cpu[query_idx].item())
        if track_id < 0:
            track_id = 10_000 + int(query_idx)
        out.append(
            {
                "rank": int(rank),
                "query_index": int(query_idx),
                "confidence": float(confs[query_idx].item()),
                "track_id": int(track_id),
            }
        )
    return out


def decode_people_meshes(
    people: list[dict[str, Any]],
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    track_palette: dict[int, int],
    use_hsi_refined: bool,
    device: torch.device,
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    if not people:
        return [], []
    pose_key = "hsi_refined_pred_poses" if use_hsi_refined and "hsi_refined_pred_poses" in predictions else "pred_poses"
    betas_key = "hsi_refined_pred_betas" if use_hsi_refined and "hsi_refined_pred_betas" in predictions else "pred_betas"
    transl_key = "hsi_refined_pred_transl_cam" if use_hsi_refined and "hsi_refined_pred_transl_cam" in predictions else "pred_transl_cam"
    query_indices = torch.as_tensor([int(item["query_index"]) for item in people], dtype=torch.long, device=device)
    poses = predictions[pose_key][0, 0, query_indices].detach()
    betas = predictions[betas_key][0, 0, query_indices].detach()
    transl = predictions[transl_key][0, 0, query_indices].detach()
    with torch.no_grad():
        vertices, _ = smpl(poses.reshape(-1, 72), betas)
    vertices = vertices + transl[:, None, :]
    meshes = [vertices[idx].detach().cpu().numpy() for idx in range(vertices.shape[0])]
    colors = []
    for item in people:
        palette_index = palette_index_for_track(int(item["track_id"]), track_palette)
        item["palette_index"] = int(palette_index)
        colors.append(PAPER_PALETTE_10[palette_index])
    return meshes, colors


def depth_to_camera_points(
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    image_tensor: torch.Tensor,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [H,W] or [H,W,1], got {tuple(depth.shape)}")
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(height, width), build_intrinsics=True)
    intrinsics_0 = intrinsics[0, 0].to(device=depth.device, dtype=depth.dtype)
    step = max(int(stride), 1)
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=depth.dtype),
        torch.arange(0, width, step, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()].clamp(min=1e-6)
    x = (xs - intrinsics_0[0, 2]) / intrinsics_0[0, 0].clamp(min=1e-6) * z
    y = (ys - intrinsics_0[1, 2]) / intrinsics_0[1, 1].clamp(min=1e-6) * z
    points = torch.stack([x, y, z], dim=-1)
    rgb = image_tensor.to(device=depth.device)
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    colors = (rgb[:, ys.long(), xs.long()].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    mask = torch.isfinite(points).all(dim=-1) & torch.isfinite(z) & (z > 1e-6) & (z <= float(max_depth))
    return points[mask].detach().cpu().numpy(), colors[mask].detach().cpu().numpy()


def write_combined_scene_ply(
    path: Path,
    env_vertices: np.ndarray,
    env_colors: np.ndarray,
    meshes: list[np.ndarray],
    faces: np.ndarray,
    mesh_colors: list[tuple[int, int, int]],
) -> None:
    vertices_parts = [np.asarray(env_vertices, dtype=np.float32).reshape(-1, 3)]
    colors_parts = [np.asarray(env_colors, dtype=np.uint8).reshape(-1, 3)]
    faces_parts = [np.empty((0, 3), dtype=np.int64)]
    offset = vertices_parts[0].shape[0]
    face_template = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for mesh_idx, mesh in enumerate(meshes):
        vertices = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(vertices).all(axis=1)
        index_map = -np.ones(vertices.shape[0], dtype=np.int64)
        index_map[finite] = np.arange(int(finite.sum()), dtype=np.int64)
        valid_faces = finite[face_template].all(axis=1)
        clean_faces = index_map[face_template[valid_faces]]
        clean_vertices = vertices[finite]
        color = np.tile(np.asarray(mesh_colors[mesh_idx % len(mesh_colors)], dtype=np.uint8).reshape(1, 3), (clean_vertices.shape[0], 1))
        vertices_parts.append(clean_vertices)
        colors_parts.append(color)
        faces_parts.append(clean_faces + offset)
        offset += clean_vertices.shape[0]
    write_ply_vertices_faces(path, np.concatenate(vertices_parts, axis=0), np.concatenate(colors_parts, axis=0), np.concatenate(faces_parts, axis=0))


def load_sidecar_frame(sidecar_root: Path, frame_id: str) -> dict[str, Any]:
    path = sidecar_root / "smpl_boxes" / f"{frame_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sidecar for frame {frame_id!r}: {path}")
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Sidecar frame must be a dict: {path}")
    return data


def frame_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
    detections = frame.get("detections", [])
    if isinstance(detections, list) and detections:
        return [dict(det, det_id=int(det.get("det_id", idx))) for idx, det in enumerate(detections)]
    out = []
    for idx, person in enumerate(frame.get("persons", [])):
        if not person.get("bbox_valid", person.get("valid", True)):
            continue
        out.append(
            {
                "det_id": int(person.get("det_id", idx)),
                "person_id": int(person.get("person_id", -1)),
                "bbox_xyxy_pixels": person.get("bbox_xyxy_pixels"),
                "bbox_cxcywh_norm": person.get("bbox_cxcywh_norm"),
                "det_score": float(person.get("det_score", person.get("track_confidence", 0.0))),
                "mask": person.get("mask"),
            }
        )
    return out


def frame_hw(frame: dict[str, Any], fallback_hw: tuple[int, int]) -> tuple[int, int]:
    if "image_hw" in frame:
        h, w = frame["image_hw"]
        return int(h), int(w)
    if "image_height" in frame and "image_width" in frame:
        return int(frame["image_height"]), int(frame["image_width"])
    return int(fallback_hw[0]), int(fallback_hw[1])


def det_xyxy(det: dict[str, Any], image_w: int, image_h: int) -> np.ndarray:
    if det.get("bbox_xyxy_pixels") is not None:
        return np.asarray(det["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
    box = np.asarray(det.get("bbox_cxcywh_norm", [0, 0, 0, 0]), dtype=np.float32).reshape(4)
    cx, cy, bw, bh = [float(v) for v in box]
    w = bw * float(max(image_w, 1))
    h = bh * float(max(image_h, 1))
    x = cx * float(max(image_w, 1))
    y = cy * float(max(image_h, 1))
    return np.asarray([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], dtype=np.float32)


def det_area(det: dict[str, Any]) -> float:
    try:
        xyxy = np.asarray(det.get("bbox_xyxy_pixels"), dtype=np.float32).reshape(4)
        return float(max(xyxy[2] - xyxy[0], 0.0) * max(xyxy[3] - xyxy[1], 0.0))
    except Exception:
        box = np.asarray(det.get("bbox_cxcywh_norm", [0, 0, 0, 0]), dtype=np.float32).reshape(4)
        return float(max(box[2], 0.0) * max(box[3], 0.0))


def load_det_patch_mask(
    sidecar_root: Path,
    det: dict[str, Any],
    geometry: ResizeGeometry,
    patch_size: int,
    threshold: float,
    min_mask_patches: int,
) -> np.ndarray | None:
    meta = det.get("mask")
    if not isinstance(meta, dict):
        return None
    raw_path = Path(str(meta.get("path", ""))).expanduser()
    if not raw_path.is_absolute():
        candidates = [
            (sidecar_root / raw_path).resolve(),
            (sidecar_root.parent / raw_path).resolve(),
            (ROOT / raw_path).resolve(),
            raw_path.resolve(),
        ]
        raw_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    key = str(meta.get("array_key", ""))
    if not raw_path.is_file() or not key:
        return None
    with np.load(raw_path) as data:
        if key not in data:
            return None
        pixel_mask = np.asarray(data[key]).astype(np.float32)
    resized = resize_mask_with_geometry(pixel_mask, geometry)
    patch_mask = pixel_mask_to_patch_mask_hw(resized, image_hw=geometry.input_hw, patch_size=patch_size, threshold=threshold)
    if int(patch_mask.sum()) < int(min_mask_patches):
        return None
    return patch_mask


def match_track_id(
    det: dict[str, Any],
    det_xyxy_value: np.ndarray,
    frame: dict[str, Any],
    image_w: int,
    image_h: int,
    iou_threshold: float,
) -> tuple[int, float]:
    if int(det.get("person_id", -1)) >= 0:
        return int(det["person_id"]), float(det.get("det_score", det.get("score", 1.0)))
    best_id = -1
    best_iou = 0.0
    best_conf = 0.0
    for person in frame.get("persons", []):
        person_id = int(person.get("person_id", -1))
        if person_id < 0 or not person.get("bbox_valid", person.get("valid", True)):
            continue
        p_xyxy = det_xyxy(person, image_w=image_w, image_h=image_h)
        iou = box_iou(det_xyxy_value, p_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_id = person_id
            best_conf = float(person.get("track_confidence", person.get("det_score", iou)))
    if best_iou < float(iou_threshold):
        return -1, 0.0
    return int(best_id), float(best_conf)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a).reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b).reshape(4)]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def palette_index_for_track(track_id: int, track_palette: dict[int, int]) -> int:
    if track_id in track_palette:
        return track_palette[track_id]
    used = set(track_palette.values())
    for idx in range(len(PAPER_PALETTE_10)):
        if idx not in used:
            track_palette[track_id] = idx
            return idx
    idx = abs(int(track_id)) % len(PAPER_PALETTE_10)
    track_palette[track_id] = idx
    return idx


def iter_selected_frames(frames_dir: Path, args: argparse.Namespace) -> list[Path]:
    frames = list(iter_image_files(frames_dir))
    start = max(int(args.start_index), 0)
    stride = max(int(args.frame_stride), 1)
    selected = frames[start::stride]
    if int(args.max_frames) > 0:
        selected = selected[: int(args.max_frames)]
    return selected


def require_smpl_model_dir(config: dict[str, Any], args: argparse.Namespace) -> str:
    value = args.smpl_model_dir or str(config.get("assets", {}).get("smpl_model_dir", ""))
    if not value:
        return require_path(config, "assets.smpl_model_dir", allow_empty=False)
    return str(resolve_project_path(value))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
