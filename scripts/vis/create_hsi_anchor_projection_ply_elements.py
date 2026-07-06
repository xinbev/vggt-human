#!/usr/bin/env python3
"""Create real-data PLY elements for HSI anchors and anchor-to-depth projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import build_model  # noqa: E402
from scripts.vis.create_hsi_local_probe_real_elements import (  # noqa: E402
    build_query_priors,
    choose_query_indices,
    compute_hsi_probe,
    load_config,
    load_image,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    resolve_checkpoint,
    resolve_project_path,
    unproject_one,
)
from scripts.vis.create_hsi_paper_ply_elements import (  # noqa: E402
    MeshBuilder,
    add_box,
    add_cylinder,
    add_uv_sphere,
)
from vggt_omega.models.heads.hsi_refinement_head import _project_points, _scale_points_to_depth  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


COLOR = {
    "camera": (35, 45, 65),
    "camera_light": (105, 118, 138),
    "human": (232, 142, 82),
    "anchor": (42, 168, 107),
    "anchor_hand": (126, 92, 210),
    "anchor_body": (26, 166, 166),
    "projection": (255, 218, 58),
    "projection_line": (244, 190, 40),
    "sample_gt": (34, 197, 94),
    "patch_grid": (148, 163, 184),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_after_translation_ray_refine.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis/paper_hsi_anchor_projection_ply_elements"))
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--person-index", type=int, default=-1)
    parser.add_argument("--person-select", choices=("rightmost", "leftmost", "confidence", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--auto-person-prior", action="store_true")
    parser.add_argument("--auto-top-k", type=int, default=2)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-half", action="store_true")
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--patch-mask-min-overlap", type=float, default=0.02)
    parser.add_argument("--depth-source", choices=("hsi", "raw"), default="hsi")
    parser.add_argument("--smpl-stage", choices=("base", "refined"), default="base")
    parser.add_argument("--depth-colormap", choices=("turbo", "inferno", "magma", "viridis", "teal"), default="turbo")
    parser.add_argument("--depth-surface-color", choices=("rgb", "colormap"), default="rgb")
    parser.add_argument("--depth-upsample", type=int, default=2)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-scene-depth", type=float, default=30.0)
    parser.add_argument("--anchor-radius-scale", type=float, default=0.009)
    parser.add_argument("--projection-radius-scale", type=float, default=0.0035)
    parser.add_argument("--mask-depth-samples", type=int, default=24)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    checkpoint = resolve_checkpoint(args, config)
    image_path = resolve_project_path(args.image)
    input_size = int(config.get("data", {}).get("image_size", args.image_size))
    patch_size = int(config.get("model", {}).get("patch_size", 16))

    priors = build_query_priors(args, image_path, input_size, patch_size, config, device, output_dir)
    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, checkpoint, device)
    model.eval()

    image_tensor, original_image = load_image(image_path, input_size)
    rgb_input = original_image.resize((input_size, input_size), Image.BILINEAR).convert("RGB")
    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=priors["boxes"],
            smpl_query_boxes_mask=priors["mask"],
            smpl_query_patch_masks=priors["patch_masks"],
        )

    base_probe = compute_hsi_probe(predictions, model, input_size)
    query_indices = choose_query_indices(predictions, priors, args)
    diagnostics = build_projection_diagnostics(predictions, model, base_probe, query_indices)
    probe, actual_smpl_stage = apply_smpl_stage(base_probe, predictions, model, args.smpl_stage)
    depth_key = "hsi_depth_hw" if args.depth_source == "hsi" and "hsi_depth_hw" in probe else "depth_hw"
    actual_depth_source = "hsi" if depth_key == "hsi_depth_hw" else "raw"

    selected = build_selected_people(probe, query_indices, depth_key)
    scale = estimate_scene_scale(probe, selected)
    radius = max(scale * float(args.anchor_radius_scale), 0.006)
    line_radius = max(scale * float(args.projection_radius_scale), 0.0025)
    camera_depth = choose_camera_plane_depth(probe, selected)

    depth_mesh = build_depth_component(
        probe=probe,
        depth_key=depth_key,
        upsample=max(int(args.depth_upsample), 1),
        stride=max(int(args.depth_stride), 1),
        max_depth=float(args.max_scene_depth),
        colormap=args.depth_colormap,
        rgb=rgb_input,
        color_source=args.depth_surface_color,
    )
    camera_mesh = MeshBuilder()
    add_camera_frustum(camera_mesh, probe["intrinsics"], input_size, camera_depth, scale)

    files: list[str] = []
    camera_path = output_dir / "00_camera_frustum.ply"
    camera_mesh.write(camera_path)
    files.append(camera_path.name)

    depth_suffix = "rgb" if args.depth_surface_color == "rgb" else args.depth_colormap
    depth_path = output_dir / f"00_depth_surface_{actual_depth_source}_{depth_suffix}.ply"
    depth_mesh.write(depth_path)
    files.append(depth_path.name)

    depth_png = depth_to_rgba_image(probe[depth_key][0, 0], args.depth_colormap)
    depth_png_path = output_dir / f"00_depth_map_{actual_depth_source}_{args.depth_colormap}.png"
    depth_png.save(depth_png_path)
    files.append(depth_png_path.name)

    collection = MeshBuilder()
    collection.merge(camera_mesh)
    collection.merge(depth_mesh)
    visual_collection = MeshBuilder()
    visual_collection.merge(camera_mesh)
    visual_collection.merge(depth_mesh)

    people_meta: list[dict[str, Any]] = []
    for rank, person in enumerate(selected):
        person_prefix = f"person{rank}_q{person['query_idx']}"
        smpl_mesh = build_smpl_component(person)
        smpl_path = output_dir / f"01_{person_prefix}_smpl_only.ply"
        smpl_mesh.write(smpl_path)
        files.append(smpl_path.name)

        anchor_only_mesh = build_anchor_component(person, radius)
        anchor_only_path = output_dir / f"01_{person_prefix}_24anchors_only.ply"
        anchor_only_mesh.write(anchor_only_path)
        files.append(anchor_only_path.name)

        anchors_mesh = build_person_anchor_component(person, radius)
        anchors_path = output_dir / f"01_{person_prefix}_smpl_24anchors.ply"
        anchors_mesh.write(anchors_path)
        files.append(anchors_path.name)

        projection_only_mesh = MeshBuilder()
        add_anchor_projection_marks(projection_only_mesh, person, radius, line_radius)
        projection_only_path = output_dir / f"02_{person_prefix}_projection_links_yellow_points_only.ply"
        projection_only_mesh.write(projection_only_path)
        files.append(projection_only_path.name)

        projection_mesh = MeshBuilder()
        projection_mesh.merge(camera_mesh)
        projection_mesh.merge(depth_mesh)
        projection_mesh.merge(anchors_mesh)
        add_anchor_projection_marks(projection_mesh, person, radius, line_radius)
        projection_path = output_dir / f"02_{person_prefix}_camera_person_depth_projection.ply"
        projection_mesh.write(projection_path)
        files.append(projection_path.name)

        mask_samples_meta = None
        query_mask = query_mask_input(priors, int(person["query_idx"]))
        if query_mask is not None and int(args.mask_depth_samples) > 0:
            mask_samples = sample_mask_depth_points(
                probe=probe,
                depth_key=depth_key,
                mask_input=query_mask,
                num_points=int(args.mask_depth_samples),
            )
            if mask_samples["points"].size:
                mask_samples_mesh = build_depth_sample_component(mask_samples["points"], radius, COLOR["sample_gt"])
                mask_samples_path = output_dir / f"02_{person_prefix}_mask_depth_samples_green_only.ply"
                mask_samples_mesh.write(mask_samples_path)
                files.append(mask_samples_path.name)

                mask_samples_scene = MeshBuilder()
                mask_samples_scene.merge(camera_mesh)
                mask_samples_scene.merge(depth_mesh)
                mask_samples_scene.merge(mask_samples_mesh)
                mask_samples_scene_path = output_dir / f"02_{person_prefix}_camera_depth_mask_samples_green.ply"
                mask_samples_scene.write(mask_samples_scene_path)
                files.append(mask_samples_scene_path.name)

                visual_collection.merge(mask_samples_mesh)
                patch_png = create_mask_sample_patch_png(
                    rgb_input=rgb_input,
                    samples=mask_samples,
                    patch_size=patch_size,
                    grid_hw=tuple(int(v) for v in probe["patch_grid_hw"].detach().cpu().tolist()),
                )
                patch_png_path = output_dir / f"02_{person_prefix}_mask_depth_samples_9patch_rgb.png"
                patch_png.save(patch_png_path)
                files.append(patch_png_path.name)

                mask_samples_meta = write_mask_sample_json(output_dir / f"mask_depth_samples_{person_prefix}.json", mask_samples)
                files.append(f"mask_depth_samples_{person_prefix}.json")

        collection.merge(anchors_mesh)
        add_anchor_projection_marks(collection, person, radius, line_radius)
        people_meta.append(person_metadata(person, mask_samples_meta))

    collection_path = output_dir / "03_hsi_anchor_projection_collection.ply"
    collection.write(collection_path)
    files.append(collection_path.name)

    visual_collection_path = output_dir / "03_hsi_mask_depth_sampling_collection.ply"
    visual_collection.write(visual_collection_path)
    files.append(visual_collection_path.name)

    manifest = {
        "purpose": "Paper-figure PLY elements for HSI 24 body anchors and camera-to-depth projection.",
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "requested_depth_source": args.depth_source,
        "actual_depth_source": actual_depth_source,
        "requested_smpl_stage": args.smpl_stage,
        "actual_smpl_stage": actual_smpl_stage,
        "depth_colormap": args.depth_colormap,
        "depth_surface_color": args.depth_surface_color,
        "projection_point_color": "bright yellow",
        "mask_depth_sample_color": "green",
        "camera_style": "classic CV frustum pyramid",
        "num_people": len(selected),
        "people": people_meta,
        "projection_diagnostics": diagnostics,
        "files": files,
        "notes": [
            "All PLY files use the same camera-space coordinate system for manual composition.",
            "Depth surface is colored by resized input RGB by default; set --depth-surface-color colormap for heatmap PLY.",
            "Yellow spheres are the anchor-corresponding depth samples after projection.",
            "Green spheres use SAM2 person masks to sample real depth points for paper visualization when SMPL projection is unreliable.",
            "The 9patch PNG files show the 3x3 patch-token neighborhoods around mask-depth sample points.",
            "The *_only.ply files are layer-only exports for manual figure assembly.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "projection_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": files}, indent=2))


def apply_smpl_stage(
    probe: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    model: torch.nn.Module,
    stage: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if stage == "base":
        return probe, "base"
    refined = rebuild_probe_geometry_from_prediction_stage(probe, predictions, model, "refined")
    if refined is None:
        return probe, "base_fallback_refined_keys_missing"
    return refined, "refined"


def rebuild_probe_geometry_from_prediction_stage(
    probe: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    model: torch.nn.Module,
    stage: str,
) -> dict[str, torch.Tensor] | None:
    if stage == "base":
        pose6d = predictions.get("pred_pose_6d")
        betas = predictions.get("pred_betas")
        transl = predictions.get("pred_transl_cam")
    elif stage == "refined":
        pose6d = predictions.get("hsi_refined_pred_pose_6d")
        betas = predictions.get("hsi_refined_pred_betas")
        transl = predictions.get("hsi_refined_pred_transl_cam")
    else:
        raise ValueError(f"Unsupported SMPL stage: {stage}")
    if not isinstance(pose6d, torch.Tensor) or not isinstance(betas, torch.Tensor) or not isinstance(transl, torch.Tensor):
        return None
    head = getattr(model, "hsi_refinement_head", None)
    if head is None:
        return None
    pose6d = pose6d.float()
    betas = betas.float()
    transl = transl.float()
    anchors = head._anchors_cam(pose6d, betas, transl)
    poses_aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
    smpl_vertices, smpl_joints = head.smpl(poses_aa.float(), betas.reshape(-1, betas.shape[-1]).float())
    smpl_vertices = smpl_vertices.to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
    smpl_joints = smpl_joints[:, :24].to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
    smpl_vertices = smpl_vertices.reshape(*pose6d.shape[:3], smpl_vertices.shape[-2], 3)
    smpl_joints = smpl_joints.reshape(*pose6d.shape[:3], smpl_joints.shape[-2], 3)
    batch_size, num_frames, num_queries, _, _ = anchors.shape
    intrinsics = probe["intrinsics"].to(device=pose6d.device, dtype=pose6d.dtype)
    flat_anchors = anchors.reshape(batch_size * num_frames * num_queries, 24, 3)
    projected = _project_points(flat_anchors, intrinsics.repeat_interleave(num_queries, dim=0)).reshape(
        batch_size, num_frames, num_queries, 24, 2
    )
    height, width = probe["depth_hw"].shape[-2:]
    image_size = int(probe["image_size"].detach().cpu())
    projected_depth = _scale_points_to_depth(projected, image_size, height, width)
    rebuilt = dict(probe)
    rebuilt.update(
        {
            "anchors": anchors.detach(),
            "smpl_vertices": smpl_vertices.detach(),
            "smpl_joints": smpl_joints.detach(),
            "projected": projected.detach(),
            "projected_depth": projected_depth.detach(),
        }
    )
    return rebuilt


def build_projection_diagnostics(
    predictions: dict[str, torch.Tensor],
    model: torch.nn.Module,
    base_probe: dict[str, torch.Tensor],
    query_indices: list[int],
) -> dict[str, Any]:
    variants = {"base": base_probe}
    refined_probe = rebuild_probe_geometry_from_prediction_stage(base_probe, predictions, model, "refined")
    if refined_probe is not None:
        variants["refined"] = refined_probe
    depth_keys = {"raw": "depth_hw"}
    if "hsi_depth_hw" in base_probe:
        depth_keys["hsi"] = "hsi_depth_hw"
    per_query: dict[str, Any] = {}
    for query_idx in query_indices:
        query_metrics: dict[str, Any] = {}
        for stage, stage_probe in variants.items():
            for depth_name, depth_key in depth_keys.items():
                scene_points, projected_depth, valid = sample_anchor_scene_points(stage_probe, query_idx, depth_key)
                anchors = stage_probe["anchors"][0, 0, query_idx].detach().float().cpu().numpy()
                query_metrics[f"{stage}_{depth_name}"] = summarize_projection_alignment(
                    anchors=anchors,
                    scene_points=scene_points,
                    projected_depth=projected_depth,
                    valid=valid,
                    depth_hw=stage_probe[depth_key][0, 0],
                )
        per_query[str(query_idx)] = query_metrics
    return {
        "meaning": "Distance between SMPL anchors and depth samples along the same projected camera rays.",
        "interpretation": {
            "base_raw": "Closest to the actual HSI tokenization step: base SMPL anchors sample raw VGGT depth.",
            "base_hsi": "Current default paper-depth visualization if --smpl-stage=base and --depth-source=hsi.",
            "refined_hsi": "Expected best alignment after HSI refines SMPL and applies scene affine depth.",
        },
        "queries": per_query,
    }


def summarize_projection_alignment(
    anchors: np.ndarray,
    scene_points: np.ndarray,
    projected_depth: np.ndarray,
    valid: np.ndarray,
    depth_hw: torch.Tensor,
) -> dict[str, Any]:
    valid = np.asarray(valid).astype(bool)
    if not bool(valid.any()):
        return {"valid_projected_anchors": 0}
    delta = np.asarray(scene_points, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)
    l2 = np.linalg.norm(delta[valid], axis=-1)
    z_delta = delta[valid, 2]
    uv = np.asarray(projected_depth, dtype=np.float32)[valid]
    height, width = depth_hw.shape[-2:]
    return {
        "valid_projected_anchors": int(valid.sum()),
        "mean_l2_m": float(np.mean(l2)),
        "median_l2_m": float(np.median(l2)),
        "max_l2_m": float(np.max(l2)),
        "mean_abs_depth_delta_m": float(np.mean(np.abs(z_delta))),
        "median_abs_depth_delta_m": float(np.median(np.abs(z_delta))),
        "mean_signed_depth_delta_m": float(np.mean(z_delta)),
        "projected_depth_uv_min": [float(uv[:, 0].min()), float(uv[:, 1].min())],
        "projected_depth_uv_max": [float(uv[:, 0].max()), float(uv[:, 1].max())],
        "depth_hw": [int(height), int(width)],
    }


def build_selected_people(probe: dict[str, torch.Tensor], query_indices: list[int], depth_key: str) -> list[dict[str, Any]]:
    selected = []
    for query_idx in query_indices:
        anchors = probe["anchors"][0, 0, query_idx].detach().float().cpu().numpy()
        smpl_vertices = probe["smpl_vertices"][0, 0, query_idx].detach().float().cpu().numpy()
        smpl_faces = probe["smpl_faces"].detach().cpu().numpy()
        scene_points, projected_depth, valid = sample_anchor_scene_points(probe, query_idx, depth_key)
        selected.append(
            {
                "query_idx": int(query_idx),
                "anchors": anchors,
                "smpl_vertices": smpl_vertices,
                "smpl_faces": smpl_faces,
                "scene_points": scene_points,
                "projected_depth": projected_depth,
                "valid_projection": valid,
            }
        )
    return selected


def sample_anchor_scene_points(probe: dict[str, torch.Tensor], query_idx: int, depth_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = probe[depth_key][0, 0]
    height, width = depth.shape[-2:]
    intr = probe["intrinsics"].reshape(-1, 3, 3)[0].to(device=depth.device, dtype=depth.dtype)
    projected_input = probe["projected"][0, 0, query_idx]
    projected_depth = probe["projected_depth"][0, 0, query_idx]
    px = projected_depth[:, 0].round().long()
    py = projected_depth[:, 1].round().long()
    valid = (
        torch.isfinite(projected_input).all(dim=-1)
        & torch.isfinite(projected_depth).all(dim=-1)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    z = depth.new_zeros(24)
    if bool(valid.any()):
        z[valid] = depth[py[valid], px[valid]]
    valid = valid & torch.isfinite(z) & (z > 1e-6)
    scene = torch.zeros(24, 3, device=depth.device, dtype=depth.dtype)
    for idx in range(24):
        if bool(valid[idx]):
            scene[idx] = unproject_one(projected_input[idx], z[idx], intr)
    return (
        scene.detach().float().cpu().numpy(),
        projected_depth.detach().float().cpu().numpy(),
        valid.detach().cpu().numpy().astype(bool),
    )


def query_mask_input(priors: dict[str, Any], query_idx: int) -> np.ndarray | None:
    masks = priors.get("person_masks_input_by_query") or {}
    mask = masks.get(int(query_idx))
    if mask is None:
        mask = masks.get(str(int(query_idx)))
    if mask is None:
        return None
    return np.asarray(mask).astype(bool)


def sample_mask_depth_points(
    probe: dict[str, torch.Tensor],
    depth_key: str,
    mask_input: np.ndarray,
    num_points: int,
) -> dict[str, np.ndarray]:
    depth = probe[depth_key][0, 0]
    height, width = depth.shape[-2:]
    mask_depth = resize_bool_mask_nearest(mask_input, (height, width))
    valid_depth = torch.isfinite(depth) & (depth > 1e-6)
    valid_np = (mask_depth & valid_depth.detach().cpu().numpy())
    ys, xs = np.nonzero(valid_np)
    if xs.size == 0:
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "uv_depth": np.empty((0, 2), dtype=np.float32),
            "uv_input": np.empty((0, 2), dtype=np.float32),
            "depth": np.empty((0,), dtype=np.float32),
        }
    chosen = farthest_sample_pixels(np.stack([xs, ys], axis=1).astype(np.float32), min(int(num_points), int(xs.size)), width, height)
    chosen_x = chosen[:, 0].round().astype(np.int64).clip(0, width - 1)
    chosen_y = chosen[:, 1].round().astype(np.int64).clip(0, height - 1)
    image_size = float(int(probe["image_size"].detach().cpu()))
    input_x = torch.as_tensor(chosen_x, device=depth.device, dtype=depth.dtype) * (image_size / float(width))
    input_y = torch.as_tensor(chosen_y, device=depth.device, dtype=depth.dtype) * (image_size / float(height))
    z = depth[torch.as_tensor(chosen_y, device=depth.device), torch.as_tensor(chosen_x, device=depth.device)]
    intr = probe["intrinsics"].reshape(-1, 3, 3)[0].to(device=depth.device, dtype=depth.dtype)
    points = []
    for idx in range(chosen_x.shape[0]):
        points.append(unproject_one(torch.stack([input_x[idx], input_y[idx]]), z[idx], intr))
    points_tensor = torch.stack(points, dim=0)
    return {
        "points": points_tensor.detach().float().cpu().numpy().astype(np.float32),
        "uv_depth": np.stack([chosen_x, chosen_y], axis=1).astype(np.float32),
        "uv_input": torch.stack([input_x, input_y], dim=-1).detach().float().cpu().numpy().astype(np.float32),
        "depth": z.detach().float().cpu().numpy().astype(np.float32),
    }


def resize_bool_mask_nearest(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    mask_tensor = torch.from_numpy(np.asarray(mask).astype(np.float32)).reshape(1, 1, *mask.shape)
    resized = torch.nn.functional.interpolate(mask_tensor, size=size_hw, mode="nearest")[0, 0]
    return resized.numpy() > 0.5


def farthest_sample_pixels(coords_xy: np.ndarray, count: int, width: int, height: int) -> np.ndarray:
    if coords_xy.shape[0] <= count:
        return coords_xy
    norm = coords_xy.copy()
    norm[:, 0] /= max(float(width - 1), 1.0)
    norm[:, 1] /= max(float(height - 1), 1.0)
    center = norm.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.linalg.norm(norm - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(norm - norm[first : first + 1], axis=1)
    for _ in range(1, count):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(norm - norm[next_idx : next_idx + 1], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return coords_xy[np.asarray(selected, dtype=np.int64)]


def build_depth_sample_component(points: np.ndarray, radius: float, color: tuple[int, int, int]) -> MeshBuilder:
    mesh = MeshBuilder()
    for point in np.asarray(points, dtype=np.float32).reshape(-1, 3):
        add_uv_sphere(mesh, point, radius * 0.85, color, rings=8, segments=14)
    return mesh


def create_mask_sample_patch_png(
    rgb_input: Image.Image,
    samples: dict[str, np.ndarray],
    patch_size: int,
    grid_hw: tuple[int, int],
) -> Image.Image:
    out = rgb_input.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = out.size
    grid_h, grid_w = grid_hw
    cell_w = float(width) / max(float(grid_w), 1.0)
    cell_h = float(height) / max(float(grid_h), 1.0)
    del patch_size

    for col in range(grid_w + 1):
        x = col * cell_w
        draw.line([(x, 0), (x, height)], fill=(*COLOR["patch_grid"], 72), width=1)
    for row in range(grid_h + 1):
        y = row * cell_h
        draw.line([(0, y), (width, y)], fill=(*COLOR["patch_grid"], 72), width=1)

    uv_input = np.asarray(samples["uv_input"], dtype=np.float32).reshape(-1, 2)
    highlighted: set[tuple[int, int]] = set()
    for u, v in uv_input:
        center_col = int(np.floor(float(u) / max(cell_w, 1e-6)))
        center_row = int(np.floor(float(v) / max(cell_h, 1e-6)))
        center_col = max(0, min(grid_w - 1, center_col))
        center_row = max(0, min(grid_h - 1, center_row))
        for row in range(max(0, center_row - 1), min(grid_h, center_row + 2)):
            for col in range(max(0, center_col - 1), min(grid_w, center_col + 2)):
                highlighted.add((row, col))

    for row, col in sorted(highlighted):
        rect = [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h]
        draw.rectangle(rect, fill=(*COLOR["sample_gt"], 54), outline=(*COLOR["sample_gt"], 180), width=2)

    for u, v in uv_input:
        draw_dot_2d(draw, (float(u), float(v)), COLOR["sample_gt"], radius=5, alpha=245)
    out.alpha_composite(overlay)
    return out


def draw_dot_2d(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
    alpha: int,
) -> None:
    x, y = xy
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, alpha), outline=(255, 255, 255, 230), width=2)


def write_mask_sample_json(path: Path, samples: dict[str, np.ndarray]) -> dict[str, Any]:
    data = {
        "num_points": int(samples["points"].shape[0]),
        "source": "SAM2 person mask resized to depth grid + HSI/VGGT depth sampling",
        "points_xyz": samples["points"].astype(float).tolist(),
        "uv_depth": samples["uv_depth"].astype(float).tolist(),
        "uv_input": samples["uv_input"].astype(float).tolist(),
        "depth_values": samples["depth"].astype(float).tolist(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"file": path.name, "num_points": int(samples["points"].shape[0])}


def estimate_scene_scale(probe: dict[str, torch.Tensor], selected: list[dict[str, Any]]) -> float:
    parts = []
    for person in selected:
        parts.append(person["smpl_vertices"])
        valid_scene = person["scene_points"][person["valid_projection"]]
        if valid_scene.size:
            parts.append(valid_scene)
    if not parts:
        return 1.0
    points = np.concatenate(parts, axis=0)
    extent = np.nanmax(points, axis=0) - np.nanmin(points, axis=0)
    return max(float(np.linalg.norm(extent)), 1.0)


def choose_camera_plane_depth(probe: dict[str, torch.Tensor], selected: list[dict[str, Any]]) -> float:
    z_values = []
    for person in selected:
        verts = person["smpl_vertices"]
        z_values.extend(verts[np.isfinite(verts).all(axis=1), 2].tolist())
    if z_values:
        return max(0.5, float(np.percentile(np.asarray(z_values), 5)) * 0.55)
    depth = probe["depth_hw"][0, 0].detach().float().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 1e-6)
    return max(0.5, float(np.percentile(depth[valid], 10)) * 0.5) if valid.any() else 1.0


def add_camera_frustum(mesh: MeshBuilder, intrinsics: torch.Tensor, image_size: int, z_plane: float, scale: float) -> None:
    intr = intrinsics.reshape(-1, 3, 3)[0].detach().float().cpu().numpy()
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    corners_uv = np.asarray(
        [[0.0, 0.0], [float(image_size), 0.0], [float(image_size), float(image_size)], [0.0, float(image_size)]],
        dtype=np.float32,
    )
    corners = []
    for u, v in corners_uv:
        corners.append([(u - cx) / max(fx, 1e-6) * z_plane, (v - cy) / max(fy, 1e-6) * z_plane, z_plane])
    corners = np.asarray(corners, dtype=np.float32)
    center = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    radius = max(scale * 0.0025, 0.003)
    add_box(mesh, np.asarray([0.0, 0.0, -0.06 * scale], dtype=np.float32), (0.08 * scale, 0.055 * scale, 0.045 * scale), COLOR["camera"])
    for corner in corners:
        add_cylinder(mesh, center, corner, radius, COLOR["camera_light"])
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        add_cylinder(mesh, start, end, radius, COLOR["camera"])


def build_depth_component(
    probe: dict[str, torch.Tensor],
    depth_key: str,
    upsample: int,
    stride: int,
    max_depth: float,
    colormap: str,
    rgb: Image.Image,
    color_source: str,
) -> MeshBuilder:
    vertices, colors, faces = depth_to_colored_surface_mesh(
        depth=probe[depth_key][0, 0],
        intrinsics=probe["intrinsics"],
        image_size=int(probe["image_size"].detach().cpu()),
        upsample=upsample,
        stride=stride,
        max_depth=max_depth,
        colormap=colormap,
        rgb=rgb,
        color_source=color_source,
    )
    mesh = MeshBuilder()
    if vertices.size:
        mesh.add_mesh(vertices, faces, colors)
    return mesh


def depth_to_colored_surface_mesh(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    upsample: int,
    stride: int,
    max_depth: float,
    colormap: str,
    rgb: Image.Image,
    color_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_hw = depth.detach().float()
    if int(upsample) > 1:
        depth_hw = torch.nn.functional.interpolate(
            depth_hw.reshape(1, 1, *depth_hw.shape[-2:]),
            scale_factor=int(upsample),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    height, width = depth_hw.shape[-2:]
    ys = torch.arange(0, height, max(int(stride), 1), device=depth_hw.device)
    xs = torch.arange(0, width, max(int(stride), 1), device=depth_hw.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    z = depth_hw[yy, xx]
    valid = torch.isfinite(z) & (z > 1e-6) & (z <= float(max_depth))
    if not bool(valid.any()):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.empty((0, 3), dtype=np.int64)
    xx_f = xx.to(dtype=depth_hw.dtype) * (float(image_size) / float(width))
    yy_f = yy.to(dtype=depth_hw.dtype) * (float(image_size) / float(height))
    intr = intrinsics.reshape(-1, 3, 3)[0].to(device=depth_hw.device, dtype=depth_hw.dtype)
    fx = intr[0, 0].clamp(min=1e-6)
    fy = intr[1, 1].clamp(min=1e-6)
    cx = intr[0, 2]
    cy = intr[1, 2]
    points = torch.stack([(xx_f - cx) / fx * z, (yy_f - cy) / fy * z, z], dim=-1)
    points_grid = points.detach().float().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid.detach().cpu().numpy()
    index_map = -np.ones(valid_np.shape, dtype=np.int64)
    index_map[valid_np] = np.arange(int(valid_np.sum()), dtype=np.int64)
    vertices = points_grid[valid_np]

    z_np = z.detach().float().cpu().numpy()
    z_valid = z_np[valid_np]
    if color_source == "rgb":
        rgb_small = np.asarray(rgb.resize((width, height), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
        colors_grid = rgb_small[yy.detach().cpu().numpy(), xx.detach().cpu().numpy()]
    else:
        colors_grid = depth_colormap_colors(z_np, z_valid, colormap)
    colors = colors_grid[valid_np]

    faces: list[list[int]] = []
    rows, cols = valid_np.shape
    for row in range(rows - 1):
        for col in range(cols - 1):
            ids = [index_map[row, col], index_map[row, col + 1], index_map[row + 1, col], index_map[row + 1, col + 1]]
            if min(ids) < 0:
                continue
            depths = np.asarray([z_np[row, col], z_np[row, col + 1], z_np[row + 1, col], z_np[row + 1, col + 1]], dtype=np.float32)
            if float(depths.max() - depths.min()) / max(float(np.mean(np.abs(depths))), 1e-6) > 0.08:
                continue
            a, b, c, d = [int(v) for v in ids]
            faces.append([a, c, b])
            faces.append([b, c, d])
    return vertices, colors.astype(np.uint8), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def depth_to_rgba_image(depth: torch.Tensor, colormap: str) -> Image.Image:
    depth_np = depth.detach().float().cpu().numpy()
    valid = np.isfinite(depth_np) & (depth_np > 1e-6)
    colors = depth_colormap_colors(depth_np, depth_np[valid], colormap).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([colors, alpha]), mode="RGBA")


def depth_colormap_colors(depth_grid: np.ndarray, valid_values: np.ndarray, colormap: str) -> np.ndarray:
    if valid_values.size:
        lo, hi = np.percentile(valid_values, [2.0, 98.0])
    else:
        lo, hi = 0.0, 1.0
    t = np.nan_to_num(np.clip((depth_grid - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    stops_by_name = {
        "turbo": [
            [48, 18, 59],
            [58, 82, 166],
            [32, 159, 181],
            [72, 193, 110],
            [245, 231, 65],
            [245, 135, 48],
            [180, 35, 38],
        ],
        "inferno": [
            [0, 0, 4],
            [40, 11, 84],
            [101, 21, 110],
            [159, 42, 99],
            [212, 72, 66],
            [245, 125, 21],
            [252, 255, 164],
        ],
        "magma": [
            [0, 0, 4],
            [28, 16, 68],
            [79, 18, 123],
            [129, 37, 129],
            [181, 54, 122],
            [229, 80, 100],
            [252, 253, 191],
        ],
        "viridis": [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        "teal": [
            [29, 74, 120],
            [21, 132, 160],
            [40, 177, 150],
            [139, 213, 168],
        ],
    }
    stops = np.asarray(stops_by_name.get(str(colormap), stops_by_name["turbo"]), dtype=np.float32)
    pos = t * float(len(stops) - 1)
    idx = np.floor(pos).astype(np.int32).clip(0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    return ((1.0 - frac) * stops[idx] + frac * stops[idx + 1]).clip(0, 255)


def build_person_anchor_component(person: dict[str, Any], radius: float) -> MeshBuilder:
    mesh = MeshBuilder()
    mesh.merge(build_smpl_component(person))
    mesh.merge(build_anchor_component(person, radius))
    return mesh


def build_smpl_component(person: dict[str, Any]) -> MeshBuilder:
    mesh = MeshBuilder()
    mesh.add_mesh(person["smpl_vertices"], person["smpl_faces"], COLOR["human"])
    return mesh


def build_anchor_component(person: dict[str, Any], radius: float) -> MeshBuilder:
    mesh = MeshBuilder()
    add_24_anchor_spheres(mesh, person["anchors"], radius)
    return mesh


def add_24_anchor_spheres(mesh: MeshBuilder, anchors: np.ndarray, radius: float) -> None:
    for idx, anchor in enumerate(np.asarray(anchors, dtype=np.float32).reshape(24, 3)):
        color = COLOR["anchor"]
        if idx in {21, 22}:
            color = COLOR["anchor_hand"]
        elif idx == 23:
            color = COLOR["anchor_body"]
        add_uv_sphere(mesh, anchor, radius, color, rings=8, segments=14)


def add_anchor_projection_marks(mesh: MeshBuilder, person: dict[str, Any], radius: float, line_radius: float) -> None:
    anchors = person["anchors"]
    scene_points = person["scene_points"]
    valid = person["valid_projection"]
    for idx in range(24):
        if not bool(valid[idx]):
            continue
        anchor = anchors[idx]
        scene = scene_points[idx]
        if not np.isfinite(scene).all() or not np.isfinite(anchor).all():
            continue
        if float(np.linalg.norm(scene - anchor)) > 1e-8:
            add_cylinder(mesh, anchor, scene, line_radius, COLOR["projection_line"])
        add_uv_sphere(mesh, scene, radius * 0.72, COLOR["projection"], rings=8, segments=14)


def person_metadata(person: dict[str, Any], mask_samples_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "query_index": int(person["query_idx"]),
        "num_anchors": 24,
        "valid_projected_anchors": int(np.asarray(person["valid_projection"]).sum()),
        "anchor_schema": {
            "0_20": "SMPL joints 1:22, non-root body joints",
            "21": "left hand center",
            "22": "right hand center",
            "23": "full body center",
        },
    }
    if mask_samples_meta is not None:
        data["mask_depth_samples"] = mask_samples_meta
    return data


if __name__ == "__main__":
    main()
