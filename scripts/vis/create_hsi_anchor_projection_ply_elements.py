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


COLOR = {
    "camera": (35, 45, 65),
    "camera_light": (105, 118, 138),
    "human": (232, 142, 82),
    "anchor": (42, 168, 107),
    "anchor_hand": (126, 92, 210),
    "anchor_body": (26, 166, 166),
    "projection": (255, 218, 58),
    "projection_line": (244, 190, 40),
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
    parser.add_argument("--depth-upsample", type=int, default=2)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-scene-depth", type=float, default=30.0)
    parser.add_argument("--anchor-radius-scale", type=float, default=0.018)
    parser.add_argument("--projection-radius-scale", type=float, default=0.0035)
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

    image_tensor, _original_image = load_image(image_path, input_size)
    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=priors["boxes"],
            smpl_query_boxes_mask=priors["mask"],
            smpl_query_patch_masks=priors["patch_masks"],
        )

    probe = compute_hsi_probe(predictions, model, input_size)
    query_indices = choose_query_indices(predictions, priors, args)
    depth_key = "hsi_depth_hw" if args.depth_source == "hsi" and "hsi_depth_hw" in probe else "depth_hw"
    actual_depth_source = "hsi" if depth_key == "hsi_depth_hw" else "raw"

    selected = build_selected_people(probe, query_indices, depth_key)
    scale = estimate_scene_scale(probe, selected)
    radius = max(scale * float(args.anchor_radius_scale), 0.012)
    line_radius = max(scale * float(args.projection_radius_scale), 0.0025)
    camera_depth = choose_camera_plane_depth(probe, selected)

    depth_mesh = build_depth_component(
        probe=probe,
        depth_key=depth_key,
        upsample=max(int(args.depth_upsample), 1),
        stride=max(int(args.depth_stride), 1),
        max_depth=float(args.max_scene_depth),
    )
    camera_mesh = MeshBuilder()
    add_camera_frustum(camera_mesh, probe["intrinsics"], input_size, camera_depth, scale)

    files: list[str] = []
    camera_path = output_dir / "00_camera_frustum.ply"
    camera_mesh.write(camera_path)
    files.append(camera_path.name)

    depth_path = output_dir / f"00_depth_surface_{actual_depth_source}_teal.ply"
    depth_mesh.write(depth_path)
    files.append(depth_path.name)

    collection = MeshBuilder()
    collection.merge(camera_mesh)
    collection.merge(depth_mesh)

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

        collection.merge(anchors_mesh)
        add_anchor_projection_marks(collection, person, radius, line_radius)
        people_meta.append(person_metadata(person))

    collection_path = output_dir / "03_hsi_anchor_projection_collection.ply"
    collection.write(collection_path)
    files.append(collection_path.name)

    manifest = {
        "purpose": "Paper-figure PLY elements for HSI 24 body anchors and camera-to-depth projection.",
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "requested_depth_source": args.depth_source,
        "actual_depth_source": actual_depth_source,
        "depth_colormap": "teal-blue-green",
        "projection_point_color": "bright yellow",
        "camera_style": "classic CV frustum pyramid",
        "num_people": len(selected),
        "people": people_meta,
        "files": files,
        "notes": [
            "All PLY files use the same camera-space coordinate system for manual composition.",
            "Depth surface is colored by a teal/blue-green depth colormap, not RGB texture.",
            "Yellow spheres are the anchor-corresponding depth samples after projection.",
            "The *_only.ply files are layer-only exports for manual figure assembly.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": files}, indent=2))


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
) -> MeshBuilder:
    vertices, colors, faces = depth_to_teal_surface_mesh(
        depth=probe[depth_key][0, 0],
        intrinsics=probe["intrinsics"],
        image_size=int(probe["image_size"].detach().cpu()),
        upsample=upsample,
        stride=stride,
        max_depth=max_depth,
    )
    mesh = MeshBuilder()
    if vertices.size:
        mesh.add_mesh(vertices, faces, colors)
    return mesh


def depth_to_teal_surface_mesh(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    upsample: int,
    stride: int,
    max_depth: float,
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
    colors_grid = teal_depth_colors(z_np, z_valid)
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


def teal_depth_colors(depth_grid: np.ndarray, valid_values: np.ndarray) -> np.ndarray:
    if valid_values.size:
        lo, hi = np.percentile(valid_values, [2.0, 98.0])
    else:
        lo, hi = 0.0, 1.0
    t = np.clip((depth_grid - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    stops = np.asarray(
        [
            [29, 74, 120],
            [21, 132, 160],
            [40, 177, 150],
            [139, 213, 168],
        ],
        dtype=np.float32,
    )
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


def person_metadata(person: dict[str, Any]) -> dict[str, Any]:
    return {
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


if __name__ == "__main__":
    main()
