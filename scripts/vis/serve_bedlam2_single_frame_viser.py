#!/usr/bin/env python3
"""Audit one processed BEDLAM-style frame with a world-space Viser view.

The viewer intentionally uses the same convention for every geometry source:
``cam/*.npz:pose`` is camera-from-world (world-to-camera), while depth and
``smplx_transl`` are camera-space quantities. It backprojects metric depth with
RGB colours, converts both the point cloud and decoded SMPL meshes through the
inverse extrinsic, then renders them together in the world frame.

This is a convention audit, not a model inference viewer. A bad alignment is
useful evidence: it usually means an inconsistent depth scale, SMPL translation
convention, or camera extrinsic convention in the processed data.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SEQUENCE_DIR = Path(
    "/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/"
    "20241213_1_250_rome_tracking_seq_000002"
)
DEFAULT_SMPL_MODEL_DIR = Path("/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smpl")
PALETTE = ((41, 98, 255), (239, 71, 111), (6, 180, 162), (255, 176, 0), (131, 90, 241), (46, 204, 113))


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import torch  # noqa: PLC0415
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415
    from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: PLC0415

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    frame_id = args.frame_id.strip() or first_frame_id(sequence_dir)
    paths = frame_paths(sequence_dir, frame_id)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(paths["depth"]), dtype=np.float32).squeeze()
    intrinsics, extrinsic = load_camera(paths["cam"])
    validate_frame_geometry(rgb, depth, intrinsics, extrinsic)
    persons = load_persons(paths["smpl"])

    smpl_model_dir = Path(args.smpl_model_dir).expanduser().resolve()
    smpl = SMPLLayer(smpl_model_dir).to(torch.device("cpu")).eval()
    mesh_cam, joints_cam = smpl_persons_to_meshes_camera(smpl, persons)
    mesh_world = [camera_points_to_world_np(vertices, extrinsic) for vertices in mesh_cam]
    joints_world = [camera_points_to_world_np(joints, extrinsic) for joints in joints_cam]
    point_world, point_colors = depth_rgb_to_world_points(
        depth, rgb, intrinsics, extrinsic, stride=max(1, args.depth_stride), max_depth=args.max_depth
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path, projection_summary = save_smpl_projection_overlay(
        output_dir, frame_id, rgb, mesh_cam, joints_cam, intrinsics
    )
    summary = build_summary(
        sequence_dir=sequence_dir,
        frame_id=frame_id,
        paths=paths,
        depth=depth,
        intrinsics=intrinsics,
        extrinsic=extrinsic,
        persons=persons,
        point_count=int(point_world.shape[0]),
        mesh_count=len(mesh_world),
        projection_summary=projection_summary,
        overlay_path=overlay_path,
    )
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    server = viser.ViserServer(port=int(args.port))
    scene = getattr(server, "scene", server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    scene.add_point_cloud("world_rgb_depth", point_world, colors=point_colors, point_size=float(args.point_size))
    add_camera_frustum(scene, vtf, extrinsic, intrinsics, args.camera_scale)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    for index, (vertices, joints) in enumerate(zip(mesh_world, joints_world, strict=True)):
        color = PALETTE[index % len(PALETTE)]
        scene.add_mesh_simple(f"world_smpl_{index}", vertices=vertices, faces=faces, color=color, opacity=float(args.mesh_opacity))
        joint_colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], joints.shape[0], axis=0)
        scene.add_point_cloud(f"world_smpl_joints_{index}", joints, colors=joint_colors, point_size=0.012)
    scene.add_label(
        "title",
        text=f"World audit: RGB-depth point cloud + GT SMPL | {sequence_dir.name} / {frame_id}",
        position=camera_center_from_extrinsic(extrinsic),
    )
    print(
        json.dumps(
            {"viewer": f"http://127.0.0.1:{int(args.port)}", "summary": str(summary_path), "projection_overlay": str(overlay_path)},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.smoke_only:
        print("[ok] BEDLAM world-geometry viewer smoke passed", flush=True)
        return
    while True:
        time.sleep(3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--frame-id", default="", help="Frame stem; defaults to the first RGB frame")
    parser.add_argument("--output-dir", default="outputs/vis/bedlam_world_geometry_audit")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--smpl-model-dir", default=str(DEFAULT_SMPL_MODEL_DIR))
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=30.0, help="Ignore larger/non-positive metric depth; <=0 keeps all positive depth")
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--camera-scale", type=float, default=0.2)
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.depth_stride <= 0:
        parser.error("--depth-stride must be positive")
    return args


def frame_paths(sequence_dir: Path, frame_id: str) -> dict[str, Path]:
    return {
        "rgb": sequence_dir / "rgb" / f"{frame_id}.png",
        "depth": sequence_dir / "depth" / f"{frame_id}.npy",
        "cam": sequence_dir / "cam" / f"{frame_id}.npz",
        "smpl": sequence_dir / "smpl" / f"{frame_id}.pkl",
    }


def first_frame_id(sequence_dir: Path) -> str:
    paths = sorted((sequence_dir / "rgb").glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG RGB frames found under {sequence_dir / 'rgb'}")
    return paths[0].stem


def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "intrinsics" not in data or "pose" not in data:
            raise ValueError(f"Camera file needs intrinsics and pose: {path}; keys={list(data.files)}")
        return np.asarray(data["intrinsics"], dtype=np.float32), np.asarray(data["pose"], dtype=np.float32)


def validate_frame_geometry(rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsic: np.ndarray) -> None:
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW after squeeze, got {depth.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth raster mismatch: rgb={rgb.shape[:2]} depth={depth.shape}")
    if intrinsics.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"Expected K=(3,3), pose=(4,4), got K={intrinsics.shape}, pose={extrinsic.shape}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsic).all():
        raise ValueError("Camera matrices contain non-finite values")
    rotation = extrinsic[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise ValueError("cam pose rotation is not orthonormal; cannot safely interpret it as camera-from-world")


def load_persons(smpl_path: Path) -> list[dict[str, Any]]:
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"Expected list[dict] in {smpl_path}, got {type(persons).__name__}")
    return persons


def depth_rgb_to_world_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    extrinsic: np.ndarray,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth > 0:
        valid &= z <= float(max_depth)
    xs_valid = xs[valid].astype(np.float32)
    ys_valid = ys[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    fx, fy = max(float(intrinsics[0, 0]), 1e-6), max(float(intrinsics[1, 1]), 1e-6)
    points_cam = np.stack(
        ((xs_valid - float(intrinsics[0, 2])) * z_valid / fx, (ys_valid - float(intrinsics[1, 2])) * z_valid / fy, z_valid), axis=1
    )
    return camera_points_to_world_np(points_cam, extrinsic), rgb[ys[valid], xs[valid]].astype(np.uint8, copy=False)


def smpl_persons_to_meshes_camera(smpl: Any, persons: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses, betas, translations = [], [], []
    for person_index, person in enumerate(persons):
        required = ("smplx_root_pose", "smplx_body_pose", "smplx_shape", "smplx_transl")
        missing = [key for key in required if key not in person]
        if missing:
            raise KeyError(f"Person {person_index} is missing SMPL fields: {missing}")
        root_pose = np.asarray(person["smplx_root_pose"], dtype=np.float32).reshape(1, 3)
        body_pose = np.asarray(person["smplx_body_pose"], dtype=np.float32).reshape(21, 3)
        poses.append(np.concatenate((root_pose, body_pose, np.zeros((2, 3), dtype=np.float32)), axis=0))
        betas.append(np.asarray(person["smplx_shape"], dtype=np.float32).reshape(-1)[:10])
        translations.append(np.asarray(person["smplx_transl"], dtype=np.float32).reshape(3))
    if not poses:
        return [], []
    import torch  # noqa: PLC0415

    with torch.inference_mode():
        vertices, joints = smpl(torch.from_numpy(np.stack(poses)), torch.from_numpy(np.stack(betas)))
    translation = np.stack(translations).astype(np.float32)
    vertices_np = vertices.cpu().numpy().astype(np.float32) + translation[:, None]
    joints_np = joints.cpu().numpy().astype(np.float32) + translation[:, None]
    return [vertices_np[index] for index in range(vertices_np.shape[0])], [joints_np[index] for index in range(joints_np.shape[0])]


def camera_points_to_world_np(points_cam: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    """Invert x_cam = R_w2c @ x_world + t_w2c for row-major point arrays."""
    rotation = np.asarray(camera_from_world[:3, :3], dtype=np.float32)
    translation = np.asarray(camera_from_world[:3, 3], dtype=np.float32)
    return ((np.asarray(points_cam, dtype=np.float32) - translation[None]) @ rotation).astype(np.float32)


def camera_center_from_extrinsic(camera_from_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(camera_from_world[:3, :3], dtype=np.float32)
    translation = np.asarray(camera_from_world[:3, 3], dtype=np.float32)
    return (-rotation.T @ translation).astype(np.float32)


def add_camera_frustum(scene: Any, transforms: Any, camera_from_world: np.ndarray, intrinsics: np.ndarray, scale: float) -> None:
    position = camera_center_from_extrinsic(camera_from_world)
    rotation_w2c = np.asarray(camera_from_world[:3, :3], dtype=np.float32)
    fov_y = float(np.degrees(2.0 * np.arctan2(max(float(intrinsics[1, 2]), 1.0), max(float(intrinsics[1, 1]), 1e-6))))
    aspect = float(max(float(intrinsics[0, 2]), 1.0) / max(float(intrinsics[1, 2]), 1.0))
    scene.add_camera_frustum(
        "camera",
        fov=fov_y,
        aspect=aspect,
        scale=float(scale),
        wxyz=transforms.SO3.from_matrix(rotation_w2c.T).wxyz,
        position=position,
        color=(255, 255, 255),
    )


def save_smpl_projection_overlay(
    output_dir: Path,
    frame_id: str,
    rgb: np.ndarray,
    meshes_cam: list[np.ndarray],
    joints_cam: list[np.ndarray],
    intrinsics: np.ndarray,
) -> tuple[Path, list[dict[str, Any]]]:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    height, width = rgb.shape[:2]
    summary: list[dict[str, Any]] = []
    for index, (vertices, joints) in enumerate(zip(meshes_cam, joints_cam, strict=True)):
        color = PALETTE[index % len(PALETTE)]
        projected_vertices, valid_vertices = project_camera_points(vertices, intrinsics, width, height)
        projected_joints, valid_joints = project_camera_points(joints, intrinsics, width, height)
        for x, y in projected_vertices[valid_vertices][::16]:
            draw.point((float(x), float(y)), fill=(*color, 72))
        for x, y in projected_joints[valid_joints]:
            draw.ellipse((float(x) - 2.5, float(y) - 2.5, float(x) + 2.5, float(y) + 2.5), fill=(*color, 245))
        visible = projected_vertices[valid_vertices]
        bbox = None
        if visible.size:
            left, top = visible.min(axis=0)
            right, bottom = visible.max(axis=0)
            bbox = [float(left), float(top), float(right), float(bottom)]
            draw.rectangle(tuple(bbox), outline=(*color, 255), width=2)
            draw.text((left + 4, top + 4), f"SMPL {index}", fill=(*color, 255))
        summary.append({"person_index": index, "visible_vertex_fraction": float(valid_vertices.mean()), "visible_joint_count": int(valid_joints.sum()), "projected_bbox_xyxy": bbox})
    path = output_dir / f"{frame_id}_smpl_projection_overlay.png"
    image.save(path)
    return path, summary


def project_camera_points(points: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    xy = np.zeros((points.shape[0], 2), dtype=np.float32)
    xy[valid, 0] = float(intrinsics[0, 0]) * points[valid, 0] / z[valid] + float(intrinsics[0, 2])
    xy[valid, 1] = float(intrinsics[1, 1]) * points[valid, 1] / z[valid] + float(intrinsics[1, 2])
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return xy, valid


def build_summary(**kwargs: Any) -> dict[str, Any]:
    paths = kwargs["paths"]
    depth = kwargs["depth"]
    extrinsic = kwargs["extrinsic"]
    valid_depth = np.isfinite(depth) & (depth > 0)
    rotation = extrinsic[:3, :3]
    return {
        "sequence_dir": str(kwargs["sequence_dir"]),
        "frame_id": kwargs["frame_id"],
        "paths": {key: str(value) for key, value in paths.items()},
        "coordinate_contract": {
            "cam_pose": "camera_from_world / world_to_camera: x_cam = R @ x_world + t",
            "depth": "metric camera Z-depth; RGB shares its pixel grid",
            "smpl": "decoded canonical mesh + stored smplx_transl in camera coordinates",
            "world_render": "depth points and SMPL vertices both use x_world = R.T @ (x_cam - t)",
        },
        "rgb_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "intrinsics": kwargs["intrinsics"].tolist(),
        "camera_center_world": camera_center_from_extrinsic(extrinsic).tolist(),
        "rotation_orthonormal_error": float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
        "depth": {
            "valid_fraction": float(valid_depth.mean()),
            "median_m": float(np.median(depth[valid_depth])) if valid_depth.any() else None,
            "p05_m": float(np.percentile(depth[valid_depth], 5)) if valid_depth.any() else None,
            "p95_m": float(np.percentile(depth[valid_depth], 95)) if valid_depth.any() else None,
        },
        "person_count": len(kwargs["persons"]),
        "smpl_translation_cam": smpl_translation_summary(kwargs["persons"]),
        "point_count": kwargs["point_count"],
        "mesh_count": kwargs["mesh_count"],
        "smpl_projection": kwargs["projection_summary"],
        "projection_overlay": str(kwargs["overlay_path"]),
    }


def smpl_translation_summary(persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for index, person in enumerate(persons):
        translation = person.get("smplx_transl")
        if translation is None:
            summary.append({"person_index": index, "translation_cam_m": None})
            continue
        value = np.asarray(translation, dtype=np.float32).reshape(-1)
        summary.append({"person_index": index, "translation_cam_m": value[:3].tolist()})
    return summary


def ensure_viser_available() -> None:
    try:
        import viser  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - server dependency.
        raise ImportError("Viser is required. Install it in the server environment before running this viewer.") from exc


if __name__ == "__main__":
    main()
