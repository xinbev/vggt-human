#!/usr/bin/env python3
"""Serve one BEDLAM2 frame in Viser.

This viewer is intentionally minimal: it loads a single frame from a processed
BEDLAM2 sequence and renders the depth point cloud, camera frustum, and
optional SMPL meshes/joints.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402


DEFAULT_SEQUENCE_DIR = Path(
    "/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/"
    "20241213_1_250_rome_tracking_seq_000002"
)
DEFAULT_SMPL_MODEL_DIR = Path("/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smpl")


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    sequence_dir = resolve_project_path(args.sequence_dir)
    frame_id = args.frame_id.strip()
    if not frame_id:
        frame_id = first_frame_id(sequence_dir)

    depth_path = sequence_dir / "depth" / f"{frame_id}.npy"
    cam_path = sequence_dir / "cam" / f"{frame_id}.npz"
    smpl_path = sequence_dir / "smpl" / f"{frame_id}.pkl"
    for path in (depth_path, cam_path, smpl_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
    with np.load(cam_path) as camera:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        extrinsic = np.asarray(camera["pose"], dtype=np.float32)

    persons = load_persons(smpl_path)
    smpl = None
    smpl_faces = None
    smpl_model_dir = resolve_project_path(args.smpl_model_dir) if args.smpl_model_dir else DEFAULT_SMPL_MODEL_DIR
    if smpl_model_dir is not None:
        smpl = SMPLLayer(smpl_model_dir).to(torch.device("cpu")).eval()
        smpl_faces = np.asarray(smpl.faces, dtype=np.int64)

    depth_points, depth_colors = depth_to_world_points(depth, intrinsics, extrinsic, stride=max(1, args.depth_stride), max_depth=args.max_depth)
    camera_position = camera_center_from_extrinsic(extrinsic)

    server = viser.ViserServer(port=int(args.port))
    if hasattr(server, "scene") and hasattr(server.scene, "set_up_direction"):
        server.scene.set_up_direction("-y")
    elif hasattr(server, "set_up_direction"):
        server.set_up_direction("-y")

    scene = scene_api(server)
    scene.add_point_cloud(name="depth_points", points=depth_points, colors=depth_colors, point_size=float(args.point_size))
    add_camera_frustum(scene, vtf, extrinsic, intrinsics, args.camera_scale)
    mesh_count = 0
    if smpl is not None:
        mesh_count = add_smpl_overlays(scene, smpl, smpl_faces, persons, args.mesh_opacity)

    summary = {
        "sequence_dir": str(sequence_dir),
        "frame_id": frame_id,
        "depth_path": str(depth_path),
        "cam_path": str(cam_path),
        "smpl_path": str(smpl_path),
        "person_count": len(persons),
        "depth_point_count": int(depth_points.shape[0]),
        "mesh_count": int(mesh_count),
        "camera_position": camera_position.tolist(),
        "output_dir": str(resolve_project_path(args.output_dir)),
    }
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"viewer": f"http://127.0.0.1:{int(args.port)}", "summary": str(summary_path)}, indent=2), flush=True)
    if args.smoke_only:
        print("[ok] BEDLAM2 single-frame Viser smoke passed", flush=True)
        return

    server.scene.add_label(name="title", text=f"{sequence_dir.name} / {frame_id}", position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    while True:
        time.sleep(3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--frame-id", default="", help="Frame stem to visualize; defaults to the first RGB frame")
    parser.add_argument("--output-dir", default="outputs/vis/bedlam2_single_frame_viser")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smpl-model-dir", default=str(DEFAULT_SMPL_MODEL_DIR))
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--camera-scale", type=float, default=0.2)
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def first_frame_id(sequence_dir: Path) -> str:
    rgb_dir = sequence_dir / "rgb"
    paths = sorted(rgb_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No RGB frames found under {rgb_dir}")
    return paths[0].stem


def load_persons(smpl_path: Path) -> list[dict[str, Any]]:
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"Expected list of persons in {smpl_path}")
    return persons


def depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsic: np.ndarray,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape[:2]
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth > 0:
        valid &= z <= float(max_depth)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=1)
    points_world = camera_points_to_world_np(points_cam, extrinsic)
    colors = depth_to_gray_colors(z)
    return points_world.astype(np.float32, copy=False), colors


def camera_points_to_world_np(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation[None, :]) @ rotation).astype(np.float32)


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return (-rotation.T @ translation).astype(np.float32)


def add_camera_frustum(scene: Any, transforms: Any, extrinsic: np.ndarray, intrinsics: np.ndarray, scale: float) -> None:
    position = camera_center_from_extrinsic(extrinsic)
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    wxyz = transforms.SO3.from_matrix(rotation.T).wxyz
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    width = max(cx * 2.0, 1.0)
    height = max(cy * 2.0, 1.0)
    fov_y = float(np.degrees(2.0 * np.arctan2(height * 0.5, max(fy, 1e-6))))
    aspect = float(width / max(height, 1e-6))
    try:
        scene.add_camera_frustum(name="camera", fov=fov_y, aspect=aspect, scale=float(scale), wxyz=wxyz, position=position, color=(255, 255, 255))
    except TypeError:
        scene.add_camera_frustum("camera", fov_y, aspect, float(scale), wxyz, position)


def add_smpl_overlays(scene: Any, smpl: SMPLLayer, faces: np.ndarray | None, persons: list[dict[str, Any]], opacity: float) -> int:
    if faces is None or len(persons) == 0:
        return 0
    vertices, joints = smpl_persons_to_meshes(smpl, persons)
    count = 0
    for idx, (verts, jnts) in enumerate(zip(vertices, joints, strict=False)):
        color = palette_color(idx)
        scene.add_mesh_simple(name=f"smpl_{idx}", vertices=verts, faces=faces, color=color, opacity=float(opacity))
        scene.add_point_cloud(name=f"smpl_joints_{idx}", points=jnts, colors=np.repeat(np.asarray(color, dtype=np.uint8)[None, :], jnts.shape[0], axis=0), point_size=0.008)
        count += 1
    return count


def smpl_persons_to_meshes(smpl: SMPLLayer, persons: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses = []
    betas = []
    transl = []
    for person in persons:
        if "smplx_root_pose" not in person or "smplx_body_pose" not in person or "smplx_shape" not in person or "smplx_transl" not in person:
            continue
        root_pose = np.asarray(person["smplx_root_pose"], dtype=np.float32).reshape(1, 3)
        body_pose = np.asarray(person["smplx_body_pose"], dtype=np.float32).reshape(21, 3)
        pad = np.zeros((2, 3), dtype=np.float32)
        pose = np.concatenate([root_pose, body_pose, pad], axis=0).reshape(24, 3)
        poses.append(pose)
        betas.append(np.asarray(person["smplx_shape"], dtype=np.float32).reshape(-1)[:10])
        transl.append(np.asarray(person["smplx_transl"], dtype=np.float32).reshape(3))
    if not poses:
        return [], []
    pose_t = torch.from_numpy(np.stack(poses, axis=0))
    beta_t = torch.from_numpy(np.stack(betas, axis=0))
    with torch.inference_mode():
        verts, joints = smpl(pose_t, beta_t)
    verts_np = verts.detach().cpu().numpy().astype(np.float32, copy=False)
    joints_np = joints.detach().cpu().numpy().astype(np.float32, copy=False)
    transl_np = np.stack(transl, axis=0).astype(np.float32, copy=False)
    verts_np = verts_np + transl_np[:, None, :]
    joints_np = joints_np + transl_np[:, None, :]
    return [verts_np[i] for i in range(verts_np.shape[0])], [joints_np[i] for i in range(joints_np.shape[0])]


def palette_color(index: int) -> tuple[int, int, int]:
    palette = [
        (41, 98, 255),
        (239, 71, 111),
        (6, 180, 162),
        (255, 176, 0),
        (131, 90, 241),
        (46, 204, 113),
    ]
    return palette[index % len(palette)]


def depth_to_gray_colors(depth_values: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    colors = np.full((values.shape[0], 3), 180, dtype=np.uint8)
    if bool(finite.any()):
        valid = values[finite]
        lo = float(np.percentile(valid, 2.0))
        hi = float(np.percentile(valid, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(valid.min())
            hi = float(valid.max())
        if hi > lo:
            norm = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
            gray = (255.0 * (1.0 - norm)).astype(np.uint8)
            colors[finite] = np.stack([gray, gray, gray], axis=1)
    return colors


def scene_api(server: Any) -> Any:
    return getattr(server, "scene", server)


def ensure_viser_available() -> None:
    try:
        import viser  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError("The Viser viewer requires the optional dependency 'viser'. Install it with `pip install viser`.") from exc


if __name__ == "__main__":
    main()
