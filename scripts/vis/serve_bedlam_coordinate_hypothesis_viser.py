#!/usr/bin/env python3
"""Enumerate BEDLAM depth/SMPL/camera coordinate hypotheses in Viser.

This diagnostic is for a processed sequence whose SMPL mesh penetrates the
depth scene. It does not alter data or choose a winner automatically. Instead,
it lets the researcher compare the physically meaningful alternatives in one
viewer. World-space alternatives overlay several frames: static scene geometry
only remains stable when the external-camera direction is correct.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
FRAME_COLORS = ((245, 245, 245), (96, 196, 255), (255, 188, 96), (184, 255, 130), (245, 130, 210))


@dataclass
class FrameData:
    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    pose: np.ndarray
    local_meshes: list[np.ndarray]
    local_joints: list[np.ndarray]
    translations: list[np.ndarray]


@dataclass(frozen=True)
class Hypothesis:
    key: str
    label: str
    description: str
    frame_limit: int
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray]
    translation_offset: Callable[[np.ndarray], np.ndarray]
    camera_pose_kind: str


def main() -> None:
    args = parse_args()
    import torch  # noqa: PLC0415
    from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: PLC0415

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    frame_ids = select_frame_ids(sequence_dir, args.frame_id, args.frame_count, args.frame_step)
    smpl_model_dir = Path(args.smpl_model_dir).expanduser().resolve()
    smpl = SMPLLayer(smpl_model_dir).to(torch.device("cpu")).eval()
    frames = [load_frame(sequence_dir, frame_id, smpl) for frame_id in frame_ids]
    hypotheses = build_hypotheses()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays = save_camera_projection_overlays(output_dir, frames[0])
    summary = build_summary(sequence_dir, frames, hypotheses, overlays)
    summary_path = output_dir / "coordinate_hypothesis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.smoke_only:
        print(json.dumps({"summary": str(summary_path), "overlays": overlays}, indent=2, ensure_ascii=False), flush=True)
        print("[ok] BEDLAM coordinate-hypothesis smoke passed", flush=True)
        return

    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    server = viser.ViserServer(port=int(args.port))
    scene = getattr(server, "scene", server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    groups: dict[str, list[Any]] = {}
    faces = np.asarray(smpl.faces, dtype=np.int64)
    for hypothesis in hypotheses:
        groups[hypothesis.key] = add_hypothesis_group(
            scene, vtf, faces, frames, hypothesis, args.depth_stride, args.max_depth, args.point_size, args.mesh_opacity
        )
    selector = add_dropdown(server, "Coordinate hypothesis", [item.label for item in hypotheses], hypotheses[0].label)
    labels = {item.label: item.key for item in hypotheses}

    def update_visibility(selected_label: str) -> None:
        selected_key = labels[selected_label]
        for key, handles in groups.items():
            set_group_visible(handles, key == selected_key)

    update_visibility(hypotheses[0].label)
    bind_update(selector, lambda event: update_visibility(getattr(event, "value", selector.value)))
    print(
        json.dumps(
            {"viewer": f"http://127.0.0.1:{int(args.port)}", "summary": str(summary_path), "overlays": overlays},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    while True:
        time.sleep(3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--frame-id", default="", help="First frame stem; defaults to the first RGB frame")
    parser.add_argument("--frame-count", type=int, default=3, help="World hypotheses overlay this many frames")
    parser.add_argument("--frame-step", type=int, default=5, help="Frame stride inside the selected sequence")
    parser.add_argument("--output-dir", default="outputs/vis/bedlam_coordinate_hypothesis_audit")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--smpl-model-dir", default=str(DEFAULT_SMPL_MODEL_DIR))
    parser.add_argument("--depth-stride", type=int, default=6)
    parser.add_argument("--max-depth", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.009)
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.frame_count <= 0 or args.frame_step <= 0 or args.depth_stride <= 0:
        parser.error("--frame-count, --frame-step, and --depth-stride must be positive")
    return args


def select_frame_ids(sequence_dir: Path, requested: str, count: int, step: int) -> list[str]:
    available = sorted(path.stem for path in (sequence_dir / "rgb").glob("*.png"))
    if not available:
        raise RuntimeError(f"No PNG RGB frames under {sequence_dir / 'rgb'}")
    if requested:
        if requested not in available:
            raise KeyError(f"Requested frame {requested!r} not found in {sequence_dir / 'rgb'}")
        start = available.index(requested)
    else:
        start = 0
    selected = available[start : start + count * step : step]
    if not selected:
        raise RuntimeError("No frames selected")
    return selected


def load_frame(sequence_dir: Path, frame_id: str, smpl: Any) -> FrameData:
    paths = {
        "rgb": sequence_dir / "rgb" / f"{frame_id}.png",
        "depth": sequence_dir / "depth" / f"{frame_id}.npy",
        "cam": sequence_dir / "cam" / f"{frame_id}.npz",
        "smpl": sequence_dir / "smpl" / f"{frame_id}.pkl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(paths["depth"]), dtype=np.float32).squeeze()
    with np.load(paths["cam"]) as data:
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        pose = np.asarray(data["pose"], dtype=np.float32)
    validate_geometry(rgb, depth, intrinsics, pose, frame_id)
    with paths["smpl"].open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL pickle must contain list[dict]: {paths['smpl']}")
    local_meshes, local_joints, translations = decode_smpl_local(smpl, persons)
    return FrameData(frame_id, rgb, depth, intrinsics, pose, local_meshes, local_joints, translations)


def validate_geometry(rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray, frame_id: str) -> None:
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError(f"{frame_id}: RGB/depth must share HxW, got rgb={rgb.shape[:2]}, depth={depth.shape}")
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError(f"{frame_id}: expected K=(3,3), pose=(4,4), got {intrinsics.shape}, {pose.shape}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(pose).all():
        raise ValueError(f"{frame_id}: non-finite camera matrix")
    if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-3):
        raise ValueError(f"{frame_id}: camera rotation is not orthonormal")


def decode_smpl_local(smpl: Any, persons: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    poses, betas, translations = [], [], []
    for index, person in enumerate(persons):
        required = ("smplx_root_pose", "smplx_body_pose", "smplx_shape", "smplx_transl")
        missing = [key for key in required if key not in person]
        if missing:
            raise KeyError(f"Person {index} is missing {missing}")
        root = np.asarray(person["smplx_root_pose"], dtype=np.float32).reshape(1, 3)
        body = np.asarray(person["smplx_body_pose"], dtype=np.float32).reshape(21, 3)
        poses.append(np.concatenate((root, body, np.zeros((2, 3), dtype=np.float32)), axis=0))
        betas.append(np.asarray(person["smplx_shape"], dtype=np.float32).reshape(-1)[:10])
        translations.append(np.asarray(person["smplx_transl"], dtype=np.float32).reshape(3))
    if not poses:
        return [], [], []
    import torch  # noqa: PLC0415

    with torch.inference_mode():
        vertices, joints = smpl(torch.from_numpy(np.stack(poses)), torch.from_numpy(np.stack(betas)))
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    joints_np = joints.cpu().numpy().astype(np.float32)
    return [vertices_np[index] for index in range(len(poses))], [joints_np[index] for index in range(len(poses))], translations


def build_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis("camera_stored", "1 Camera: stored translation", "Depth and SMPL both remain in camera coordinates. Tests K, depth scale, pose and stored smplx_transl.", 1, identity_points, zero_offset, "camera"),
        Hypothesis("camera_minus_t", "2 Camera: stored translation - pose.t", "Tests the historical adapter alternative: stored smplx_transl may contain an extra pose translation.", 1, identity_points, negative_pose_translation, "camera"),
        Hypothesis("world_w2c_stored", "3 World: pose is W2C, stored translation", "x_world = R^T (x_cam - t). Static geometry should align across frames if pose is world-to-camera.", 0, w2c_to_world, zero_offset, "w2c"),
        Hypothesis("world_w2c_minus_t", "4 World: pose is W2C, stored translation - pose.t", "W2C world transform plus the historical extra-translation correction.", 0, w2c_to_world, negative_pose_translation, "w2c"),
        Hypothesis("world_c2w_stored", "5 World: pose is C2W, stored translation", "x_world = R x_cam + t. Static geometry should align across frames only if pose is camera-to-world.", 0, c2w_to_world, zero_offset, "c2w"),
        Hypothesis("world_c2w_minus_t", "6 World: pose is C2W, stored translation - pose.t", "C2W world transform plus the historical extra-translation correction.", 0, c2w_to_world, negative_pose_translation, "c2w"),
    ]


def identity_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    del pose
    return np.asarray(points, dtype=np.float32)


def w2c_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float32) - pose[:3, 3][None]) @ pose[:3, :3]


def c2w_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ pose[:3, :3].T + pose[:3, 3][None]


def zero_offset(pose: np.ndarray) -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def negative_pose_translation(pose: np.ndarray) -> np.ndarray:
    return -np.asarray(pose[:3, 3], dtype=np.float32)


def depth_rgb_to_camera_points(frame: FrameData, stride: int, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = frame.depth[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth > 0:
        valid &= z <= float(max_depth)
    z = z[valid].astype(np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    fx, fy = max(float(frame.intrinsics[0, 0]), 1e-6), max(float(frame.intrinsics[1, 1]), 1e-6)
    points = np.stack(((xs - frame.intrinsics[0, 2]) * z / fx, (ys - frame.intrinsics[1, 2]) * z / fy, z), axis=1)
    return points.astype(np.float32), frame.rgb[ys.astype(np.int64), xs.astype(np.int64)].astype(np.uint8)


def add_hypothesis_group(
    scene: Any,
    transforms: Any,
    faces: np.ndarray,
    frames: list[FrameData],
    hypothesis: Hypothesis,
    depth_stride: int,
    max_depth: float,
    point_size: float,
    mesh_opacity: float,
) -> list[Any]:
    handles: list[Any] = []
    active_frames = frames[: hypothesis.frame_limit] if hypothesis.frame_limit else frames
    for frame_index, frame in enumerate(active_frames):
        points_cam, colors = depth_rgb_to_camera_points(frame, depth_stride, max_depth)
        points = hypothesis.transform(points_cam, frame.pose)
        handles.append(scene.add_point_cloud(f"{hypothesis.key}/depth/{frame.frame_id}", points=points, colors=colors, point_size=point_size))
        offset = hypothesis.translation_offset(frame.pose)
        for person_index, (mesh_local, joints_local, translation) in enumerate(zip(frame.local_meshes, frame.local_joints, frame.translations, strict=True)):
            mesh = hypothesis.transform(mesh_local + translation[None] + offset[None], frame.pose)
            joints = hypothesis.transform(joints_local + translation[None] + offset[None], frame.pose)
            color = PALETTE[person_index % len(PALETTE)]
            handles.append(scene.add_mesh_simple(f"{hypothesis.key}/smpl/{frame.frame_id}_{person_index}", vertices=mesh, faces=faces, color=color, opacity=mesh_opacity))
            joint_colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], joints.shape[0], axis=0)
            handles.append(scene.add_point_cloud(f"{hypothesis.key}/joints/{frame.frame_id}_{person_index}", points=joints, colors=joint_colors, point_size=0.011))
        if hypothesis.camera_pose_kind != "camera":
            handles.append(add_camera_frustum(scene, transforms, f"{hypothesis.key}/camera/{frame.frame_id}", frame, hypothesis.camera_pose_kind, 0.17, FRAME_COLORS[frame_index % len(FRAME_COLORS)]))
    first = active_frames[0]
    title_position = hypothesis.transform(np.zeros((1, 3), dtype=np.float32), first.pose)[0]
    handles.append(scene.add_label(f"{hypothesis.key}/label", text=hypothesis.description, position=title_position))
    return handles


def add_camera_frustum(scene: Any, transforms: Any, name: str, frame: FrameData, pose_kind: str, scale: float, color: tuple[int, int, int]) -> Any:
    rotation = frame.pose[:3, :3]
    translation = frame.pose[:3, 3]
    if pose_kind == "w2c":
        position = -rotation.T @ translation
        rotation_c2w = rotation.T
    else:
        position = translation
        rotation_c2w = rotation
    fov_y = float(np.degrees(2.0 * np.arctan2(max(float(frame.intrinsics[1, 2]), 1.0), max(float(frame.intrinsics[1, 1]), 1e-6))))
    aspect = float(max(float(frame.intrinsics[0, 2]), 1.0) / max(float(frame.intrinsics[1, 2]), 1.0))
    return scene.add_camera_frustum(name=name, fov=fov_y, aspect=aspect, scale=scale, wxyz=transforms.SO3.from_matrix(rotation_c2w).wxyz, position=position, color=color)


def save_camera_projection_overlays(output_dir: Path, frame: FrameData) -> dict[str, str]:
    overlays = {}
    for key, offset in (("stored", zero_offset(frame.pose)), ("stored_minus_pose_t", negative_pose_translation(frame.pose))):
        image = Image.fromarray(frame.rgb, mode="RGB")
        draw = ImageDraw.Draw(image, mode="RGBA")
        height, width = frame.rgb.shape[:2]
        for person_index, (mesh_local, joints_local, translation) in enumerate(zip(frame.local_meshes, frame.local_joints, frame.translations, strict=True)):
            mesh_cam = mesh_local + translation[None] + offset[None]
            joints_cam = joints_local + translation[None] + offset[None]
            color = PALETTE[person_index % len(PALETTE)]
            vertices_xy, vertices_valid = project_camera_points(mesh_cam, frame.intrinsics, width, height)
            joints_xy, joints_valid = project_camera_points(joints_cam, frame.intrinsics, width, height)
            for x, y in vertices_xy[vertices_valid][::16]:
                draw.point((float(x), float(y)), fill=(*color, 72))
            for x, y in joints_xy[joints_valid]:
                draw.ellipse((float(x) - 2.5, float(y) - 2.5, float(x) + 2.5, float(y) + 2.5), fill=(*color, 245))
        path = output_dir / f"{frame.frame_id}_{key}_projection_overlay.png"
        image.save(path)
        overlays[key] = str(path)
    return overlays


def project_camera_points(points: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    xy = np.zeros((points.shape[0], 2), dtype=np.float32)
    xy[valid, 0] = intrinsics[0, 0] * points[valid, 0] / z[valid] + intrinsics[0, 2]
    xy[valid, 1] = intrinsics[1, 1] * points[valid, 1] / z[valid] + intrinsics[1, 2]
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return xy, valid


def build_summary(sequence_dir: Path, frames: list[FrameData], hypotheses: list[Hypothesis], overlays: dict[str, str]) -> dict[str, Any]:
    return {
        "sequence_dir": str(sequence_dir),
        "coordinate_note": "The viewer enumerates hypotheses; no hypothesis is automatically claimed correct.",
        "frames": [
            {
                "frame_id": frame.frame_id,
                "depth_median": float(np.median(frame.depth[np.isfinite(frame.depth) & (frame.depth > 0)])),
                "pose_translation": frame.pose[:3, 3].tolist(),
                "smpl_translation": [value.tolist() for value in frame.translations],
            }
            for frame in frames
        ],
        "hypotheses": [{"key": item.key, "label": item.label, "description": item.description} for item in hypotheses],
        "projection_overlays": overlays,
    }


def add_dropdown(server: Any, name: str, options: list[str], initial: str) -> Any:
    api = getattr(server, "gui", server)
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


def ensure_viser_available() -> None:
    try:
        import viser  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - server dependency.
        raise ImportError("Viser is required in the server environment") from exc


if __name__ == "__main__":
    main()
