"""Attach RICH ground-truth SMPL meshes to a cached viewer scene."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


RICH_GT_COLOR = (64, 192, 255)


def attach_rich_gt_smpl(
    scene: dict[str, Any],
    *,
    support_root: str | Path,
    smplx_model_dir: str | Path,
    smplx_to_smpl_path: str | Path,
    sequence: str = "",
    coordinate_source: str = "hsi_scaled",
    color: tuple[int, int, int] = RICH_GT_COLOR,
    device: str = "cuda",
) -> dict[str, Any]:
    """Decode RICH GT SMPL-X and place each mesh in the cache viewer frame."""
    if coordinate_source not in {"hsi_scaled", "raw_vggt"}:
        raise ValueError(f"Unsupported GT coordinate source: {coordinate_source}")
    frames = list(scene.get("frames", []))
    if not frames:
        raise ValueError("Cannot attach RICH GT SMPL to an empty scene")

    support = Path(support_root).expanduser().resolve()
    labels_path = support / "rich_test_labels.pt"
    cameras_path = support / "cam2params.pt"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing RICH labels: {labels_path}")
    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing RICH camera parameters: {cameras_path}")

    import torch
    import smplx

    labels = _torch_load(labels_path)
    resolved_sequence = sequence.strip() or _infer_sequence(frames)
    if resolved_sequence not in labels:
        candidates = sorted(str(key) for key in labels if resolved_sequence in str(key))
        suffix = f" Similar sequences: {candidates[:5]}" if candidates else ""
        raise KeyError(f"RICH sequence not found: {resolved_sequence}.{suffix}")
    label = labels[resolved_sequence]
    params = label.get("gt_smplx_params")
    if not isinstance(params, dict):
        raise ValueError(f"{resolved_sequence}: missing gt_smplx_params")

    cameras = _torch_load(cameras_path)
    camera_key = _rich_camera_key(resolved_sequence)
    if camera_key not in cameras:
        raise KeyError(f"RICH camera calibration entry not found: {camera_key}")
    gt_w2c, _ = cameras[camera_key]
    gt_w2c = np.asarray(_to_numpy(gt_w2c), dtype=np.float32).reshape(4, 4)

    source_ids = [_source_frame_index(frame) for frame in frames]
    label_frame_ids = np.asarray(_to_numpy(label.get("frame_id")), dtype=np.int64).reshape(-1)
    label_positions = {int(source_id): index for index, source_id in enumerate(label_frame_ids.tolist())}
    missing = [source_id for source_id in source_ids if source_id not in label_positions]
    if missing:
        raise ValueError(
            f"{resolved_sequence}: cache contains {len(missing)} frames absent from RICH frame_id; "
            f"first missing source indices={missing[:8]}"
        )

    gender = str(label.get("gender", "")).lower()
    if gender not in {"male", "female"}:
        raise ValueError(f"{resolved_sequence}: unsupported RICH gender {gender!r}")
    body_model = smplx.create(
        str(Path(smplx_model_dir).expanduser().resolve()),
        model_type="smplx",
        gender=gender,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device).eval()
    mapping = _load_mapping(Path(smplx_to_smpl_path), device)
    faces = _cached_faces(frames)
    positions = [label_positions[source_id] for source_id in source_ids]
    vertex_chunks = []
    with torch.no_grad():
        for start in range(0, len(positions), 32):
            frame_params = _select_params(params, positions[start : start + 32], device)
            output = body_model(**frame_params)
            vertex_chunks.append(torch.einsum("sv,bvc->bsc", mapping, output.vertices).cpu())
    smpl_vertices = torch.cat(vertex_chunks, dim=0)
    smpl_vertices_np = smpl_vertices.detach().float().cpu().numpy().astype(np.float32, copy=False)

    for frame, vertices_world_rich in zip(frames, smpl_vertices_np, strict=True):
        vertices_cam = _transform_points(vertices_world_rich, gt_w2c)
        extrinsic_key = "hsi_extrinsic" if coordinate_source == "hsi_scaled" else "raw_extrinsic"
        viewer_extrinsic = np.asarray(frame.get(extrinsic_key), dtype=np.float32).reshape(4, 4)
        frame["gt_vertices"] = _camera_points_to_world(vertices_cam, viewer_extrinsic)
        frame["gt_faces"] = faces
        frame["gt_color"] = tuple(int(value) for value in color)
        frame["gt_smpl_available"] = True

    metadata = {
        "dataset": "RICH",
        "sequence": resolved_sequence,
        "coordinate_source": coordinate_source,
        "camera_key": camera_key,
        "gt_frame_count": len(frames),
        "source_frame_indices": source_ids,
        "model_gender": gender,
        "color": tuple(int(value) for value in color),
    }
    scene["rich_gt_smpl"] = metadata
    return metadata


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_mapping(path: Path, device: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMPL-X to SMPL mapping: {path}")
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as file:
            value = pickle.load(file, encoding="latin1")
    else:
        value = _torch_load(path)
    if isinstance(value, dict):
        if "matrix" not in value:
            raise KeyError(f"SMPL-X to SMPL mapping dict has no 'matrix': {path}")
        value = value["matrix"]
    if hasattr(value, "toarray"):
        value = value.toarray()
    import torch

    mapping = torch.as_tensor(value, dtype=torch.float32, device=device)
    if mapping.ndim != 2 or tuple(mapping.shape) != (6890, 10475):
        raise ValueError(f"Expected SMPL-X to SMPL mapping [6890,10475], got {tuple(mapping.shape)}")
    return mapping


def _select_params(params: dict[str, Any], positions: list[int], device: str) -> dict[str, Any]:
    import torch

    selected: dict[str, Any] = {}
    for key in ("global_orient", "body_pose", "betas", "transl"):
        if key not in params:
            raise KeyError(f"RICH GT SMPL-X parameters missing {key!r}")
        value = torch.as_tensor(params[key], dtype=torch.float32)
        if key == "betas" and (value.ndim == 1 or value.shape[0] == 1):
            value = value.reshape(1, -1).expand(len(positions), -1)
        else:
            value = value[positions]
        selected[key] = value.to(device)
    selected["global_orient"] = selected["global_orient"].reshape(len(positions), 3)
    selected["body_pose"] = selected["body_pose"].reshape(len(positions), 63)
    selected["betas"] = selected["betas"].reshape(len(positions), -1)[:, :10]
    selected["transl"] = selected["transl"].reshape(len(positions), 3)
    return selected


def _infer_sequence(frames: list[dict[str, Any]]) -> str:
    for frame in frames:
        parts = Path(str(frame.get("image", ""))).parts
        for index, part in enumerate(parts[:-2]):
            if part in {"test", "train", "val"} and index + 2 < len(parts):
                return "/".join(parts[index : index + 3])
    raise ValueError("Cannot infer RICH sequence from cached frame image paths; pass --rich-sequence")


def _rich_camera_key(sequence: str) -> str:
    parts = sequence.strip("/").split("/")
    if len(parts) != 3 or not parts[2].startswith("cam_"):
        raise ValueError(f"Expected RICH sequence split/recording/cam_XX, got {sequence!r}")
    scene = parts[1].split("_", 1)[0]
    return f"{scene}_{int(parts[2].removeprefix('cam_'))}"


def _source_frame_index(frame: dict[str, Any]) -> int:
    value = int(frame.get("source_frame_index", -1))
    if value >= 0:
        return value
    stem = Path(str(frame.get("image", frame.get("frame_id", "-1")))).stem
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(f"Cannot infer source RICH frame index from cache frame: {frame.get('frame_id')!r}") from exc


def _cached_faces(frames: list[dict[str, Any]]) -> np.ndarray:
    for frame in frames:
        for person in frame.get("people", []):
            if person.get("faces") is not None:
                faces = np.asarray(person["faces"], dtype=np.int32).reshape(-1, 3)
                if faces.size:
                    return faces
    raise ValueError("Cached scene has no SMPL faces to display RICH GT")


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return np.asarray(points, dtype=np.float32) @ matrix[:3, :3].T + matrix[:3, 3]


def _camera_points_to_world(points_cam: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    matrix = np.asarray(world_to_camera, dtype=np.float32).reshape(4, 4)
    return (np.asarray(points_cam, dtype=np.float32) - matrix[:3, 3]) @ matrix[:3, :3]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
