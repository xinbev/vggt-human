"""Raw 3DPW test data and gender-specific camera-space GT construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image
import torch

from vggt_omega.data.geometry import ResizeGeometry, compute_resize_geometry, resize_image_with_geometry, transform_intrinsics
from vggt_omega.utils.rotation import axis_angle_to_rotmat, rotation_matrix_to_axis_angle


@dataclass(frozen=True)
class ThreeDPWTestSequence:
    name: str
    path: Path
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(len(self.metadata["poses"][0]))

    @property
    def persons(self) -> int:
        return int(len(self.metadata["poses"]))


def load_test_sequences(root: str | Path, sequence_filter: str = "") -> list[ThreeDPWTestSequence]:
    root_path = Path(root)
    paths = sorted((root_path / "sequenceFiles" / "test").glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No raw 3DPW test pkls below {root_path / 'sequenceFiles' / 'test'}")
    query = sequence_filter.lower().strip()
    sequences = []
    for path in paths:
        if query and query not in path.stem.lower():
            continue
        with path.open("rb") as file:
            metadata = pickle.load(file, encoding="latin1")
        sequences.append(ThreeDPWTestSequence(path.stem, path, metadata))
    if not sequences:
        raise ValueError(f"sequence_filter={sequence_filter!r} matched no raw 3DPW test sequence")
    return sequences


def frame_path(root: str | Path, sequence: ThreeDPWTestSequence, frame_index: int) -> Path:
    image_root = Path(root) / "imageFiles" / sequence.name
    candidates = [image_root / f"image_{frame_index:05d}.jpg"]
    ids = sequence.metadata.get("img_frame_ids")
    if ids is not None and frame_index < len(ids):
        candidates.append(image_root / f"image_{int(ids[frame_index]):05d}.jpg")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"3DPW RGB is missing for {sequence.name} frame={frame_index}; tried {candidates}")


def load_processed_frame(path: Path, intrinsics: np.ndarray, resolution: int, patch_size: int, resize_mode: str) -> tuple[torch.Tensor, torch.Tensor, ResizeGeometry]:
    image = Image.open(path).convert("RGB")
    geometry = compute_resize_geometry((image.height, image.width), image_resolution=resolution, patch_size=patch_size, mode=resize_mode)
    resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return tensor, transform_intrinsics(intrinsics, geometry), geometry


def raw_openpose_2d(sequence: ThreeDPWTestSequence, frame_index: int, geometry: ResizeGeometry) -> torch.Tensor:
    people = sequence.persons
    out = torch.zeros(people, 18, 3, dtype=torch.float32)
    x1, y1, _, _ = geometry.crop_xyxy
    sx, sy = geometry.scale_xy
    for person in range(people):
        poses2d = np.asarray(sequence.metadata["poses2d"][person], dtype=np.float32)
        if poses2d.shape[1] == 3:
            poses2d = poses2d.transpose(0, 2, 1)
        frame = poses2d[frame_index]
        out[person, :, 0] = torch.from_numpy((frame[:, 0] - x1) * sx)
        out[person, :, 1] = torch.from_numpy((frame[:, 1] - y1) * sy)
        out[person, :, 2] = torch.from_numpy(frame[:, 2])
    return out


@torch.no_grad()
def decode_gt_camera_space(
    sequence: ThreeDPWTestSequence,
    frame_index: int,
    smpl_layers: dict[str, torch.nn.Module],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Human3R-style GT: gender SMPL, root camera rotation, camera translation."""
    meta = sequence.metadata
    t_w2c = np.asarray(meta["cam_poses"][frame_index], dtype=np.float32).reshape(4, 4)
    rotation, translation = t_w2c[:3, :3], t_w2c[:3, 3]
    vertices, joints, valid = [], [], []
    for person in range(sequence.persons):
        is_valid = bool(np.asarray(meta["campose_valid"][person])[frame_index])
        if not is_valid:
            continue
        pose = np.asarray(meta["poses"][person][frame_index], dtype=np.float32).reshape(24, 3)
        betas = np.asarray(meta["betas"][person], dtype=np.float32).reshape(-1)[:10]
        raw_trans = np.asarray(meta["trans"][person][frame_index], dtype=np.float32).reshape(3)
        gender = "male" if str(meta["genders"][person]).lower().startswith("m") else "female"
        root_rotation = axis_angle_to_rotmat(torch.from_numpy(pose[0])).cpu().numpy()
        root_camera = rotation_matrix_to_axis_angle(torch.from_numpy(rotation @ root_rotation)).cpu().numpy().reshape(3)
        pose_camera = np.concatenate((root_camera[None], pose[1:]), axis=0).reshape(1, 72)
        layer = smpl_layers[gender]
        local_vertices, local_joints = layer(torch.from_numpy(pose_camera).to(device).float(), torch.from_numpy(betas[None]).to(device).float())
        local_vertices, local_joints = local_vertices[0], local_joints[0, :24]
        root_after_trans = local_joints[0] + torch.from_numpy(raw_trans).to(device)
        camera_root = torch.from_numpy(rotation).to(device) @ root_after_trans + torch.from_numpy(translation).to(device)
        camera_vertices = local_vertices + torch.from_numpy(raw_trans).to(device) - root_after_trans + camera_root
        camera_joints = local_joints + torch.from_numpy(raw_trans).to(device) - root_after_trans + camera_root
        vertices.append(camera_vertices)
        joints.append(camera_joints)
        valid.append(person)
    if not vertices:
        empty_v = torch.empty(0, 6890, 3, device=device)
        empty_j = torch.empty(0, 24, 3, device=device)
        return empty_v, empty_j, torch.empty(0, dtype=torch.long, device=device)
    return torch.stack(vertices), torch.stack(joints), torch.tensor(valid, dtype=torch.long, device=device)
