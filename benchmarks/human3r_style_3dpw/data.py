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
    poses, betas, genders, person_ids = gt_camera_parameters(sequence, frame_index, device)
    vertices, joints = [], []
    for row, person in enumerate(person_ids.tolist()):
        layer = smpl_layers[genders[row]]
        local_vertices, local_joints = layer(poses[row : row + 1], betas[row : row + 1])
        # The camera-space root translation is reconstructed by subtracting
        # the zero-translation local root from the already camera-space joint.
        camera_root = gt_camera_root_translation(sequence, frame_index, person, local_joints[0, 0], device)
        camera_vertices = local_vertices[0] + camera_root
        camera_joints = local_joints[0, :24] + camera_root
        vertices.append(camera_vertices)
        joints.append(camera_joints)
    if not vertices:
        empty_v = torch.empty(0, 6890, 3, device=device)
        empty_j = torch.empty(0, 24, 3, device=device)
        return empty_v, empty_j, torch.empty(0, dtype=torch.long, device=device)
    return torch.stack(vertices), torch.stack(joints), person_ids


def gt_camera_parameters(
    sequence: ThreeDPWTestSequence,
    frame_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Return camera-orientated SMPL pose, betas, gender and original person ID."""
    meta = sequence.metadata
    t_w2c = np.asarray(meta["cam_poses"][frame_index], dtype=np.float32).reshape(4, 4)
    rotation = t_w2c[:3, :3]
    poses, betas, genders, person_ids = [], [], [], []
    for person in range(sequence.persons):
        if not bool(np.asarray(meta["campose_valid"][person])[frame_index]):
            continue
        pose = np.asarray(meta["poses"][person][frame_index], dtype=np.float32).reshape(24, 3)
        root_rotation = axis_angle_to_rotmat(torch.from_numpy(pose[0])).cpu().numpy()
        root_camera = rotation_matrix_to_axis_angle(torch.from_numpy(rotation @ root_rotation)).cpu().numpy().reshape(3)
        poses.append(np.concatenate((root_camera[None], pose[1:]), axis=0).reshape(72))
        betas.append(np.asarray(meta["betas"][person], dtype=np.float32).reshape(-1)[:10])
        genders.append("male" if str(meta["genders"][person]).lower().startswith("m") else "female")
        person_ids.append(person)
    if not poses:
        return (
            torch.empty(0, 72, device=device),
            torch.empty(0, 10, device=device),
            [],
            torch.empty(0, dtype=torch.long, device=device),
        )
    return (
        torch.from_numpy(np.stack(poses)).to(device).float(),
        torch.from_numpy(np.stack(betas)).to(device).float(),
        genders,
        torch.tensor(person_ids, dtype=torch.long, device=device),
    )


def gt_camera_root_translation(
    sequence: ThreeDPWTestSequence,
    frame_index: int,
    person: int,
    local_root_joint: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Human3R-style conversion from raw 3DPW world translation to camera root."""
    meta = sequence.metadata
    t_w2c = np.asarray(meta["cam_poses"][frame_index], dtype=np.float32).reshape(4, 4)
    rotation, translation = torch.from_numpy(t_w2c[:3, :3]).to(device), torch.from_numpy(t_w2c[:3, 3]).to(device)
    raw_trans = torch.from_numpy(np.asarray(meta["trans"][person][frame_index], dtype=np.float32)).to(device)
    root_after_raw_trans = local_root_joint + raw_trans
    camera_root = rotation @ root_after_raw_trans + translation
    return camera_root - local_root_joint
