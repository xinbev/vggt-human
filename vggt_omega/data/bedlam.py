import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import sys
from vggt_omega.data.bedlam_boxes import extract_person_id
from vggt_omega.utils.rotation import axis_angle_to_rot6d
import numpy
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric


TRACK_SOURCE_SLOT = 0
TRACK_SOURCE_PERSON_INDEX = 1
TRACK_SOURCE_EXPLICIT_ID = 2
TRACK_ID_NAMESPACE = 1_000_000_000

# Some BEDLAM pickle files may reference NumPy 2.x module names. Register the
# compatibility aliases only after project imports have loaded torch, because
# setting numpy._core before torch import can segfault in this environment.
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


class BedlamDataset(Dataset):
    """Read preprocessed BEDLAM-style sequence data for VGGT-Omega training.

    Expected layout:
      root/<split>/<sequence>/rgb/frame_*.png
      root/<split>/<sequence>/depth/frame_*.npy       optional
      root/<split>/<sequence>/cam/frame_*.npz         optional unless intrinsics are needed later
      root/<split>/<sequence>/smpl/frame_*.pkl        optional; empty people if missing

    Images are returned as raw float RGB tensors in [0, 1]. The model aggregator
    owns ImageNet normalization, so the dataset intentionally does not normalize.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "Training",
        sequence_length: int = 2,
        stride: int = 1,
        image_size: int = 518,
        max_humans: int = 20,
        require_smpl: bool = True,
        require_depth: bool = False,
        boxes_root: str | Path | None = None,
        require_boxes: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = split
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.max_humans = int(max_humans)
        self.require_smpl = require_smpl
        self.require_depth = require_depth
        self.boxes_root = Path(boxes_root).expanduser() if boxes_root else None
        self.require_boxes = require_boxes
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self._sequences = _build_sequence_index(self.root, split)
        self._index: list[tuple[int, int]] = []
        for seq_idx, (_, frame_ids) in enumerate(self._sequences):
            max_start = len(frame_ids) - (self.sequence_length - 1) * self.stride
            for frame_idx in range(max_start):
                self._index.append((seq_idx, frame_idx))
        if not self._index:
            raise RuntimeError(
                f"No trainable frame windows found for split={split!r}, "
                f"sequence_length={self.sequence_length}, stride={self.stride}."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start_idx = self._index[idx]
        seq_dir, frame_ids = self._sequences[seq_idx]
        selected = [frame_ids[start_idx + step * self.stride] for step in range(self.sequence_length)]

        images = []
        depths = []
        intrinsics = []
        persons_per_frame = []
        boxes_per_frame = []
        for frame_id in selected:
            rgb_path = seq_dir / "rgb" / f"{frame_id}.png"
            depth_path = seq_dir / "depth" / f"{frame_id}.npy"
            cam_path = seq_dir / "cam" / f"{frame_id}.npz"
            smpl_path = seq_dir / "smpl" / f"{frame_id}.pkl"
            box_path = self._box_path(seq_dir, frame_id) if self.boxes_root is not None else None

            image, orig_hw = _load_rgb_tensor(rgb_path, self.image_size)
            images.append(image)
            depths.append(_load_depth_tensor(depth_path, self.image_size, self.require_depth))
            intrinsics.append(_load_intrinsics(cam_path, orig_hw, self.image_size))
            persons_per_frame.append(_load_persons(smpl_path, self.require_smpl))
            boxes_per_frame.append(_load_box_persons(box_path, self.require_boxes) if box_path is not None else None)

        smpl = _build_smpl_targets(persons_per_frame, self.max_humans)
        boxes = _build_box_targets(boxes_per_frame, persons_per_frame, self.max_humans, self.require_boxes)
        return {
            "images": torch.stack(images, dim=0),
            "gt_depth": torch.stack(depths, dim=0),
            "K_scal3r": torch.stack(intrinsics, dim=0),
            "gt_pose_6d": smpl["pose_6d"],
            "gt_betas": smpl["betas"],
            "gt_transl_cam": smpl["transl_cam"],
            "gt_cam_trans": smpl["transl_cam"],
            "smpl_mask": smpl["smpl_mask"],
            "gt_boxes": boxes["boxes"],
            "boxes_mask": boxes["boxes_mask"],
            "person_ids": boxes["person_ids"],
            "person_id_mask": boxes["person_id_mask"],
            "gt_track_ids": boxes["gt_track_ids"],
            "gt_track_mask": boxes["gt_track_mask"],
            "gt_track_source": boxes["gt_track_source"],
            "gt_track_quality": boxes["gt_track_quality"],
        }

    def _box_path(self, seq_dir: Path, frame_id: str) -> Path:
        sequence_name = seq_dir.relative_to(self.root / self.split)
        return self.boxes_root / self.split / sequence_name / "smpl_boxes" / f"{frame_id}.pkl"


def bedlam_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty BEDLAM batch")
    out: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([_require_tensor(item[key], key) for item in batch], dim=0)
    return out


def _build_sequence_index(root: Path, split: str) -> list[tuple[Path, list[str]]]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")
    sequences = []
    for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        frames = sorted(path.stem for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if frames:
            sequences.append((seq_dir, frames))
    if not sequences:
        raise RuntimeError(f"No valid BEDLAM sequences found under {split_dir}")
    return sequences


def _load_rgb_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"RGB frame not found: {path}")
    image = Image.open(path).convert("RGB")
    orig_hw = (image.height, image.width)
    image = image.resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_hw


def _load_depth_tensor(path: Path, size: int, require_depth: bool) -> torch.Tensor:
    if not path.is_file():
        if require_depth:
            raise FileNotFoundError(f"Depth frame not found: {path}")
        return torch.zeros(1, size, size, dtype=torch.float32)
    depth = np.load(path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map from {path}, got {depth.shape}")
    image = Image.fromarray(depth, mode="F").resize((int(size), int(size)), Image.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)


def _load_intrinsics(path: Path, orig_hw: tuple[int, int], size: int) -> torch.Tensor:
    if path.is_file():
        data = np.load(path)
        if "intrinsics" not in data:
            raise ValueError(f"Camera file missing 'intrinsics': {path}")
        intrinsics = data["intrinsics"].astype(np.float32)
    else:
        focal = float(size)
        center = (float(size) - 1.0) * 0.5
        intrinsics = np.asarray([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float32)
        return torch.from_numpy(intrinsics)

    src_h, src_w = orig_hw
    scaled = intrinsics.copy()
    scaled[0, 0] *= float(size) / float(src_w)
    scaled[0, 2] *= float(size) / float(src_w)
    scaled[1, 1] *= float(size) / float(src_h)
    scaled[1, 2] *= float(size) / float(src_h)
    return torch.from_numpy(scaled)


def _load_persons(path: Path, require_smpl: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        if require_smpl:
            raise FileNotFoundError(f"SMPL annotation not found: {path}")
        return []
    with path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL annotation must be a list of person dicts: {path}")
    return persons


def _load_box_persons(path: Path, require_boxes: bool) -> list[dict[str, Any]] | None:
    if not path.is_file():
        if require_boxes:
            raise FileNotFoundError(f"Preprocessed bbox annotation not found: {path}. Run scripts/preprocess/prepare_bedlam_boxes.py first.")
        return None
    with path.open("rb") as file:
        data = pickle.load(file)
    persons = data.get("persons") if isinstance(data, dict) else None
    if not isinstance(persons, list):
        raise TypeError(f"Preprocessed bbox annotation must contain a persons list: {path}")
    return persons


def _build_smpl_targets(persons_per_frame: list[list[dict[str, Any]]], max_humans: int) -> dict[str, torch.Tensor]:
    pose_frames = []
    beta_frames = []
    transl_cam_frames = []
    mask_frames = []
    for persons in persons_per_frame:
        poses = torch.zeros(max_humans, 144, dtype=torch.float32)
        betas = torch.zeros(max_humans, 10, dtype=torch.float32)
        transl_cam = torch.zeros(max_humans, 3, dtype=torch.float32)
        mask = torch.zeros(max_humans, dtype=torch.bool)
        for person_idx, person in enumerate(persons[:max_humans]):
            poses[person_idx] = _person_pose_6d(person)
            betas[person_idx] = torch.as_tensor(person["smplx_shape"], dtype=torch.float32).reshape(-1)[:10]
            transl_cam[person_idx] = torch.as_tensor(person["smplx_transl"], dtype=torch.float32).reshape(3)
            mask[person_idx] = True
        pose_frames.append(poses)
        beta_frames.append(betas)
        transl_cam_frames.append(transl_cam)
        mask_frames.append(mask)
    return {
        "pose_6d": torch.stack(pose_frames, dim=0),
        "betas": torch.stack(beta_frames, dim=0),
        "transl_cam": torch.stack(transl_cam_frames, dim=0),
        "smpl_mask": torch.stack(mask_frames, dim=0),
    }


def _build_box_targets(
    boxes_per_frame: list[list[dict[str, Any]] | None],
    persons_per_frame: list[list[dict[str, Any]]],
    max_humans: int,
    require_boxes: bool,
) -> dict[str, torch.Tensor]:
    explicit_by_person_index: dict[int, int] = {}
    explicit_by_slot: dict[int, int] = {}
    for frame_idx, persons in enumerate(persons_per_frame):
        box_persons = boxes_per_frame[frame_idx]
        for person_idx, person in enumerate(persons[:max_humans]):
            box_person = box_persons[person_idx] if box_persons is not None and person_idx < len(box_persons) else None
            explicit_id, explicit_valid = _extract_explicit_person_id(box_person, person)
            if not explicit_valid:
                continue
            sidecar_index, sidecar_valid = _extract_person_index(box_person)
            if sidecar_valid:
                explicit_by_person_index.setdefault(sidecar_index, explicit_id)
            explicit_by_slot.setdefault(person_idx, explicit_id)

    box_frames = []
    box_mask_frames = []
    person_id_frames = []
    person_id_mask_frames = []
    track_source_frames = []
    for frame_idx, persons in enumerate(persons_per_frame):
        boxes = torch.zeros(max_humans, 4, dtype=torch.float32)
        boxes_mask = torch.zeros(max_humans, dtype=torch.bool)
        person_ids = torch.full((max_humans,), -1, dtype=torch.long)
        person_id_mask = torch.zeros(max_humans, dtype=torch.bool)
        track_source = torch.full((max_humans,), -1, dtype=torch.long)
        box_persons = boxes_per_frame[frame_idx]
        for person_idx, person in enumerate(persons[:max_humans]):
            box_person = None
            if box_persons is not None and person_idx < len(box_persons):
                box_person = box_persons[person_idx]
                if bool(box_person.get("bbox_valid", False)):
                    boxes[person_idx] = torch.as_tensor(box_person["bbox_cxcywh_norm"], dtype=torch.float32).reshape(4).clamp(0.0, 1.0)
                    boxes_mask[person_idx] = True
            track_id, source = _resolve_track_id(box_person, person, person_idx, explicit_by_person_index, explicit_by_slot)
            person_ids[person_idx] = track_id
            person_id_mask[person_idx] = True
            track_source[person_idx] = source
            if require_boxes and not boxes_mask[person_idx]:
                raise ValueError(
                    "Valid SMPL person is missing a preprocessed bbox. "
                    "Run scripts/preprocess/prepare_bedlam_boxes.py with usable bbox/j2d annotations."
                )
        box_frames.append(boxes)
        box_mask_frames.append(boxes_mask)
        person_id_frames.append(person_ids)
        person_id_mask_frames.append(person_id_mask)
        track_source_frames.append(track_source)
    track_quality = _compute_track_quality(persons_per_frame, person_id_frames, person_id_mask_frames, track_source_frames, max_humans)
    track_ids = torch.stack(person_id_frames, dim=0)
    track_mask = torch.stack(person_id_mask_frames, dim=0)
    track_source = torch.stack(track_source_frames, dim=0)
    return {
        "boxes": torch.stack(box_frames, dim=0),
        "boxes_mask": torch.stack(box_mask_frames, dim=0),
        "person_ids": track_ids,
        "person_id_mask": track_mask,
        "gt_track_ids": track_ids,
        "gt_track_mask": track_mask,
        "gt_track_source": track_source,
        "gt_track_quality": track_quality,
    }


def _extract_explicit_person_id(box_person: dict[str, Any] | None, person: dict[str, Any]) -> tuple[int, bool]:
    if box_person is not None and bool(box_person.get("person_id_valid", False)):
        try:
            person_id = int(box_person.get("person_id", -1))
            if person_id >= 0:
                return person_id, True
        except (TypeError, ValueError):
            pass
    return extract_person_id(person)


def _extract_person_index(box_person: dict[str, Any] | None) -> tuple[int, bool]:
    if box_person is None or "person_index" not in box_person:
        return -1, False
    try:
        return int(box_person["person_index"]), True
    except (TypeError, ValueError):
        return -1, False


def _resolve_track_id(
    box_person: dict[str, Any] | None,
    person: dict[str, Any],
    person_idx: int,
    explicit_by_person_index: dict[int, int],
    explicit_by_slot: dict[int, int],
) -> tuple[int, int]:
    explicit_id, explicit_valid = _extract_explicit_person_id(box_person, person)
    if explicit_valid:
        return _namespaced_track_id(TRACK_SOURCE_EXPLICIT_ID, explicit_id), TRACK_SOURCE_EXPLICIT_ID

    sidecar_index, sidecar_valid = _extract_person_index(box_person)
    if sidecar_valid and sidecar_index in explicit_by_person_index:
        return _namespaced_track_id(TRACK_SOURCE_EXPLICIT_ID, explicit_by_person_index[sidecar_index]), TRACK_SOURCE_EXPLICIT_ID
    if person_idx in explicit_by_slot:
        return _namespaced_track_id(TRACK_SOURCE_EXPLICIT_ID, explicit_by_slot[person_idx]), TRACK_SOURCE_EXPLICIT_ID
    if sidecar_valid:
        return _namespaced_track_id(TRACK_SOURCE_PERSON_INDEX, sidecar_index), TRACK_SOURCE_PERSON_INDEX
    return _namespaced_track_id(TRACK_SOURCE_SLOT, person_idx), TRACK_SOURCE_SLOT


def _namespaced_track_id(source: int, raw_id: int) -> int:
    if raw_id < 0:
        return -1
    return int(source) * TRACK_ID_NAMESPACE + int(raw_id)


def _compute_track_quality(
    persons_per_frame: list[list[dict[str, Any]]],
    track_id_frames: list[torch.Tensor],
    track_mask_frames: list[torch.Tensor],
    track_source_frames: list[torch.Tensor],
    max_humans: int,
) -> torch.Tensor:
    quality_frames = []
    previous: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for frame_idx, persons in enumerate(persons_per_frame):
        quality = torch.zeros(max_humans, dtype=torch.float32)
        track_ids = track_id_frames[frame_idx]
        track_mask = track_mask_frames[frame_idx]
        track_source = track_source_frames[frame_idx]
        for person_idx, person in enumerate(persons[:max_humans]):
            if not bool(track_mask[person_idx]):
                continue
            track_id = int(track_ids[person_idx])
            source = int(track_source[person_idx])
            base_quality = _track_source_base_quality(source)
            transl = torch.as_tensor(person["smplx_transl"], dtype=torch.float32).reshape(3)
            betas = torch.as_tensor(person["smplx_shape"], dtype=torch.float32).reshape(-1)[:10]
            if track_id in previous:
                prev_transl, prev_betas = previous[track_id]
                transl_dist = torch.linalg.norm(transl - prev_transl).item()
                beta_l1 = torch.mean(torch.abs(betas - prev_betas)).item()
                continuity = 1.0 / (1.0 + float(transl_dist) + 0.2 * float(beta_l1))
                quality[person_idx] = float(min(base_quality, continuity))
            else:
                quality[person_idx] = base_quality
            previous[track_id] = (transl, betas)
        quality_frames.append(quality)
    return torch.stack(quality_frames, dim=0)


def _track_source_base_quality(source: int) -> float:
    if source == TRACK_SOURCE_EXPLICIT_ID:
        return 1.0
    if source == TRACK_SOURCE_PERSON_INDEX:
        return 0.95
    return 0.75


def _person_pose_6d(person: dict[str, Any]) -> torch.Tensor:
    root_pose = torch.as_tensor(person["smplx_root_pose"], dtype=torch.float32).reshape(1, 3)
    body_pose = torch.as_tensor(person["smplx_body_pose"], dtype=torch.float32).reshape(21, 3)
    aa_22 = torch.cat([root_pose, body_pose], dim=0)
    pose_6d_22 = axis_angle_to_rot6d(aa_22).reshape(22, 6)
    identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32).expand(2, -1)
    return torch.cat([pose_6d_22, identity_6d], dim=0).reshape(144)


def _require_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
