from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.utils.rotation import axis_angle_to_rot6d


class ThreeDPWDataset(Dataset):
    """Read compact 3DPW SMPL-base annotations without copying image files.

    The annotation cache is produced by
    ``scripts/preprocess/prepare_3dpw_smpl_base.py`` and stores camera-space
    SMPL targets plus 2D query boxes. Images are loaded from the original
    ``imageFiles`` directory.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_root: str | Path,
        split: str = "train",
        sequence_length: int = 1,
        stride: int = 1,
        image_size: int = 518,
        max_humans: int = 2,
        require_boxes: bool = True,
        require_smpl: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.annotation_root = Path(annotation_root).expanduser()
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.max_humans = int(max_humans)
        self.require_boxes = bool(require_boxes)
        self.require_smpl = bool(require_smpl)
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        annot_path = self.annotation_root / f"{self.split}.pkl"
        if not annot_path.is_file():
            raise FileNotFoundError(
                f"3DPW annotation cache not found: {annot_path}. "
                "Run scripts/preprocess/prepare_3dpw_smpl_base.sh first."
            )
        with annot_path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict) or "frames" not in data:
            raise TypeError(f"Invalid 3DPW annotation cache: {annot_path}")
        self.frames: dict[str, dict[str, Any]] = data["frames"]
        self._sequences = self._build_sequence_index()
        self._index: list[tuple[int, int]] = []
        for seq_idx, (_, frame_keys) in enumerate(self._sequences):
            max_start = len(frame_keys) - (self.sequence_length - 1) * self.stride
            for frame_idx in range(max(max_start, 0)):
                self._index.append((seq_idx, frame_idx))
        if not self._index:
            raise RuntimeError(
                f"No 3DPW windows found for split={split!r}, "
                f"sequence_length={sequence_length}, stride={stride}."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start_idx = self._index[idx]
        _, frame_keys = self._sequences[seq_idx]
        selected = [frame_keys[start_idx + step * self.stride] for step in range(self.sequence_length)]

        images = []
        intrinsics = []
        persons_per_frame = []
        for frame_key in selected:
            frame = self.frames[frame_key]
            image, orig_hw = _load_rgb_tensor(self.root / "imageFiles" / frame["image_relpath"], self.image_size)
            images.append(image)
            intrinsics.append(_scale_intrinsics(torch.as_tensor(frame["K"], dtype=torch.float32), orig_hw, self.image_size))
            persons_per_frame.append(frame.get("persons", []))

        targets = _build_targets(persons_per_frame, self.max_humans, self.require_boxes, self.require_smpl)
        return {
            "images": torch.stack(images, dim=0),
            "gt_depth": torch.zeros(self.sequence_length, 1, self.image_size, self.image_size, dtype=torch.float32),
            "K_scal3r": torch.stack(intrinsics, dim=0),
            **targets,
        }

    def _build_sequence_index(self) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for key in self.frames:
            seq = key.split("/", 1)[0]
            grouped.setdefault(seq, []).append(key)
        sequences = []
        for seq, keys in sorted(grouped.items()):
            keys.sort(key=_frame_sort_key)
            sequences.append((seq, keys))
        return sequences


def threedpw_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty 3DPW batch")
    return {key: torch.stack([_require_tensor(item[key], key) for item in batch], dim=0) for key in batch[0].keys()}


def _build_targets(
    persons_per_frame: list[list[dict[str, Any]]],
    max_humans: int,
    require_boxes: bool,
    require_smpl: bool,
) -> dict[str, torch.Tensor]:
    pose_frames = []
    beta_frames = []
    transl_frames = []
    smpl_mask_frames = []
    box_frames = []
    box_mask_frames = []
    id_frames = []
    id_mask_frames = []
    source_frames = []
    quality_frames = []
    for persons in persons_per_frame:
        poses = torch.zeros(max_humans, 144, dtype=torch.float32)
        betas = torch.zeros(max_humans, 10, dtype=torch.float32)
        transl = torch.zeros(max_humans, 3, dtype=torch.float32)
        smpl_mask = torch.zeros(max_humans, dtype=torch.bool)
        boxes = torch.zeros(max_humans, 4, dtype=torch.float32)
        boxes_mask = torch.zeros(max_humans, dtype=torch.bool)
        ids = torch.full((max_humans,), -1, dtype=torch.long)
        ids_mask = torch.zeros(max_humans, dtype=torch.bool)
        source = torch.full((max_humans,), 2, dtype=torch.long)
        quality = torch.zeros(max_humans, dtype=torch.float32)

        valid_persons = [person for person in persons if person.get("smpl_valid", True)]
        valid_persons.sort(key=lambda item: float(item.get("smpl_transl", [0.0, 0.0, 1e6])[2]))
        for slot, person in enumerate(valid_persons[:max_humans]):
            root = torch.as_tensor(person["smpl_root_pose"], dtype=torch.float32).reshape(1, 3)
            body = torch.as_tensor(person["smpl_body_pose"], dtype=torch.float32).reshape(23, 3)
            aa = torch.cat([root, body], dim=0)
            poses[slot] = axis_angle_to_rot6d(aa).reshape(144)
            betas[slot] = torch.as_tensor(person["smpl_shape"], dtype=torch.float32).reshape(-1)[:10]
            transl[slot] = torch.as_tensor(person["smpl_transl"], dtype=torch.float32).reshape(3)
            smpl_mask[slot] = True
            ids[slot] = int(person.get("person_id", slot))
            ids_mask[slot] = True
            quality[slot] = 1.0
            if bool(person.get("bbox_valid", False)):
                boxes[slot] = torch.as_tensor(person["bbox_cxcywh_norm"], dtype=torch.float32).reshape(4).clamp(0.0, 1.0)
                boxes_mask[slot] = True
            elif require_boxes:
                raise ValueError("3DPW person is missing a valid 2D bbox; rebuild annotations with a lower keypoint threshold.")

        if require_smpl and not bool(smpl_mask.any()):
            raise ValueError("3DPW frame has no valid SMPL person")
        pose_frames.append(poses)
        beta_frames.append(betas)
        transl_frames.append(transl)
        smpl_mask_frames.append(smpl_mask)
        box_frames.append(boxes)
        box_mask_frames.append(boxes_mask)
        id_frames.append(ids)
        id_mask_frames.append(ids_mask)
        source_frames.append(source)
        quality_frames.append(quality)

    track_ids = torch.stack(id_frames, dim=0)
    track_mask = torch.stack(id_mask_frames, dim=0)
    transl_cam = torch.stack(transl_frames, dim=0)
    return {
        "gt_pose_6d": torch.stack(pose_frames, dim=0),
        "gt_betas": torch.stack(beta_frames, dim=0),
        "gt_transl_cam": transl_cam,
        "gt_cam_trans": transl_cam,
        "smpl_mask": torch.stack(smpl_mask_frames, dim=0),
        "gt_boxes": torch.stack(box_frames, dim=0),
        "boxes_mask": torch.stack(box_mask_frames, dim=0),
        "person_ids": track_ids,
        "person_id_mask": track_mask,
        "gt_track_ids": track_ids,
        "gt_track_mask": track_mask,
        "gt_track_source": torch.stack(source_frames, dim=0),
        "gt_track_quality": torch.stack(quality_frames, dim=0),
    }


def _load_rgb_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"3DPW RGB frame not found: {path}")
    image = Image.open(path).convert("RGB")
    orig_hw = (image.height, image.width)
    image = image.resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_hw


def _scale_intrinsics(intrinsics: torch.Tensor, orig_hw: tuple[int, int], size: int) -> torch.Tensor:
    src_h, src_w = orig_hw
    scaled = intrinsics.clone().float()
    scaled[0, 0] *= float(size) / float(max(src_w, 1))
    scaled[0, 2] *= float(size) / float(max(src_w, 1))
    scaled[1, 1] *= float(size) / float(max(src_h, 1))
    scaled[1, 2] *= float(size) / float(max(src_h, 1))
    return scaled


def _frame_sort_key(key: str) -> tuple[str, int]:
    seq, name = key.split("/", 1)
    match = re.search(r"(\d+)", name)
    return seq, int(match.group(1)) if match else 0


def _require_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
