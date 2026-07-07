from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.data.geometry import (
    ResizeGeometry,
    collate_smpl_geometry_batch,
    compute_resize_geometry,
    resize_image_with_geometry,
    transform_intrinsics,
    transform_xyxy_to_normalized_cxcywh,
)
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
        image_size: int = 512,
        image_resolution: int | None = None,
        resize_mode: str = "balanced",
        max_humans: int = 2,
        require_boxes: bool = True,
        require_smpl: bool = True,
        sam2_patch_masks_root: str | Path | None = None,
        require_sam2_patch_masks: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.annotation_root = Path(annotation_root).expanduser()
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_resolution = int(image_resolution or image_size)
        self.resize_mode = str(resize_mode or "balanced")
        self.image_size = self.image_resolution
        self.max_humans = int(max_humans)
        self.require_boxes = bool(require_boxes)
        self.require_smpl = bool(require_smpl)
        self.require_sam2_patch_masks = bool(require_sam2_patch_masks)
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
        self.sam2_patch_masks = _load_sam2_patch_masks(
            sam2_patch_masks_root,
            split=self.split,
            require=self.require_sam2_patch_masks,
        )
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
        patch_masks_per_frame = []
        orig_hws = []
        for frame_key in selected:
            frame = self.frames[frame_key]
            image, orig_hw, geometry = _load_rgb_tensor(self.root / "imageFiles" / frame["image_relpath"], self.image_resolution, 16, self.resize_mode)
            images.append(image)
            orig_hws.append(orig_hw)
            intrinsics.append(_scale_intrinsics(torch.as_tensor(frame["K"], dtype=torch.float32), orig_hw, geometry))
            persons_per_frame.append(_map_person_boxes(frame.get("persons", []), orig_hw, geometry))
            if self.sam2_patch_masks is not None:
                patch_masks_per_frame.append(self.sam2_patch_masks["frames"].get(frame_key, {}))

        sample_num_patches = (int(images[0].shape[-2]) // 16) * (int(images[0].shape[-1]) // 16)
        targets = _build_targets(
            persons_per_frame,
            self.max_humans,
            self.require_boxes,
            self.require_smpl,
            patch_masks_per_frame if self.sam2_patch_masks is not None else None,
            sample_num_patches if self.sam2_patch_masks is not None else 0,
        )
        intrinsics_tensor = torch.stack(intrinsics, dim=0)
        return {
            "images": torch.stack(images, dim=0),
            "gt_depth": torch.zeros(self.sequence_length, 1, images[0].shape[-2], images[0].shape[-1], dtype=torch.float32),
            "K_scal3r": intrinsics_tensor,
            "gt_intrinsics": intrinsics_tensor,
            "valid_hw": torch.tensor([list(image.shape[-2:]) for image in images], dtype=torch.long),
            "image_hw": torch.tensor([list(image.shape[-2:]) for image in images], dtype=torch.long),
            "orig_hw": torch.tensor([list(hw) for hw in orig_hws], dtype=torch.long),
            "patch_size": torch.tensor(16, dtype=torch.long),
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
    patch_size = int(batch[0].get("patch_size", torch.tensor(16)).reshape(-1)[0].item())
    return collate_smpl_geometry_batch(batch, patch_size=patch_size)


def _build_targets(
    persons_per_frame: list[list[dict[str, Any]]],
    max_humans: int,
    require_boxes: bool,
    require_smpl: bool,
    patch_masks_per_frame: list[dict[int, np.ndarray]] | None = None,
    num_patches: int = 0,
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
    patch_mask_frames = []
    patch_mask_valid_frames = []
    for frame_idx, persons in enumerate(persons_per_frame):
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
        patch_masks = torch.zeros(max_humans, int(num_patches), dtype=torch.bool) if num_patches > 0 else None
        patch_masks_valid = torch.zeros(max_humans, dtype=torch.bool) if num_patches > 0 else None
        frame_patch_masks = patch_masks_per_frame[frame_idx] if patch_masks_per_frame is not None else {}

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
            if patch_masks is not None and patch_masks_valid is not None:
                mask = frame_patch_masks.get(int(ids[slot].item()))
                if mask is not None:
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
                    if int(mask_tensor.numel()) != int(patch_masks.shape[-1]):
                        continue
                    patch_masks[slot] = mask_tensor
                    patch_masks_valid[slot] = bool(patch_masks[slot].any())
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
        if patch_masks is not None and patch_masks_valid is not None:
            patch_mask_frames.append(patch_masks)
            patch_mask_valid_frames.append(patch_masks_valid)

    track_ids = torch.stack(id_frames, dim=0)
    track_mask = torch.stack(id_mask_frames, dim=0)
    transl_cam = torch.stack(transl_frames, dim=0)
    out = {
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
    if patch_mask_frames:
        out["smpl_query_patch_masks"] = torch.stack(patch_mask_frames, dim=0)
        out["smpl_query_patch_masks_valid"] = torch.stack(patch_mask_valid_frames, dim=0)
    return out


def _load_sam2_patch_masks(root: str | Path | None, split: str, require: bool) -> dict[str, Any] | None:
    if root is None or str(root).strip() == "":
        if require:
            raise ValueError("3DPW SAM2 patch masks are required but sam2_patch_masks_root is empty")
        return None
    root_path = Path(root).expanduser()
    path = root_path / f"{split}.pkl"
    if not path.is_file():
        if require:
            raise FileNotFoundError(f"3DPW SAM2 patch-mask cache not found: {path}")
        return None
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict) or "frames" not in data:
        raise TypeError(f"Invalid 3DPW SAM2 patch-mask cache: {path}")
    num_patches = int(data.get("num_patches", 0))
    if num_patches <= 0:
        raise ValueError(f"Invalid num_patches in SAM2 cache {path}: {num_patches}")
    frames: dict[str, dict[int, np.ndarray]] = {}
    for frame_key, packed_by_person in data["frames"].items():
        frame_masks: dict[int, np.ndarray] = {}
        if not isinstance(packed_by_person, dict):
            continue
        for raw_pid, item in packed_by_person.items():
            if not isinstance(item, dict) or "bits" not in item:
                continue
            item_num_patches = int(item.get("num_patches", num_patches))
            mask = np.unpackbits(np.asarray(item["bits"], dtype=np.uint8), count=item_num_patches).astype(np.bool_)
            frame_masks[int(raw_pid)] = mask
        frames[str(frame_key)] = frame_masks
    return {
        "num_patches": num_patches,
        "frames": frames,
        "path": str(path),
    }


def _load_rgb_tensor(path: Path, image_resolution: int, patch_size: int, resize_mode: str) -> tuple[torch.Tensor, tuple[int, int], ResizeGeometry]:
    if not path.is_file():
        raise FileNotFoundError(f"3DPW RGB frame not found: {path}")
    image = Image.open(path).convert("RGB")
    orig_hw = (image.height, image.width)
    geometry = compute_resize_geometry(orig_hw, image_resolution=image_resolution, patch_size=patch_size, mode=resize_mode)
    resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_hw, geometry


def _scale_intrinsics(intrinsics: torch.Tensor, orig_hw: tuple[int, int], geometry: ResizeGeometry) -> torch.Tensor:
    del orig_hw
    return transform_intrinsics(intrinsics, geometry)


def _map_person_boxes(persons: list[dict[str, Any]], orig_hw: tuple[int, int], geometry: ResizeGeometry) -> list[dict[str, Any]]:
    out = []
    image_h, image_w = orig_hw
    for person in persons:
        mapped = dict(person)
        xyxy = None
        if "bbox_xyxy_pixels" in mapped:
            xyxy = np.asarray(mapped["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
        elif "bbox_cxcywh_norm" in mapped:
            xyxy = _cxcywh_norm_to_xyxy(np.asarray(mapped["bbox_cxcywh_norm"], dtype=np.float32), image_w, image_h)
        if xyxy is not None:
            box, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometry)
            mapped["bbox_cxcywh_norm"] = box.tolist()
            mapped["bbox_valid"] = bool(valid)
        out.append(mapped)
    return out


def _cxcywh_norm_to_xyxy(box: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box.reshape(4)]
    bw = w * float(max(image_w, 1))
    bh = h * float(max(image_h, 1))
    x1 = (cx * float(max(image_w, 1))) - 0.5 * bw
    y1 = (cy * float(max(image_h, 1))) - 0.5 * bh
    return np.asarray([x1, y1, x1 + bw, y1 + bh], dtype=np.float32)


def _frame_sort_key(key: str) -> tuple[str, int]:
    seq, name = key.split("/", 1)
    match = re.search(r"(\d+)", name)
    return seq, int(match.group(1)) if match else 0


def _require_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
