import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import sys
from vggt_omega.data.bedlam_boxes import extract_best_box, extract_person_id
from vggt_omega.data.geometry import (
    ResizeGeometry,
    collate_smpl_geometry_batch,
    compute_resize_geometry,
    default_intrinsics_for_geometry,
    resize_depth_with_geometry,
    resize_image_with_geometry,
    resize_mask_with_geometry,
    pixel_mask_to_patch_mask_hw,
    transform_intrinsics,
    transform_xyxy_to_normalized_cxcywh,
)
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
        image_size: int = 512,
        image_resolution: int | None = None,
        resize_mode: str = "balanced",
        max_humans: int = 20,
        require_smpl: bool = True,
        require_depth: bool = False,
        boxes_root: str | Path | None = None,
        require_boxes: bool = False,
        query_source: str = "persons",
        patch_size: int = 16,
        mask_patch_threshold: float = 0.10,
        min_mask_patches: int = 4,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = split
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_resolution = int(image_resolution or image_size)
        self.resize_mode = str(resize_mode or "balanced")
        self.image_size = self.image_resolution
        self.max_humans = int(max_humans)
        self.require_smpl = require_smpl
        self.require_depth = require_depth
        self.boxes_root = Path(boxes_root).expanduser() if boxes_root else None
        self.require_boxes = require_boxes
        self.query_source = str(query_source or "persons")
        self.patch_size = int(patch_size)
        self.mask_patch_threshold = float(mask_patch_threshold)
        self.min_mask_patches = int(min_mask_patches)
        if self.query_source not in {"persons", "detections"}:
            raise ValueError(f"Unsupported BEDLAM query_source: {self.query_source!r}")
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
        box_frames = []
        geometries = []
        for frame_id in selected:
            rgb_path = seq_dir / "rgb" / f"{frame_id}.png"
            depth_path = seq_dir / "depth" / f"{frame_id}.npy"
            cam_path = seq_dir / "cam" / f"{frame_id}.npz"
            smpl_path = seq_dir / "smpl" / f"{frame_id}.pkl"
            box_path = self._box_path(seq_dir, frame_id) if self.boxes_root is not None else None

            image, orig_hw, geometry = _load_rgb_tensor(rgb_path, self.image_resolution, self.patch_size, self.resize_mode)
            geometries.append(geometry)
            images.append(image)
            depths.append(_load_depth_tensor(depth_path, geometry, self.require_depth))
            intrinsics.append(_load_intrinsics(cam_path, orig_hw, geometry))
            persons_per_frame.append(_load_persons(smpl_path, self.require_smpl))
            box_frame = _load_box_frame(box_path, self.require_boxes) if box_path is not None else None
            box_frames.append(box_frame)
            boxes_per_frame.append(_frame_persons(box_frame, geometry))

        smpl = _build_smpl_targets(persons_per_frame, self.max_humans)
        boxes = _build_box_targets(boxes_per_frame, persons_per_frame, geometries, self.max_humans, self.require_boxes)
        sample = {
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
        if self.query_source == "detections":
            query = _build_detection_query_targets(
                box_frames=box_frames,
                max_humans=self.max_humans,
                image_size=self.image_size,
                geometries=geometries,
                patch_size=self.patch_size,
                mask_patch_threshold=self.mask_patch_threshold,
                min_mask_patches=self.min_mask_patches,
                sidecar_root=self.boxes_root,
            )
            external = _build_external_prior_targets(box_frames, query["smpl_query_boxes"], query["smpl_query_boxes_mask"], geometries=geometries)
            sample.update(query)
            sample.update(external)
        sample["valid_hw"] = torch.tensor([list(image.shape[-2:]) for image in images], dtype=torch.long)
        sample["image_hw"] = sample["valid_hw"].clone()
        sample["orig_hw"] = torch.tensor([list(orig_hw)] * len(images), dtype=torch.long)
        sample["patch_size"] = torch.tensor(self.patch_size, dtype=torch.long)
        return sample

    def _box_path(self, seq_dir: Path, frame_id: str) -> Path:
        sequence_name = seq_dir.relative_to(self.root / self.split)
        return self.boxes_root / self.split / sequence_name / "smpl_boxes" / f"{frame_id}.pkl"


def bedlam_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    patch_size = int(batch[0].get("patch_size", torch.tensor(16)).reshape(-1)[0].item())
    return collate_smpl_geometry_batch(batch, patch_size=patch_size)


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


def _load_rgb_tensor(path: Path, image_resolution: int, patch_size: int, resize_mode: str) -> tuple[torch.Tensor, tuple[int, int], ResizeGeometry]:
    if not path.is_file():
        raise FileNotFoundError(f"RGB frame not found: {path}")
    image = Image.open(path).convert("RGB")
    orig_hw = (image.height, image.width)
    geometry = compute_resize_geometry(orig_hw, image_resolution=image_resolution, patch_size=patch_size, mode=resize_mode)
    resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_hw, geometry


def _load_depth_tensor(path: Path, geometry: ResizeGeometry, require_depth: bool) -> torch.Tensor:
    if not path.is_file():
        if require_depth:
            raise FileNotFoundError(f"Depth frame not found: {path}")
        return torch.zeros(1, *geometry.input_hw, dtype=torch.float32)
    depth = np.load(path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map from {path}, got {depth.shape}")
    return torch.from_numpy(resize_depth_with_geometry(depth, geometry).copy()).unsqueeze(0)


def _load_intrinsics(path: Path, orig_hw: tuple[int, int], geometry: ResizeGeometry) -> torch.Tensor:
    if path.is_file():
        data = np.load(path)
        if "intrinsics" not in data:
            raise ValueError(f"Camera file missing 'intrinsics': {path}")
        intrinsics = data["intrinsics"].astype(np.float32)
    else:
        del orig_hw
        return default_intrinsics_for_geometry(geometry)
    return transform_intrinsics(intrinsics, geometry)


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


def _load_box_frame(path: Path | None, require_boxes: bool) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        if require_boxes:
            raise FileNotFoundError(f"Preprocessed bbox annotation not found: {path}. Run scripts/preprocess/prepare_bedlam_boxes.py first.")
        return None
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Preprocessed bbox annotation must be a frame dict: {path}")
    return data


def _frame_persons(frame: dict[str, Any] | None, geometry: ResizeGeometry) -> list[dict[str, Any]] | None:
    if frame is None:
        return None
    persons = frame.get("persons")
    if not isinstance(persons, list):
        return None
    out = []
    for person in persons:
        mapped = dict(person)
        if bool(mapped.get("bbox_valid", False)):
            xyxy = mapped.get("bbox_xyxy_pixels")
            if xyxy is None and "bbox_cxcywh_norm" in mapped:
                image_h, image_w = _frame_hw(frame)
                xyxy = _cxcywh_norm_to_xyxy(np.asarray(mapped["bbox_cxcywh_norm"], dtype=np.float32), image_w, image_h)
            if xyxy is not None:
                box, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometry)
                mapped["bbox_cxcywh_norm"] = box.tolist()
                mapped["bbox_valid"] = bool(valid)
        out.append(mapped)
    return out


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
    geometries: list[ResizeGeometry],
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
            if not boxes_mask[person_idx]:
                fallback_box, fallback_valid = _fallback_person_box(person, geometries[frame_idx])
                if fallback_valid:
                    boxes[person_idx] = torch.as_tensor(fallback_box, dtype=torch.float32).reshape(4).clamp(0.0, 1.0)
                    boxes_mask[person_idx] = True
            track_id, source = _resolve_track_id(box_person, person, person_idx, explicit_by_person_index, explicit_by_slot)
            person_ids[person_idx] = track_id
            person_id_mask[person_idx] = True
            track_source[person_idx] = source
            if require_boxes and not boxes_mask[person_idx]:
                raise ValueError(
                    "Valid SMPL person is missing a preprocessed bbox. "
                    f"frame_index={frame_idx} person_index={person_idx}. "
                    "Run scripts/preprocess/prepare_bedlam_boxes.py with usable bbox/j2d annotations, "
                    "or verify the BEDLAM SMPL pickle contains bbox/j2d fallback fields."
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


def _fallback_person_box(person: dict[str, Any], geometry: ResizeGeometry) -> tuple[np.ndarray, bool]:
    try:
        raw = extract_best_box(person, geometry.orig_hw)
    except (KeyError, TypeError, ValueError):
        return np.zeros(4, dtype=np.float32), False
    if not bool(raw.get("bbox_valid", False)):
        return np.zeros(4, dtype=np.float32), False
    xyxy = raw.get("bbox_xyxy_pixels")
    if xyxy is None:
        return np.zeros(4, dtype=np.float32), False
    return transform_xyxy_to_normalized_cxcywh(xyxy, geometry)


def _build_detection_query_targets(
    box_frames: list[dict[str, Any] | None],
    max_humans: int,
    image_size: int,
    geometries: list[ResizeGeometry],
    patch_size: int,
    mask_patch_threshold: float,
    min_mask_patches: int,
    sidecar_root: Path | None,
) -> dict[str, torch.Tensor]:
    num_frames = len(box_frames)
    del image_size
    image_hw = (
        max(int(geometry.input_hw[0]) for geometry in geometries),
        max(int(geometry.input_hw[1]) for geometry in geometries),
    )
    grid_h = int(image_hw[0]) // int(patch_size)
    grid_w = int(image_hw[1]) // int(patch_size)
    num_patches = grid_h * grid_w
    boxes = torch.zeros(num_frames, max_humans, 4, dtype=torch.float32)
    box_mask = torch.zeros(num_frames, max_humans, dtype=torch.bool)
    scores = torch.zeros(num_frames, max_humans, dtype=torch.float32)
    det_ids = torch.full((num_frames, max_humans), -1, dtype=torch.long)
    patch_masks = torch.zeros(num_frames, max_humans, num_patches, dtype=torch.bool)
    patch_masks_valid = torch.zeros(num_frames, max_humans, dtype=torch.bool)

    for frame_idx, frame in enumerate(box_frames):
        if frame is None:
            continue
        image_h, image_w = _frame_hw(frame)
        detections = sorted(
            _frame_detections(frame),
            key=lambda det: (float(det.get("det_score", det.get("score", 0.0))), _det_area(det)),
            reverse=True,
        )[:max_humans]
        geometry = geometries[frame_idx]
        for slot, det in enumerate(detections):
            model_box, model_box_valid = _det_model_cxcywh(det, image_w, image_h, geometry)
            if not model_box_valid:
                continue
            boxes[frame_idx, slot] = torch.as_tensor(model_box, dtype=torch.float32).clamp(0.0, 1.0)
            box_mask[frame_idx, slot] = True
            scores[frame_idx, slot] = float(det.get("det_score", det.get("score", 0.0)))
            det_ids[frame_idx, slot] = int(det.get("det_id", slot))
            patch_mask = _load_detection_patch_mask(
                det,
                sidecar_root=sidecar_root,
                geometry=geometry,
                image_hw=image_hw,
                patch_size=patch_size,
                threshold=mask_patch_threshold,
                min_mask_patches=min_mask_patches,
            )
            if patch_mask is not None:
                patch_masks[frame_idx, slot] = torch.as_tensor(patch_mask.reshape(-1), dtype=torch.bool)
                patch_masks_valid[frame_idx, slot] = True
    return {
        "smpl_query_boxes": boxes,
        "smpl_query_boxes_mask": box_mask,
        "smpl_query_scores": scores,
        "smpl_query_det_ids": det_ids,
        "smpl_query_patch_masks": patch_masks,
        "smpl_query_patch_masks_valid": patch_masks_valid,
    }


def _build_external_prior_targets(
    box_frames: list[dict[str, Any] | None],
    query_boxes: torch.Tensor,
    query_mask: torch.Tensor,
    geometries: list[ResizeGeometry] | None = None,
    iou_threshold: float = 0.50,
) -> dict[str, torch.Tensor]:
    num_frames, max_humans = query_boxes.shape[:2]
    ids = torch.full((num_frames, max_humans), -1, dtype=torch.long)
    mask = torch.zeros(num_frames, max_humans, dtype=torch.bool)
    conf = torch.zeros(num_frames, max_humans, dtype=torch.float32)
    for frame_idx, frame in enumerate(box_frames):
        if frame is None:
            continue
        image_h, image_w = _frame_hw(frame)
        geometry = geometries[frame_idx] if geometries is not None else None
        persons = [
            person
            for person in frame.get("persons", [])
            if person.get("valid", True)
            and person.get("person_id_valid", True)
            and int(person.get("person_id", -1)) >= 0
            and person.get("bbox_valid", True)
        ]
        used: set[int] = set()
        for slot in range(max_humans):
            if not bool(query_mask[frame_idx, slot]):
                continue
            if geometry is not None:
                q_xyxy = _cxcywh_norm_to_xyxy(query_boxes[frame_idx, slot].numpy(), geometry.input_hw[1], geometry.input_hw[0])
            else:
                q_xyxy = _cxcywh_norm_to_xyxy(query_boxes[frame_idx, slot].numpy(), image_w, image_h)
            best_idx = -1
            best_iou = 0.0
            for person_idx, person in enumerate(persons):
                if person_idx in used:
                    continue
                p_xyxy = np.asarray(person.get("bbox_xyxy_pixels", [0, 0, 0, 0]), dtype=np.float32)
                if geometry is not None:
                    p_box, p_valid = transform_xyxy_to_normalized_cxcywh(p_xyxy, geometry)
                    if not p_valid:
                        continue
                    p_xyxy = _cxcywh_norm_to_xyxy(p_box, geometry.input_hw[1], geometry.input_hw[0])
                iou = _box_iou(q_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = person_idx
            if best_idx < 0 or best_iou < float(iou_threshold):
                continue
            person = persons[best_idx]
            used.add(best_idx)
            ids[frame_idx, slot] = int(person["person_id"])
            mask[frame_idx, slot] = True
            conf[frame_idx, slot] = float(person.get("track_confidence", person.get("det_score", best_iou)))
    return {
        "external_track_ids": ids,
        "external_track_mask": mask,
        "external_track_confidence": conf,
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


def _frame_hw(frame: dict[str, Any]) -> tuple[int, int]:
    if "image_hw" in frame:
        h, w = frame["image_hw"]
        return int(h), int(w)
    persons = frame.get("persons", [])
    if persons:
        return int(persons[0].get("image_height", 0)), int(persons[0].get("image_width", 0))
    return 0, 0


def _frame_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
    detections = frame.get("detections", [])
    if isinstance(detections, list) and detections:
        return [dict(det, det_id=int(det.get("det_id", idx))) for idx, det in enumerate(detections)]
    out = []
    for idx, person in enumerate(frame.get("persons", [])):
        if not person.get("bbox_valid", person.get("valid", True)):
            continue
        out.append(
            {
                "det_id": int(person.get("det_id", idx)),
                "bbox_xyxy_pixels": person.get("bbox_xyxy_pixels"),
                "bbox_cxcywh_norm": person.get("bbox_cxcywh_norm"),
                "det_score": float(person.get("det_score", person.get("track_confidence", 0.0))),
                "mask": person.get("mask"),
            }
        )
    return out


def _det_area(det: dict[str, Any]) -> float:
    if "bbox_xyxy_pixels" in det:
        x1, y1, x2, y2 = np.asarray(det["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
        return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
    box = np.asarray(det.get("bbox_cxcywh_norm", [0, 0, 0, 0]), dtype=np.float32)
    return float(max(box[2], 0.0) * max(box[3], 0.0))


def _det_cxcywh(det: dict[str, Any], image_w: int, image_h: int) -> list[float]:
    if "bbox_cxcywh_norm" in det:
        return [float(v) for v in det["bbox_cxcywh_norm"]]
    if "bbox_xyxy_pixels" in det:
        x1, y1, x2, y2 = np.asarray(det["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
        width = max(float(image_w), 1.0)
        height = max(float(image_h), 1.0)
        bw = max(float(x2 - x1), 0.0)
        bh = max(float(y2 - y1), 0.0)
        return [float((x1 + 0.5 * bw) / width), float((y1 + 0.5 * bh) / height), float(bw / width), float(bh / height)]
    raise ValueError("Detection is missing bbox_cxcywh_norm and bbox_xyxy_pixels")


def _det_model_cxcywh(det: dict[str, Any], image_w: int, image_h: int, geometry: ResizeGeometry) -> tuple[np.ndarray, bool]:
    if "bbox_xyxy_pixels" in det:
        xyxy = np.asarray(det["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
    elif "bbox_cxcywh_norm" in det:
        xyxy = _cxcywh_norm_to_xyxy(np.asarray(det["bbox_cxcywh_norm"], dtype=np.float32), image_w, image_h)
    else:
        raise ValueError("Detection is missing bbox_cxcywh_norm and bbox_xyxy_pixels")
    return transform_xyxy_to_normalized_cxcywh(xyxy, geometry)


def _load_detection_patch_mask(
    det: dict[str, Any],
    sidecar_root: Path | None,
    geometry: ResizeGeometry,
    image_hw: tuple[int, int],
    patch_size: int,
    threshold: float,
    min_mask_patches: int,
) -> np.ndarray | None:
    meta = det.get("mask")
    if not isinstance(meta, dict):
        return None
    path = Path(str(meta.get("path", ""))).expanduser()
    if not path.is_absolute() and sidecar_root is not None:
        direct = (sidecar_root / path).resolve()
        path = direct if direct.is_file() else (sidecar_root.parent / path).resolve()
    if not path.is_file():
        return None
    key = str(meta.get("array_key", ""))
    if not key:
        return None
    with np.load(path) as data:
        if key not in data:
            return None
        pixel_mask = np.asarray(data[key]).astype(bool)
    resized_mask = resize_mask_with_geometry(pixel_mask, geometry)
    patch_mask = pixel_mask_to_patch_mask_hw(resized_mask, image_hw, patch_size, threshold)
    if int(patch_mask.sum()) < int(min_mask_patches):
        return None
    return patch_mask


def _cxcywh_norm_to_xyxy(box: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box.reshape(4)]
    bw = w * float(max(image_w, 1))
    bh = h * float(max(image_h, 1))
    x1 = (cx * float(max(image_w, 1))) - 0.5 * bw
    y1 = (cy * float(max(image_h, 1))) - 0.5 * bh
    return np.asarray([x1, y1, x1 + bw, y1 + bh], dtype=np.float32)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a.reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in b.reshape(4)]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return float(inter / max(area_a + area_b - inter, 1e-6))
