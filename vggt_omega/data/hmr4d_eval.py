from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.tracking.query_builder import build_detection_query_tensors_from_sidecar, build_external_track_prior_from_sidecar


EMDB1_NAMES = [
    "P1_14_outdoor_climb",
    "P2_23_outdoor_hug_tree",
    "P3_31_outdoor_workout",
    "P3_32_outdoor_soccer_warmup_a",
    "P3_33_outdoor_soccer_warmup_b",
    "P5_42_indoor_dancing",
    "P5_44_indoor_rom",
    "P6_49_outdoor_big_stairs_down",
    "P6_50_outdoor_workout",
    "P6_51_outdoor_dancing",
    "P7_57_outdoor_rock_chair",
    "P7_59_outdoor_rom",
    "P7_60_outdoor_workout",
    "P8_64_outdoor_skateboard",
    "P8_68_outdoor_handstand",
    "P8_69_outdoor_cartwheel",
    "P9_76_outdoor_sitting",
]

EMDB2_NAMES = [
    "P0_09_outdoor_walk",
    "P2_19_indoor_walk_off_mvs",
    "P2_20_outdoor_walk",
    "P2_24_outdoor_long_walk",
    "P3_27_indoor_walk_off_mvs",
    "P3_28_outdoor_walk_lunges",
    "P3_29_outdoor_stairs_up",
    "P3_30_outdoor_stairs_down",
    "P4_35_indoor_walk",
    "P4_36_outdoor_long_walk",
    "P4_37_outdoor_run_circle",
    "P5_40_indoor_walk_big_circle",
    "P6_48_outdoor_walk_downhill",
    "P6_49_outdoor_big_stairs_down",
    "P7_55_outdoor_walk",
    "P7_56_outdoor_stairs_up_down",
    "P7_57_outdoor_rock_chair",
    "P7_58_outdoor_parcours",
    "P7_61_outdoor_sit_lie_walk",
    "P8_64_outdoor_skateboard",
    "P8_65_outdoor_walk_straight",
    "P9_77_outdoor_stairs_up",
    "P9_78_outdoor_stairs_up_down",
    "P9_79_outdoor_walk_rectangle",
    "P9_80_outdoor_walk_big_circle",
]


@dataclass(frozen=True, slots=True)
class HMR4DSequenceRecord:
    dataset_key: str
    dataset_id: str
    vid: str
    safe_vid: str
    length: int
    label: dict[str, Any]


class HMR4DSupportEvalDataset(Dataset):
    """VGGT-Omega RGB/query adapter for GVHMR-style hmr4d_support eval labels.

    This is an adapter rewrite of the GVHMR evaluation dataset idea.  It does
    not import GVHMR modules; it only consumes the support files that GVHMR
    documents for EMDB, RICH and 3DPW.
    """

    def __init__(
        self,
        dataset: str,
        support_root: str | Path,
        frames_root: str | Path,
        sidecar_root: str | Path | None = None,
        sequence_length: int = 32,
        stride: int = 1,
        image_size: int = 518,
        max_humans: int = 1,
        patch_size: int = 16,
        full_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = _canonical_dataset_key(dataset)
        self.support_root = Path(support_root).expanduser()
        self.frames_root = Path(frames_root).expanduser()
        self.sidecar_root = Path(sidecar_root).expanduser() if sidecar_root else None
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.max_humans = int(max_humans)
        self.patch_size = int(patch_size)
        self.full_sequence = bool(full_sequence)
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if self.max_humans <= 0:
            raise ValueError(f"max_humans must be positive, got {max_humans}")

        self.records = self._load_records()
        self._index: list[tuple[int, int, int]] = []
        for record_idx, record in enumerate(self.records):
            window = record.length if self.full_sequence else min(self.sequence_length, record.length)
            max_start = record.length - (window - 1) * self.stride
            for start in range(max(max_start, 0)):
                self._index.append((record_idx, start, window))
        if not self._index:
            raise RuntimeError(f"No eval windows for dataset={self.dataset!r} under {self.support_root}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, start, window = self._index[idx]
        record = self.records[record_idx]
        frame_indices = [start + step * self.stride for step in range(window)]
        images = []
        intrinsics = []
        orig_hw = None
        for frame_idx in frame_indices:
            image, hw = _load_rgb_tensor(self._frame_path(record, frame_idx), self.image_size)
            images.append(image)
            orig_hw = hw
            intrinsics.append(_scale_intrinsics(self._full_intrinsics(record), hw, self.image_size))
        if orig_hw is None:
            raise RuntimeError("Empty HMR4D eval window")

        gt_boxes, boxes_mask = self._fallback_query_boxes(record, frame_indices, orig_hw)
        sample: dict[str, Any] = {
            "images": torch.stack(images, dim=0),
            "K_scal3r": torch.stack(intrinsics, dim=0),
            "gt_depth": torch.zeros(window, 1, self.image_size, self.image_size, dtype=torch.float32),
            "gt_boxes": gt_boxes,
            "boxes_mask": boxes_mask,
            "smpl_mask": boxes_mask.clone(),
            "person_ids": torch.zeros(window, self.max_humans, dtype=torch.long),
            "person_id_mask": boxes_mask.clone(),
            "gt_track_ids": torch.zeros(window, self.max_humans, dtype=torch.long),
            "gt_track_mask": boxes_mask.clone(),
            "meta": {
                "dataset_key": record.dataset_key,
                "dataset_id": record.dataset_id,
                "vid": record.vid,
                "safe_vid": record.safe_vid,
                "start": int(start),
                "frame_indices": [int(v) for v in frame_indices],
            },
            "eval_mask": self._select_eval_mask(record, frame_indices),
            "eval_label": self._select_label(record, frame_indices),
        }
        sample.update(self._query_from_sidecar_or_fallback(record, frame_indices, gt_boxes, boxes_mask))
        return sample

    def _load_records(self) -> list[HMR4DSequenceRecord]:
        if self.dataset in {"emdb1", "emdb2"}:
            labels = _torch_load(self.support_root / "emdb_vit_v4.pt")
            names = EMDB1_NAMES if self.dataset == "emdb1" else EMDB2_NAMES
            return [
                HMR4DSequenceRecord(self.dataset, f"EMDB_{1 if self.dataset == 'emdb1' else 2}", vid, _safe_vid(vid), int(len(labels[vid]["mask"])), labels[vid])
                for vid in names
                if vid in labels
            ]
        if self.dataset == "rich":
            labels = _torch_load(self.support_root / "rich_test_labels.pt")
            preproc = _torch_load(self.support_root / "rich_test_preproc.pt")
            merged = []
            for vid, label in sorted(labels.items()):
                item = dict(label)
                if vid in preproc:
                    item.update(
                        {
                            "bbx_xys": preproc[vid].get("bbx_xys"),
                            "kp2d": preproc[vid].get("kp2d"),
                            "img_wh": preproc[vid].get("img_wh"),
                        }
                    )
                merged.append(HMR4DSequenceRecord("rich", "RICH", vid, _safe_vid(vid), int(len(label["frame_id"])), item))
            return merged
        if self.dataset == "3dpw":
            labels = _torch_load(self.support_root / "test_3dpw_gt_labels.pt")
            vid2bbx = _torch_load(self.support_root / "preproc_test_bbx.pt")
            vid2kp2d = _torch_load(self.support_root / "preproc_test_kp2d_v0.pt")
            merged = []
            for vid, label in sorted(labels.items()):
                item = dict(label)
                if vid in vid2bbx:
                    item["bbx_xys"] = vid2bbx[vid].get("bbx_xys")
                if vid in vid2kp2d:
                    item["kp2d"] = vid2kp2d[vid]
                merged.append(HMR4DSequenceRecord("3dpw", "3DPW", vid, _safe_vid(vid), int(len(label["mask_wham"])), item))
            return merged
        raise ValueError(f"Unsupported HMR4D eval dataset: {self.dataset}")

    def _frame_path(self, record: HMR4DSequenceRecord, frame_idx: int) -> Path:
        candidates = [
            self.frames_root / record.dataset_key / record.safe_vid / "rgb" / f"{frame_idx:06d}.png",
            self.frames_root / record.dataset_id / record.safe_vid / "rgb" / f"{frame_idx:06d}.png",
            self.frames_root / record.safe_vid / "rgb" / f"{frame_idx:06d}.png",
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            "HMR4D eval RGB frame not found. Expected one of: "
            + ", ".join(str(path) for path in candidates)
            + ". Run scripts/preprocess/extract_hmr4d_eval_frames.py first."
        )

    def _full_intrinsics(self, record: HMR4DSequenceRecord) -> torch.Tensor:
        label = record.label
        if "K_fullimg" in label:
            return torch.as_tensor(label["K_fullimg"], dtype=torch.float32).reshape(3, 3)
        if "K" in label:
            return torch.as_tensor(label["K"], dtype=torch.float32).reshape(3, 3)
        return _default_intrinsics(self.image_size)

    def _fallback_query_boxes(self, record: HMR4DSequenceRecord, frame_indices: list[int], orig_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = torch.zeros(len(frame_indices), self.max_humans, 4, dtype=torch.float32)
        mask = torch.zeros(len(frame_indices), self.max_humans, dtype=torch.bool)
        bbx = record.label.get("bbx_xys")
        if bbx is None and self.dataset == "3dpw":
            return boxes, mask
        if bbx is None:
            return boxes, mask
        bbx_t = torch.as_tensor(bbx, dtype=torch.float32)
        src_h, src_w = orig_hw
        for out_idx, frame_idx in enumerate(frame_indices):
            if frame_idx >= bbx_t.shape[0]:
                continue
            x, y, size = [float(v) for v in bbx_t[frame_idx].reshape(-1)[:3]]
            boxes[out_idx, 0] = torch.tensor([x / max(src_w, 1), y / max(src_h, 1), size / max(src_w, 1), size / max(src_h, 1)], dtype=torch.float32).clamp(0.0, 1.0)
            mask[out_idx, 0] = True
        return boxes, mask

    def _select_eval_mask(self, record: HMR4DSequenceRecord, frame_indices: list[int]) -> torch.Tensor:
        label = record.label
        raw = label.get("mask", label.get("mask_wham"))
        if raw is None:
            return torch.ones(len(frame_indices), dtype=torch.bool)
        mask = torch.as_tensor(raw).bool()
        return torch.stack([mask[i] if i < mask.numel() else torch.tensor(False) for i in frame_indices])

    def _select_label(self, record: HMR4DSequenceRecord, frame_indices: list[int]) -> dict[str, Any]:
        label = record.label
        keys = ("smpl_params", "gt_smplx_params", "T_w2c", "K_fullimg", "K", "gender", "frame_id", "bbx_xys", "kp2d", "img_wh")
        out: dict[str, Any] = {}
        for key in keys:
            if key not in label:
                continue
            out[key] = _select_value(label[key], frame_indices)
        return out

    def _query_from_sidecar_or_fallback(
        self,
        record: HMR4DSequenceRecord,
        frame_indices: list[int],
        gt_boxes: torch.Tensor,
        boxes_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sidecar = self._sidecar_dir(record)
        if sidecar is not None:
            frame_ids = [f"{idx:06d}" for idx in frame_indices]
            query = build_detection_query_tensors_from_sidecar(
                sidecar,
                frame_ids=frame_ids,
                max_humans=self.max_humans,
                image_size=self.image_size,
                patch_size=self.patch_size,
            )
            prior = build_external_track_prior_from_sidecar(sidecar, query)
            return {
                key: value.squeeze(0) if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1 else value
                for payload in (query, prior)
                for key, value in payload.items()
                if isinstance(value, torch.Tensor)
            }
        num_patches = (self.image_size // self.patch_size) ** 2
        return {
            "smpl_query_boxes": gt_boxes,
            "smpl_query_boxes_mask": boxes_mask,
            "smpl_query_scores": boxes_mask.to(dtype=torch.float32),
            "smpl_query_det_ids": torch.arange(self.max_humans, dtype=torch.long).reshape(1, -1).expand(gt_boxes.shape[0], -1),
            "smpl_query_patch_masks": torch.zeros(gt_boxes.shape[0], self.max_humans, num_patches, dtype=torch.bool),
            "smpl_query_patch_masks_valid": torch.zeros(gt_boxes.shape[0], self.max_humans, dtype=torch.bool),
            "external_track_ids": torch.full((gt_boxes.shape[0], self.max_humans), -1, dtype=torch.long),
            "external_track_mask": torch.zeros(gt_boxes.shape[0], self.max_humans, dtype=torch.bool),
            "external_track_confidence": torch.zeros(gt_boxes.shape[0], self.max_humans, dtype=torch.float32),
        }

    def _sidecar_dir(self, record: HMR4DSequenceRecord) -> Path | None:
        if self.sidecar_root is None:
            return None
        candidates = [
            self.sidecar_root / record.dataset_key / record.safe_vid,
            self.sidecar_root / record.dataset_id / record.safe_vid,
            self.sidecar_root / record.safe_vid,
        ]
        for path in candidates:
            if (path / "smpl_boxes").is_dir():
                return path
        return None


def hmr4d_eval_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty HMR4D eval batch")
    return {key: _collate_values([item[key] for item in batch]) for key in batch[0].keys()}


def _collate_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, dict):
        return {key: _collate_values([value[key] for value in values]) for key in first.keys()}
    return values


def _load_rgb_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
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


def _default_intrinsics(size: int) -> torch.Tensor:
    focal = float(size)
    center = (float(size) - 1.0) * 0.5
    return torch.tensor([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _select_value(value: Any, frame_indices: list[int]) -> Any:
    if isinstance(value, dict):
        return {key: _select_value(item, frame_indices) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value[frame_indices] if value.ndim > 0 and value.shape[0] >= max(frame_indices, default=-1) + 1 else value
    if isinstance(value, np.ndarray):
        selected = value[frame_indices] if value.ndim > 0 and value.shape[0] >= max(frame_indices, default=-1) + 1 else value
        return torch.as_tensor(selected)
    return value


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"HMR4D support file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _canonical_dataset_key(dataset: str) -> str:
    key = str(dataset).lower().replace("_", "").replace("-", "")
    if key in {"emdb1", "emdbsplit1"}:
        return "emdb1"
    if key in {"emdb2", "emdbsplit2"}:
        return "emdb2"
    if key == "rich":
        return "rich"
    if key in {"3dpw", "threedpw"}:
        return "3dpw"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _safe_vid(vid: str) -> str:
    return str(vid).replace("/", "__").replace("\\", "__").replace(" ", "_")
