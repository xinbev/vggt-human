from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


BBOX_KEYS = ("bbox_xyxy", "bbox", "box", "person_bbox", "bbox_xywh")
J2D_KEYS = ("j2ds", "joints2d", "joints_2d", "keypoints2d", "keypoints_2d")
PERSON_ID_KEYS = ("person_id", "id", "track_id", "trackid", "pid")


def extract_person_id(person: dict[str, Any]) -> tuple[int, bool]:
    for key in PERSON_ID_KEYS:
        if key not in person:
            continue
        value = person[key]
        if value is None:
            continue
        try:
            return int(value), True
        except (TypeError, ValueError):
            return stable_string_id(str(value)), True
    return -1, False


def stable_string_id(value: str) -> int:
    h = 2166136261
    for char in value:
        h ^= ord(char)
        h = (h * 16777619) & 0x7FFFFFFF
    return int(h)


def extract_person_box(person: dict[str, Any], image_hw: tuple[int, int], bbox_format: str = "auto") -> tuple[torch.Tensor | None, str]:
    for key in BBOX_KEYS:
        if key not in person:
            continue
        box = _as_box_tensor(person[key])
        if box is None:
            continue
        fmt = bbox_format
        if fmt == "auto":
            fmt = "xywh" if key.endswith("xywh") else _infer_box_format(box, image_hw)
        xyxy = _box_to_xyxy_pixels(box, image_hw, fmt)
        repaired = repair_xyxy_box(xyxy, image_hw)
        if repaired is not None:
            return xyxy_to_cxcywh_norm(repaired, image_hw), key
    return None, "missing"


def extract_j2d_box(person: dict[str, Any], image_hw: tuple[int, int]) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    for key in J2D_KEYS:
        if key not in person:
            continue
        j2ds = _as_j2d_tensor(person[key])
        if j2ds is None or j2ds.numel() == 0:
            continue
        coords = j2ds[..., :2].float()
        visibility = _j2d_visibility(j2ds, image_hw)
        if not visibility.any():
            continue
        visible_coords = coords[visibility]
        x1y1 = visible_coords.amin(dim=0)
        x2y2 = visible_coords.amax(dim=0)
        xyxy = torch.cat([x1y1, x2y2], dim=0)
        repaired = repair_xyxy_box(xyxy, image_hw)
        if repaired is None:
            continue
        mask = visibility[:, None].expand(-1, 2).contiguous()
        return xyxy_to_cxcywh_norm(repaired, image_hw), mask, key
    return None, None, "missing"


def repair_xyxy_box(box: torch.Tensor, image_hw: tuple[int, int], min_size: float = 1.0) -> torch.Tensor | None:
    h, w = image_hw
    box = box.float().reshape(4).clone()
    if not torch.isfinite(box).all():
        return None
    x1 = torch.minimum(box[0], box[2]).clamp(0.0, float(w - 1))
    y1 = torch.minimum(box[1], box[3]).clamp(0.0, float(h - 1))
    x2 = torch.maximum(box[0], box[2]).clamp(0.0, float(w - 1))
    y2 = torch.maximum(box[1], box[3]).clamp(0.0, float(h - 1))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return torch.stack([x1, y1, x2, y2])


def xyxy_to_cxcywh_norm(box: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    x1, y1, x2, y2 = box.float().unbind(-1)
    cx = (x1 + x2) * 0.5 / float(w)
    cy = (y1 + y2) * 0.5 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return torch.stack([cx, cy, bw, bh], dim=-1).clamp(0.0, 1.0)


def cxcywh_norm_to_xyxy(box: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    cx, cy, bw, bh = box.float().unbind(-1)
    half_w = bw * float(w) * 0.5
    half_h = bh * float(h) * 0.5
    x = cx * float(w)
    y = cy * float(h)
    return torch.stack([x - half_w, y - half_h, x + half_w, y + half_h], dim=-1)


def extract_best_box(person: dict[str, Any], image_hw: tuple[int, int]) -> dict[str, Any]:
    bbox, source = extract_person_box(person, image_hw)
    j2ds = None
    j2ds_mask = None
    if bbox is None:
        bbox, j2ds_mask, source = extract_j2d_box(person, image_hw)
        j2ds = _first_j2d(person)
    person_id, person_id_valid = extract_person_id(person)
    xyxy = cxcywh_norm_to_xyxy(bbox, image_hw) if bbox is not None else torch.zeros(4)
    out: dict[str, Any] = {
        "person_id": person_id,
        "person_id_valid": person_id_valid,
        "bbox_cxcywh_norm": bbox.tolist() if bbox is not None else [0.0, 0.0, 0.0, 0.0],
        "bbox_xyxy_pixels": xyxy.tolist(),
        "bbox_valid": bbox is not None,
        "bbox_source": source,
    }
    if j2ds is not None:
        out["j2ds"] = j2ds[..., :2].float().tolist()
    if j2ds_mask is not None:
        out["j2ds_mask"] = j2ds_mask.bool().tolist()
    return out


def optional_smpl_projection_box(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        import smplx  # noqa: F401
    except ImportError as exc:
        raise ImportError("--use-smpl-projection requires smplx and valid SMPL/SMPL-X assets") from exc
    raise NotImplementedError(
        "SMPL projection bbox generation needs dataset-specific camera/body-model wiring. "
        "Use existing bbox or 2D keypoint annotations, or implement this path for your BEDLAM schema."
    )


def relative_sequence_name(seq_dir: Path, split_dir: Path) -> str:
    return seq_dir.relative_to(split_dir).as_posix()


def _as_box_tensor(value: Any) -> torch.Tensor | None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if tensor.numel() < 4:
        return None
    return tensor[:4]


def _as_j2d_tensor(value: Any) -> torch.Tensor | None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.ndim == 1:
        if tensor.numel() % 3 == 0:
            tensor = tensor.reshape(-1, 3)
        elif tensor.numel() % 2 == 0:
            tensor = tensor.reshape(-1, 2)
    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        return None
    return tensor


def _first_j2d(person: dict[str, Any]) -> torch.Tensor | None:
    for key in J2D_KEYS:
        if key in person:
            return _as_j2d_tensor(person[key])
    return None


def _j2d_visibility(j2ds: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    coords = j2ds[..., :2]
    valid = torch.isfinite(coords).all(dim=-1)
    valid = valid & (coords[..., 0] >= 0) & (coords[..., 0] < float(w)) & (coords[..., 1] >= 0) & (coords[..., 1] < float(h))
    if j2ds.shape[-1] >= 3:
        valid = valid & (j2ds[..., 2] > 0)
    return valid


def _infer_box_format(box: torch.Tensor, image_hw: tuple[int, int]) -> str:
    h, w = image_hw
    if float(box.max()) <= 2.0:
        return "xyxy_norm" if box[2] > box[0] and box[3] > box[1] else "xywh_norm"
    if box[2] > box[0] and box[3] > box[1] and box[2] <= w * 1.2 and box[3] <= h * 1.2:
        return "xyxy"
    return "xywh"


def _box_to_xyxy_pixels(box: torch.Tensor, image_hw: tuple[int, int], fmt: str) -> torch.Tensor:
    h, w = image_hw
    box = box.float()
    if fmt in {"xyxy_norm", "norm_xyxy"}:
        scale = box.new_tensor([w, h, w, h], dtype=box.dtype)
        return box * scale
    if fmt in {"xywh_norm", "norm_xywh"}:
        scale = box.new_tensor([w, h, w, h], dtype=box.dtype)
        box = box * scale
        fmt = "xywh"
    if fmt == "xywh":
        x, y, bw, bh = box.unbind(0)
        return torch.stack([x, y, x + bw, y + bh])
    if fmt == "cxcywh_norm":
        return cxcywh_norm_to_xyxy(box, image_hw)
    if fmt == "cxcywh":
        cx, cy, bw, bh = box.unbind(0)
        return torch.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5])
    if fmt != "xyxy":
        raise ValueError(f"Unsupported bbox format: {fmt}")
    return box
