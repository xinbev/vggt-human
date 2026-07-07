from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .clip_builder import load_sidecar_frames
from .schema import box_area_xyxy, xyxy_to_cxcywh_norm


def build_detection_query_tensors_from_sidecar(
    sidecar_root: str | Path,
    frame_ids: list[str] | None = None,
    max_humans: int | None = None,
    image_size: int = 518,
    image_hw: tuple[int, int] | None = None,
    patch_size: int = 16,
    mask_patch_threshold: float = 0.10,
    min_mask_patches: int = 4,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Build per-frame SMPL query tensors from detection sidecar data.

    Query slots are independent per frame. They are ordered by detection score
    and area, not by cross-frame identity.
    """
    root = Path(sidecar_root).expanduser()
    frames = _select_frames(load_sidecar_frames(root), frame_ids)
    if not frames:
        raise ValueError(f"No sidecar frames found under {root}")

    q = int(max_humans or _max_detections(frames) or 1)
    s = len(frames)
    target_hw = _coerce_image_hw(image_hw, image_size)
    grid_h = int(target_hw[0]) // int(patch_size)
    grid_w = int(target_hw[1]) // int(patch_size)
    num_patches = grid_h * grid_w

    boxes = torch.zeros((1, s, q, 4), dtype=torch.float32, device=device)
    box_mask = torch.zeros((1, s, q), dtype=torch.bool, device=device)
    scores = torch.zeros((1, s, q), dtype=torch.float32, device=device)
    det_ids = torch.full((1, s, q), -1, dtype=torch.long, device=device)
    patch_masks = torch.zeros((1, s, q, num_patches), dtype=torch.bool, device=device)
    patch_mask_valid = torch.zeros((1, s, q), dtype=torch.bool, device=device)

    for frame_slot, frame in enumerate(frames):
        detections = _frame_detections(frame)
        detections = sorted(
            detections,
            key=lambda det: (float(det.get("det_score", det.get("score", 0.0))), _det_area(det)),
            reverse=True,
        )[:q]
        image_h, image_w = _frame_hw(frame)
        for slot, det in enumerate(detections):
            box = _det_cxcywh(det, image_w, image_h)
            boxes[0, frame_slot, slot] = torch.as_tensor(box, dtype=torch.float32, device=device).clamp(0.0, 1.0)
            box_mask[0, frame_slot, slot] = True
            scores[0, frame_slot, slot] = float(det.get("det_score", det.get("score", 0.0)))
            det_ids[0, frame_slot, slot] = int(det.get("det_id", slot))
            mask = _load_detection_mask(root, det, frame, target_hw, patch_size, mask_patch_threshold, min_mask_patches)
            if mask is not None:
                patch_masks[0, frame_slot, slot] = torch.as_tensor(mask.reshape(-1), dtype=torch.bool, device=device)
                patch_mask_valid[0, frame_slot, slot] = True

    return {
        "smpl_query_boxes": boxes,
        "smpl_query_boxes_mask": box_mask,
        "smpl_query_scores": scores,
        "smpl_query_det_ids": det_ids,
        "smpl_query_patch_masks": patch_masks,
        "smpl_query_patch_masks_valid": patch_mask_valid,
        "frame_ids": [str(frame["frame_id"]) for frame in frames],
        "frame_indices": [int(frame.get("frame_index", idx)) for idx, frame in enumerate(frames)],
    }


def build_external_track_prior_from_sidecar(
    sidecar_root: str | Path,
    query_tensors: dict[str, torch.Tensor | list[str] | list[int]],
    iou_threshold: float = 0.50,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Match per-frame query boxes to external BoostTrack observations."""
    root = Path(sidecar_root).expanduser()
    frame_ids = [str(item) for item in query_tensors.get("frame_ids", [])]
    frames = _select_frames(load_sidecar_frames(root), frame_ids or None)
    if not frames:
        raise ValueError(f"No sidecar frames found under {root}")
    query_boxes = _require_tensor(query_tensors["smpl_query_boxes"], "smpl_query_boxes").detach().cpu()
    query_mask = _require_tensor(query_tensors["smpl_query_boxes_mask"], "smpl_query_boxes_mask").detach().cpu().bool()
    _, s, q, _ = query_boxes.shape
    ids = torch.full((1, s, q), -1, dtype=torch.long, device=device)
    mask = torch.zeros((1, s, q), dtype=torch.bool, device=device)
    conf = torch.zeros((1, s, q), dtype=torch.float32, device=device)

    for frame_slot, frame in enumerate(frames[:s]):
        image_h, image_w = _frame_hw(frame)
        persons = [
            person
            for person in frame.get("persons", [])
            if person.get("valid", True)
            and person.get("person_id_valid", True)
            and int(person.get("person_id", -1)) >= 0
            and person.get("bbox_valid", True)
        ]
        used: set[int] = set()
        for slot in range(q):
            if not bool(query_mask[0, frame_slot, slot]):
                continue
            q_xyxy = _cxcywh_norm_to_xyxy(query_boxes[0, frame_slot, slot].numpy(), image_w, image_h)
            best_idx = -1
            best_iou = 0.0
            for person_idx, person in enumerate(persons):
                if person_idx in used:
                    continue
                p_xyxy = np.asarray(person.get("bbox_xyxy_pixels", [0, 0, 0, 0]), dtype=np.float32)
                iou = _box_iou(q_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = person_idx
            if best_idx < 0 or best_iou < float(iou_threshold):
                continue
            person = persons[best_idx]
            used.add(best_idx)
            ids[0, frame_slot, slot] = int(person["person_id"])
            mask[0, frame_slot, slot] = True
            conf[0, frame_slot, slot] = float(person.get("track_confidence", person.get("det_score", best_iou)))

    return {
        "external_track_ids": ids,
        "external_track_mask": mask,
        "external_track_confidence": conf,
    }


def _select_frames(frames: list[dict[str, Any]], frame_ids: list[str] | None) -> list[dict[str, Any]]:
    if frame_ids is not None:
        requested = set(str(item) for item in frame_ids)
        frames = [frame for frame in frames if str(frame["frame_id"]) in requested]
    frames.sort(key=lambda item: int(item.get("frame_index", 0)))
    return frames


def _max_detections(frames: list[dict[str, Any]]) -> int:
    return max((len(_frame_detections(frame)) for frame in frames), default=0)


def _frame_detections(frame: dict[str, Any]) -> list[dict[str, Any]]:
    detections = frame.get("detections", [])
    if detections:
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


def _frame_hw(frame: dict[str, Any]) -> tuple[int, int]:
    if "image_hw" in frame:
        h, w = frame["image_hw"]
        return int(h), int(w)
    persons = frame.get("persons", [])
    if persons:
        return int(persons[0].get("image_height", 0)), int(persons[0].get("image_width", 0))
    return 0, 0


def _det_area(det: dict[str, Any]) -> float:
    if "bbox_xyxy_pixels" in det:
        return box_area_xyxy(det["bbox_xyxy_pixels"])
    box = np.asarray(det.get("bbox_cxcywh_norm", [0, 0, 0, 0]), dtype=np.float32)
    return float(max(box[2], 0.0) * max(box[3], 0.0))


def _det_cxcywh(det: dict[str, Any], image_w: int, image_h: int) -> list[float]:
    if "bbox_cxcywh_norm" in det:
        return [float(v) for v in det["bbox_cxcywh_norm"]]
    if "bbox_xyxy_pixels" in det:
        return xyxy_to_cxcywh_norm(det["bbox_xyxy_pixels"], image_w, image_h)
    raise ValueError("Detection is missing bbox_cxcywh_norm and bbox_xyxy_pixels")


def _load_detection_mask(
    root: Path,
    det: dict[str, Any],
    frame: dict[str, Any],
    image_hw: tuple[int, int],
    patch_size: int,
    threshold: float,
    min_mask_patches: int,
) -> np.ndarray | None:
    meta = det.get("mask")
    if not isinstance(meta, dict):
        return None
    path = Path(str(meta.get("path", ""))).expanduser()
    if not path.is_absolute():
        direct = (root / path).resolve()
        output_parent = (root.parent / path).resolve()
        path = direct if direct.is_file() else output_parent
    if not path.is_file():
        return None
    key = str(meta.get("array_key", ""))
    if not key:
        return None
    with np.load(path) as data:
        if key not in data:
            return None
        pixel_mask = np.asarray(data[key]).astype(bool)
    patch_mask = pixel_mask_to_patch_mask(pixel_mask, image_hw=image_hw, patch_size=patch_size, threshold=threshold)
    if int(patch_mask.sum()) < int(min_mask_patches):
        return None
    return patch_mask


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    image_size: int | None = None,
    patch_size: int = 16,
    threshold: float = 0.10,
    image_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    if pixel_mask.ndim != 2:
        raise ValueError(f"Expected 2D pixel mask, got {pixel_mask.shape}")
    import cv2

    target_h, target_w = _coerce_image_hw(image_hw, int(image_size or pixel_mask.shape[0]))
    resized = cv2.resize(pixel_mask.astype(np.float32), (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    grid_h = int(target_h) // int(patch_size)
    grid_w = int(target_w) // int(patch_size)
    usable_h = grid_h * int(patch_size)
    usable_w = grid_w * int(patch_size)
    resized = resized[:usable_h, :usable_w]
    patch = resized.reshape(grid_h, int(patch_size), grid_w, int(patch_size)).mean(axis=(1, 3))
    return patch >= float(threshold)


def _coerce_image_hw(image_hw: tuple[int, int] | None, image_size: int) -> tuple[int, int]:
    if image_hw is None:
        return int(image_size), int(image_size)
    return int(image_hw[0]), int(image_hw[1])


def _cxcywh_norm_to_xyxy(box: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box.reshape(4)]
    bw = w * float(max(image_w, 1))
    bh = h * float(max(image_h, 1))
    x1 = (cx * float(max(image_w, 1))) - 0.5 * bw
    y1 = (cy * float(max(image_h, 1))) - 0.5 * bh
    x2 = x1 + bw
    y2 = y1 + bh
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


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


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    return value
