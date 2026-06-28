from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def clamp_box_xyxy(box: list[float] | tuple[float, ...] | np.ndarray, width: int, height: int) -> list[float]:
    arr = np.asarray(box, dtype=np.float32).reshape(4)
    arr[[0, 2]] = np.clip(arr[[0, 2]], 0.0, float(width))
    arr[[1, 3]] = np.clip(arr[[1, 3]], 0.0, float(height))
    if arr[2] < arr[0]:
        arr[0], arr[2] = arr[2], arr[0]
    if arr[3] < arr[1]:
        arr[1], arr[3] = arr[3], arr[1]
    return [float(v) for v in arr]


def xyxy_to_cxcywh_norm(box: list[float] | tuple[float, ...] | np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = clamp_box_xyxy(box, width, height)
    bw = max(x2 - x1, 0.0)
    bh = max(y2 - y1, 0.0)
    return [
        float((x1 + 0.5 * bw) / max(width, 1)),
        float((y1 + 0.5 * bh) / max(height, 1)),
        float(bw / max(width, 1)),
        float(bh / max(height, 1)),
    ]


def box_area_xyxy(box: list[float] | tuple[float, ...] | np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
    return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))


@dataclass(slots=True)
class Detection:
    bbox_xyxy: list[float]
    score: float
    class_id: int = 0
    class_name: str = "person"
    source: str = "detector"
    det_id: int = -1
    mask: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, image_width: int | None = None, image_height: int | None = None) -> dict[str, Any]:
        out = {
            "det_id": int(self.det_id),
            "bbox_xyxy_pixels": [float(v) for v in self.bbox_xyxy],
            "det_score": float(self.score),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "det_source": self.source,
        }
        if image_width is not None and image_height is not None:
            out["bbox_cxcywh_norm"] = xyxy_to_cxcywh_norm(self.bbox_xyxy, image_width, image_height)
        if self.mask is not None:
            out["mask"] = self.mask
        out.update(self.extra)
        return out


@dataclass(slots=True)
class TrackObservation:
    frame_id: str
    frame_index: int
    person_id: int
    bbox_xyxy: list[float]
    bbox_cxcywh_norm: list[float]
    det_score: float
    track_confidence: float
    image_width: int
    image_height: int
    valid: bool = True
    missing_count: int = 0
    bbox_source: str = "yolo_torchscript+boosttrack"
    mask: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "frame_id": self.frame_id,
            "frame_index": int(self.frame_index),
            "person_id": int(self.person_id),
            "person_id_valid": self.valid and self.person_id >= 0,
            "bbox_xyxy_pixels": [float(v) for v in self.bbox_xyxy],
            "bbox_cxcywh_norm": [float(v) for v in self.bbox_cxcywh_norm],
            "bbox_valid": bool(self.valid and box_area_xyxy(self.bbox_xyxy) > 0.0),
            "bbox_source": self.bbox_source,
            "det_score": float(self.det_score),
            "track_confidence": float(self.track_confidence),
            "missing_count": int(self.missing_count),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "valid": bool(self.valid),
        }
        if self.mask is not None:
            out["mask"] = self.mask
        out.update(self.extra)
        return out


@dataclass(slots=True)
class FrameObservations:
    frame_id: str
    frame_index: int
    image_path: str
    image_width: int
    image_height: int
    detections: list[Detection]
    persons: list[TrackObservation]

    def to_sidecar_frame(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": int(self.frame_index),
            "image_path": self.image_path,
            "image_hw": [int(self.image_height), int(self.image_width)],
            "detections": [det.to_dict(self.image_width, self.image_height) for det in self.detections],
            "persons": [obs.to_dict() for obs in self.persons],
        }
