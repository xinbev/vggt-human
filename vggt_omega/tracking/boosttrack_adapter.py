from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from .schema import Detection, TrackObservation, xyxy_to_cxcywh_norm


BOOSTTRACK_WEIGHT_NAMES = (
    "mot17_sbs_S50.pth",
    "mot20_sbs_S50.pth",
    "dance_sbs_S50.pth",
    "osnet_ain_ms_d_c.pth.tar",
)


@contextlib.contextmanager
def temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BoostTrackPersonTracker:
    """Thin adapter around third_party/BoostTrack without importing its detector."""

    def __init__(
        self,
        boosttrack_root: str | Path,
        weights_root: str | Path | None = None,
        dataset: str = "mot17",
        test_dataset: bool = False,
        use_reid: bool = True,
        use_ecc: bool = False,
        min_box_area: float = 10.0,
        aspect_ratio_thresh: float = 1.6,
        det_thresh: float | None = None,
        iou_threshold: float | None = None,
        max_age: int | None = None,
        min_hits: int | None = None,
        auto_link_weights: bool = True,
    ) -> None:
        self.boosttrack_root = Path(boosttrack_root).expanduser().resolve()
        if not self.boosttrack_root.is_dir():
            raise FileNotFoundError(f"BoostTrack root not found: {self.boosttrack_root}")
        self.weights_root = Path(weights_root).expanduser().resolve() if weights_root else None
        if self.weights_root is not None and not self.weights_root.is_dir():
            raise FileNotFoundError(f"BoostTrack weights root not found: {self.weights_root}")
        if auto_link_weights and self.weights_root is not None:
            self._ensure_weight_links()

        self._extend_boosttrack_python_path()
        with temporary_cwd(self.boosttrack_root):
            from default_settings import GeneralSettings
            from tracker.boost_track import BoostTrack

            GeneralSettings.values["dataset"] = dataset
            GeneralSettings.values["test_dataset"] = bool(test_dataset)
            GeneralSettings.values["use_embedding"] = bool(use_reid)
            GeneralSettings.values["use_ecc"] = bool(use_ecc)
            if det_thresh is not None:
                GeneralSettings.values["det_thresh"] = float(det_thresh)
                GeneralSettings.dataset_specific_settings.setdefault(dataset, {})["det_thresh"] = float(det_thresh)
            if iou_threshold is not None:
                GeneralSettings.values["iou_threshold"] = float(iou_threshold)
            if max_age is not None:
                GeneralSettings.values["max_age"] = int(max_age)
            if min_hits is not None:
                GeneralSettings.values["min_hits"] = int(min_hits)
            GeneralSettings.values["min_box_area"] = float(min_box_area)
            GeneralSettings.values["aspect_ratio_thresh"] = float(aspect_ratio_thresh)
            self._general_settings = GeneralSettings
            self._tracker_cls = BoostTrack
            self._tracker = BoostTrack(video_name=None)

        self.min_box_area = float(min_box_area)
        self.aspect_ratio_thresh = float(aspect_ratio_thresh)

    def reset(self, video_name: str | None = None) -> None:
        with temporary_cwd(self.boosttrack_root):
            self._tracker = self._tracker_cls(video_name=video_name)

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: list[Detection],
        frame_id: str,
        frame_index: int,
        video_name: str,
    ) -> list[TrackObservation]:
        height, width = frame_bgr.shape[:2]
        dets = self._detections_to_array(detections)
        img_tensor = torch.from_numpy(frame_bgr).permute(2, 0, 1).contiguous().float().unsqueeze(0)
        tag = f"{video_name}:{frame_index + 1}"
        with temporary_cwd(self.boosttrack_root):
            targets = self._tracker.update(dets, img_tensor, frame_bgr, tag)
        observations: list[TrackObservation] = []
        if targets is None or len(targets) == 0:
            return observations
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)

        for target in targets:
            if target.shape[0] < 5:
                continue
            x1, y1, x2, y2 = [float(v) for v in target[:4]]
            person_id = int(target[4])
            track_conf = float(target[5]) if target.shape[0] > 5 else 1.0
            if not self._passes_geometry_filter(x1, y1, x2, y2):
                continue
            det_score = self._nearest_detection_score([x1, y1, x2, y2], detections)
            box = [x1, y1, x2, y2]
            observations.append(
                TrackObservation(
                    frame_id=frame_id,
                    frame_index=frame_index,
                    person_id=person_id,
                    bbox_xyxy=box,
                    bbox_cxcywh_norm=xyxy_to_cxcywh_norm(box, width=width, height=height),
                    det_score=det_score,
                    track_confidence=track_conf,
                    image_width=width,
                    image_height=height,
                    valid=True,
                    missing_count=0,
                    bbox_source="yolo_torchscript+boosttrack",
                )
            )
        observations.sort(key=lambda item: item.person_id)
        return observations

    def dump_cache(self) -> None:
        with temporary_cwd(self.boosttrack_root):
            self._tracker.dump_cache()
            embedder = getattr(self._tracker, "embedder", None)
            if embedder is not None:
                embedder.dump_cache()

    def _detections_to_array(self, detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray([[*det.bbox_xyxy, det.score] for det in detections], dtype=np.float32)

    def _passes_geometry_filter(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width * height < self.min_box_area:
            return False
        if height <= 0.0:
            return False
        return width / height <= self.aspect_ratio_thresh

    def _nearest_detection_score(self, box: list[float], detections: list[Detection]) -> float:
        if not detections:
            return 0.0
        target = np.asarray(box, dtype=np.float32)
        best_iou = -1.0
        best_score = 0.0
        for det in detections:
            iou = _iou_xyxy(target, np.asarray(det.bbox_xyxy, dtype=np.float32))
            if iou > best_iou:
                best_iou = iou
                best_score = float(det.score)
        return best_score

    def _extend_boosttrack_python_path(self) -> None:
        paths = [
            self.boosttrack_root,
            self.boosttrack_root / "external" / "YOLOX",
            self.boosttrack_root / "external" / "deep-person-reid",
        ]
        for path in paths:
            path_str = str(path)
            if path.is_dir() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _ensure_weight_links(self) -> None:
        assert self.weights_root is not None
        target_dir = self.boosttrack_root / "external" / "weights"
        target_dir.mkdir(parents=True, exist_ok=True)
        missing: list[str] = []
        for name in BOOSTTRACK_WEIGHT_NAMES:
            source = self.weights_root / name
            target = target_dir / name
            if target.exists():
                continue
            if not source.is_file():
                missing.append(name)
                continue
            try:
                target.symlink_to(source)
            except OSError:
                try:
                    os.link(source, target)
                except OSError:
                    missing.append(name)
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                "BoostTrack ReID weights are missing or could not be linked into "
                f"{target_dir}: {joined}. Either copy/link them there or run with --no-reid."
            )


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0
