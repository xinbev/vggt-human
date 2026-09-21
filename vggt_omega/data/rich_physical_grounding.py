from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from vggt_omega.data.geometry import (
    ResizeGeometry,
    compute_resize_geometry,
    resize_image_with_geometry,
    transform_xyxy_to_normalized_cxcywh,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class RichPhysicalGroundingSequence:
    vid: str
    split: str
    recording: str
    camera: str
    camera_id: int
    frame_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    label: dict[str, Any]
    scene_asset: str
    scan_path: Path
    calibration_path: Path
    multicam2world_path: Path


@dataclass(frozen=True, slots=True)
class RichFrameBatch:
    images: torch.Tensor
    query_boxes: torch.Tensor
    query_mask: torch.Tensor
    track_ids: torch.Tensor
    track_mask: torch.Tensor
    label_positions: tuple[int, ...]
    source_frame_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    resize_geometry: ResizeGeometry


class RichPhysicalGroundingDataset:
    """Read label-selected RGB frames and target-person boxes for RICH.

    ``rich_test_labels.pt['frame_id']`` is interpreted as an index into the
    sorted official image list of the corresponding camera directory. This is
    the same contract checked by ``check_rich_physical_grounding_assets.py``.
    Only the RICH target person is exposed, in query slot zero.
    """

    def __init__(
        self,
        official_root: str | Path,
        support_root: str | Path,
        image_resolution: int = 512,
        patch_size: int = 16,
        resize_mode: str = "balanced",
        max_humans: int = 20,
        sequence_filters: Iterable[str] | None = None,
        max_sequences: int = 0,
        frame_stride: int = 1,
        max_frames_per_sequence: int = 0,
    ) -> None:
        self.official_root = Path(official_root).expanduser()
        self.support_root = Path(support_root).expanduser()
        self.image_resolution = int(image_resolution)
        self.patch_size = int(patch_size)
        self.resize_mode = str(resize_mode)
        self.max_humans = int(max_humans)
        self.frame_stride = int(frame_stride)
        self.max_frames_per_sequence = int(max_frames_per_sequence)
        if self.max_humans <= 0:
            raise ValueError("max_humans must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")

        labels_path = self.support_root / "rich_test_labels.pt"
        preproc_path = self.support_root / "rich_test_preproc.pt"
        labels = _load_torch(labels_path)
        preproc = _load_torch(preproc_path)
        if not isinstance(labels, dict) or not labels:
            raise RuntimeError(f"Unexpected RICH labels payload: {labels_path}")
        if not isinstance(preproc, dict) or not preproc:
            raise RuntimeError(f"Unexpected RICH preprocessing payload: {preproc_path}")

        filters = tuple(str(value).strip().strip("/") for value in (sequence_filters or ()) if str(value).strip())
        records = []
        for vid, raw_label in sorted(labels.items()):
            if filters and not any(_matches_sequence_filter(str(vid), value) for value in filters):
                continue
            label = dict(raw_label)
            if vid in preproc:
                for key in ("bbx_xys", "kp2d", "img_wh"):
                    if key in preproc[vid]:
                        label[key] = preproc[vid][key]
            records.append(self._build_record(str(vid), label))
        if max_sequences > 0:
            records = records[: int(max_sequences)]
        if not records:
            requested = ", ".join(filters) if filters else "<all>"
            raise RuntimeError(f"No RICH camera-view sequences matched: {requested}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def selected_label_positions(self, record: RichPhysicalGroundingSequence) -> list[int]:
        positions = list(range(0, len(record.frame_ids), self.frame_stride))
        if self.max_frames_per_sequence > 0:
            positions = positions[: self.max_frames_per_sequence]
        return positions

    def load_batch(
        self,
        record: RichPhysicalGroundingSequence,
        label_positions: Iterable[int],
    ) -> RichFrameBatch:
        positions = tuple(int(value) for value in label_positions)
        if not positions:
            raise ValueError("Cannot load an empty RICH frame batch")
        if min(positions) < 0 or max(positions) >= len(record.frame_ids):
            raise IndexError(f"Label positions outside [0, {len(record.frame_ids)}): {positions}")

        bbx = record.label.get("bbx_xys")
        if bbx is None:
            raise KeyError(f"RICH preprocessing has no bbx_xys for {record.vid}")
        bbx_tensor = torch.as_tensor(bbx, dtype=torch.float32)
        if bbx_tensor.ndim < 2 or bbx_tensor.shape[0] < len(record.frame_ids) or bbx_tensor.shape[-1] < 3:
            raise ValueError(f"Unexpected bbx_xys shape for {record.vid}: {tuple(bbx_tensor.shape)}")
        raw_eval_mask = record.label.get("mask")
        eval_mask = (
            torch.ones(len(record.frame_ids), dtype=torch.bool)
            if raw_eval_mask is None
            else torch.as_tensor(raw_eval_mask, dtype=torch.bool).reshape(-1)
        )
        if eval_mask.numel() < len(record.frame_ids):
            raise ValueError(f"Unexpected eval mask length for {record.vid}: {eval_mask.numel()}")

        images: list[torch.Tensor] = []
        boxes = torch.zeros(len(positions), self.max_humans, 4, dtype=torch.float32)
        mask = torch.zeros(len(positions), self.max_humans, dtype=torch.bool)
        geometries: list[ResizeGeometry] = []
        paths = []
        source_ids = []
        for output_index, label_position in enumerate(positions):
            source_id = int(record.frame_ids[label_position])
            path = record.image_paths[source_id]
            with Image.open(path) as image_file:
                image = image_file.convert("RGB")
                geometry = compute_resize_geometry(
                    (image.height, image.width),
                    image_resolution=self.image_resolution,
                    patch_size=self.patch_size,
                    mode=self.resize_mode,
                )
                resized = resize_image_with_geometry(image, geometry, Image.Resampling.BICUBIC)
                array = np.asarray(resized, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            geometries.append(geometry)
            paths.append(path)
            source_ids.append(source_id)

            center_x, center_y, size = [float(value) for value in bbx_tensor[label_position].reshape(-1)[:3]]
            xyxy = np.asarray(
                [center_x - 0.5 * size, center_y - 0.5 * size, center_x + 0.5 * size, center_y + 0.5 * size],
                dtype=np.float32,
            )
            transformed, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometry)
            boxes[output_index, 0] = torch.as_tensor(transformed, dtype=torch.float32)
            mask[output_index, 0] = bool(valid) and bool(eval_mask[label_position])

        input_hws = {geometry.input_hw for geometry in geometries}
        if len(input_hws) != 1:
            raise RuntimeError(f"RICH batch produced inconsistent input shapes: {sorted(input_hws)}")
        track_ids = torch.full((len(positions), self.max_humans), -1, dtype=torch.long)
        track_ids[:, 0] = 0
        return RichFrameBatch(
            images=torch.stack(images, dim=0),
            query_boxes=boxes,
            query_mask=mask,
            track_ids=track_ids,
            track_mask=mask.clone(),
            label_positions=positions,
            source_frame_ids=tuple(source_ids),
            image_paths=tuple(paths),
            resize_geometry=geometries[0],
        )

    def _build_record(self, vid: str, label: dict[str, Any]) -> RichPhysicalGroundingSequence:
        parts = vid.split("/")
        if len(parts) != 3:
            raise ValueError(f"Unexpected RICH key {vid!r}; expected split/recording/camera")
        split, recording, camera = parts
        try:
            camera_id = int(camera.removeprefix("cam_"))
        except ValueError as exc:
            raise ValueError(f"Invalid RICH camera name in {vid!r}: {camera!r}") from exc
        image_dir = self.official_root / split / recording / camera
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing RICH image directory: {image_dir}")
        image_paths = tuple(sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))
        if not image_paths:
            raise FileNotFoundError(f"No RICH images found in: {image_dir}")
        frame_ids_tensor = torch.as_tensor(label.get("frame_id", []), dtype=torch.long).flatten()
        if frame_ids_tensor.numel() == 0:
            raise ValueError(f"Empty frame_id for {vid}")
        frame_ids = tuple(int(value) for value in frame_ids_tensor.tolist())
        if min(frame_ids) < 0 or max(frame_ids) >= len(image_paths):
            raise IndexError(
                f"frame_id range [{min(frame_ids)}, {max(frame_ids)}] is invalid for "
                f"{vid} with {len(image_paths)} images"
            )

        scene = recording.split("_", 1)[0]
        scene_asset = resolve_scene_asset_name(recording)
        scan_name = (
            f"scan_{scene_asset.removeprefix('LectureHall_')}_scene_camcoord.ply"
            if scene == "LectureHall"
            else "scan_camcoord.ply"
        )
        return RichPhysicalGroundingSequence(
            vid=vid,
            split=split,
            recording=recording,
            camera=camera,
            camera_id=camera_id,
            frame_ids=frame_ids,
            image_paths=image_paths,
            label=label,
            scene_asset=scene_asset,
            scan_path=self.official_root / "scan_calibration" / scene / scan_name,
            calibration_path=self.official_root / "scan_calibration" / scene / "calibration" / f"{camera_id:03d}.xml",
            multicam2world_path=self.official_root / "multicam2world" / f"{scene_asset}_multicam2world.json",
        )


def resolve_scene_asset_name(recording: str) -> str:
    scene = recording.split("_", 1)[0]
    if scene != "LectureHall":
        return scene
    if "wipingchairs" in recording or "reparingprojector" in recording:
        return "LectureHall_chair"
    return "LectureHall_yoga"


def _matches_sequence_filter(vid: str, requested: str) -> bool:
    normalized_vid = vid.strip("/")
    normalized_requested = requested.strip("/")
    parts = normalized_vid.split("/")
    recording = parts[1] if len(parts) == 3 else ""
    camera_view = "/".join(parts[1:]) if len(parts) == 3 else ""
    return normalized_requested in {normalized_vid, recording, camera_view}


def _load_torch(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RICH support file: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
