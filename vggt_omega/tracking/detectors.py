from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

if os.environ.get("VGGT_OMEGA_DISABLE_CV2", "").strip().lower() in {"1", "true", "yes"}:
    cv2 = None
else:
    try:
        import cv2
    except (ImportError, AttributeError, SystemError):
        cv2 = None

from .schema import Detection, clamp_box_xyxy


@dataclass(slots=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    out_width: int
    out_height: int
    in_width: int
    in_height: int


class TorchScriptYOLOPersonDetector:
    """Person detector for YOLO-style TorchScript exports.

    The parser accepts the two common export layouts:
    - post-NMS: [N, 6] or [B, N, 6] as xyxy, score, class_id
    - raw YOLOv8: [N, 84] / [B, N, 84] or transposed [B, 84, N]
      as cx, cy, w, h, class scores.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        image_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        person_class_id: int = 0,
        max_detections: int = 300,
        half: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"YOLO TorchScript checkpoint not found: {self.checkpoint}")
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.person_class_id = int(person_class_id)
        self.max_detections = int(max_detections)
        self.half = bool(half and self.device.type == "cuda")
        self.model = torch.jit.load(str(self.checkpoint), map_location=self.device).eval()
        if self.half:
            self.model.half()

    @torch.inference_mode()
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        tensor, meta = self._preprocess(frame_bgr)
        output = self.model(tensor)
        boxes, scores, classes = self._parse_output(output)
        if boxes.numel() == 0:
            return []

        person_mask = classes == self.person_class_id
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        classes = classes[person_mask]
        keep = scores >= self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        if boxes.numel() == 0:
            return []

        keep_idx = nms(boxes.float(), scores.float(), self.iou_threshold)[: self.max_detections]
        boxes = self._scale_boxes_to_original(boxes[keep_idx], meta)
        scores = scores[keep_idx]
        classes = classes[keep_idx]

        detections: list[Detection] = []
        height, width = frame_bgr.shape[:2]
        for box, score, class_id in zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()):
            xyxy = clamp_box_xyxy(box.tolist(), width=width, height=height)
            detections.append(
                Detection(
                    bbox_xyxy=xyxy,
                    score=float(score),
                    class_id=int(class_id),
                    class_name="person",
                    source="yolo_torchscript",
                )
            )
        return detections

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, LetterboxMeta]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape [H,W,3], got {frame_bgr.shape}")
        in_height, in_width = frame_bgr.shape[:2]
        scale = min(self.image_size / max(in_width, 1), self.image_size / max(in_height, 1))
        resized_width = int(round(in_width * scale))
        resized_height = int(round(in_height * scale))
        resized = _resize_bgr_linear(frame_bgr, (resized_width, resized_height))
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        pad_x = (self.image_size - resized_width) // 2
        pad_y = (self.image_size - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = _bgr_to_rgb(canvas)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.half:
            tensor = tensor.half()
        meta = LetterboxMeta(
            scale=float(scale),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            out_width=self.image_size,
            out_height=self.image_size,
            in_width=int(in_width),
            in_height=int(in_height),
        )
        return tensor, meta

    def _parse_output(self, output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = self._unwrap_output(output)
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Unsupported YOLO TorchScript output shape: {tuple(tensor.shape)}")
        if tensor.shape[0] in {5, 6, 84, 85} and tensor.shape[1] > tensor.shape[0]:
            tensor = tensor.transpose(0, 1)
        tensor = tensor.float()
        if tensor.shape[1] < 6:
            raise ValueError(f"YOLO output must have at least 6 columns, got {tuple(tensor.shape)}")

        if tensor.shape[1] == 6:
            boxes = tensor[:, :4]
            scores = tensor[:, 4]
            classes = tensor[:, 5].long()
            return self._ensure_xyxy(boxes), scores, classes

        boxes_xywh = tensor[:, :4]
        class_scores = tensor[:, 4:]
        if tensor.shape[1] == 85:
            scores_no_obj, classes_no_obj = tensor[:, 4:].max(dim=1)
            scores_with_obj, classes_with_obj = tensor[:, 5:].max(dim=1)
            scores_with_obj = scores_with_obj * tensor[:, 4].sigmoid().clamp(0.0, 1.0)
            use_obj = scores_with_obj.mean() > scores_no_obj.mean() * 0.5
            if use_obj:
                scores = scores_with_obj
                classes = classes_with_obj.long()
            else:
                scores = scores_no_obj
                classes = classes_no_obj.long()
        else:
            scores, classes = class_scores.max(dim=1)
            classes = classes.long()
        boxes = self._xywh_to_xyxy(boxes_xywh)
        return boxes, scores, classes

    def _unwrap_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("pred", "preds", "output", "outputs", "detections"):
                if key in output:
                    return self._unwrap_output(output[key])
            first = next(iter(output.values()))
            return self._unwrap_output(first)
        if isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("YOLO TorchScript returned an empty output list")
            return self._unwrap_output(output[0])
        raise TypeError(f"Unsupported YOLO TorchScript output type: {type(output)!r}")

    def _ensure_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        looks_xyxy = bool(((x2 > x1) & (y2 > y1)).float().mean() > 0.8)
        if looks_xyxy:
            return boxes
        return self._xywh_to_xyxy(boxes)

    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, width, height = boxes.unbind(dim=1)
        return torch.stack((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0), dim=1)

    def _scale_boxes_to_original(self, boxes: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
        scaled = boxes.clone().float()
        if float(scaled.max().detach().cpu()) <= 2.0:
            scale_vec = torch.tensor(
                [meta.out_width, meta.out_height, meta.out_width, meta.out_height],
                dtype=scaled.dtype,
                device=scaled.device,
            )
            scaled = scaled * scale_vec
        scaled[:, [0, 2]] -= meta.pad_x
        scaled[:, [1, 3]] -= meta.pad_y
        scaled[:, :4] /= max(meta.scale, 1e-8)
        scaled[:, [0, 2]] = scaled[:, [0, 2]].clamp(0.0, float(meta.in_width))
        scaled[:, [1, 3]] = scaled[:, [1, 3]].clamp(0.0, float(meta.in_height))
        return scaled


def _resize_bgr_linear(frame_bgr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(frame_bgr, size_wh, interpolation=cv2.INTER_LINEAR)
    rgb = _bgr_to_rgb(frame_bgr)
    resampling = getattr(Image, "Resampling", Image)
    resized_rgb = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize(size_wh, resample=resampling.BILINEAR),
        dtype=np.uint8,
    )
    return np.ascontiguousarray(resized_rgb[..., ::-1])


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(frame_bgr[..., ::-1])
