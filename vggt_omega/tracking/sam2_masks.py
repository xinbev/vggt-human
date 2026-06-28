from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .schema import Detection, TrackObservation


class SAM2BoxMaskPredictor:
    """SAM2 image predictor prompted by tracked person boxes."""

    def __init__(
        self,
        sam2_root: str | Path,
        checkpoint: str | Path,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        multimask_output: bool = True,
    ) -> None:
        self.sam2_root = Path(sam2_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.sam2_root.is_dir():
            raise FileNotFoundError(f"SAM2 root not found: {self.sam2_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        sam2_root_str = str(self.sam2_root)
        if sam2_root_str not in sys.path:
            sys.path.insert(0, sam2_root_str)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.multimask_output = bool(multimask_output)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(model_cfg, str(self.checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    @torch.inference_mode()
    def predict_for_observations(
        self,
        frame_bgr: np.ndarray,
        observations: list[TrackObservation],
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float | int]]]:
        if not observations:
            return {}, {}
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masks: dict[int, np.ndarray] = {}
        metadata: dict[int, dict[str, float | int]] = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            self.predictor.set_image(frame_rgb)
            for obs in observations:
                box = np.asarray(obs.bbox_xyxy, dtype=np.float32)
                pred_masks, pred_scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=self.multimask_output,
                    return_logits=False,
                )
                best_idx = int(np.argmax(pred_scores))
                mask = np.asarray(pred_masks[best_idx] > 0, dtype=np.bool_)
                masks[int(obs.person_id)] = mask
                metadata[int(obs.person_id)] = {
                    "sam2_score": float(pred_scores[best_idx]),
                    "mask_area": int(mask.sum()),
                    "mask_height": int(mask.shape[0]),
                    "mask_width": int(mask.shape[1]),
                }
        return masks, metadata

    @torch.inference_mode()
    def predict_for_detections(
        self,
        frame_bgr: np.ndarray,
        detections: list[Detection],
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float | int]]]:
        if not detections:
            return {}, {}
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masks: dict[int, np.ndarray] = {}
        metadata: dict[int, dict[str, float | int]] = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            self.predictor.set_image(frame_rgb)
            for det_idx, det in enumerate(detections):
                det_id = int(det.det_id if det.det_id >= 0 else det_idx)
                box = np.asarray(det.bbox_xyxy, dtype=np.float32)
                pred_masks, pred_scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=self.multimask_output,
                    return_logits=False,
                )
                best_idx = int(np.argmax(pred_scores))
                mask = np.asarray(pred_masks[best_idx] > 0, dtype=np.bool_)
                masks[det_id] = mask
                metadata[det_id] = {
                    "sam2_score": float(pred_scores[best_idx]),
                    "mask_area": int(mask.sum()),
                    "mask_height": int(mask.shape[0]),
                    "mask_width": int(mask.shape[1]),
                }
        return masks, metadata


def save_frame_masks(path: Path, masks: dict[int, np.ndarray], prefix: str = "person") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if prefix == "det":
        arrays = {f"{prefix}_{item_id:06d}": mask.astype(np.uint8) for item_id, mask in sorted(masks.items())}
    else:
        arrays = {f"{prefix}_{item_id}": mask.astype(np.uint8) for item_id, mask in sorted(masks.items())}
    np.savez_compressed(path, **arrays)
