from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class HumanTrackState:
    person_id: int
    last_seen_frame: int
    bbox_xyxy: list[float]
    bbox_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    missing_count: int = 0
    state: str = "active"
    pose: list[float] | None = None
    betas: list[float] | None = None
    transl_cam: list[float] | None = None
    hsi_refined_pose: list[float] | None = None
    hsi_refined_betas: list[float] | None = None
    hsi_refined_transl_cam: list[float] | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": int(self.person_id),
            "state": self.state,
            "last_seen_frame": int(self.last_seen_frame),
            "missing_count": int(self.missing_count),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "bbox_velocity": [float(v) for v in self.bbox_velocity],
            "pose": self.pose,
            "betas": self.betas,
            "transl_cam": self.transl_cam,
            "hsi_refined_pose": self.hsi_refined_pose,
            "hsi_refined_betas": self.hsi_refined_betas,
            "hsi_refined_transl_cam": self.hsi_refined_transl_cam,
            "confidence": float(self.confidence),
        }


class HSITrackMemory:
    """Model-feedback memory for HSI-aware matching or tracklet stitching."""

    def __init__(self, max_lost: int = 60, ema: float = 0.8) -> None:
        self.max_lost = int(max_lost)
        self.ema = float(ema)
        self.tracks: dict[int, HumanTrackState] = {}

    def update_observations(self, frame_index: int, observations: list[dict[str, Any]]) -> None:
        visible_ids: set[int] = set()
        for obs in observations:
            if not obs.get("valid", True) or not obs.get("person_id_valid", True):
                continue
            person_id = int(obs["person_id"])
            visible_ids.add(person_id)
            box = [float(v) for v in obs["bbox_xyxy_pixels"]]
            prev = self.tracks.get(person_id)
            if prev is None:
                self.tracks[person_id] = HumanTrackState(
                    person_id=person_id,
                    last_seen_frame=frame_index,
                    bbox_xyxy=box,
                    confidence=float(obs.get("track_confidence", obs.get("det_score", 0.0))),
                )
                continue
            old_box = np.asarray(prev.bbox_xyxy, dtype=np.float32)
            new_box = np.asarray(box, dtype=np.float32)
            prev.bbox_velocity = [float(v) for v in (new_box - old_box)]
            prev.bbox_xyxy = box
            prev.last_seen_frame = frame_index
            prev.missing_count = 0
            prev.state = "active"
            prev.confidence = float(obs.get("track_confidence", prev.confidence))

        for person_id, state in self.tracks.items():
            if person_id in visible_ids:
                continue
            state.missing_count = max(frame_index - state.last_seen_frame, 0)
            state.state = "lost" if state.missing_count <= self.max_lost else "dead"

    def update_from_model_outputs(
        self,
        frame_indices: list[int],
        slot_track_ids: torch.Tensor,
        slot_track_mask: torch.Tensor,
        outputs: dict[str, torch.Tensor],
    ) -> None:
        """Write SMPL/HSI outputs back into memory.

        Args:
            frame_indices: S frame indices for the clip.
            slot_track_ids: [S,Q] track ids.
            slot_track_mask: [S,Q] valid mask.
            outputs: VGGTOmega output dict, preferably containing both base and HSI keys.
        """
        track_ids = slot_track_ids.detach().cpu()
        track_mask = slot_track_mask.detach().cpu().bool()
        for s, frame_index in enumerate(frame_indices):
            for q in range(track_ids.shape[1]):
                if not bool(track_mask[s, q]):
                    continue
                pid = int(track_ids[s, q])
                state = self.tracks.get(pid)
                if state is None:
                    continue
                state.last_seen_frame = int(frame_index)
                self._assign_tensor_list(state, "pose", outputs, "pred_poses", s, q)
                self._assign_tensor_list(state, "betas", outputs, "pred_betas", s, q)
                self._assign_tensor_list(state, "transl_cam", outputs, "pred_transl_cam", s, q)
                self._assign_tensor_list(state, "hsi_refined_pose", outputs, "hsi_refined_pred_poses", s, q)
                self._assign_tensor_list(state, "hsi_refined_betas", outputs, "hsi_refined_pred_betas", s, q)
                self._assign_tensor_list(state, "hsi_refined_transl_cam", outputs, "hsi_refined_pred_transl_cam", s, q)
                if "pred_confs" in outputs:
                    conf = outputs["pred_confs"].detach().float().cpu()
                    state.confidence = float(conf.reshape(conf.shape[0], conf.shape[1], conf.shape[2], -1)[0, s, q].mean())

    def to_dict(self) -> dict[str, Any]:
        return {str(pid): state.to_dict() for pid, state in sorted(self.tracks.items())}

    def _assign_tensor_list(
        self,
        state: HumanTrackState,
        attr: str,
        outputs: dict[str, torch.Tensor],
        frame_slot: int,
        query_slot: int,
    ) -> None:
        key_map = {
            "pose": "pred_poses",
            "betas": "pred_betas",
            "transl_cam": "pred_transl_cam",
            "hsi_refined_pose": "hsi_refined_pred_poses",
            "hsi_refined_betas": "hsi_refined_pred_betas",
            "hsi_refined_transl_cam": "hsi_refined_pred_transl_cam",
        }
        key = key_map[attr]
        if key not in outputs:
            return
        tensor = outputs[key].detach().float().cpu()
        while tensor.ndim > 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 4:
            values = tensor[0, frame_slot, query_slot].reshape(-1).tolist()
        elif tensor.ndim == 3:
            values = tensor[frame_slot, query_slot].reshape(-1).tolist()
        else:
            return
        old = getattr(state, attr)
        if old is None or len(old) != len(values):
            setattr(state, attr, [float(v) for v in values])
            return
        blended = [self.ema * float(a) + (1.0 - self.ema) * float(b) for a, b in zip(old, values)]
        setattr(state, attr, blended)
