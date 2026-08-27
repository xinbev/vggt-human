"""Explicit, opt-in bridge from tracked pipeline outputs to the temporal refiner.

Nothing in the VGGT/HSI/TRSTR forward path imports this adapter.  Deployment
code may instantiate it after loading a standalone checkpoint, retain the
existing outputs as the fallback, and call :meth:`refine_tracked_batch` only
for complete offline clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vggt_omega.models import TemporalRefinerConfig, TemporalSMPLRefiner


class SMPLTemporalRefinementAdapter:
    """Refine person tracks while preserving the project's frame-major layout."""

    def __init__(self, refiner: TemporalSMPLRefiner) -> None:
        self.refiner = refiner

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str = "cpu") -> "SMPLTemporalRefinementAdapter":
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        if checkpoint.get("format") != "smpl_temporal_refiner_v1":
            raise ValueError(f"Unsupported temporal-refiner checkpoint: {checkpoint_path}")
        config = TemporalRefinerConfig(**checkpoint["model_config"])
        refiner = TemporalSMPLRefiner(config).to(device)
        refiner.load_state_dict(checkpoint["model_state"], strict=True)
        refiner.eval()
        return cls(refiner)

    @torch.no_grad()
    def refine_tracked_batch(
        self,
        pose_6d: torch.Tensor,
        transl_cam: torch.Tensor,
        betas: torch.Tensor,
        track_ids: torch.Tensor,
        person_valid: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine a complete offline batch with stable track IDs.

        Args use the current project convention: pose ``[B,S,Q,144]``,
        translation ``[B,S,Q,3]`` and IDs ``[B,S,Q]``.  A person can move
        between slots across frames: all matching IDs are gathered into one
        temporal track before refinement and scattered back afterwards.  Any
        invalid/missing position stays byte-for-byte equal to the input.
        """
        if pose_6d.ndim != 4 or pose_6d.shape[-1] != 144:
            raise ValueError(f"pose_6d must be [B,S,Q,144], got {tuple(pose_6d.shape)}")
        if transl_cam.shape != (*pose_6d.shape[:3], 3):
            raise ValueError("transl_cam must align with pose_6d")
        if betas.shape[:3] != pose_6d.shape[:3] or betas.shape[-1] < 10:
            raise ValueError("betas must be [B,S,Q,>=10] aligned with pose_6d")
        if track_ids.shape != pose_6d.shape[:3]:
            raise ValueError("track_ids must be [B,S,Q] aligned with pose_6d")
        batch, steps, slots = pose_6d.shape[:3]
        if steps > self.refiner.config.window_size:
            raise ValueError(
                f"clip has {steps} steps but checkpoint supports at most {self.refiner.config.window_size}; "
                "apply it with overlapping offline windows."
            )
        valid = track_ids >= 0 if person_valid is None else (person_valid.to(dtype=torch.bool) & (track_ids >= 0))
        refined_pose = pose_6d.clone()
        refined_transl = transl_cam.clone()
        refined_betas = betas.clone()
        gathered: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        tracks_pose: list[torch.Tensor] = []
        tracks_transl: list[torch.Tensor] = []
        tracks_betas: list[torch.Tensor] = []
        tracks_valid: list[torch.Tensor] = []
        tracks_confidence: list[torch.Tensor] = []
        for batch_index in range(batch):
            ids = torch.unique(track_ids[batch_index][valid[batch_index]])
            for track_id in ids.tolist():
                locations = (track_ids[batch_index] == int(track_id)) & valid[batch_index]
                # Tracker output must have at most one detection for a person
                # per frame.  Ambiguous frames are excluded rather than mixed.
                per_frame_count = locations.sum(dim=1)
                track_valid = per_frame_count == 1
                slot_for_frame = locations.to(dtype=torch.long).argmax(dim=1)
                track_pose = torch.zeros(steps, 144, device=pose_6d.device, dtype=pose_6d.dtype)
                track_transl = torch.zeros(steps, 3, device=transl_cam.device, dtype=transl_cam.dtype)
                track_betas = torch.zeros(steps, 10, device=betas.device, dtype=betas.dtype)
                if bool(track_valid.any()):
                    frame_idx = torch.nonzero(track_valid, as_tuple=False).squeeze(-1)
                    slot_idx = slot_for_frame[frame_idx]
                    track_pose[frame_idx] = pose_6d[batch_index, frame_idx, slot_idx]
                    track_transl[frame_idx] = transl_cam[batch_index, frame_idx, slot_idx]
                    track_betas[frame_idx] = betas[batch_index, frame_idx, slot_idx, :10]
                if confidence is None:
                    track_confidence = track_valid.to(dtype=pose_6d.dtype)
                else:
                    track_confidence = confidence[batch_index, torch.arange(steps, device=pose_6d.device), slot_for_frame]
                    track_confidence = track_confidence * track_valid.to(dtype=track_confidence.dtype)
                gathered.append((batch_index, int(track_id), slot_for_frame, track_valid))
                tracks_pose.append(track_pose)
                tracks_transl.append(track_transl)
                tracks_betas.append(track_betas)
                tracks_valid.append(track_valid)
                tracks_confidence.append(track_confidence)
        if not gathered:
            return {
                "smpl_temporal_refined_pose_6d": refined_pose,
                "smpl_temporal_refined_pred_transl_cam": refined_transl,
                "smpl_temporal_refined_betas": refined_betas,
                "smpl_temporal_applied": valid.new_zeros((batch, steps, slots)),
            }
        outputs = self.refiner(
            torch.stack(tracks_pose),
            torch.stack(tracks_transl),
            torch.stack(tracks_betas),
            torch.stack(tracks_valid),
            torch.stack(tracks_confidence),
        )
        applied = valid.new_zeros((batch, steps, slots))
        for track_index, (batch_index, _, slots_by_frame, track_valid) in enumerate(gathered):
            frame_idx = torch.nonzero(track_valid, as_tuple=False).squeeze(-1)
            slot_idx = slots_by_frame[frame_idx]
            refined_pose[batch_index, frame_idx, slot_idx] = outputs["refined_pose_6d"][track_index, frame_idx]
            refined_transl[batch_index, frame_idx, slot_idx] = outputs["refined_transl"][track_index, frame_idx]
            applied[batch_index, frame_idx, slot_idx] = True
        return {
            "smpl_temporal_refined_pose_6d": refined_pose,
            "smpl_temporal_refined_pred_transl_cam": refined_transl,
            "smpl_temporal_refined_betas": refined_betas,
            "smpl_temporal_applied": applied,
            "smpl_temporal_pose_gate": outputs["pose_gate"],
            "smpl_temporal_transl_gate": outputs["transl_gate"],
            "smpl_temporal_uncertainty": outputs["uncertainty"],
        }
