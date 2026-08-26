from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TRSTRTrackState:
    track_id: int
    last_frame: int
    translation: torch.Tensor
    velocity: torch.Tensor
    confidence: float
    region_token: torch.Tensor | None = None


class HSITRSTRTrackMemory:
    """Detached persistent translation memory keyed by track ID."""

    def __init__(self, max_gap: int = 30) -> None:
        self.max_gap = int(max_gap)
        self.states: dict[int, TRSTRTrackState] = {}

    def get(
        self,
        track_id: int,
        frame_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TRSTRTrackState | None:
        state = self.states.get(int(track_id))
        if state is None:
            return None
        gap = int(frame_index) - int(state.last_frame)
        if gap <= 0 or gap > self.max_gap:
            if gap > self.max_gap:
                self.states.pop(int(track_id), None)
            return None
        return TRSTRTrackState(
            track_id=state.track_id,
            last_frame=state.last_frame,
            translation=state.translation.to(device=device, dtype=dtype),
            velocity=state.velocity.to(device=device, dtype=dtype),
            confidence=float(state.confidence),
            region_token=None if state.region_token is None else state.region_token.to(device=device, dtype=dtype),
        )

    def update(
        self,
        track_id: int,
        frame_index: int,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        confidence: float,
        region_token: torch.Tensor | None = None,
    ) -> None:
        self.states[int(track_id)] = TRSTRTrackState(
            track_id=int(track_id),
            last_frame=int(frame_index),
            translation=translation.detach().float().cpu(),
            velocity=velocity.detach().float().cpu(),
            confidence=float(confidence),
            region_token=None if region_token is None else region_token.detach().float().cpu(),
        )

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self.states.clear()
        else:
            self.states.pop(int(track_id), None)

    def __len__(self) -> int:
        return len(self.states)
