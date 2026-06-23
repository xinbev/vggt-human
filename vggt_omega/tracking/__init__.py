"""Video person observation and tracking helpers for VGGT-Omega."""

from .clip_builder import build_clip_tensors_from_sidecar
from .schema import Detection, FrameObservations, TrackObservation

__all__ = [
    "Detection",
    "FrameObservations",
    "TrackObservation",
    "build_clip_tensors_from_sidecar",
]
