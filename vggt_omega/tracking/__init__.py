"""Video person observation and tracking helpers for VGGT-Omega."""

from .clip_builder import build_clip_tensors_from_sidecar
from .query_builder import build_detection_query_tensors_from_sidecar, build_external_track_prior_from_sidecar
from .schema import Detection, FrameObservations, TrackObservation
from .smpl_track_assigner import BaseSMPLTrackAssigner

__all__ = [
    "BaseSMPLTrackAssigner",
    "Detection",
    "FrameObservations",
    "TrackObservation",
    "build_clip_tensors_from_sidecar",
    "build_detection_query_tensors_from_sidecar",
    "build_external_track_prior_from_sidecar",
]
