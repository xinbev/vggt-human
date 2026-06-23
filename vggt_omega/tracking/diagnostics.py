from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .schema import FrameObservations


@dataclass(slots=True)
class TrackingDiagnostics:
    total_frames: int = 0
    total_detections: int = 0
    total_observations: int = 0
    max_people_per_frame: int = 0
    active_track_ids: set[int] = field(default_factory=set)
    track_lengths: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    track_first_frame: dict[int, int] = field(default_factory=dict)
    track_last_frame: dict[int, int] = field(default_factory=dict)
    track_gaps: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    confidence_values: list[float] = field(default_factory=list)

    def update(self, frame: FrameObservations) -> None:
        self.total_frames += 1
        self.total_detections += len(frame.detections)
        self.total_observations += len(frame.persons)
        self.max_people_per_frame = max(self.max_people_per_frame, len(frame.persons))
        current_ids = {obs.person_id for obs in frame.persons if obs.valid and obs.person_id >= 0}
        self.active_track_ids.update(current_ids)
        for obs in frame.persons:
            if not obs.valid or obs.person_id < 0:
                continue
            pid = obs.person_id
            self.track_lengths[pid] += 1
            self.track_first_frame.setdefault(pid, frame.frame_index)
            previous = self.track_last_frame.get(pid)
            if previous is not None and frame.frame_index - previous > 1:
                self.track_gaps[pid].append(frame.frame_index - previous - 1)
            self.track_last_frame[pid] = frame.frame_index
            self.confidence_values.append(float(obs.track_confidence))

    def to_dict(self) -> dict[str, Any]:
        gaps = [gap for values in self.track_gaps.values() for gap in values]
        return {
            "total_frames": int(self.total_frames),
            "total_detections": int(self.total_detections),
            "total_observations": int(self.total_observations),
            "num_tracks": int(len(self.active_track_ids)),
            "max_people_per_frame": int(self.max_people_per_frame),
            "track_lengths": {str(key): int(value) for key, value in sorted(self.track_lengths.items())},
            "track_first_frame": {str(key): int(value) for key, value in sorted(self.track_first_frame.items())},
            "track_last_frame": {str(key): int(value) for key, value in sorted(self.track_last_frame.items())},
            "track_gap_count": int(len(gaps)),
            "track_gap_max": int(max(gaps) if gaps else 0),
            "track_gap_mean": float(sum(gaps) / len(gaps)) if gaps else 0.0,
            "track_confidence_mean": float(sum(self.confidence_values) / len(self.confidence_values))
            if self.confidence_values
            else 0.0,
        }
