from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class StitchCandidate:
    source_id: int
    target_id: int
    score: float
    gap: int
    center_norm: float
    size_log_delta: float
    pred_iou: float


def postprocess_sidecar_tracks(
    sidecar_root: str | Path,
    max_gap: int = 30,
    center_thresh: float = 1.25,
    size_log_thresh: float = 0.70,
    min_score: float = 0.25,
    compact_ids: bool = True,
) -> dict[str, Any]:
    root = Path(sidecar_root).expanduser()
    frame_paths = sorted((root / "smpl_boxes").glob("*.pkl"))
    if not frame_paths:
        raise FileNotFoundError(f"No smpl_boxes found under {root / 'smpl_boxes'}")
    frames = [_load_pickle(path) for path in frame_paths]
    tracklets = _collect_tracklets(frames)
    candidates = _build_stitch_candidates(tracklets, max_gap, center_thresh, size_log_thresh, min_score)
    raw_to_group = _select_stitches(tracklets, candidates)
    final_id_map = _build_final_id_map(tracklets, raw_to_group, compact_ids)
    changed = _rewrite_frames(frames, final_id_map)
    for path, frame in zip(frame_paths, frames):
        with path.open("wb") as file:
            pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
    _rewrite_observations_jsonl(root / "observations.jsonl", frames)
    track_id_map = {
        str(raw_id): int(final_id)
        for raw_id, final_id in sorted(final_id_map.items())
        if int(raw_id) != int(final_id)
    }
    stitch_rows = [
        {
            "source_id": item.source_id,
            "target_id": item.target_id,
            "score": item.score,
            "gap": item.gap,
            "center_norm": item.center_norm,
            "size_log_delta": item.size_log_delta,
            "pred_iou": item.pred_iou,
            "final_id": final_id_map[item.source_id],
        }
        for item in candidates
        if final_id_map.get(item.source_id) == final_id_map.get(item.target_id)
    ]
    (root / "track_id_postprocess.json").write_text(
        json.dumps({"track_id_map": track_id_map, "stitches": stitch_rows}, indent=2),
        encoding="utf-8",
    )
    diagnostics = summarize_frames(frames)
    return {
        "tracklet_stitching": {
            "enabled": True,
            "num_raw_tracklets": len(tracklets),
            "num_final_tracks": diagnostics["num_tracks"],
            "num_changed_observations": int(changed),
            "track_id_map": track_id_map,
            "stitches": stitch_rows,
            "params": {
                "max_gap": int(max_gap),
                "center_thresh": float(center_thresh),
                "size_log_thresh": float(size_log_thresh),
                "min_score": float(min_score),
                "compact_ids": bool(compact_ids),
            },
        },
        "diagnostics": diagnostics,
    }


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    track_lengths: dict[int, int] = {}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    gaps: dict[int, list[int]] = {}
    confs: list[float] = []
    total_detections = 0
    total_observations = 0
    max_people = 0
    for frame in frames:
        frame_idx = int(frame.get("frame_index", 0))
        total_detections += len(frame.get("detections", []))
        persons = [p for p in frame.get("persons", []) if p.get("valid", True) and p.get("person_id_valid", True)]
        total_observations += len(persons)
        max_people = max(max_people, len(persons))
        for person in persons:
            pid = int(person["person_id"])
            track_lengths[pid] = track_lengths.get(pid, 0) + 1
            first.setdefault(pid, frame_idx)
            prev = last.get(pid)
            if prev is not None and frame_idx - prev > 1:
                gaps.setdefault(pid, []).append(frame_idx - prev - 1)
            last[pid] = frame_idx
            confs.append(float(person.get("track_confidence", 0.0)))
    all_gaps = [gap for values in gaps.values() for gap in values]
    return {
        "total_frames": len(frames),
        "total_detections": int(total_detections),
        "total_observations": int(total_observations),
        "num_tracks": len(track_lengths),
        "max_people_per_frame": int(max_people),
        "track_lengths": {str(k): int(v) for k, v in sorted(track_lengths.items())},
        "track_first_frame": {str(k): int(v) for k, v in sorted(first.items())},
        "track_last_frame": {str(k): int(v) for k, v in sorted(last.items())},
        "track_gap_count": int(len(all_gaps)),
        "track_gap_max": int(max(all_gaps) if all_gaps else 0),
        "track_gap_mean": float(sum(all_gaps) / len(all_gaps)) if all_gaps else 0.0,
        "track_confidence_mean": float(sum(confs) / len(confs)) if confs else 0.0,
    }


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected frame sidecar dict: {path}")
    return data


def _collect_tracklets(frames: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    tracklets: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        frame_idx = int(frame.get("frame_index", 0))
        for person in frame.get("persons", []):
            if not person.get("valid", True) or not person.get("person_id_valid", True) or not person.get("bbox_valid", True):
                continue
            pid = int(person["person_id"])
            person["_frame_index_for_postprocess"] = frame_idx
            tracklets.setdefault(pid, []).append(person)
    for observations in tracklets.values():
        observations.sort(key=lambda item: int(item["_frame_index_for_postprocess"]))
    return tracklets


def _build_stitch_candidates(
    tracklets: dict[int, list[dict[str, Any]]],
    max_gap: int,
    center_thresh: float,
    size_log_thresh: float,
    min_score: float,
) -> list[StitchCandidate]:
    candidates: list[StitchCandidate] = []
    for source_id, source_obs in tracklets.items():
        if not source_obs:
            continue
        source_last = source_obs[-1]
        source_last_frame = int(source_last["_frame_index_for_postprocess"])
        for target_id, target_obs in tracklets.items():
            if source_id == target_id or not target_obs:
                continue
            target_first = target_obs[0]
            target_first_frame = int(target_first["_frame_index_for_postprocess"])
            gap = target_first_frame - source_last_frame - 1
            if gap < 0 or gap > max_gap:
                continue
            pred_box = _predict_box_at(source_obs, gap + 1)
            target_box = _box(target_first)
            center_norm = _center_distance_norm(pred_box, target_box)
            size_delta = _size_log_delta(pred_box, target_box)
            pred_iou = _iou(pred_box, target_box)
            if center_norm > center_thresh or size_delta > size_log_thresh:
                continue
            center_score = max(0.0, 1.0 - center_norm / max(center_thresh, 1e-6))
            size_score = max(0.0, 1.0 - size_delta / max(size_log_thresh, 1e-6))
            gap_score = max(0.0, 1.0 - gap / max(max_gap + 1, 1))
            score = 0.50 * center_score + 0.25 * size_score + 0.15 * pred_iou + 0.10 * gap_score
            if score >= min_score:
                candidates.append(StitchCandidate(source_id, target_id, score, gap, center_norm, size_delta, pred_iou))
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _select_stitches(tracklets: dict[int, list[dict[str, Any]]], candidates: list[StitchCandidate]) -> dict[int, int]:
    parent = {track_id: track_id for track_id in tracklets}
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for candidate in candidates:
        if candidate.source_id in used_sources or candidate.target_id in used_targets:
            continue
        if _find(parent, candidate.source_id) == _find(parent, candidate.target_id):
            continue
        parent[_find(parent, candidate.target_id)] = _find(parent, candidate.source_id)
        used_sources.add(candidate.source_id)
        used_targets.add(candidate.target_id)
    return {track_id: _find(parent, track_id) for track_id in tracklets}


def _build_final_id_map(tracklets: dict[int, list[dict[str, Any]]], raw_to_group: dict[int, int], compact_ids: bool) -> dict[int, int]:
    if not compact_ids:
        return {raw_id: int(group_id) for raw_id, group_id in raw_to_group.items()}
    groups: dict[int, list[int]] = {}
    for raw_id, group_id in raw_to_group.items():
        groups.setdefault(group_id, []).append(raw_id)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            min(int(tracklets[raw_id][0]["_frame_index_for_postprocess"]) for raw_id in item[1]),
            min(item[1]),
        ),
    )
    group_to_final = {group_id: idx + 1 for idx, (group_id, _) in enumerate(ordered)}
    return {raw_id: group_to_final[group_id] for raw_id, group_id in raw_to_group.items()}


def _rewrite_frames(frames: list[dict[str, Any]], final_id_map: dict[int, int]) -> int:
    changed = 0
    for frame in frames:
        for person in frame.get("persons", []):
            if "person_id" not in person:
                continue
            raw_id = int(person["person_id"])
            final_id = int(final_id_map.get(raw_id, raw_id))
            person.pop("_frame_index_for_postprocess", None)
            if final_id != raw_id:
                person.setdefault("track_id_before_postprocess", raw_id)
                person["person_id"] = final_id
                changed += 1
    return changed


def _rewrite_observations_jsonl(path: Path, frames: list[dict[str, Any]]) -> None:
    if not path.parent.is_dir():
        return
    with path.open("w", encoding="utf-8") as file:
        for frame in frames:
            for person in frame.get("persons", []):
                file.write(json.dumps(person, ensure_ascii=False) + "\n")


def _predict_box_at(observations: list[dict[str, Any]], steps: int) -> np.ndarray:
    last = _box(observations[-1])
    if len(observations) < 2:
        return last
    prev = _box(observations[-2])
    frame_delta = max(
        int(observations[-1]["_frame_index_for_postprocess"]) - int(observations[-2]["_frame_index_for_postprocess"]),
        1,
    )
    velocity = (last - prev) / float(frame_delta)
    return last + velocity * float(max(steps, 1))


def _box(person: dict[str, Any]) -> np.ndarray:
    return np.asarray(person["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)


def _center_distance_norm(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float32)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
    scale = 0.5 * (_diag(a) + _diag(b))
    return float(np.linalg.norm(ac - bc) / max(scale, 1e-6))


def _size_log_delta(a: np.ndarray, b: np.ndarray) -> float:
    aw = max(float(a[2] - a[0]), 1e-6)
    ah = max(float(a[3] - a[1]), 1e-6)
    bw = max(float(b[2] - b[0]), 1e-6)
    bh = max(float(b[3] - b[1]), 1e-6)
    return float(abs(math.log(aw / bw)) + abs(math.log(ah / bh)))


def _diag(box: np.ndarray) -> float:
    return float(np.hypot(max(float(box[2] - box[0]), 0.0), max(float(box[3] - box[1]), 0.0)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _find(parent: dict[int, int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item
