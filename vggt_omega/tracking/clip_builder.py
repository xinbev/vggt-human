from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from .io import read_observations_jsonl


def build_clip_tensors_from_sidecar(
    sidecar_root: str | Path,
    frame_ids: list[str] | None = None,
    max_humans: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Build VGGTOmega SMPL prior tensors from video tracking sidecar files.

    Returns:
        smpl_query_boxes: [1,S,Q,4] normalized cxcywh.
        smpl_query_boxes_mask: [1,S,Q].
        smpl_track_ids: [1,S,Q].
        smpl_track_mask: [1,S,Q].
    """
    root = Path(sidecar_root).expanduser()
    frames = load_sidecar_frames(root)
    if frame_ids is not None:
        requested = set(frame_ids)
        frames = [frame for frame in frames if str(frame["frame_id"]) in requested]
    if not frames:
        raise ValueError(f"No sidecar frames found under {root}")
    frames.sort(key=lambda item: int(item.get("frame_index", 0)))

    track_ids = sorted(
        {
            int(person["person_id"])
            for frame in frames
            for person in frame.get("persons", [])
            if person.get("valid", True) and person.get("person_id_valid", True) and int(person.get("person_id", -1)) >= 0
        }
    )
    if max_humans is not None:
        track_ids = track_ids[: int(max_humans)]
    if not track_ids:
        q = int(max_humans or 1)
    else:
        q = len(track_ids)
    slot_for_id = {person_id: slot for slot, person_id in enumerate(track_ids)}
    s = len(frames)
    boxes = torch.zeros((1, s, q, 4), dtype=torch.float32, device=device)
    box_mask = torch.zeros((1, s, q), dtype=torch.bool, device=device)
    ids = torch.full((1, s, q), -1, dtype=torch.long, device=device)
    id_mask = torch.zeros((1, s, q), dtype=torch.bool, device=device)

    for frame_slot, frame in enumerate(frames):
        for person in frame.get("persons", []):
            person_id = int(person.get("person_id", -1))
            if person_id not in slot_for_id:
                continue
            slot = slot_for_id[person_id]
            ids[0, frame_slot, slot] = person_id
            if person.get("valid", True) and person.get("person_id_valid", True):
                id_mask[0, frame_slot, slot] = True
            if person.get("bbox_valid", person.get("valid", True)):
                box = torch.as_tensor(person["bbox_cxcywh_norm"], dtype=torch.float32, device=device)
                boxes[0, frame_slot, slot] = box.clamp(0.0, 1.0)
                box_mask[0, frame_slot, slot] = True

    return {
        "smpl_query_boxes": boxes,
        "smpl_query_boxes_mask": box_mask,
        "smpl_track_ids": ids,
        "smpl_track_mask": id_mask,
        "frame_ids": [str(frame["frame_id"]) for frame in frames],
        "frame_indices": [int(frame.get("frame_index", idx)) for idx, frame in enumerate(frames)],
        "slot_track_ids": track_ids,
    }


def load_sidecar_frames(root: Path) -> list[dict[str, Any]]:
    pkl_dir = root / "smpl_boxes"
    if pkl_dir.is_dir():
        frames: list[dict[str, Any]] = []
        for path in sorted(pkl_dir.glob("*.pkl")):
            with path.open("rb") as file:
                frames.append(pickle.load(file))
        return frames
    jsonl = root / "observations.jsonl"
    if jsonl.is_file():
        rows = read_observations_jsonl(jsonl)
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            frame_id = str(row["frame_id"])
            frame = grouped.setdefault(
                frame_id,
                {
                    "frame_id": frame_id,
                    "frame_index": int(row.get("frame_index", len(grouped))),
                    "image_hw": [int(row.get("image_height", 0)), int(row.get("image_width", 0))],
                    "persons": [],
                },
            )
            frame["persons"].append(row)
        return list(grouped.values())
    raise FileNotFoundError(f"Expected {pkl_dir} or {jsonl}")
