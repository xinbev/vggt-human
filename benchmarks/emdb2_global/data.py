"""Native EMDB-2 annotation loading and gender-specific world-SMPL decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np


# Adapted from the public Human3R/GVHMR EMDB-2 protocol list. Keep order stable.
EMDB2_ANNOTATIONS = (
    "P2/19_indoor_walk_off_mvs/P2_19_indoor_walk_off_mvs_data.pkl",
    "P3/29_outdoor_stairs_up/P3_29_outdoor_stairs_up_data.pkl",
    "P4/35_indoor_walk/P4_35_indoor_walk_data.pkl",
    "P7/55_outdoor_walk/P7_55_outdoor_walk_data.pkl",
    "P9/80_outdoor_walk_big_circle/P9_80_outdoor_walk_big_circle_data.pkl",
    "P9/77_outdoor_stairs_up/P9_77_outdoor_stairs_up_data.pkl",
    "P9/79_outdoor_walk_rectangle/P9_79_outdoor_walk_rectangle_data.pkl",
    "P7/57_outdoor_rock_chair/P7_57_outdoor_rock_chair_data.pkl",
    "P2/24_outdoor_long_walk/P2_24_outdoor_long_walk_data.pkl",
    "P3/30_outdoor_stairs_down/P3_30_outdoor_stairs_down_data.pkl",
    "P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl",
    "P6/49_outdoor_big_stairs_down/P6_49_outdoor_big_stairs_down_data.pkl",
    "P9/78_outdoor_stairs_up_down/P9_78_outdoor_stairs_up_down_data.pkl",
    "P7/56_outdoor_stairs_up_down/P7_56_outdoor_stairs_up_down_data.pkl",
    "P2/20_outdoor_walk/P2_20_outdoor_walk_data.pkl",
    "P3/27_indoor_walk_off_mvs/P3_27_indoor_walk_off_mvs_data.pkl",
    "P4/37_outdoor_run_circle/P4_37_outdoor_run_circle_data.pkl",
    "P5/40_indoor_walk_big_circle/P5_40_indoor_walk_big_circle_data.pkl",
    "P6/48_outdoor_walk_downhill/P6_48_outdoor_walk_downhill_data.pkl",
    "P0/09_outdoor_walk/P0_09_outdoor_walk_data.pkl",
    "P3/28_outdoor_walk_lunges/P3_28_outdoor_walk_lunges_data.pkl",
    "P7/58_outdoor_parcours/P7_58_outdoor_parcours_data.pkl",
    "P7/61_outdoor_sit_lie_walk/P7_61_outdoor_sit_lie_walk_data.pkl",
    "P8/64_outdoor_skateboard/P8_64_outdoor_skateboard_data.pkl",
    "P8/65_outdoor_walk_straight/P8_65_outdoor_walk_straight_data.pkl",
)


@dataclass(frozen=True)
class EMDB2Sequence:
    name: str
    annotation_path: Path
    gender: str
    frame_count: int
    good_frame_mask: np.ndarray
    poses_root_world: np.ndarray
    poses_body: np.ndarray
    betas: np.ndarray
    transl_world: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray

    @property
    def good_frame_indices(self) -> np.ndarray:
        return np.flatnonzero(self.good_frame_mask).astype(np.int64, copy=False)

    @property
    def safe_name(self) -> str:
        return self.name.replace("/", "_")


def load_emdb2_sequences(root: str | Path, sequence_filter: str = "") -> list[EMDB2Sequence]:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(f"EMDB root not found: {root_path}")
    query = str(sequence_filter).lower().strip()
    sequences: list[EMDB2Sequence] = []
    missing: list[Path] = []
    for relative in EMDB2_ANNOTATIONS:
        path = root_path / relative
        if not path.is_file():
            missing.append(path)
            continue
        sequence = load_emdb_sequence(path)
        if query and query not in sequence.name.lower() and query not in sequence.safe_name.lower():
            continue
        sequences.append(sequence)
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"EMDB-2 protocol is incomplete below {root_path}: missing={len(missing)}\n{preview}"
        )
    if not sequences:
        raise ValueError(f"sequence_filter={sequence_filter!r} matched no EMDB-2 protocol sequence")
    return sequences


def load_emdb_sequence(path: str | Path) -> EMDB2Sequence:
    annotation_path = Path(path)
    with annotation_path.open("rb") as file:
        raw: dict[str, Any] = pickle.load(file, encoding="latin1")
    required = ("n_frames", "good_frames_mask", "camera", "smpl", "gender", "name")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{annotation_path} missing EMDB keys: {missing}")
    frame_count = int(raw["n_frames"])
    if not bool(raw.get("emdb2", False)):
        raise ValueError(f"{annotation_path} is not marked as an EMDB-2 sequence")
    mask = np.asarray(raw["good_frames_mask"], dtype=bool).reshape(-1)
    smpl = raw["smpl"]
    camera = raw["camera"]
    root = np.asarray(smpl["poses_root"], dtype=np.float32).reshape(frame_count, 3)
    body = np.asarray(smpl["poses_body"], dtype=np.float32).reshape(frame_count, 69)
    transl = np.asarray(smpl["trans"], dtype=np.float32).reshape(frame_count, 3)
    betas_raw = np.asarray(smpl["betas"], dtype=np.float32).reshape(-1)[:10]
    if mask.shape[0] != frame_count:
        raise ValueError(f"good_frames_mask length {mask.shape[0]} != n_frames {frame_count}")
    world_to_camera = np.asarray(camera["extrinsics"], dtype=np.float32).reshape(frame_count, 4, 4)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32).reshape(3, 3)
    name = str(raw["name"]).replace("_", "/", 1)
    gender = "male" if str(raw["gender"]).lower().startswith("m") else "female"
    return EMDB2Sequence(
        name=name,
        annotation_path=annotation_path,
        gender=gender,
        frame_count=frame_count,
        good_frame_mask=mask,
        poses_root_world=root,
        poses_body=body,
        betas=betas_raw,
        transl_world=transl,
        intrinsics=intrinsics,
        world_to_camera=world_to_camera,
    )


def decode_gt_world_joints(
    sequence: EMDB2Sequence,
    frame_indices: np.ndarray,
    smpl_layer: Any,
    device: Any,
    chunk_size: int = 512,
) -> np.ndarray:
    """Decode native EMDB world SMPL into this project's SMPL-24 joints."""
    import torch

    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.empty((0, 24, 3), dtype=np.float32)
    if indices.min() < 0 or indices.max() >= sequence.frame_count:
        raise IndexError(f"Frame index outside {sequence.name}: {indices.min()}..{indices.max()}")
    pose = np.concatenate(
        [sequence.poses_root_world[indices], sequence.poses_body[indices]], axis=-1
    ).astype(np.float32, copy=False)
    betas = np.repeat(sequence.betas[None], indices.size, axis=0).astype(np.float32, copy=False)
    transl = sequence.transl_world[indices].astype(np.float32, copy=False)
    decoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, max(int(chunk_size), 1)):
            end = min(indices.size, start + max(int(chunk_size), 1))
            pose_t = torch.from_numpy(pose[start:end]).to(device=device, dtype=torch.float32)
            beta_t = torch.from_numpy(betas[start:end]).to(device=device, dtype=torch.float32)
            transl_t = torch.from_numpy(transl[start:end]).to(device=device, dtype=torch.float32)
            _, joints = smpl_layer(pose_t, beta_t)
            joints_world = joints[:, :24] + transl_t[:, None, :]
            decoded.append(joints_world.detach().float().cpu().numpy())
    return np.concatenate(decoded, axis=0).astype(np.float32, copy=False)
