"""Sequence-level SMPL dataset for the standalone temporal refiner.

Supported sources are the native EMDB ``*_data.pkl`` files and the native
3DPW ``sequenceFiles/<split>/*.pkl`` files documented in
``docs/smpl_temporal_refiner_design.md``.  The loader keeps the original
camera matrices without guessing whether they are world-to-camera or
camera-to-world; the first training stage only consumes pose/trans labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, Iterable
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from vggt_omega.utils.rotation import axis_angle_to_rot6d


@dataclass(frozen=True)
class _PersonSequence:
    dataset_name: str
    dataset_id: int
    sequence_name: str
    source_path: str
    person_id: int
    pose_aa: np.ndarray  # [T,72]
    transl: np.ndarray  # [T,3]
    betas: np.ndarray  # [10]
    valid: np.ndarray  # [T]
    intrinsics: np.ndarray  # [3,3]
    extrinsics: np.ndarray  # [T,4,4]
    frame_ids: np.ndarray  # [T]


def _as_float32(value: Any, shape_tail: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"{field} must end with shape {shape_tail}, got {array.shape}")
    return array


def _normalise_betas(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 10:
        raise ValueError(f"SMPL betas must contain at least 10 values, got {array.size}")
    return array[:10].copy()


def _normalise_extrinsics(value: Any, length: int, source: str) -> np.ndarray:
    if value is None:
        return np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0)
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4, 4):
        return np.repeat(array[None], length, axis=0)
    if array.shape == (length, 4, 4):
        return array
    raise ValueError(f"{source} camera poses must be [T,4,4] or [4,4], got {array.shape}")


def _stable_partition(key: str, validation_fraction: float) -> bool:
    """Return true when a complete person track belongs to validation."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return fraction < validation_fraction


class SMPLTemporalPickleDataset(Dataset[dict[str, torch.Tensor]]):
    """Yield one temporally contiguous person track window per item.

    Args:
        sources: A list of mappings such as ``{"name": "3dpw", "root":
            "/.../3DPW/sequenceFiles/train"}`` and ``{"name": "emdb",
            "root": "/.../emdb"}``.  EMDB roots are searched recursively for
            ``*_data.pkl``; 3DPW roots are searched recursively for ``*.pkl``.
        partition: ``"train"`` or ``"val"``.  Splitting is by complete
            person track, preventing adjacent windows from leaking between
            train and validation.
    """

    def __init__(
        self,
        sources: Iterable[dict[str, Any]] | None = None,
        window_size: int = 9,
        stride: int = 1,
        partition: str = "train",
        validation_fraction: float = 0.1,
        min_valid_frames: int | None = None,
        records: Iterable[_PersonSequence] | None = None,
        source_file_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.partition = str(partition).lower()
        self.validation_fraction = float(validation_fraction)
        self.min_valid_frames = int(min_valid_frames or self.window_size)
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3 for bidirectional refinement")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.partition not in {"train", "val"}:
            raise ValueError("partition must be 'train' or 'val'")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0,1)")

        if (sources is None) == (records is None):
            raise ValueError("Specify exactly one of sources or records")
        if records is None:
            assert sources is not None
            all_records, file_counts = self.load_records(sources)
        else:
            all_records = list(records)
            file_counts = dict(source_file_counts or {})
        self.source_file_counts = file_counts

        selected: list[_PersonSequence] = []
        for record in all_records:
            # Split an entire source sequence together.  In particular, all
            # people in a multi-person 3DPW pkl stay on the same side, so a
            # shared motion/camera sequence cannot leak through person_idx.
            is_val = _stable_partition(
                f"{record.dataset_name}:{record.source_path}:{record.sequence_name}", self.validation_fraction
            )
            if (self.partition == "val") == is_val:
                selected.append(record)
        self.records = selected
        self.index: list[tuple[int, int]] = []
        span = (self.window_size - 1) * self.stride + 1
        for record_index, record in enumerate(self.records):
            for start in range(0, max(0, len(record.pose_aa) - span + 1)):
                indices = start + np.arange(self.window_size) * self.stride
                if int(record.valid[indices].sum()) >= self.min_valid_frames:
                    self.index.append((record_index, start))
        if not self.index:
            raise RuntimeError(
                f"No valid {self.partition} windows: window_size={self.window_size}, stride={self.stride}. "
                "Check pkl roots and validity masks."
            )

    @classmethod
    def load_records(cls, sources: Iterable[dict[str, Any]]) -> tuple[list[_PersonSequence], dict[str, int]]:
        """Parse all native pkl files once and return reusable person tracks."""
        records: list[_PersonSequence] = []
        file_counts: dict[str, int] = defaultdict(int)
        for dataset_id, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TypeError("Each data source must be a mapping")
            name = str(source.get("name", "")).strip().lower()
            root = Path(str(source.get("root", "")).strip()).expanduser()
            if name not in {"3dpw", "emdb"}:
                raise ValueError(f"Unsupported temporal source {name!r}; expected '3dpw' or 'emdb'")
            if not root.is_dir():
                raise FileNotFoundError(f"{name} pickle root does not exist: {root}")
            source_records, source_file_count = cls._load_source(name, dataset_id, root)
            records.extend(source_records)
            file_counts[name] += source_file_count
        return records, dict(file_counts)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable, pre-DataLoader data audit."""
        window_counts = Counter(record_index for record_index, _ in self.index)
        by_dataset: dict[str, dict[str, int]] = {}
        sequences: dict[str, set[str]] = defaultdict(set)
        for record_index, record in enumerate(self.records):
            item = by_dataset.setdefault(
                record.dataset_name,
                {
                    "pickle_files": int(self.source_file_counts.get(record.dataset_name, 0)),
                    "person_tracks": 0,
                    "frames_total": 0,
                    "frames_valid": 0,
                    "frames_invalid": 0,
                    "windows": 0,
                },
            )
            item["person_tracks"] += 1
            item["frames_total"] += int(record.valid.size)
            item["frames_valid"] += int(record.valid.sum())
            item["frames_invalid"] += int((~record.valid).sum())
            item["windows"] += int(window_counts.get(record_index, 0))
            sequences[record.dataset_name].add(record.source_path + "::" + record.sequence_name)
        for dataset_name, item in by_dataset.items():
            item["sequences"] = len(sequences[dataset_name])
        return {
            "partition": self.partition,
            "window_size": self.window_size,
            "stride": self.stride,
            "min_valid_frames": self.min_valid_frames,
            "total_person_tracks": len(self.records),
            "total_windows": len(self.index),
            "by_dataset": dict(sorted(by_dataset.items())),
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start = self.index[index]
        record = self.records[record_index]
        indices = start + np.arange(self.window_size) * self.stride
        pose_aa = torch.from_numpy(record.pose_aa[indices].copy())
        pose_6d = axis_angle_to_rot6d(pose_aa.reshape(self.window_size, 24, 3)).reshape(self.window_size, 144)
        return {
            "target_pose_6d": pose_6d.to(dtype=torch.float32),
            "target_transl": torch.from_numpy(record.transl[indices].copy()).to(dtype=torch.float32),
            "target_betas": torch.from_numpy(np.repeat(record.betas[None], self.window_size, axis=0).copy()),
            "valid_mask": torch.from_numpy(record.valid[indices].copy()).to(dtype=torch.bool),
            "intrinsics": torch.from_numpy(np.repeat(record.intrinsics[None], self.window_size, axis=0).copy()),
            "camera_extrinsics": torch.from_numpy(record.extrinsics[indices].copy()),
            "frame_indices": torch.from_numpy(record.frame_ids[indices].copy()).to(dtype=torch.long),
            "dataset_id": torch.tensor(record.dataset_id, dtype=torch.long),
            "person_id": torch.tensor(record.person_id, dtype=torch.long),
        }

    @staticmethod
    def _load_source(name: str, dataset_id: int, root: Path) -> tuple[list[_PersonSequence], int]:
        paths = sorted(root.rglob("*_data.pkl" if name == "emdb" else "*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No {name} pickle files found below {root}")
        records: list[_PersonSequence] = []
        for path in paths:
            try:
                with path.open("rb") as file:
                    payload = pickle.load(file, encoding="latin1") if name == "3dpw" else pickle.load(file)
                parsed = _parse_emdb(payload, dataset_id, path) if name == "emdb" else _parse_3dpw(payload, dataset_id, path)
                records.extend(parsed)
            except (KeyError, TypeError, ValueError, pickle.UnpicklingError) as error:
                raise RuntimeError(f"Failed to parse {name} pkl: {path}: {error}") from error
        return records, len(paths)


def _parse_emdb(payload: dict[str, Any], dataset_id: int, path: Path) -> list[_PersonSequence]:
    smpl = payload["smpl"]
    pose = np.concatenate(
        (_as_float32(smpl["poses_root"], (3,), "EMDB smpl.poses_root"), _as_float32(smpl["poses_body"], (69,), "EMDB smpl.poses_body")),
        axis=-1,
    )
    transl = _as_float32(smpl["trans"], (3,), "EMDB smpl.trans")
    if pose.shape != (len(transl), 72):
        raise ValueError(f"EMDB pose/trans length mismatch: {pose.shape} vs {transl.shape}")
    camera = payload["camera"]
    intrinsics = _as_float32(camera["intrinsics"], (3, 3), "EMDB camera.intrinsics")
    extrinsics = _normalise_extrinsics(camera.get("extrinsics"), len(pose), "EMDB")
    valid = np.asarray(payload.get("good_frames_mask", np.ones(len(pose), dtype=bool)), dtype=bool).reshape(-1)
    if len(valid) != len(pose):
        raise ValueError("EMDB good_frames_mask length does not match SMPL sequence")
    valid &= np.isfinite(pose).all(axis=1) & np.isfinite(transl).all(axis=1)
    sequence_name = str(payload.get("name", path.stem))
    return [
        _PersonSequence(
            dataset_name="emdb",
            dataset_id=dataset_id,
            sequence_name=sequence_name,
            source_path=str(path),
            person_id=0,
            pose_aa=pose,
            transl=transl,
            betas=_normalise_betas(smpl["betas"]),
            valid=valid,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            frame_ids=np.arange(len(pose), dtype=np.int64),
        )
    ]


def _parse_3dpw(payload: dict[str, Any], dataset_id: int, path: Path) -> list[_PersonSequence]:
    poses = payload["poses"]
    trans = payload["trans"]
    betas = payload["betas"]
    valid_list = payload.get("campose_valid")
    if not (len(poses) == len(trans) == len(betas)):
        raise ValueError("3DPW poses/trans/betas person counts differ")
    intrinsics = _as_float32(payload["cam_intrinsics"], (3, 3), "3DPW cam_intrinsics")
    frame_ids = np.asarray(payload.get("img_frame_ids"), dtype=np.int64).reshape(-1)
    sequence_raw = payload.get("sequence", path.stem)
    sequence_name = str(np.asarray(sequence_raw).reshape(-1)[0])
    records: list[_PersonSequence] = []
    for person_id, (person_pose, person_trans, person_betas) in enumerate(zip(poses, trans, betas)):
        pose = _as_float32(person_pose, (72,), f"3DPW poses[{person_id}]")
        transl = _as_float32(person_trans, (3,), f"3DPW trans[{person_id}]")
        if pose.shape != (len(transl), 72):
            raise ValueError(f"3DPW person {person_id} pose/trans mismatch: {pose.shape} vs {transl.shape}")
        if len(frame_ids) != len(pose):
            raise ValueError("3DPW img_frame_ids length does not match SMPL sequence")
        valid = np.ones(len(pose), dtype=bool)
        if valid_list is not None:
            person_valid = np.asarray(valid_list[person_id], dtype=bool).reshape(-1)
            if len(person_valid) != len(pose):
                raise ValueError(f"3DPW campose_valid[{person_id}] length does not match SMPL sequence")
            valid &= person_valid
        valid &= np.isfinite(pose).all(axis=1) & np.isfinite(transl).all(axis=1)
        records.append(
            _PersonSequence(
                dataset_name="3dpw",
                dataset_id=dataset_id,
                sequence_name=sequence_name,
                source_path=str(path),
                person_id=person_id,
                pose_aa=pose,
                transl=transl,
                betas=_normalise_betas(person_betas),
                valid=valid,
                intrinsics=intrinsics,
                extrinsics=_normalise_extrinsics(payload.get("cam_poses"), len(pose), "3DPW"),
                frame_ids=frame_ids,
            )
        )
    return records
