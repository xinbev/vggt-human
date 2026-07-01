from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.utils.rotation import axis_angle_to_rot6d


IMAGE_KEYS = ("imgname", "image_path", "img_path", "image", "image_name", "filename", "frame_name")
POSE_KEYS = ("pose_cam", "smplx_pose_cam", "smpl_pose_cam", "pose", "smplx_pose", "smpl_pose", "poses")
ROOT_KEYS = ("root_orient", "global_orient", "smplx_root_pose", "smpl_root_pose")
BODY_KEYS = ("body_pose", "smplx_body_pose", "smpl_body_pose")
BETA_KEYS = ("shape", "betas", "beta", "smplx_shape", "smpl_shape")
TRANSL_KEYS = ("trans_cam", "transl_cam", "cam_trans", "trans", "translation", "smplx_transl", "smpl_transl")
CAM_EXT_KEYS = ("cam_ext", "camera_ext", "extrinsics", "cam_extrinsics")
INTRINSIC_KEYS = ("cam_int", "cam_intrinsics", "intrinsics", "K", "camera_intrinsics")
BBOX_KEYS = ("bbox", "bbox_xyxy", "bboxes", "person_bbox", "box")
CENTER_KEYS = ("center", "bbox_center")
SCALE_KEYS = ("scale", "bbox_scale")
PROJ_VERTS_KEYS = ("proj_verts", "projected_vertices", "verts2d", "vertices2d")
J2D_KEYS = ("joints2d", "joints_2d", "keypoints2d", "gtkps", "j2d")
PERSON_ID_KEYS = ("person_id", "person_ids", "track_id", "subject_id", "sub")


class HFBedlamDataset(Dataset):
    """Read HuggingFace BEDLAM images plus all_npz annotations.

    The expected server layout is:

    ``images_root/<scene>/png/<seq_name>/<seq_name>_<frame>.png``
    ``npz_root/<scene>.npz``

    The raw BEDLAM NPZ variants are not perfectly standardized, so this loader
    accepts several common key aliases and raises a clear error when a required
    field is missing.
    """

    def __init__(
        self,
        images_root: str | Path,
        npz_root: str | Path,
        sequence_length: int = 1,
        stride: int = 1,
        image_size: int = 518,
        max_humans: int = 20,
        require_boxes: bool = True,
        require_smpl: bool = True,
        bbox_expand: float = 0.15,
        transl_add_cam_ext: bool = True,
        max_npz_files: int = 0,
        max_frames: int = 0,
    ) -> None:
        super().__init__()
        self.images_root = Path(images_root).expanduser()
        self.npz_root = Path(npz_root).expanduser()
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.max_humans = int(max_humans)
        self.require_boxes = bool(require_boxes)
        self.require_smpl = bool(require_smpl)
        self.bbox_expand = float(bbox_expand)
        self.transl_add_cam_ext = bool(transl_add_cam_ext)
        self.max_npz_files = int(max_npz_files)
        self.max_frames = int(max_frames)
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"HF BEDLAM images root not found: {self.images_root}")
        if not self.npz_root.is_dir():
            raise FileNotFoundError(f"HF BEDLAM NPZ root not found: {self.npz_root}")

        self._npz_files = sorted(self.npz_root.glob("*.npz"))
        if self.max_npz_files > 0:
            self._npz_files = self._npz_files[: self.max_npz_files]
        if not self._npz_files:
            raise RuntimeError(f"No .npz annotation files found under {self.npz_root}")

        self._frames = self._build_frames()
        self._sequences = self._build_sequence_index()
        self._index: list[tuple[int, int]] = []
        for seq_idx, (_, frame_ids) in enumerate(self._sequences):
            max_start = len(frame_ids) - (self.sequence_length - 1) * self.stride
            for frame_idx in range(max(max_start, 0)):
                self._index.append((seq_idx, frame_idx))
        if not self._index:
            raise RuntimeError(
                "No HF BEDLAM windows found. "
                f"frames={len(self._frames)} sequence_length={self.sequence_length} stride={self.stride}"
            )
        self._cache_path: Path | None = None
        self._cache_data: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start_idx = self._index[idx]
        _, frame_ids = self._sequences[seq_idx]
        selected = [frame_ids[start_idx + step * self.stride] for step in range(self.sequence_length)]

        images = []
        intrinsics = []
        persons_per_frame = []
        for frame_id in selected:
            frame = self._frames[frame_id]
            image, orig_hw = _load_rgb_tensor(frame["image_path"], self.image_size)
            images.append(image)
            data = self._load_npz(frame["npz_path"])
            intrinsics.append(_scale_intrinsics(_frame_intrinsics(data, frame["person_indices"][0], self.image_size), orig_hw, self.image_size))
            persons = [_load_person(data, person_idx, orig_hw, self.bbox_expand, self.transl_add_cam_ext) for person_idx in frame["person_indices"]]
            persons_per_frame.append(persons)

        targets = _build_targets(persons_per_frame, self.max_humans, self.require_boxes, self.require_smpl)
        intrinsics_tensor = torch.stack(intrinsics, dim=0)
        return {
            "images": torch.stack(images, dim=0),
            "gt_depth": torch.zeros(self.sequence_length, 1, self.image_size, self.image_size, dtype=torch.float32),
            "K_scal3r": intrinsics_tensor,
            "gt_intrinsics": intrinsics_tensor,
            **targets,
        }

    def _build_frames(self) -> dict[str, dict[str, Any]]:
        frames: dict[str, dict[str, Any]] = {}
        allowed_limited_frames: set[str] = set()
        for npz_path in self._npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                image_key = _first_key(data, IMAGE_KEYS)
                if image_key is None:
                    raise KeyError(
                        f"{npz_path} has no image-name key. "
                        f"Tried {IMAGE_KEYS}. Run scripts/diagnostics/inspect_hf_bedlam_npz.sh and send the keys."
                    )
                names = _as_string_array(data[image_key])
                for person_idx, raw_name in enumerate(names):
                    rel = _normalize_image_relpath(raw_name, npz_path.stem)
                    image_path = _resolve_image_path(self.images_root, rel, npz_path.stem)
                    frame_key = _frame_key(rel, npz_path.stem)
                    if self.max_frames > 0:
                        if frame_key not in allowed_limited_frames and len(allowed_limited_frames) >= self.max_frames:
                            continue
                        allowed_limited_frames.add(frame_key)
                    record = frames.setdefault(
                        frame_key,
                        {
                            "npz_path": npz_path,
                            "image_relpath": rel,
                            "image_path": image_path,
                            "person_indices": [],
                        },
                    )
                    record["person_indices"].append(person_idx)
        for key, frame in frames.items():
            if not Path(frame["image_path"]).is_file():
                raise FileNotFoundError(f"HF BEDLAM image not found for frame {key}: {frame['image_path']}")
        return frames

    def _build_sequence_index(self) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for frame_key, frame in self._frames.items():
            seq = str(Path(frame["image_relpath"]).parent)
            grouped.setdefault(seq, []).append(frame_key)
        sequences = []
        for seq, keys in sorted(grouped.items()):
            keys.sort(key=_natural_sort_key)
            sequences.append((seq, keys))
        return sequences

    def _load_npz(self, path: Path) -> dict[str, np.ndarray]:
        if self._cache_path == path and self._cache_data is not None:
            return self._cache_data
        with np.load(path, allow_pickle=True) as data:
            loaded = {key: data[key] for key in data.files}
        self._cache_path = path
        self._cache_data = loaded
        return loaded


def hf_bedlam_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty HF BEDLAM batch")
    return {key: torch.stack([_require_tensor(item[key], key) for item in batch], dim=0) for key in batch[0].keys()}


def _load_person(
    data: dict[str, np.ndarray],
    person_idx: int,
    orig_hw: tuple[int, int],
    bbox_expand: float,
    transl_add_cam_ext: bool,
) -> dict[str, Any]:
    pose_aa = _person_pose_axis_angle(data, person_idx)
    betas = _field_value(data, BETA_KEYS, person_idx, required=True).reshape(-1)[:10].astype(np.float32)
    transl = _person_translation(data, person_idx, transl_add_cam_ext)
    bbox = _person_bbox(data, person_idx, orig_hw, bbox_expand)
    person_id = _person_id(data, person_idx)
    return {
        "pose_aa": pose_aa,
        "betas": betas,
        "transl_cam": transl,
        "bbox_cxcywh_norm": bbox,
        "bbox_valid": bbox is not None,
        "person_id": person_id,
        "person_id_valid": person_id >= 0,
    }


def _person_translation(data: dict[str, np.ndarray], person_idx: int, transl_add_cam_ext: bool) -> np.ndarray:
    transl = _field_value(data, TRANSL_KEYS, person_idx, required=True).reshape(-1)[:3].astype(np.float32)
    if not transl_add_cam_ext:
        return transl
    cam_ext = _field_value(data, CAM_EXT_KEYS, person_idx, required=False)
    if cam_ext is None:
        return transl
    cam_ext = np.asarray(cam_ext, dtype=np.float32).reshape(4, 4)
    return (transl + cam_ext[:3, 3].astype(np.float32)).astype(np.float32)


def _person_pose_axis_angle(data: dict[str, np.ndarray], person_idx: int) -> np.ndarray:
    pose = _field_value(data, POSE_KEYS, person_idx, required=False)
    if pose is not None:
        pose = pose.reshape(-1).astype(np.float32)
        if pose.size < 66:
            raise ValueError(f"HF BEDLAM pose vector must have at least 66 values, got {pose.size}")
        root = pose[:3]
        body = pose[3 : 3 + 21 * 3]
        return np.concatenate([root, body], axis=0).astype(np.float32)
    root = _field_value(data, ROOT_KEYS, person_idx, required=True).reshape(-1)[:3].astype(np.float32)
    body = _field_value(data, BODY_KEYS, person_idx, required=True).reshape(-1)[: 21 * 3].astype(np.float32)
    if body.size < 21 * 3:
        raise ValueError(f"HF BEDLAM body pose must have at least 63 values, got {body.size}")
    return np.concatenate([root, body], axis=0).astype(np.float32)


def _person_bbox(data: dict[str, np.ndarray], person_idx: int, orig_hw: tuple[int, int], bbox_expand: float) -> np.ndarray | None:
    bbox = _field_value(data, BBOX_KEYS, person_idx, required=False)
    if bbox is not None:
        return _bbox_to_cxcywh_norm(np.asarray(bbox).reshape(-1)[:4], orig_hw, bbox_expand)
    proj_verts = _field_value(data, PROJ_VERTS_KEYS, person_idx, required=False)
    if proj_verts is not None:
        box = _points_to_cxcywh_norm(proj_verts, orig_hw, bbox_expand)
        if box is not None:
            return box
    j2d = _field_value(data, J2D_KEYS, person_idx, required=False)
    if j2d is not None:
        box = _points_to_cxcywh_norm(j2d, orig_hw, bbox_expand)
        if box is not None:
            return box
    center = _field_value(data, CENTER_KEYS, person_idx, required=False)
    scale = _field_value(data, SCALE_KEYS, person_idx, required=False)
    if center is not None and scale is not None:
        return _center_scale_to_cxcywh_norm(center, scale, orig_hw)
    return None


def _points_to_cxcywh_norm(points_value: np.ndarray, orig_hw: tuple[int, int], bbox_expand: float) -> np.ndarray | None:
    points = np.asarray(points_value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        points = points.reshape(-1, points.shape[-1])
    xy = points[:, :2]
    if points.shape[1] >= 3:
        valid = points[:, 2] > 0.2
    else:
        valid = np.isfinite(xy).all(axis=1)
    valid &= np.isfinite(xy).all(axis=1)
    if not bool(valid.any()):
        return None
    x1, y1 = xy[valid].min(axis=0)
    x2, y2 = xy[valid].max(axis=0)
    return _xyxy_to_cxcywh_norm(np.asarray([x1, y1, x2, y2], dtype=np.float32), orig_hw, bbox_expand)


def _person_id(data: dict[str, np.ndarray], person_idx: int) -> int:
    value = _field_value(data, PERSON_ID_KEYS, person_idx, required=False)
    if value is None:
        return int(person_idx)
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return int(person_idx)


def _frame_intrinsics(data: dict[str, np.ndarray], person_idx: int, image_size: int) -> np.ndarray:
    value = _field_value(data, INTRINSIC_KEYS, person_idx, required=False)
    if value is None:
        focal = float(image_size)
        center = (float(image_size) - 1.0) * 0.5
        return np.asarray([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.size >= 4:
        fx, fy, cx, cy = arr.reshape(-1)[:4]
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported intrinsics value shape: {arr.shape}")


def _field_value(data: dict[str, np.ndarray], aliases: tuple[str, ...], person_idx: int, required: bool) -> np.ndarray | None:
    key = _first_key(data, aliases)
    if key is None:
        if required:
            raise KeyError(f"HF BEDLAM annotation missing required keys {aliases}; available={sorted(data.keys())}")
        return None
    value = np.asarray(data[key])
    if value.ndim == 0:
        return value
    if value.ndim >= 1 and value.shape[0] > person_idx:
        return np.asarray(value[person_idx])
    if value.shape == (3, 3):
        return value
    return value


def _first_key(data: Any, aliases: tuple[str, ...]) -> str | None:
    keys = set(data.files) if hasattr(data, "files") else set(data.keys())
    for key in aliases:
        if key in keys:
            return key
    return None


def _as_string_array(value: np.ndarray) -> list[str]:
    arr = np.asarray(value)
    return [str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in arr.reshape(-1).tolist()]


def _normalize_image_relpath(raw_name: str, scene_name: str) -> str:
    text = str(raw_name).replace("\\", "/").strip()
    markers = ("training_images/", "validation_images/", "test_images/")
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[1]
    text = text.lstrip("./")
    if text.startswith(scene_name + "/"):
        return text
    if text.startswith("png/"):
        return f"{scene_name}/{text}"
    if re.match(r"seq_\d+/", text):
        return f"{scene_name}/png/{text}"
    if text.endswith(".png") and "/" not in text:
        seq = text.rsplit("_", 1)[0]
        return f"{scene_name}/png/{seq}/{text}"
    return text


def _resolve_image_path(images_root: Path, rel: str, scene_name: str) -> Path:
    direct = images_root / rel
    if direct.is_file():
        return direct
    alt = images_root / scene_name / rel
    if alt.is_file():
        return alt
    return direct


def _frame_key(rel: str, scene_name: str) -> str:
    path = Path(rel)
    return f"{scene_name}/{path.parent.as_posix()}/{path.stem}"


def _natural_sort_key(text: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _load_rgb_tensor(path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"HF BEDLAM RGB frame not found: {path}")
    image = Image.open(path).convert("RGB")
    orig_hw = (image.height, image.width)
    image = image.resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_hw


def _scale_intrinsics(intrinsics: np.ndarray, orig_hw: tuple[int, int], size: int) -> torch.Tensor:
    src_h, src_w = orig_hw
    scaled = np.asarray(intrinsics, dtype=np.float32).copy().reshape(3, 3)
    scaled[0, 0] *= float(size) / float(max(src_w, 1))
    scaled[0, 2] *= float(size) / float(max(src_w, 1))
    scaled[1, 1] *= float(size) / float(max(src_h, 1))
    scaled[1, 2] *= float(size) / float(max(src_h, 1))
    return torch.from_numpy(scaled)


def _bbox_to_cxcywh_norm(raw_box: np.ndarray, orig_hw: tuple[int, int], expand: float) -> np.ndarray:
    values = np.asarray(raw_box, dtype=np.float32).reshape(4)
    x0, y0, a, b = [float(v) for v in values]
    h, w = orig_hw
    if a > x0 and b > y0:
        xyxy = np.asarray([x0, y0, a, b], dtype=np.float32)
    else:
        xyxy = np.asarray([x0, y0, x0 + max(a, 0.0), y0 + max(b, 0.0)], dtype=np.float32)
    return _xyxy_to_cxcywh_norm(xyxy, orig_hw, expand)


def _center_scale_to_cxcywh_norm(center: np.ndarray, scale: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    h, w = orig_hw
    center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
    scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
    if center_arr.size < 2 or scale_arr.size < 1:
        raise ValueError(f"Invalid center/scale bbox values: center={center_arr.shape} scale={scale_arr.shape}")
    cx, cy = float(center_arr[0]), float(center_arr[1])
    side = float(scale_arr[0]) * 200.0
    side = max(side, 1.0)
    x1 = cx - 0.5 * side
    y1 = cy - 0.5 * side
    x2 = cx + 0.5 * side
    y2 = cy + 0.5 * side
    x1 = max(x1, 0.0)
    y1 = max(y1, 0.0)
    x2 = min(x2, float(max(w - 1, 1)))
    y2 = min(y2, float(max(h - 1, 1)))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return np.asarray([(x1 + 0.5 * bw) / max(w, 1), (y1 + 0.5 * bh) / max(h, 1), bw / max(w, 1), bh / max(h, 1)], dtype=np.float32)


def _xyxy_to_cxcywh_norm(xyxy: np.ndarray, orig_hw: tuple[int, int], expand: float) -> np.ndarray:
    h, w = orig_hw
    x1, y1, x2, y2 = [float(v) for v in xyxy.reshape(4)]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = 0.5 * float(expand) * bw
    pad_y = 0.5 * float(expand) * bh
    x1 = max(x1 - pad_x, 0.0)
    y1 = max(y1 - pad_y, 0.0)
    x2 = min(x2 + pad_x, float(max(w - 1, 1)))
    y2 = min(y2 + pad_y, float(max(h - 1, 1)))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return np.asarray([(x1 + 0.5 * bw) / max(w, 1), (y1 + 0.5 * bh) / max(h, 1), bw / max(w, 1), bh / max(h, 1)], dtype=np.float32)


def _build_targets(
    persons_per_frame: list[list[dict[str, Any]]],
    max_humans: int,
    require_boxes: bool,
    require_smpl: bool,
) -> dict[str, torch.Tensor]:
    pose_frames = []
    beta_frames = []
    transl_frames = []
    smpl_mask_frames = []
    box_frames = []
    box_mask_frames = []
    id_frames = []
    id_mask_frames = []
    source_frames = []
    quality_frames = []
    for persons in persons_per_frame:
        poses = torch.zeros(max_humans, 144, dtype=torch.float32)
        betas = torch.zeros(max_humans, 10, dtype=torch.float32)
        transl = torch.zeros(max_humans, 3, dtype=torch.float32)
        smpl_mask = torch.zeros(max_humans, dtype=torch.bool)
        boxes = torch.zeros(max_humans, 4, dtype=torch.float32)
        boxes_mask = torch.zeros(max_humans, dtype=torch.bool)
        ids = torch.full((max_humans,), -1, dtype=torch.long)
        ids_mask = torch.zeros(max_humans, dtype=torch.bool)
        source = torch.full((max_humans,), 2, dtype=torch.long)
        quality = torch.zeros(max_humans, dtype=torch.float32)
        sorted_persons = sorted(persons, key=lambda item: float(np.asarray(item["transl_cam"]).reshape(3)[2]))
        for slot, person in enumerate(sorted_persons[:max_humans]):
            aa_22 = torch.as_tensor(person["pose_aa"], dtype=torch.float32).reshape(22, 3)
            pose_6d_22 = axis_angle_to_rot6d(aa_22).reshape(22, 6)
            identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32).expand(2, -1)
            poses[slot] = torch.cat([pose_6d_22, identity_6d], dim=0).reshape(144)
            betas[slot] = torch.as_tensor(person["betas"], dtype=torch.float32).reshape(-1)[:10]
            transl[slot] = torch.as_tensor(person["transl_cam"], dtype=torch.float32).reshape(3)
            smpl_mask[slot] = True
            ids[slot] = int(person.get("person_id", slot))
            ids_mask[slot] = True
            quality[slot] = 1.0
            if person.get("bbox_valid", False) and person.get("bbox_cxcywh_norm") is not None:
                boxes[slot] = torch.as_tensor(person["bbox_cxcywh_norm"], dtype=torch.float32).reshape(4).clamp(0.0, 1.0)
                boxes_mask[slot] = True
            elif require_boxes:
                raise ValueError("HF BEDLAM person is missing bbox/joints2d; inspect NPZ keys or set require_boxes=false.")
        if require_smpl and not bool(smpl_mask.any()):
            raise ValueError("HF BEDLAM frame has no valid SMPL person")
        pose_frames.append(poses)
        beta_frames.append(betas)
        transl_frames.append(transl)
        smpl_mask_frames.append(smpl_mask)
        box_frames.append(boxes)
        box_mask_frames.append(boxes_mask)
        id_frames.append(ids)
        id_mask_frames.append(ids_mask)
        source_frames.append(source)
        quality_frames.append(quality)
    track_ids = torch.stack(id_frames, dim=0)
    track_mask = torch.stack(id_mask_frames, dim=0)
    transl_cam = torch.stack(transl_frames, dim=0)
    return {
        "gt_pose_6d": torch.stack(pose_frames, dim=0),
        "gt_betas": torch.stack(beta_frames, dim=0),
        "gt_transl_cam": transl_cam,
        "gt_cam_trans": transl_cam,
        "smpl_mask": torch.stack(smpl_mask_frames, dim=0),
        "gt_boxes": torch.stack(box_frames, dim=0),
        "boxes_mask": torch.stack(box_mask_frames, dim=0),
        "person_ids": track_ids,
        "person_id_mask": track_mask,
        "gt_track_ids": track_ids,
        "gt_track_mask": track_mask,
        "gt_track_source": torch.stack(source_frames, dim=0),
        "gt_track_quality": torch.stack(quality_frames, dim=0),
    }


def _require_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
