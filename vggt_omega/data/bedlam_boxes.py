from __future__ import annotations

import builtins
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch


BBOX_KEYS = ("bbox_xyxy", "bbox", "box", "person_bbox", "bbox_xywh")
J2D_KEYS = ("j2ds", "joints2d", "joints_2d", "keypoints2d", "keypoints_2d")
PERSON_ID_KEYS = ("person_id", "id", "track_id", "trackid", "pid")


def extract_person_id(person: dict[str, Any]) -> tuple[int, bool]:
    for key in PERSON_ID_KEYS:
        if key not in person:
            continue
        value = person[key]
        if value is None:
            continue
        try:
            return int(value), True
        except (TypeError, ValueError):
            return stable_string_id(str(value)), True
    return -1, False


def stable_string_id(value: str) -> int:
    h = 2166136261
    for char in value:
        h ^= ord(char)
        h = (h * 16777619) & 0x7FFFFFFF
    return int(h)


def extract_person_box(person: dict[str, Any], image_hw: tuple[int, int], bbox_format: str = "auto") -> tuple[torch.Tensor | None, str]:
    for key in BBOX_KEYS:
        if key not in person:
            continue
        box = _as_box_tensor(person[key])
        if box is None:
            continue
        fmt = bbox_format
        if fmt == "auto":
            fmt = "xywh" if key.endswith("xywh") else _infer_box_format(box, image_hw)
        xyxy = _box_to_xyxy_pixels(box, image_hw, fmt)
        repaired = repair_xyxy_box(xyxy, image_hw)
        if repaired is not None:
            return xyxy_to_cxcywh_norm(repaired, image_hw), key
    return None, "missing"


def extract_j2d_box(person: dict[str, Any], image_hw: tuple[int, int]) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    for key in J2D_KEYS:
        if key not in person:
            continue
        j2ds = _as_j2d_tensor(person[key])
        if j2ds is None or j2ds.numel() == 0:
            continue
        coords = j2ds[..., :2].float()
        visibility = _j2d_visibility(j2ds, image_hw)
        if not visibility.any():
            continue
        visible_coords = coords[visibility]
        x1y1 = visible_coords.amin(dim=0)
        x2y2 = visible_coords.amax(dim=0)
        xyxy = torch.cat([x1y1, x2y2], dim=0)
        repaired = repair_xyxy_box(xyxy, image_hw)
        if repaired is None:
            continue
        mask = visibility[:, None].expand(-1, 2).contiguous()
        return xyxy_to_cxcywh_norm(repaired, image_hw), mask, key
    return None, None, "missing"


def repair_xyxy_box(box: torch.Tensor, image_hw: tuple[int, int], min_size: float = 1.0) -> torch.Tensor | None:
    h, w = image_hw
    box = box.float().reshape(4).clone()
    if not torch.isfinite(box).all():
        return None
    x1 = torch.minimum(box[0], box[2]).clamp(0.0, float(w - 1))
    y1 = torch.minimum(box[1], box[3]).clamp(0.0, float(h - 1))
    x2 = torch.maximum(box[0], box[2]).clamp(0.0, float(w - 1))
    y2 = torch.maximum(box[1], box[3]).clamp(0.0, float(h - 1))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return torch.stack([x1, y1, x2, y2])


def xyxy_to_cxcywh_norm(box: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    x1, y1, x2, y2 = box.float().unbind(-1)
    cx = (x1 + x2) * 0.5 / float(w)
    cy = (y1 + y2) * 0.5 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return torch.stack([cx, cy, bw, bh], dim=-1).clamp(0.0, 1.0)


def cxcywh_norm_to_xyxy(box: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    cx, cy, bw, bh = box.float().unbind(-1)
    half_w = bw * float(w) * 0.5
    half_h = bh * float(h) * 0.5
    x = cx * float(w)
    y = cy * float(h)
    return torch.stack([x - half_w, y - half_h, x + half_w, y + half_h], dim=-1)


def extract_best_box(person: dict[str, Any], image_hw: tuple[int, int]) -> dict[str, Any]:
    bbox, source = extract_person_box(person, image_hw)
    j2ds = None
    j2ds_mask = None
    if bbox is None:
        bbox, j2ds_mask, source = extract_j2d_box(person, image_hw)
        j2ds = _first_j2d(person)
    person_id, person_id_valid = extract_person_id(person)
    xyxy = cxcywh_norm_to_xyxy(bbox, image_hw) if bbox is not None else torch.zeros(4)
    out: dict[str, Any] = {
        "person_id": person_id,
        "person_id_valid": person_id_valid,
        "bbox_cxcywh_norm": bbox.tolist() if bbox is not None else [0.0, 0.0, 0.0, 0.0],
        "bbox_xyxy_pixels": xyxy.tolist(),
        "bbox_valid": bbox is not None,
        "bbox_source": source,
    }
    if j2ds is not None:
        out["j2ds"] = j2ds[..., :2].float().tolist()
    if j2ds_mask is not None:
        out["j2ds_mask"] = j2ds_mask.bool().tolist()
    return out


def extract_visibility_stats(
    person: dict[str, Any],
    image_hw: tuple[int, int],
    bbox_xyxy_pixels: Any | None = None,
) -> dict[str, Any]:
    """Return lightweight visible-person diagnostics for BEDLAM sidecars."""
    j2ds = None
    j2d_source = "missing"
    for key in J2D_KEYS:
        if key not in person:
            continue
        j2ds = _as_j2d_tensor(person[key])
        if j2ds is not None:
            j2d_source = key
            break

    visible_joints = 0
    total_joints = 0
    if j2ds is not None:
        visible = _j2d_visibility(j2ds, image_hw)
        visible_joints = int(visible.sum().item())
        total_joints = int(visible.numel())

    bbox_area = 0.0
    if bbox_xyxy_pixels is not None:
        try:
            xyxy = torch.as_tensor(bbox_xyxy_pixels, dtype=torch.float32).reshape(4)
            repaired = repair_xyxy_box(xyxy, image_hw)
            if repaired is not None:
                width = max(float(repaired[2] - repaired[0]), 0.0)
                height = max(float(repaired[3] - repaired[1]), 0.0)
                bbox_area = float(width * height)
        except (TypeError, ValueError):
            bbox_area = 0.0

    return {
        "has_j2d_visibility": j2ds is not None,
        "j2d_source": j2d_source,
        "visible_joints": visible_joints,
        "total_joints": total_joints,
        "bbox_area_pixels": bbox_area,
    }


def build_smpl_model_cache(
    model_dir: str | Path,
    device: torch.device | str = "cpu",
    genders: tuple[str, ...] = ("neutral", "male", "female"),
) -> dict[str, Any]:
    """Build gender-specific SMPL models used by BEDLAM projection preprocessing."""
    _ensure_legacy_smpl_compat()
    try:
        import smplx
    except ImportError as exc:
        raise ImportError("--use-smpl-projection requires the `smplx` package") from exc

    root = resolve_smpl_model_dir(model_dir)
    models: dict[str, Any] = {}
    for gender in genders:
        try:
            models[gender.lower()] = smplx.create(str(root), "smpl", gender=gender, num_betas=10).to(device).eval()
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                raise ImportError(
                    "--use-smpl-projection requires scipy because smplx loads SMPL .pkl files "
                    "that contain scipy sparse matrices. Install scipy in the active environment."
                ) from exc
            raise
    return models


def resolve_smpl_model_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "smpl" / "SMPL_NEUTRAL.pkl").is_file():
        return p
    if (p / "SMPL_NEUTRAL.pkl").is_file():
        return p.parent
    raise FileNotFoundError(
        "Could not locate SMPL model files under "
        f"{p}. Expected either <dir>/smpl/SMPL_NEUTRAL.pkl or <dir>/SMPL_NEUTRAL.pkl."
    )


def optional_smpl_projection_box(
    person: dict[str, Any],
    image_hw: tuple[int, int],
    smpl_model_dir: str | Path,
    intrinsics: Any,
    smpl_models: dict[str, Any] | None = None,
    projection_source: str = "vertices",
) -> dict[str, Any]:
    """Project a BEDLAM SMPL body to image pixels and return sidecar bbox data.

    BEDLAM preprocessing stores ``smplx_transl`` in camera coordinates in this
    project, so the frame camera extrinsic is intentionally not applied here.
    """
    if projection_source not in {"vertices", "joints"}:
        raise ValueError(f"Unsupported projection_source={projection_source!r}")
    models = smpl_models if smpl_models is not None else build_smpl_model_cache(smpl_model_dir)
    model = _select_smpl_model(models, person)
    device = next(model.parameters()).device

    root_pose = torch.as_tensor(person["smplx_root_pose"], dtype=torch.float32, device=device).reshape(1, 3)
    body_pose_21 = torch.as_tensor(person["smplx_body_pose"], dtype=torch.float32, device=device).reshape(21, 3)
    betas = torch.as_tensor(person["smplx_shape"], dtype=torch.float32, device=device).reshape(-1)[:10].reshape(1, 10)
    transl = torch.as_tensor(person["smplx_transl"], dtype=torch.float32, device=device).reshape(1, 3)
    body_pose = torch.cat([body_pose_21, body_pose_21.new_zeros(2, 3)], dim=0).reshape(1, 69)

    with torch.no_grad():
        smpl_out = model(betas=betas, global_orient=root_pose, body_pose=body_pose, transl=transl)
    vertices = smpl_out.vertices[0].detach().cpu().float()
    joints = smpl_out.joints[0, :24].detach().cpu().float()

    k = torch.as_tensor(intrinsics, dtype=torch.float32).reshape(3, 3)
    bbox_points = vertices if projection_source == "vertices" else joints
    xyxy = _projected_points_to_xyxy(bbox_points, k, image_hw)
    repaired = repair_xyxy_box(xyxy, image_hw) if xyxy is not None else None
    if repaired is None:
        return {
            "bbox_cxcywh_norm": [0.0, 0.0, 0.0, 0.0],
            "bbox_xyxy_pixels": [0.0, 0.0, 0.0, 0.0],
            "bbox_valid": False,
            "bbox_source": f"smpl_projection_{projection_source}_invalid",
        }

    j2ds, j2ds_mask = _project_points(joints, k, image_hw)
    return {
        "bbox_cxcywh_norm": xyxy_to_cxcywh_norm(repaired, image_hw).tolist(),
        "bbox_xyxy_pixels": repaired.tolist(),
        "bbox_valid": True,
        "bbox_source": f"smpl_projection_{projection_source}",
        "j2ds": j2ds.tolist(),
        "j2ds_mask": j2ds_mask[:, None].expand(-1, 2).contiguous().tolist(),
    }


def relative_sequence_name(seq_dir: Path, split_dir: Path) -> str:
    return seq_dir.relative_to(split_dir).as_posix()


def _ensure_legacy_smpl_compat() -> None:
    alias_map = {
        "bool": np.bool_,
        "int": builtins.int,
        "float": builtins.float,
        "complex": builtins.complex,
        "object": builtins.object,
        "unicode": builtins.str,
        "str": builtins.str,
    }
    for name, value in alias_map.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]


def _select_smpl_model(models: dict[str, Any], person: dict[str, Any]) -> Any:
    gender = str(person.get("smplx_gender", "neutral")).lower()
    if gender in models:
        return models[gender]
    if "neutral" in models:
        return models["neutral"]
    return next(iter(models.values()))


def _projected_points_to_xyxy(points3d: torch.Tensor, intrinsics: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor | None:
    points2d, visible = _project_points(points3d, intrinsics, image_hw, require_in_image=False)
    if not visible.any():
        return None
    coords = points2d[visible]
    return torch.cat([coords.amin(dim=0), coords.amax(dim=0)], dim=0)


def _project_points(
    points3d: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    require_in_image: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_hw
    points = points3d.float()
    z = points[..., 2]
    visible = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    normalized = points / z.clamp(min=1e-6).unsqueeze(-1)
    points2d = torch.matmul(normalized, intrinsics.float().t())[..., :2]
    visible = visible & torch.isfinite(points2d).all(dim=-1)
    if require_in_image:
        visible = visible & (points2d[..., 0] >= 0.0) & (points2d[..., 0] < float(w)) & (points2d[..., 1] >= 0.0) & (points2d[..., 1] < float(h))
    return points2d, visible


def _as_box_tensor(value: Any) -> torch.Tensor | None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if tensor.numel() < 4:
        return None
    return tensor[:4]


def _as_j2d_tensor(value: Any) -> torch.Tensor | None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.ndim == 1:
        if tensor.numel() % 3 == 0:
            tensor = tensor.reshape(-1, 3)
        elif tensor.numel() % 2 == 0:
            tensor = tensor.reshape(-1, 2)
    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        return None
    return tensor


def _first_j2d(person: dict[str, Any]) -> torch.Tensor | None:
    for key in J2D_KEYS:
        if key in person:
            return _as_j2d_tensor(person[key])
    return None


def _j2d_visibility(j2ds: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    coords = j2ds[..., :2]
    valid = torch.isfinite(coords).all(dim=-1)
    valid = valid & (coords[..., 0] >= 0) & (coords[..., 0] < float(w)) & (coords[..., 1] >= 0) & (coords[..., 1] < float(h))
    if j2ds.shape[-1] >= 3:
        valid = valid & (j2ds[..., 2] > 0)
    return valid


def _infer_box_format(box: torch.Tensor, image_hw: tuple[int, int]) -> str:
    h, w = image_hw
    if float(box.max()) <= 2.0:
        return "xyxy_norm" if box[2] > box[0] and box[3] > box[1] else "xywh_norm"
    if box[2] > box[0] and box[3] > box[1] and box[2] <= w * 1.2 and box[3] <= h * 1.2:
        return "xyxy"
    return "xywh"


def _box_to_xyxy_pixels(box: torch.Tensor, image_hw: tuple[int, int], fmt: str) -> torch.Tensor:
    h, w = image_hw
    box = box.float()
    if fmt in {"xyxy_norm", "norm_xyxy"}:
        scale = box.new_tensor([w, h, w, h], dtype=box.dtype)
        return box * scale
    if fmt in {"xywh_norm", "norm_xywh"}:
        scale = box.new_tensor([w, h, w, h], dtype=box.dtype)
        box = box * scale
        fmt = "xywh"
    if fmt == "xywh":
        x, y, bw, bh = box.unbind(0)
        return torch.stack([x, y, x + bw, y + bh])
    if fmt == "cxcywh_norm":
        return cxcywh_norm_to_xyxy(box, image_hw)
    if fmt == "cxcywh":
        cx, cy, bw, bh = box.unbind(0)
        return torch.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5])
    if fmt != "xyxy":
        raise ValueError(f"Unsupported bbox format: {fmt}")
    return box
