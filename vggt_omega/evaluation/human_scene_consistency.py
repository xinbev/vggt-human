"""SHOW-style human--scene consistency metrics.

This module is an adapted rewrite of the evaluation logic in the SHOW
reference repository.  It intentionally has no import-time dependency on the
reference checkout under ``.paper``.

The released SHOW code and the paper formula are not identical.  This module
uses the released evaluator by default because that is the protocol needed for
Table 3 comparison:

``show_code``
    Scene-point-to-rendered-body nearest-neighbour distance, normalized by the
    mean scene depth and multiplied by 100.  This matches the released code's
    ``chamfer_norm_pct`` value used by its evaluation callbacks.

``paper`` (optional audit)
    Rendered-body-to-scene nearest-neighbour distance, normalized by the mean
    absolute body-depth deviation.  This is algebraically equivalent to Eq. 19
    in the paper (ratio of the two sums).

HS-V is the mean absolute difference between the x/y coordinate variances of
the visible body surface and masked scene points.  Official SHOW key aliases
(``chamfer_norm_pct_p5_95`` and ``xy_scale_mse_p5_95`` variants) are emitted
alongside the compact Table 3 aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F


VisibilityBackend = Literal["pytorch3d", "vertex_zbuffer"]
InvalidPolicy = Literal["show_zero", "skip_nan"]


@dataclass(frozen=True, slots=True)
class HumanSceneConsistencyConfig:
    percentile_trims: tuple[float, ...] = (5.0, 10.0)
    visibility_backend: VisibilityBackend = "pytorch3d"
    invalid_policy: InvalidPolicy = "show_zero"
    compute_paper_audit: bool = False
    depth_conf_threshold: float = 0.05
    scene_point_stride: int = 1
    distance_chunk_size: int = 2048
    min_body_points: int = 1
    min_scene_points: int = 1
    min_points_for_percentile: int = 20
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not self.percentile_trims:
            raise ValueError("percentile_trims must not be empty")
        if any(not math.isfinite(float(trim)) or not (0.0 <= float(trim) < 50.0) for trim in self.percentile_trims):
            raise ValueError(f"Each percentile trim must satisfy 0 <= trim < 50, got {self.percentile_trims}")
        if len({float(trim) for trim in self.percentile_trims}) != len(self.percentile_trims):
            raise ValueError(f"percentile_trims contains duplicates: {self.percentile_trims}")
        if self.visibility_backend not in {"pytorch3d", "vertex_zbuffer"}:
            raise ValueError(f"Unsupported visibility_backend: {self.visibility_backend!r}")
        if self.invalid_policy not in {"show_zero", "skip_nan"}:
            raise ValueError(f"Unsupported invalid_policy: {self.invalid_policy!r}")
        for name in (
            "scene_point_stride",
            "distance_chunk_size",
            "min_body_points",
            "min_scene_points",
            "min_points_for_percentile",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {self.eps}")
        if not math.isfinite(float(self.depth_conf_threshold)):
            raise ValueError(f"depth_conf_threshold must be finite, got {self.depth_conf_threshold}")


@torch.no_grad()
def compute_human_scene_consistency(
    mesh_vertices_cam: torch.Tensor,
    scene_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    human_mask: torch.Tensor,
    *,
    faces: torch.Tensor | None = None,
    scene_valid_mask: torch.Tensor | None = None,
    config: HumanSceneConsistencyConfig | None = None,
) -> dict[str, Any]:
    """Compute per-person SHOW-style metrics in one camera frame.

    Args:
        mesh_vertices_cam: Full predicted body mesh, ``[V, 3]``, in camera
            coordinates.  Coordinates and ``scene_depth`` must share a scale.
        scene_depth: Predicted scene depth, ``[H, W]``, in the same camera
            coordinate convention as the body mesh.
        intrinsics: Camera intrinsics, ``[3, 3]``, calibrated for ``[H, W]``.
        human_mask: Independent human segmentation mask.  It is resized with
            nearest-neighbour interpolation when needed.
        scene_valid_mask: Optional depth-confidence/valid-image mask combined
            with ``human_mask`` before depth unprojection.
        faces: Mesh triangle indices, required by the ``pytorch3d`` backend.
        config: Metric and numerical settings.

    Returns:
        A flat dictionary suitable for CSV serialization.  ``hs_cf{trim}``
        follows the released SHOW code.  With the default ``show_zero`` policy,
        invalid examples reproduce SHOW's zero-valued contribution and also
        carry an explicit invalid reason.  ``skip_nan`` is a non-official audit
        option.  Paper Eq. 19 values are computed only when explicitly enabled.
    """

    cfg = config or HumanSceneConsistencyConfig()
    depth = canonical_depth_2d(scene_depth).float()
    vertices = _points_2d(mesh_vertices_cam, "mesh_vertices_cam").float()
    K = torch.as_tensor(intrinsics, device=depth.device, dtype=torch.float32)
    if tuple(K.shape) != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3], got {tuple(K.shape)}")

    mask = resize_human_mask(human_mask, tuple(depth.shape)).to(device=depth.device)
    if scene_valid_mask is not None:
        mask = mask & resize_human_mask(scene_valid_mask, tuple(depth.shape)).to(device=depth.device)
    body_points = visible_body_surface_points(
        vertices.to(device=depth.device),
        K,
        image_hw=tuple(depth.shape),
        faces=faces,
        backend=cfg.visibility_backend,
    )
    scene_points = masked_depth_to_camera_points(
        depth,
        K,
        mask,
        stride=cfg.scene_point_stride,
    )

    result: dict[str, Any] = {
        "visibility_backend": cfg.visibility_backend,
        "body_point_count": int(body_points.shape[0]),
        "scene_point_count_unfiltered": int(scene_points.shape[0]),
    }
    if body_points.shape[0] < int(cfg.min_body_points):
        return _invalid_result(
            result,
            cfg.percentile_trims,
            "insufficient_body_points",
            cfg.invalid_policy,
            cfg.compute_paper_audit,
        )
    if scene_points.shape[0] < int(cfg.min_scene_points):
        return _invalid_result(
            result,
            cfg.percentile_trims,
            "insufficient_scene_points",
            cfg.invalid_policy,
            cfg.compute_paper_audit,
        )

    for trim in cfg.percentile_trims:
        label = percentile_label(trim)
        filtered_scene = filter_depth_percentile(
            scene_points,
            trim=float(trim),
            min_points_for_percentile=int(cfg.min_points_for_percentile),
        )
        result[f"scene_point_count_{label}"] = int(filtered_scene.shape[0])
        if filtered_scene.shape[0] < int(cfg.min_scene_points):
            _set_invalid_metrics(result, label, cfg.invalid_policy, cfg.compute_paper_audit)
            result[f"valid_{label}"] = False
            result[f"valid_show_code{label}"] = False
            if cfg.compute_paper_audit:
                result[f"valid_paper{label}"] = False
            result[f"invalid_reason_{label}"] = "insufficient_filtered_scene_points"
            _set_official_show_aliases(result, label, trim)
            continue

        metrics = compute_from_visible_points(
            body_points,
            filtered_scene,
            distance_chunk_size=int(cfg.distance_chunk_size),
            eps=float(cfg.eps),
            compute_paper_audit=bool(cfg.compute_paper_audit),
        )
        result[f"valid_{label}"] = bool(metrics.pop("valid"))
        for name, value in metrics.items():
            result[f"{name}{label}"] = value
        if not result[f"valid_{label}"]:
            result[f"invalid_reason_{label}"] = "degenerate_normalizer"
        _set_official_show_aliases(result, label, trim)

    result["valid"] = any(bool(result.get(f"valid_{percentile_label(t)}", False)) for t in cfg.percentile_trims)
    result["invalid_reason"] = "" if result["valid"] else "no_valid_percentile_range"
    return result


@torch.no_grad()
def compute_from_visible_points(
    body_points_cam: torch.Tensor,
    scene_points_cam: torch.Tensor,
    *,
    distance_chunk_size: int = 2048,
    eps: float = 1e-8,
    compute_paper_audit: bool = True,
) -> dict[str, float | bool]:
    """Compute both published and released-code protocols from point sets."""

    body = _valid_camera_points(body_points_cam)
    scene = _valid_camera_points(scene_points_cam)
    if body.numel() == 0 or scene.numel() == 0:
        out: dict[str, float | bool] = {
            "valid": False,
            "valid_show_code": False,
            "hs_cf": math.nan,
            "hs_v": math.nan,
            "chamfer_scene_to_body_m": math.nan,
        }
        if compute_paper_audit:
            out.update(
                {
                    "valid_paper": False,
                    "hs_cf_paper": math.nan,
                    "hs_cf_paper_pct": math.nan,
                    "chamfer_body_to_scene_m": math.nan,
                }
            )
        return out

    scene_to_body = directed_nearest_mean(scene, body, chunk_size=distance_chunk_size)
    body_to_scene = (
        directed_nearest_mean(body, scene, chunk_size=distance_chunk_size)
        if compute_paper_audit
        else body.new_tensor(float("nan"))
    )

    mean_scene_depth = scene[:, 2].mean()
    body_depth_mad = (body[:, 2] - body[:, 2].mean()).abs().mean()
    show_valid = bool(torch.isfinite(mean_scene_depth) and mean_scene_depth > eps)
    paper_valid = bool(compute_paper_audit and torch.isfinite(body_depth_mad) and body_depth_mad > eps)

    hs_cf = (scene_to_body / mean_scene_depth) * 100.0 if show_valid else scene_to_body.new_tensor(float("nan"))
    hs_cf_paper = body_to_scene / body_depth_mad if paper_valid else body_to_scene.new_tensor(float("nan"))
    body_xy_var = body[:, :2].var(dim=0, unbiased=False)
    scene_xy_var = scene[:, :2].var(dim=0, unbiased=False)
    hs_v = (body_xy_var - scene_xy_var).abs().mean()

    hs_v_valid = bool(torch.isfinite(hs_v))
    out = {
        # The released-code protocol is the primary Table 3 comparison.
        "valid": bool(show_valid and hs_v_valid),
        "valid_show_code": bool(show_valid and hs_v_valid),
        "hs_cf": float(hs_cf.detach().cpu()),
        "hs_v": float(hs_v.detach().cpu()),
        "chamfer_scene_to_body_m": float(scene_to_body.detach().cpu()),
    }
    if compute_paper_audit:
        out.update(
            {
                "valid_paper": bool(paper_valid and hs_v_valid),
                "hs_cf_paper": float(hs_cf_paper.detach().cpu()),
                "hs_cf_paper_pct": float((hs_cf_paper * 100.0).detach().cpu()),
                "chamfer_body_to_scene_m": float(body_to_scene.detach().cpu()),
            }
        )
    return out


@torch.no_grad()
def directed_nearest_mean(source: torch.Tensor, target: torch.Tensor, *, chunk_size: int = 2048) -> torch.Tensor:
    """Mean distance from every source point to its nearest target point.

    Both axes are chunked, avoiding the quadratic peak allocation of a single
    ``torch.cdist(source, target)`` call while preserving the exact result.
    """

    source = _points_2d(source, "source").float()
    target = _points_2d(target, "target").to(device=source.device, dtype=source.dtype)
    if source.shape[0] == 0 or target.shape[0] == 0:
        return source.new_tensor(float("nan"))
    step = max(int(chunk_size), 1)
    total = source.new_zeros(())
    count = 0
    for source_start in range(0, int(source.shape[0]), step):
        source_chunk = source[source_start : source_start + step]
        best = source.new_full((source_chunk.shape[0],), float("inf"))
        for target_start in range(0, int(target.shape[0]), step):
            target_chunk = target[target_start : target_start + step]
            best = torch.minimum(best, torch.cdist(source_chunk, target_chunk).amin(dim=1))
        total = total + best.sum()
        count += int(best.numel())
    return total / max(count, 1)


@torch.no_grad()
def masked_depth_to_camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    human_mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """Unproject valid masked depth pixels to camera-frame XYZ points."""

    depth = canonical_depth_2d(depth).float()
    K = torch.as_tensor(intrinsics, device=depth.device, dtype=depth.dtype)
    mask = resize_human_mask(human_mask, tuple(depth.shape)).to(device=depth.device)
    step = max(int(stride), 1)
    height, width = int(depth.shape[0]), int(depth.shape[1])
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=depth.dtype),
        torch.arange(0, width, step, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()]
    selected = mask[ys.long(), xs.long()]
    fx = K[0, 0].clamp_min(1e-8)
    fy = K[1, 1].clamp_min(1e-8)
    x = (xs - K[0, 2]) * z / fx
    y = (ys - K[1, 2]) * z / fy
    points = torch.stack((x, y, z), dim=-1)
    valid = selected & torch.isfinite(points).all(dim=-1) & (z > 0)
    return points[valid]


@torch.no_grad()
def visible_body_surface_points(
    vertices_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    faces: torch.Tensor | None,
    backend: VisibilityBackend,
) -> torch.Tensor:
    if backend == "pytorch3d":
        if faces is None:
            raise ValueError("faces are required for visibility_backend='pytorch3d'")
        return _rasterize_mesh_surface_pytorch3d(vertices_cam, faces, intrinsics, image_hw)
    if backend == "vertex_zbuffer":
        return _visible_vertices_zbuffer(vertices_cam, intrinsics, image_hw)
    raise ValueError(f"Unsupported visibility backend: {backend!r}")


@torch.no_grad()
def render_mesh_silhouette(
    vertices_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    faces: torch.Tensor,
) -> torch.Tensor:
    """Render a dense mesh silhouette with SHOW's PyTorch3D convention."""

    _, mask = _rasterize_mesh_depth_pytorch3d(vertices_cam, faces, intrinsics, image_hw)
    if tuple(mask.shape) != tuple(image_hw):
        mask = F.interpolate(mask.float()[None, None], size=image_hw, mode="nearest")[0, 0] > 0.5
    return mask


def canonical_depth_2d(depth: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(depth)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"depth must reduce to [H,W], got {tuple(torch.as_tensor(depth).shape)}")
    return value


def resize_human_mask(mask: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    value = torch.as_tensor(mask)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"human_mask must reduce to [H,W], got {tuple(torch.as_tensor(mask).shape)}")
    if tuple(value.shape) != tuple(image_hw):
        value = F.interpolate(value.float()[None, None], size=image_hw, mode="nearest")[0, 0]
    return value > 0.5


def filter_depth_percentile(points: torch.Tensor, *, trim: float, min_points_for_percentile: int = 20) -> torch.Tensor:
    points = _valid_camera_points(points)
    trim = float(trim)
    if not (0.0 <= trim < 50.0):
        raise ValueError(f"trim must satisfy 0 <= trim < 50, got {trim}")
    if trim <= 0.0 or points.shape[0] < int(min_points_for_percentile):
        return points
    z = points[:, 2]
    low = torch.quantile(z, trim / 100.0)
    high = torch.quantile(z, 1.0 - trim / 100.0)
    return points[(z >= low) & (z <= high)]


def percentile_label(trim: float) -> str:
    value = float(trim)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def percentile_range_label(trim: float) -> str:
    return f"p{percentile_label(trim)}_{percentile_label(100.0 - float(trim))}"


def _valid_camera_points(points: torch.Tensor) -> torch.Tensor:
    value = _points_2d(points, "points")
    valid = torch.isfinite(value).all(dim=-1) & (value[:, 2] > 0)
    return value[valid]


def _points_2d(points: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(points)
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {tuple(value.shape)}")
    return value


@torch.no_grad()
def _visible_vertices_zbuffer(
    vertices_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Dependency-free visibility approximation using one vertex per pixel."""

    vertices = _valid_camera_points(vertices_cam)
    if vertices.numel() == 0:
        return vertices
    K = torch.as_tensor(intrinsics, device=vertices.device, dtype=vertices.dtype)
    height, width = int(image_hw[0]), int(image_hw[1])
    z = vertices[:, 2]
    x = torch.round(K[0, 0] * vertices[:, 0] / z + K[0, 2]).long()
    y = torch.round(K[1, 1] * vertices[:, 1] / z + K[1, 2]).long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not bool(valid.any()):
        return vertices[:0]
    vertices, x, y, z = vertices[valid], x[valid], y[valid], z[valid]
    pixel = (y * width + x).detach().cpu().numpy()
    depth = z.detach().float().cpu().numpy()
    import numpy as np

    order = np.lexsort((depth, pixel))
    sorted_pixel = pixel[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
    indices = torch.as_tensor(order[keep], device=vertices.device, dtype=torch.long)
    return vertices[indices]


@torch.no_grad()
def _rasterize_mesh_surface_pytorch3d(
    vertices_cam: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Rasterize a body depth map following SHOW's released renderer setup."""

    depth, rendered = _rasterize_mesh_depth_pytorch3d(vertices_cam, faces, intrinsics, image_hw)
    return masked_depth_to_camera_points(depth, intrinsics, rendered, stride=1)


@torch.no_grad()
def _rasterize_mesh_depth_pytorch3d(
    vertices_cam: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SHOW-convention rendered depth and foreground mask."""

    try:
        from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
        from pytorch3d.structures import Meshes
    except ImportError as exc:
        raise RuntimeError(
            "The official SHOW visibility backend requires pytorch3d. "
            "Use visibility_backend='vertex_zbuffer' only for dependency-free diagnostics."
        ) from exc

    vertices = _points_2d(vertices_cam, "vertices_cam").float()
    device = vertices.device
    faces_tensor = torch.as_tensor(faces, device=device, dtype=torch.int64)
    if faces_tensor.ndim != 2 or faces_tensor.shape[-1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {tuple(faces_tensor.shape)}")
    K = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)

    # The SHOW release constructs its renderer canvas as 2*principal-point
    # rather than receiving the source image size explicitly.  Preserve that
    # behavior for the official-comparison backend, with image_hw as a guard for
    # malformed intrinsics.
    fallback_height, fallback_width = int(image_hw[0]), int(image_hw[1])
    width = int(round(float(K[0, 2].detach().cpu()) * 2.0))
    height = int(round(float(K[1, 2].detach().cpu()) * 2.0))
    width = width if width > 0 else fallback_width
    height = height if height > 0 else fallback_height

    # SHOW's renderer mirrors the principal point before rasterization and
    # flips the resulting buffers back into the ordinary image convention.
    K4 = torch.zeros((1, 4, 4), device=device, dtype=torch.float32)
    K4[0, :3, :3] = K
    K4[0, 0, 2] = float(width) - K[0, 2]
    K4[0, 1, 2] = float(height) - K[1, 2]
    K4[0, 2, 2] = 0.0
    K4[0, 2, 3] = 1.0
    K4[0, 3, 2] = 1.0
    cameras = PerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device, dtype=torch.float32)[None],
        T=torch.zeros((1, 3), device=device, dtype=torch.float32),
        K=K4,
        image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),
        in_ndc=False,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=1e-5,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    mesh = Meshes(verts=[vertices], faces=[faces_tensor])
    fragments = rasterizer(mesh)
    depth = torch.flip(fragments.zbuf[0, ..., 0], dims=(0, 1))
    rendered = torch.flip(fragments.pix_to_face[0, ..., 0] >= 0, dims=(0, 1))
    depth = torch.where(rendered & (depth > 0), depth, torch.zeros_like(depth))
    return depth, rendered


def _invalid_result(
    result: dict[str, Any],
    trims: Sequence[float],
    reason: str,
    invalid_policy: InvalidPolicy,
    compute_paper_audit: bool,
) -> dict[str, Any]:
    result["valid"] = False
    result["invalid_reason"] = reason
    for trim in trims:
        label = percentile_label(trim)
        result[f"valid_{label}"] = False
        result[f"valid_show_code{label}"] = False
        if compute_paper_audit:
            result[f"valid_paper{label}"] = False
        result[f"invalid_reason_{label}"] = reason
        _set_invalid_metrics(result, label, invalid_policy, compute_paper_audit)
        _set_official_show_aliases(result, label, trim)
    return result


def _set_invalid_metrics(
    result: dict[str, Any],
    label: str,
    invalid_policy: InvalidPolicy,
    compute_paper_audit: bool,
) -> None:
    official_value = 0.0 if invalid_policy == "show_zero" else math.nan
    for name in ("hs_cf", "hs_v", "chamfer_scene_to_body_m"):
        result[f"{name}{label}"] = official_value
    if compute_paper_audit:
        for name in ("hs_cf_paper", "hs_cf_paper_pct", "chamfer_body_to_scene_m"):
            result[f"{name}{label}"] = math.nan


def _set_official_show_aliases(result: dict[str, Any], label: str, trim: float) -> None:
    suffix = percentile_range_label(trim)
    result[f"chamfer_abs_{suffix}"] = result[f"chamfer_scene_to_body_m{label}"]
    result[f"chamfer_norm_pct_{suffix}"] = result[f"hs_cf{label}"]
    result[f"xy_scale_mse_{suffix}"] = result[f"hs_v{label}"]
