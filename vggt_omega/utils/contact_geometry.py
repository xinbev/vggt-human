from __future__ import annotations

import torch
import torch.nn.functional as F


def build_sole_vertex_indices(template_vertices: torch.Tensor, count_per_foot: int = 48) -> torch.Tensor:
    """Return deterministic left/right sole indices from the neutral template."""
    vertices = template_vertices.detach().float().reshape(-1, 3)
    count = min(max(int(count_per_foot), 1), max(int(vertices.shape[0] // 2), 1))
    x_mid = vertices[:, 0].median()
    sides = []
    for side_mask in (vertices[:, 0] >= x_mid, vertices[:, 0] < x_mid):
        candidates = torch.nonzero(side_mask, as_tuple=False).reshape(-1)
        order = torch.argsort(vertices[candidates, 1])[:count]
        sides.append(candidates[order])
    return torch.stack(sides, dim=0).long()


@torch.no_grad()
def estimate_local_support_planes(
    depth_flat: torch.Tensor,
    intrinsics_flat: torch.Tensor,
    foot_points_cam: torch.Tensor,
    frame_idx: torch.Tensor,
    image_size_hw: tuple[int, int],
    window_size: int = 21,
    min_points: int = 24,
    center_exclusion_radius: int = 2,
    max_rmse_m: float = 0.05,
    max_depth_m: float = 20.0,
    max_point_depth_delta_m: float = 0.75,
    min_up_component: float = 0.25,
    exclusion_mask: torch.Tensor | None = None,
    depth_confidence: torch.Tensor | None = None,
    min_depth_confidence: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Fit robust local planes around [M,2,3] foot centers in camera space."""
    depth = canonical_depth(depth_flat).float()
    if depth.ndim != 3:
        raise ValueError(f"depth_flat must resolve to [F,H,W], got {tuple(depth.shape)}")
    points = foot_points_cam.float()
    if points.ndim != 3 or points.shape[1:] != (2, 3):
        raise ValueError(f"foot_points_cam must have shape [M,2,3], got {tuple(points.shape)}")
    frame_idx = frame_idx.long().reshape(-1)
    if frame_idx.shape[0] != points.shape[0]:
        raise ValueError("frame_idx length must match foot_points_cam")
    intr = intrinsics_flat.float()
    if intr.ndim != 3 or intr.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics_flat must have shape [F,3,3], got {tuple(intr.shape)}")

    height, width = depth.shape[-2:]
    image_h, image_w = int(image_size_hw[0]), int(image_size_hw[1])
    point_intr = intr[frame_idx]
    z = points[..., 2].clamp(min=1e-6)
    px_image = point_intr[:, None, 0, 0] * points[..., 0] / z + point_intr[:, None, 0, 2]
    py_image = point_intr[:, None, 1, 1] * points[..., 1] / z + point_intr[:, None, 1, 2]
    px = px_image * (float(width) / float(image_w))
    py = py_image * (float(height) / float(image_h))

    window = max(int(window_size), 3)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=points.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, -1)
    oy = oy.reshape(1, 1, -1)
    cx = px.round().long()
    cy = py.round().long()
    xs_raw = cx[..., None] + ox
    ys_raw = cy[..., None] + oy
    valid = (xs_raw >= 0) & (xs_raw < width) & (ys_raw >= 0) & (ys_raw < height)
    if center_exclusion_radius > 0:
        valid = valid & ((ox.abs() > int(center_exclusion_radius)) | (oy.abs() > int(center_exclusion_radius)))
    xs = xs_raw.clamp(0, width - 1)
    ys = ys_raw.clamp(0, height - 1)
    sampled = depth[frame_idx[:, None, None], ys, xs]
    valid = valid & torch.isfinite(sampled) & (sampled > 1e-6)
    if float(max_depth_m) > 0.0:
        valid = valid & (sampled <= float(max_depth_m)) & (points[..., 2:3] <= float(max_depth_m))
    if float(max_point_depth_delta_m) > 0.0:
        valid = valid & ((sampled - points[..., 2:3]).abs() <= float(max_point_depth_delta_m))
    if exclusion_mask is not None:
        mask = exclusion_mask.bool()
        valid = valid & ~mask[frame_idx[:, None, None], ys, xs]
    if depth_confidence is not None and float(min_depth_confidence) > 0.0:
        confidence = canonical_depth(depth_confidence).float()
        if confidence.shape != depth.shape:
            raise ValueError(
                f"depth_confidence must match depth shape {tuple(depth.shape)}, got {tuple(confidence.shape)}"
            )
        sampled_confidence = confidence[frame_idx[:, None, None], ys, xs]
        valid = valid & torch.isfinite(sampled_confidence) & (sampled_confidence >= float(min_depth_confidence))

    fx = point_intr[:, None, None, 0, 0].clamp(min=1e-6)
    fy = point_intr[:, None, None, 1, 1].clamp(min=1e-6)
    kcx = point_intr[:, None, None, 0, 2]
    kcy = point_intr[:, None, None, 1, 2]
    image_x = xs.float() * (float(image_w) / float(width))
    image_y = ys.float() * (float(image_h) / float(height))
    scene_x = (image_x - kcx) * sampled / fx
    scene_y = (image_y - kcy) * sampled / fy
    scene = torch.stack([scene_x, scene_y, sampled], dim=-1)

    center, normal, rmse, plane_valid = _robust_plane_fit(scene, valid, min_points=min_points)
    # Camera +Y points down in the image, so an upward support normal has -Y.
    normal = torch.where(normal[..., 1:2] > 0.0, -normal, normal)
    signed = ((points - center) * normal).sum(dim=-1)
    plane_valid = (
        plane_valid
        & torch.isfinite(signed)
        & (rmse <= float(max_rmse_m))
        & ((-normal[..., 1]) >= float(min_up_component))
    )
    return {
        "center": center.to(dtype=foot_points_cam.dtype),
        "normal": normal.to(dtype=foot_points_cam.dtype),
        "rmse": rmse.to(dtype=foot_points_cam.dtype),
        "signed": signed.to(dtype=foot_points_cam.dtype),
        "valid": plane_valid,
        "point_count": valid.sum(dim=-1),
    }


def canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim in {3, 4}:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _robust_plane_fit(
    points: torch.Tensor,
    valid: torch.Tensor,
    min_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center, normal = _weighted_plane(points, valid)
    residual = ((points - center[..., None, :]) * normal[..., None, :]).sum(dim=-1).abs()
    masked = torch.where(valid, residual, torch.full_like(residual, float("nan")))
    median = torch.nanmedian(masked, dim=-1, keepdim=True).values
    mad = torch.nanmedian((masked - median).abs(), dim=-1, keepdim=True).values.clamp(min=0.005)
    trimmed = valid & (residual <= median + 3.0 * mad)
    center, normal = _weighted_plane(points, trimmed)
    residual = ((points - center[..., None, :]) * normal[..., None, :]).sum(dim=-1)
    count = trimmed.sum(dim=-1)
    rmse = torch.sqrt(
        (residual.square() * trimmed.to(dtype=residual.dtype)).sum(dim=-1) / count.clamp(min=1).to(dtype=residual.dtype)
    )
    plane_valid = (count >= max(int(min_points), 3)) & torch.isfinite(rmse)
    return center, F.normalize(normal, dim=-1, eps=1e-6), rmse, plane_valid


def _weighted_plane(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valid.to(dtype=points.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    center = (points * weights[..., None]).sum(dim=-2) / denom
    centered = (points - center[..., None, :]) * weights[..., None]
    cov = centered.transpose(-1, -2) @ centered / denom[..., None]
    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    cov = cov + eye.reshape(1, 1, 3, 3) * 1e-6
    _, evecs = torch.linalg.eigh(cov.float())
    return center, evecs[..., 0].to(dtype=points.dtype)
