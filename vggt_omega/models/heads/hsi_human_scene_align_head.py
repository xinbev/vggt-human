import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class HSIHumanSceneAlignHead(nn.Module):
    """Lightweight SMPL translation aligner for metric HSI scene depth.

    The head keeps pose/betas fixed and predicts a small camera-space root
    translation update from explicit SMPL-to-scene point residual statistics.
    """

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 256,
        num_sample_vertices: int = 96,
        local_window: int = 7,
        max_ray_delta_m: float = 0.35,
        max_tangent_delta_m: float = 0.12,
        use_delta_gate: bool = True,
        overwrite_refined: bool = True,
        base_source: str = "hsi_refined",
        max_correspondence_distance_m: float = 0.35,
        gt_max_correspondence_distance_m: float = 3.5,
        min_depth_confidence: float = 0.0,
        residual_mad_multiplier: float = 3.0,
        max_depth_m: float = 20.0,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSIHumanSceneAlignHead requires smpl_model_dir")
        if local_window % 2 != 1:
            raise ValueError(f"hsi_align_local_window must be odd, got {local_window}")
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for param in self.smpl.parameters():
            param.requires_grad = False

        self.hidden_dim = int(hidden_dim)
        self.local_window = int(local_window)
        self.max_ray_delta_m = float(max_ray_delta_m)
        self.max_tangent_delta_m = float(max_tangent_delta_m)
        self.use_delta_gate = bool(use_delta_gate)
        self.overwrite_refined = bool(overwrite_refined)
        self.base_source = str(base_source or "hsi_refined").lower()
        if self.base_source not in {"pred", "hsi_refined"}:
            raise ValueError(f"Unsupported HSI align base source: {self.base_source!r}")
        self.max_correspondence_distance_m = float(max_correspondence_distance_m)
        self.gt_max_correspondence_distance_m = float(gt_max_correspondence_distance_m)
        self.min_depth_confidence = float(min_depth_confidence)
        self.residual_mad_multiplier = float(residual_mad_multiplier)
        self.max_depth_m = float(max_depth_m)
        self.image_size = int(image_size)

        vertex_indices = _deterministic_fps_indices(
            self.smpl.layer.v_template.detach().float(),
            int(num_sample_vertices),
        )
        self.register_buffer("sample_vertex_indices", vertex_indices, persistent=False)

        # Metric point residuals contain the scale information needed by this
        # head.  Deliberately omit scene scale/bias scalars to prevent a
        # gt-depth versus VGGT-depth domain shortcut.
        feature_dim = 3 + 1 + 1 + 1 + 3 + 3 + 3 + 3 + 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)))
        self.gate_head = _biased_last_linear(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)),
            bias=2.0,
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        image_size_hw: tuple[int, int] | None = None,
        intrinsics_override: torch.Tensor | None = None,
        depth_is_metric: bool = False,
        person_boxes: torch.Tensor | None = None,
        depth_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        use_hsi_base = self.base_source == "hsi_refined"
        pose6d = predictions.get("hsi_refined_pred_pose_6d") if use_hsi_base else None
        poses = predictions.get("hsi_refined_pred_poses") if use_hsi_base else None
        betas = predictions.get("hsi_refined_pred_betas") if use_hsi_base else None
        base_transl = predictions.get("hsi_refined_pred_transl_cam") if use_hsi_base else None
        pose6d = predictions.get("pred_pose_6d") if pose6d is None else pose6d
        poses = predictions.get("pred_poses") if poses is None else poses
        betas = predictions.get("pred_betas") if betas is None else betas
        base_transl = predictions.get("pred_transl_cam") if base_transl is None else base_transl
        confs = predictions.get("pred_confs")
        if pose6d is None or poses is None or betas is None or base_transl is None or confs is None:
            raise ValueError("HSI human-scene align requires SMPL pose/betas/transl/confs")
        if depth is None:
            raise ValueError("HSI human-scene align requires depth")
        if pose_enc is None:
            raise ValueError("HSI human-scene align requires pose_enc")

        pose6d = pose6d.float()
        poses = poses.float()
        betas = betas.float()
        base_transl = base_transl.float()
        confs = confs.float()
        batch_size, num_frames, num_queries, _ = base_transl.shape
        depth_hw = _canonical_depth(depth).float()
        if depth_hw.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Expected depth [B,S,H,W], got {tuple(depth_hw.shape)}")
        height, width = depth_hw.shape[-2:]
        image_size_hw = _resolve_image_size_hw(image_size_hw, self.image_size)
        intrinsics = _resolve_intrinsics(
            pose_enc,
            image_size_hw=image_size_hw,
            batch_size=batch_size,
            num_frames=num_frames,
            device=base_transl.device,
            dtype=base_transl.dtype,
            intrinsics_override=intrinsics_override,
        )

        metric_depth = depth_hw if depth_is_metric else self._metric_depth(predictions, depth_hw)
        sample_points = self._sample_smpl_points(poses, betas, base_transl)
        scene_points, valid = _local_scene_points(
            depth_hw=metric_depth,
            points_cam=sample_points,
            intrinsics=intrinsics,
            image_size_hw=image_size_hw,
            local_window=self.local_window,
            person_boxes=person_boxes,
            depth_confidence=depth_confidence,
            min_depth_confidence=self.min_depth_confidence,
            max_correspondence_distance_m=(
                self.gt_max_correspondence_distance_m if depth_is_metric else self.max_correspondence_distance_m
            ),
            residual_mad_multiplier=self.residual_mad_multiplier,
            max_depth_m=self.max_depth_m,
        )
        valid = valid & (confs[..., :1].unsqueeze(-2) > 0.0)
        residual = scene_points - sample_points
        valid_f = valid.to(dtype=base_transl.dtype)
        denom = valid_f.sum(dim=-2).clamp(min=1.0)
        mean_residual = (residual * valid_f).sum(dim=-2) / denom
        mean_abs_residual = (residual.abs() * valid_f).sum(dim=-2) / denom
        base_point_dist = torch.linalg.norm(residual, dim=-1, keepdim=True)
        mean_base_dist = (base_point_dist * valid_f).sum(dim=-2) / denom
        depth_residual = residual[..., 2:3]
        mean_depth_residual = (depth_residual * valid_f).sum(dim=-2) / denom
        valid_ratio = valid_f.mean(dim=-2)

        ray, tangent_x, tangent_y = _camera_basis(base_transl)
        projected_center = _project_points(
            base_transl.reshape(-1, 1, 3),
            intrinsics.repeat_interleave(num_queries, dim=0),
        ).reshape(batch_size, num_frames, num_queries, 2)
        proj_norm = torch.stack(
            [
                projected_center[..., 0] / max(float(image_size_hw[1] - 1), 1.0),
                projected_center[..., 1] / max(float(image_size_hw[0] - 1), 1.0),
            ],
            dim=-1,
        )
        base_norm = torch.linalg.norm(base_transl, dim=-1, keepdim=True)
        features = torch.cat(
            [
                base_transl,
                base_norm,
                confs.clamp(min=0.0, max=1.0),
                valid_ratio,
                mean_residual,
                mean_abs_residual,
                ray,
                tangent_x,
                tangent_y,
                proj_norm,
            ],
            dim=-1,
        )
        hidden = self.mlp(features)
        raw_delta = torch.tanh(self.delta_head(hidden))
        coeff = torch.stack(
            [
                raw_delta[..., 0] * self.max_ray_delta_m,
                raw_delta[..., 1] * self.max_tangent_delta_m,
                raw_delta[..., 2] * self.max_tangent_delta_m,
            ],
            dim=-1,
        )
        delta = coeff[..., :1] * ray + coeff[..., 1:2] * tangent_x + coeff[..., 2:3] * tangent_y
        gate = torch.sigmoid(self.gate_head(hidden))
        if not self.use_delta_gate:
            gate = torch.ones_like(gate)
        delta = gate * delta
        refined_transl = base_transl + delta

        refined_points = sample_points + delta.unsqueeze(-2)
        refined_residual = scene_points - refined_points
        refined_point_dist = torch.linalg.norm(refined_residual, dim=-1, keepdim=True)
        base_loss = _masked_mean(_smooth_l1_abs(base_point_dist), valid_f)
        refined_loss = _masked_mean(_smooth_l1_abs(refined_point_dist), valid_f)

        outputs = {
            "hsi_align_base_pred_transl_cam": base_transl,
            "hsi_align_refined_pred_transl_cam": refined_transl,
            "hsi_align_delta_transl_cam": delta,
            "hsi_align_delta_coeff": coeff,
            "hsi_align_gate": gate,
            "hsi_align_valid_ratio": valid_ratio,
            "hsi_align_depth_is_metric": base_transl.new_tensor(float(depth_is_metric)),
            "hsi_align_base_point_l1": base_loss.detach(),
            "hsi_align_refined_point_l1": refined_loss.detach(),
            "hsi_align_point_l1_delta": (refined_loss - base_loss).detach(),
            "loss_hsi_align_point": refined_loss,
            "loss_hsi_align_delta_reg": _smooth_l1_abs(delta).mean(),
            "loss_hsi_align_no_worse": F.relu(refined_point_dist - base_point_dist.detach() - 0.005).mul(valid_f).sum()
            / valid_f.sum().clamp(min=1.0),
        }
        if self.overwrite_refined:
            outputs["hsi_refined_pred_pose_6d"] = pose6d
            outputs["hsi_refined_pred_poses"] = poses
            outputs["hsi_refined_pred_betas"] = betas
            outputs["hsi_refined_pred_transl_cam"] = refined_transl
        return outputs

    def _metric_depth(self, predictions: dict[str, torch.Tensor], depth_hw: torch.Tensor) -> torch.Tensor:
        scale = predictions.get("hsi_scene_scale")
        bias = predictions.get("hsi_scene_depth_bias")
        if scale is None or bias is None:
            return depth_hw
        while scale.ndim < depth_hw.ndim:
            scale = scale.unsqueeze(-1)
        while bias.ndim < depth_hw.ndim:
            bias = bias.unsqueeze(-1)
        return depth_hw * scale.to(device=depth_hw.device, dtype=depth_hw.dtype) + bias.to(device=depth_hw.device, dtype=depth_hw.dtype)

    def _sample_smpl_points(self, poses: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_queries = transl.shape[:3]
        flat_poses = poses.reshape(-1, 72)
        flat_betas = betas.reshape(-1, betas.shape[-1])
        vertices, joints = self.smpl(flat_poses.float(), flat_betas.float())
        vertices = vertices.to(device=transl.device, dtype=transl.dtype)
        joints = joints[:, :24].to(device=transl.device, dtype=transl.dtype)
        sampled = vertices[:, self.sample_vertex_indices.to(vertices.device)]
        points = torch.cat([joints, sampled], dim=1) + transl.reshape(-1, 1, 3)
        return points.reshape(batch_size, num_frames, num_queries, points.shape[1], 3)


def _canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _resolve_image_size_hw(image_size_hw: tuple[int, int] | None, fallback_image_size: int) -> tuple[int, int]:
    if image_size_hw is None:
        return int(fallback_image_size), int(fallback_image_size)
    return int(image_size_hw[0]), int(image_size_hw[1])


def _resolve_intrinsics(
    pose_enc: torch.Tensor,
    image_size_hw: tuple[int, int],
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    intrinsics_override: torch.Tensor | None = None,
) -> torch.Tensor:
    if intrinsics_override is None:
        _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=image_size_hw, build_intrinsics=True)
        return intrinsics.reshape(-1, 3, 3).to(device=device, dtype=dtype)
    intrinsics = intrinsics_override.to(device=device, dtype=dtype)
    if intrinsics.ndim == 4:
        if tuple(intrinsics.shape[:2]) != (batch_size, num_frames) or tuple(intrinsics.shape[-2:]) != (3, 3):
            raise ValueError(
                "HSI align intrinsics_override must have shape [B,S,3,3], "
                f"got {tuple(intrinsics.shape)} expected B={batch_size}, S={num_frames}"
            )
        return intrinsics.reshape(-1, 3, 3)
    if intrinsics.ndim == 3 and tuple(intrinsics.shape[-2:]) == (3, 3):
        if int(intrinsics.shape[0]) == batch_size * num_frames:
            return intrinsics.reshape(-1, 3, 3)
        if int(intrinsics.shape[0]) == num_frames and batch_size == 1:
            return intrinsics.reshape(-1, 3, 3)
    raise ValueError(f"Unsupported HSI align intrinsics_override shape: {tuple(intrinsics.shape)}")


def _project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[:, None, 0, 0] * points_cam[..., 0] / z + intrinsics[:, None, 0, 2]
    y = intrinsics[:, None, 1, 1] * points_cam[..., 1] / z + intrinsics[:, None, 1, 2]
    return torch.stack([x, y], dim=-1)


def _local_scene_points(
    depth_hw: torch.Tensor,
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    local_window: int,
    person_boxes: torch.Tensor | None = None,
    depth_confidence: torch.Tensor | None = None,
    min_depth_confidence: float = 0.0,
    max_correspondence_distance_m: float = 0.35,
    residual_mad_multiplier: float = 3.0,
    max_depth_m: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_queries, num_points, _ = points_cam.shape
    flat_frames = batch_size * num_frames
    height, width = depth_hw.shape[-2:]
    image_h, image_w = int(image_size_hw[0]), int(image_size_hw[1])
    flat_points = points_cam.reshape(flat_frames * num_queries, num_points, 3)
    projected = _project_points(flat_points, intrinsics.repeat_interleave(num_queries, dim=0))
    projected = projected.reshape(batch_size, num_frames, num_queries, num_points, 2)
    front_visible = _coarse_front_surface_mask(projected, points_cam[..., 2], image_size_hw)
    depth_xy = projected * projected.new_tensor([float(width) / float(image_w), float(height) / float(image_h)])
    center_x = depth_xy[..., 0].round().long()
    center_y = depth_xy[..., 1].round().long()
    radius = max(int(local_window), 1) // 2
    offsets = torch.arange(-radius, radius + 1, device=points_cam.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, 1, 1, -1)
    oy = oy.reshape(1, 1, 1, 1, -1)
    xs_raw = center_x[..., None] + ox
    ys_raw = center_y[..., None] + oy
    in_bounds = (xs_raw >= 0) & (xs_raw < width) & (ys_raw >= 0) & (ys_raw < height)
    xs = xs_raw.clamp(0, width - 1)
    ys = ys_raw.clamp(0, height - 1)
    frame_idx = torch.arange(flat_frames, device=points_cam.device).reshape(batch_size, num_frames, 1, 1, 1)
    flat_depth = depth_hw.reshape(flat_frames, height, width)
    local_depth = flat_depth[frame_idx, ys, xs]
    valid = (
        in_bounds
        & torch.isfinite(local_depth)
        & (local_depth > 1e-6)
        & (points_cam[..., 2:3] > 1e-6)
        & front_visible[..., None]
    )
    if float(max_depth_m) > 0.0:
        valid = valid & (local_depth <= float(max_depth_m))
    if depth_confidence is not None and float(min_depth_confidence) > 0.0:
        conf_hw = _canonical_depth(depth_confidence).reshape(flat_frames, height, width)
        local_conf = conf_hw[frame_idx, ys, xs]
        valid = valid & torch.isfinite(local_conf) & (local_conf >= float(min_depth_confidence))
    if person_boxes is not None:
        boxes = person_boxes.to(device=points_cam.device, dtype=points_cam.dtype)
        if boxes.ndim != 4 or tuple(boxes.shape[:3]) != (batch_size, num_frames, num_queries):
            raise ValueError(
                "person_boxes must have shape [B,S,Q,4], "
                f"got {tuple(boxes.shape)} expected {(batch_size, num_frames, num_queries, 4)}"
            )
        center = boxes[..., :2]
        size = boxes[..., 2:].clamp(min=1e-6)
        box_min = (center - 0.55 * size) * boxes.new_tensor([float(width), float(height)])
        box_max = (center + 0.55 * size) * boxes.new_tensor([float(width), float(height)])
        valid = valid & (xs >= box_min[..., 0, None, None]) & (xs <= box_max[..., 0, None, None])
        valid = valid & (ys >= box_min[..., 1, None, None]) & (ys <= box_max[..., 1, None, None])

    intr = intrinsics.reshape(batch_size, num_frames, 1, 1, 1, 3, 3)
    fx = intr[..., 0, 0].clamp(min=1e-6)
    fy = intr[..., 1, 1].clamp(min=1e-6)
    cx = intr[..., 0, 2]
    cy = intr[..., 1, 2]
    image_x = xs.to(dtype=local_depth.dtype) * (float(image_w) / float(width))
    image_y = ys.to(dtype=local_depth.dtype) * (float(image_h) / float(height))
    scene_x = (image_x - cx.to(dtype=local_depth.dtype)) * local_depth / fx.to(dtype=local_depth.dtype)
    scene_y = (image_y - cy.to(dtype=local_depth.dtype)) * local_depth / fy.to(dtype=local_depth.dtype)
    scene_xyz = torch.stack([scene_x, scene_y, local_depth], dim=-1)
    dist = torch.linalg.norm(scene_xyz - points_cam[..., None, :].to(dtype=scene_xyz.dtype), dim=-1)
    if float(max_correspondence_distance_m) > 0.0:
        valid = valid & (dist <= float(max_correspondence_distance_m))
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    nearest_idx = dist.argmin(dim=-1)
    nearest_dist = dist.gather(dim=-1, index=nearest_idx[..., None]).squeeze(-1)
    has_valid = torch.isfinite(nearest_dist)
    gather_idx = nearest_idx[..., None, None].expand(*nearest_idx.shape, 1, 3)
    nearest_xyz = scene_xyz.gather(dim=-2, index=gather_idx).squeeze(-2)
    # Reject per-person residual outliers after local nearest sampling.  This
    # catches limbs whose projected window landed on background or another
    # person without requiring GT visibility at inference time.
    finite_dist = torch.where(has_valid, nearest_dist, torch.full_like(nearest_dist, float("nan")))
    median = torch.nanmedian(finite_dist, dim=-1, keepdim=True).values
    abs_dev = (finite_dist - median).abs()
    mad = torch.nanmedian(abs_dev, dim=-1, keepdim=True).values.clamp(min=0.01)
    robust_valid = nearest_dist <= (median + float(residual_mad_multiplier) * mad)
    has_valid = has_valid & torch.nan_to_num(robust_valid, nan=False)
    nearest_xyz = torch.where(has_valid[..., None], nearest_xyz, points_cam)
    return nearest_xyz.to(dtype=points_cam.dtype), has_valid[..., None]


def _coarse_front_surface_mask(
    projected: torch.Tensor,
    point_depth: torch.Tensor,
    image_size_hw: tuple[int, int],
    grid_size: int = 32,
    tolerance_m: float = 0.10,
) -> torch.Tensor:
    image_h, image_w = float(image_size_hw[0]), float(image_size_hw[1])
    flat_xy = projected.reshape(-1, projected.shape[-2], 2)
    flat_z = point_depth.reshape(-1, point_depth.shape[-1])
    gx = torch.floor(flat_xy[..., 0] / max(image_w, 1.0) * grid_size).long()
    gy = torch.floor(flat_xy[..., 1] / max(image_h, 1.0) * grid_size).long()
    in_bounds = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size) & (flat_z > 1e-6)
    cell = (gy.clamp(0, grid_size - 1) * grid_size + gx.clamp(0, grid_size - 1)).long()
    min_depth = flat_z.new_full((flat_z.shape[0], grid_size * grid_size), float("inf"))
    source = torch.where(in_bounds, flat_z, torch.full_like(flat_z, float("inf")))
    min_depth.scatter_reduce_(1, cell, source, reduce="amin", include_self=True)
    nearest = min_depth.gather(1, cell)
    visible = in_bounds & (flat_z <= nearest + float(tolerance_m))
    return visible.reshape(projected.shape[:-1])


def _camera_basis(transl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ray = F.normalize(transl, dim=-1, eps=1e-6)
    x_axis = torch.zeros_like(ray)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(ray)
    y_axis[..., 1] = 1.0
    tangent_x = x_axis - (x_axis * ray).sum(dim=-1, keepdim=True) * ray
    fallback = y_axis - (y_axis * ray).sum(dim=-1, keepdim=True) * ray
    tangent_x = torch.where(torch.linalg.norm(tangent_x, dim=-1, keepdim=True) > 1e-4, tangent_x, fallback)
    tangent_x = F.normalize(tangent_x, dim=-1, eps=1e-6)
    tangent_y = F.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1, eps=1e-6)
    return ray, tangent_x, tangent_y


def _broadcast_affine_feature(
    value: torch.Tensor | None,
    transl: torch.Tensor,
    default: float,
) -> torch.Tensor:
    if value is None:
        return transl.new_full((*transl.shape[:3], 1), float(default))
    value = value.to(device=transl.device, dtype=transl.dtype)
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    if value.ndim == 3:
        return value.reshape(*transl.shape[:2], 1, 1).expand(*transl.shape[:3], 1)
    if value.ndim == 4 and value.shape[2] == 1:
        return value.expand(*transl.shape[:3], 1)
    raise ValueError(f"Unsupported affine feature shape: {tuple(value.shape)}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _smooth_l1_abs(values: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    return F.smooth_l1_loss(values, torch.zeros_like(values), beta=float(beta), reduction="none")


def _zero_last_linear(module: nn.Sequential) -> nn.Sequential:
    last = module[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return module


def _biased_last_linear(module: nn.Sequential, bias: float) -> nn.Sequential:
    last = module[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(bias))
    return module


def _deterministic_fps_indices(vertices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    num_vertices = int(vertices.shape[0])
    count = min(int(count), num_vertices)
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = 0
    distances = torch.full((num_vertices,), float("inf"), dtype=vertices.dtype)
    for idx in range(1, count):
        last = vertices[selected[idx - 1]]
        distances = torch.minimum(distances, torch.linalg.norm(vertices - last, dim=-1))
        selected[idx] = int(torch.argmax(distances).item())
    return selected
