import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.models.token_layout import AggregatorTokenLayout
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class HSIRefinementHead(nn.Module):
    """GRAFT-style geometry-only HSI refinement for SMPL and scene scale.

    This is a project-local concept rewrite. It keeps VGGT patch tokens as the
    scene stream, uses SMPL body anchors as HSI tokens, and predicts residual
    human parameters plus a per-frame affine depth correction.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 5,
        num_heads: int = 8,
        num_iters: int = 3,
        scene_window: int = 3,
        probe_mode: str = "projected",
        probe_window: int = 9,
        smpl_model_dir: str = "",
        image_size: int = 518,
        use_delta_gate: bool = False,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSIRefinementHead requires smpl_model_dir")
        if scene_window % 2 != 1:
            raise ValueError(f"hsi_scene_window must be odd, got {scene_window}")
        self.hidden_dim = int(hidden_dim)
        self.num_iters = int(num_iters)
        self.scene_window = int(scene_window)
        self.probe_mode = str(probe_mode)
        self.probe_window = int(probe_window)
        self.image_size = int(image_size)
        self.use_delta_gate = bool(use_delta_gate)
        if self.probe_mode not in {"projected", "local_nearest"}:
            raise ValueError(f"Unsupported hsi_probe_mode: {self.probe_mode}")
        if self.probe_window % 2 != 1:
            raise ValueError(f"hsi_probe_window must be odd, got {self.probe_window}")
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for param in self.smpl.parameters():
            param.requires_grad = False

        full_body_indices = _deterministic_fps_indices(self.smpl.layer.v_template.detach().float(), 27)
        self.register_buffer("full_body_vertex_indices", full_body_indices, persistent=False)

        self.scene_projs = nn.ModuleList([nn.Linear(dim_in, hidden_dim // 4) for _ in range(4)])
        token_input_dim = 144 + 10 + 3 + 3 + 2 + 3 + 3 + 1 + 3 + 1
        self.token_mlp = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [HSITransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.pose_delta = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 144)))
        self.betas_delta = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 10)))
        self.transl_delta = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)))
        self.scale_delta = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.bias_delta = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.contact_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.delta_gate = _biased_last_linear(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)),
            bias=4.0,
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        token_layout: AggregatorTokenLayout,
        smpl_outputs: dict[str, torch.Tensor],
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        required = ("pred_pose_6d", "pred_poses", "pred_betas", "pred_transl_cam", "pred_confs")
        for key in required:
            if key not in smpl_outputs:
                raise ValueError(f"HSI refinement requires SMPL output {key}")
        if depth is None:
            raise ValueError("HSI refinement requires VGGT depth; set model.enable_depth=true")
        if pose_enc is None:
            raise ValueError("HSI refinement requires pose_enc; set model.enable_camera=true")

        pose6d = smpl_outputs["pred_pose_6d"].float()
        betas = smpl_outputs["pred_betas"].float()
        transl = smpl_outputs["pred_transl_cam"].float()
        confs = smpl_outputs["pred_confs"].float()
        batch_size, num_frames, num_queries, _ = pose6d.shape
        flat_frames = batch_size * num_frames

        depth_hw = _canonical_depth(depth).float()
        height, width = depth_hw.shape[-2:]
        if depth_hw.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Expected depth shape [B,S,H,W], got {tuple(depth_hw.shape)}")
        intrinsics = _flatten_intrinsics(pose_enc, self.image_size).to(device=pose6d.device, dtype=pose6d.dtype)

        scene_features = self._build_scene_features(aggregated_tokens_list, token_layout)
        local_scene_tokens = self._gather_local_scene_tokens(
            scene_features,
            pose6d,
            betas,
            transl,
            depth_hw,
            intrinsics,
            height,
            width,
        )
        hsi_tokens, token_aux = self._tokenize(pose6d, betas, transl, depth_hw, intrinsics, height, width)
        flat_tokens = hsi_tokens.reshape(flat_frames * num_queries, 24, self.hidden_dim)
        flat_scene = local_scene_tokens.reshape(flat_frames * num_queries, 24, self.scene_window * self.scene_window, self.hidden_dim)

        refined_pose6d = pose6d
        refined_betas = betas
        refined_transl = transl
        contact_logits = None
        per_query_log_scale = None
        per_query_bias = None
        gate = None
        tokens = flat_tokens
        for _ in range(max(self.num_iters, 1)):
            for block in self.blocks:
                tokens = block(tokens, flat_scene)
            pooled = tokens.mean(dim=1).reshape(flat_frames, num_queries, self.hidden_dim)
            gate = torch.sigmoid(self.delta_gate(pooled)).reshape(batch_size, num_frames, num_queries, 1)
            if not self.use_delta_gate:
                gate = torch.ones_like(gate)
            refined_pose6d = refined_pose6d + gate * 0.01 * self.pose_delta(pooled).reshape(batch_size, num_frames, num_queries, 144)
            refined_betas = refined_betas + gate * 0.01 * self.betas_delta(pooled).reshape(batch_size, num_frames, num_queries, 10)
            refined_transl = refined_transl + gate * 0.05 * self.transl_delta(pooled).reshape(batch_size, num_frames, num_queries, 3)
            per_query_log_scale = self.scale_delta(pooled).reshape(batch_size, num_frames, num_queries, 1)
            per_query_bias = self.bias_delta(pooled).reshape(batch_size, num_frames, num_queries, 1)
            contact_logits = self.contact_head(tokens).reshape(batch_size, num_frames, num_queries, 24, 1)

        weights = confs.clamp(min=0.0)
        denom = weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        log_scale = (per_query_log_scale * weights).sum(dim=2) / denom.squeeze(2)
        depth_bias = (per_query_bias * weights).sum(dim=2) / denom.squeeze(2)
        scene_scale = torch.exp(log_scale.clamp(min=-3.0, max=3.0))
        refined_poses = rot6d_to_axis_angle(refined_pose6d.reshape(-1, 24, 6)).reshape(batch_size, num_frames, num_queries, 72)
        return {
            "hsi_refined_pred_pose_6d": refined_pose6d,
            "hsi_refined_pred_poses": refined_poses,
            "hsi_refined_pred_betas": refined_betas,
            "hsi_refined_pred_transl_cam": refined_transl,
            "hsi_scene_scale": scene_scale,
            "hsi_scene_depth_bias": depth_bias,
            "hsi_contact_logits": contact_logits,
            "hsi_anchor_depth_residual": token_aux["depth_residual"],
            "hsi_per_query_scene_log_scale": per_query_log_scale,
            "hsi_per_query_scene_depth_bias": per_query_bias,
            "hsi_refine_gate": gate,
        }

    def _build_scene_features(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        token_layout: AggregatorTokenLayout,
    ) -> torch.Tensor:
        features = []
        cached = [tokens for tokens in aggregated_tokens_list if tokens is not None]
        if len(cached) < 4:
            raise ValueError("HSI refinement expects four cached aggregator layers")
        for tokens, proj in zip(cached[-4:], self.scene_projs, strict=True):
            patch_tokens = tokens[:, :, token_layout.patch_start :].float()
            batch_size, num_frames, num_patches, _ = patch_tokens.shape
            features.append(proj(patch_tokens).reshape(batch_size * num_frames, num_patches, -1))
        return torch.cat(features, dim=-1)

    def _gather_local_scene_tokens(
        self,
        scene_features: torch.Tensor,
        pose6d: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        depth_hw: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        anchors = self._anchors_cam(pose6d, betas, transl)
        num_queries = anchors.shape[2]
        projected = _project_points(
            anchors.reshape(anchors.shape[0] * anchors.shape[1] * num_queries, 24, 3),
            intrinsics.repeat_interleave(num_queries, dim=0),
        )
        projected = projected.reshape(*anchors.shape[:3], 24, 2)
        projected = _scale_points_to_depth(projected, self.image_size, height, width)
        grid_h = height // 16
        grid_w = width // 16
        center_x = (projected[..., 0] / float(width) * float(grid_w)).floor().long().clamp(0, grid_w - 1)
        center_y = (projected[..., 1] / float(height) * float(grid_h)).floor().long().clamp(0, grid_h - 1)
        radius = self.scene_window // 2
        offsets = torch.arange(-radius, radius + 1, device=pose6d.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        xs = (center_x[..., None] + ox.reshape(1, 1, 1, 1, -1)).clamp(0, grid_w - 1)
        ys = (center_y[..., None] + oy.reshape(1, 1, 1, 1, -1)).clamp(0, grid_h - 1)
        patch_idx = (ys * grid_w + xs).reshape(scene_features.shape[0], pose6d.shape[2] * 24 * self.scene_window * self.scene_window)
        gather_idx = patch_idx[..., None].expand(-1, -1, scene_features.shape[-1])
        gathered = scene_features.gather(dim=1, index=gather_idx)
        return gathered.reshape(scene_features.shape[0], pose6d.shape[2], 24, self.scene_window * self.scene_window, scene_features.shape[-1])

    def _tokenize(
        self,
        pose6d: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        depth_hw: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchors = self._anchors_cam(pose6d, betas, transl)
        batch_size, num_frames, num_queries, _, _ = anchors.shape
        flat_frames = batch_size * num_frames
        flat_anchors = anchors.reshape(flat_frames * num_queries, 24, 3)
        flat_intrinsics = intrinsics.repeat_interleave(num_queries, dim=0)
        projected = _project_points(flat_anchors, flat_intrinsics).reshape(batch_size, num_frames, num_queries, 24, 2)
        projected_depth = _scale_points_to_depth(projected, self.image_size, height, width)
        px = projected_depth[..., 0].round().long().clamp(0, width - 1)
        py = projected_depth[..., 1].round().long().clamp(0, height - 1)
        frame_idx = torch.arange(flat_frames, device=pose6d.device).reshape(batch_size, num_frames, 1, 1).expand(-1, -1, num_queries, 24)
        flat_depth = depth_hw.reshape(flat_frames, height, width)
        z_scene = flat_depth[frame_idx.reshape(-1), py.reshape(-1), px.reshape(-1)].reshape(batch_size, num_frames, num_queries, 24)
        normals = _estimate_depth_normals(depth_hw, intrinsics, height, width)
        scene_normals = normals.reshape(flat_frames, height, width, 3)[frame_idx.reshape(-1), py.reshape(-1), px.reshape(-1)].reshape(
            batch_size, num_frames, num_queries, 24, 3
        )
        scene_points = _unproject_pixels(projected[..., 0], projected[..., 1], z_scene, intrinsics, num_queries)
        if self.probe_mode == "local_nearest":
            scene_points, scene_normals = _local_nearest_scene_probe(
                depth_hw=depth_hw,
                normals_hw=normals,
                anchors=anchors,
                projected_depth=projected_depth,
                intrinsics=intrinsics,
                image_size=self.image_size,
                window_size=self.probe_window,
            )
        offset = scene_points - anchors
        distance = torch.linalg.norm(offset, dim=-1, keepdim=True)
        depth_residual = (scene_points[..., 2] - anchors[..., 2]).unsqueeze(-1)
        proj_norm = torch.stack(
            [projected_depth[..., 0] / max(float(width - 1), 1.0), projected_depth[..., 1] / max(float(height - 1), 1.0)],
            dim=-1,
        )
        params = torch.cat([pose6d, betas, transl], dim=-1).unsqueeze(3).expand(-1, -1, -1, 24, -1)
        token_input = torch.cat(
            [params, anchors, proj_norm, scene_points, offset, distance, scene_normals, depth_residual],
            dim=-1,
        )
        tokens = self.token_mlp(token_input)
        return tokens, {"depth_residual": depth_residual, "probe_distance": distance}

    def _anchors_cam(self, pose6d: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_queries, _ = pose6d.shape
        poses = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, joints = self.smpl(poses.float(), betas.reshape(-1, betas.shape[-1]).float())
        vertices = vertices.to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
        joints = joints[:, :24].to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
        body = joints[:, 1:22]
        left_hand = 0.5 * (joints[:, 20] + joints[:, 22])
        right_hand = 0.5 * (joints[:, 21] + joints[:, 23])
        full_body = vertices[:, self.full_body_vertex_indices.to(vertices.device)].mean(dim=1)
        anchors = torch.cat([body, left_hand[:, None], right_hand[:, None], full_body[:, None]], dim=1)
        return anchors.reshape(batch_size, num_frames, num_queries, 24, 3)


class HSITransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, tokens: torch.Tensor, local_scene_tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.self_attn(self.self_norm(tokens), self.self_norm(tokens), self.self_norm(tokens), need_weights=False)[0]
        q = self.cross_norm(tokens).reshape(tokens.shape[0] * tokens.shape[1], 1, tokens.shape[2])
        kv = local_scene_tokens.reshape(tokens.shape[0] * tokens.shape[1], local_scene_tokens.shape[2], local_scene_tokens.shape[3])
        cross = self.cross_attn(q, kv, kv, need_weights=False)[0].reshape_as(tokens)
        tokens = tokens + cross
        return tokens + self.ffn(self.ffn_norm(tokens))


def _canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _flatten_intrinsics(pose_enc: torch.Tensor, image_size: int) -> torch.Tensor:
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(image_size, image_size), build_intrinsics=True)
    return intrinsics.reshape(-1, 3, 3)


def _project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[:, None, 0, 0] * points_cam[..., 0] / z + intrinsics[:, None, 0, 2]
    y = intrinsics[:, None, 1, 1] * points_cam[..., 1] / z + intrinsics[:, None, 1, 2]
    return torch.stack([x, y], dim=-1)


def _unproject_pixels(px: torch.Tensor, py: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor, num_queries: int) -> torch.Tensor:
    batch_size, num_frames = px.shape[:2]
    flat_intrinsics = intrinsics.reshape(batch_size, num_frames, 1, 1, 3, 3).expand(-1, -1, num_queries, 24, -1, -1)
    fx = flat_intrinsics[..., 0, 0]
    fy = flat_intrinsics[..., 1, 1]
    cx = flat_intrinsics[..., 0, 2]
    cy = flat_intrinsics[..., 1, 2]
    x = (px - cx) / fx.clamp(min=1e-6) * z
    y = (py - cy) / fy.clamp(min=1e-6) * z
    return torch.stack([x, y, z], dim=-1)


def _scale_points_to_depth(points_2d: torch.Tensor, image_size: int, depth_height: int, depth_width: int) -> torch.Tensor:
    scale = points_2d.new_tensor(
        [
            float(depth_width) / float(image_size),
            float(depth_height) / float(image_size),
        ]
    )
    return points_2d * scale


def _estimate_depth_normals(depth_hw: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    dzdx = F.pad(depth_hw[..., :, 2:] - depth_hw[..., :, :-2], (1, 1, 0, 0)) * 0.5
    dzdy = F.pad(depth_hw[..., 2:, :] - depth_hw[..., :-2, :], (0, 0, 1, 1)) * 0.5
    normals = torch.stack([-dzdx, -dzdy, torch.ones_like(depth_hw)], dim=-1)
    return F.normalize(normals, dim=-1)


def _local_nearest_scene_probe(
    depth_hw: torch.Tensor,
    normals_hw: torch.Tensor,
    anchors: torch.Tensor,
    projected_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_queries, num_tokens, _ = anchors.shape
    flat_frames = batch_size * num_frames
    height, width = depth_hw.shape[-2:]
    radius = max(int(window_size), 1) // 2
    offsets = torch.arange(-radius, radius + 1, device=anchors.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, 1, 1, -1)
    oy = oy.reshape(1, 1, 1, 1, -1)

    center_x = projected_depth[..., 0].round().long()
    center_y = projected_depth[..., 1].round().long()
    xs = center_x[..., None] + ox
    ys = center_y[..., None] + oy
    local_valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs.clamp(0, width - 1)
    ys = ys.clamp(0, height - 1)

    frame_idx = torch.arange(flat_frames, device=anchors.device).reshape(batch_size, num_frames, 1, 1, 1)
    flat_depth = depth_hw.reshape(flat_frames, height, width)
    flat_normals = normals_hw.reshape(flat_frames, height, width, 3)
    local_depth = flat_depth[frame_idx, ys, xs]
    local_valid = local_valid & torch.isfinite(local_depth) & (local_depth > 1e-6)

    flat_intrinsics = intrinsics.reshape(batch_size, num_frames, 1, 1, 1, 3, 3)
    fx = flat_intrinsics[..., 0, 0].clamp(min=1e-6)
    fy = flat_intrinsics[..., 1, 1].clamp(min=1e-6)
    cx = flat_intrinsics[..., 0, 2]
    cy = flat_intrinsics[..., 1, 2]
    image_x = xs.to(dtype=local_depth.dtype) * (float(image_size) / float(width))
    image_y = ys.to(dtype=local_depth.dtype) * (float(image_size) / float(height))
    scene_x = (image_x - cx.to(dtype=local_depth.dtype)) * local_depth / fx.to(dtype=local_depth.dtype)
    scene_y = (image_y - cy.to(dtype=local_depth.dtype)) * local_depth / fy.to(dtype=local_depth.dtype)
    scene_xyz = torch.stack([scene_x, scene_y, local_depth], dim=-1)

    dist = torch.linalg.norm(scene_xyz - anchors[..., None, :].to(dtype=scene_xyz.dtype), dim=-1)
    dist = torch.where(local_valid, dist, torch.full_like(dist, float("inf")))
    nearest_idx = dist.argmin(dim=-1)
    has_valid = torch.isfinite(dist.gather(dim=-1, index=nearest_idx[..., None]).squeeze(-1))
    gather_xyz = nearest_idx[..., None, None].expand(*nearest_idx.shape, 1, 3)
    nearest_xyz = scene_xyz.gather(dim=-2, index=gather_xyz).squeeze(-2)
    local_normals = flat_normals[frame_idx.expand_as(xs), ys, xs]
    nearest_normals = local_normals.gather(dim=-2, index=gather_xyz).squeeze(-2)

    fallback_x = center_x.clamp(0, width - 1)
    fallback_y = center_y.clamp(0, height - 1)
    fallback_depth = flat_depth[
        torch.arange(flat_frames, device=anchors.device).reshape(batch_size, num_frames, 1, 1).expand(-1, -1, num_queries, num_tokens),
        fallback_y,
        fallback_x,
    ]
    fallback_image_x = fallback_x.to(dtype=local_depth.dtype) * (float(image_size) / float(width))
    fallback_image_y = fallback_y.to(dtype=local_depth.dtype) * (float(image_size) / float(height))
    fallback_xyz = _unproject_pixels(fallback_image_x, fallback_image_y, fallback_depth, intrinsics, num_queries)
    fallback_normals = flat_normals[
        torch.arange(flat_frames, device=anchors.device).reshape(batch_size, num_frames, 1, 1).expand(-1, -1, num_queries, num_tokens),
        fallback_y,
        fallback_x,
    ]
    nearest_xyz = torch.where(has_valid[..., None], nearest_xyz, fallback_xyz)
    nearest_normals = torch.where(has_valid[..., None], nearest_normals, fallback_normals)
    nearest_normals = F.normalize(torch.nan_to_num(nearest_normals, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    return nearest_xyz.to(dtype=anchors.dtype), nearest_normals.to(dtype=anchors.dtype)


def _deterministic_fps_indices(vertices: torch.Tensor, count: int) -> torch.Tensor:
    verts = vertices.reshape(-1, 3)
    first = torch.argmin(verts[:, 1])
    indices = [first]
    dist = torch.full((verts.shape[0],), float("inf"), dtype=verts.dtype, device=verts.device)
    for _ in range(1, int(count)):
        last = verts[indices[-1]]
        dist = torch.minimum(dist, torch.linalg.norm(verts - last, dim=-1))
        indices.append(torch.argmax(dist))
    return torch.stack(indices).long()


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
