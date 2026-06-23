import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.token_layout import AggregatorTokenLayout
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class CameraRayTranslationRefiner(nn.Module):
    """Camera-aware residual translation branch in a ray-aligned basis.

    The branch keeps the original camera-space translation as the anchor, then
    predicts a bounded residual along the box-center camera ray and its tangent
    plane. It also exposes a depth prior from bbox height and focal length, but
    uses it as a learnable residual feature, not as raw-depth supervision.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = 512,
        max_ray_delta_m: float = 0.60,
        max_tangent_delta_m: float = 0.35,
        max_log_depth_delta: float = 0.50,
        max_box_prior_weight: float = 0.50,
        human_height_prior_m: float = 1.70,
        use_log_depth: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.max_ray_delta_m = float(max_ray_delta_m)
        self.max_tangent_delta_m = float(max_tangent_delta_m)
        self.max_log_depth_delta = float(max_log_depth_delta)
        self.max_box_prior_weight = float(max_box_prior_weight)
        self.human_height_prior_m = float(human_height_prior_m)
        self.use_log_depth = bool(use_log_depth)
        extra_dim = 144 + 10 + 3 + 1 + 2 + 1 + 3 + 4 + 3 + 4 + 2
        out_dim = 6
        self.hidden_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + extra_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        _zero_last_linear(self.mlp)

    def forward(
        self,
        hidden: torch.Tensor,
        base_transl: torch.Tensor,
        pose6d: torch.Tensor,
        betas: torch.Tensor,
        boxes: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or base_transl.ndim != 3 or boxes.ndim != 3:
            raise ValueError(
                "CameraRayTranslationRefiner expects hidden/base_transl/boxes with shapes "
                f"(N,Q,C)/(N,Q,3)/(N,Q,4), got {hidden.shape}, {base_transl.shape}, {boxes.shape}"
            )
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")
        if hidden.shape[:2] != base_transl.shape[:2] or hidden.shape[:2] != boxes.shape[:2]:
            raise ValueError("Translation refiner hidden/base_transl/boxes must share (N,Q)")
        if pose6d.shape[:2] != hidden.shape[:2] or pose6d.shape[-1] != 144:
            raise ValueError(f"Expected pose6d shape (N,Q,144), got {pose6d.shape}")
        if betas.shape[:2] != hidden.shape[:2] or betas.shape[-1] != 10:
            raise ValueError(f"Expected betas shape (N,Q,10), got {betas.shape}")
        if intrinsics.shape[0] != hidden.shape[0]:
            raise ValueError(f"Intrinsics frame count {intrinsics.shape[0]} does not match hidden {hidden.shape[0]}")

        boxes = boxes.to(device=hidden.device, dtype=hidden.dtype).clamp(0.0, 1.0)
        intrinsics = intrinsics.to(device=hidden.device, dtype=hidden.dtype)
        base_transl = base_transl.to(device=hidden.device, dtype=hidden.dtype)
        pose6d = pose6d.to(device=hidden.device, dtype=hidden.dtype)
        betas = betas.to(device=hidden.device, dtype=hidden.dtype)
        ray, tangent_x, tangent_y, k_features = _camera_ray_basis(boxes, intrinsics, self.image_size)

        ray_depth = (base_transl * ray).sum(dim=-1, keepdim=True)
        tangent_coord_x = (base_transl * tangent_x).sum(dim=-1, keepdim=True)
        tangent_coord_y = (base_transl * tangent_y).sum(dim=-1, keepdim=True)
        box_area = (boxes[..., 2:3] * boxes[..., 3:4]).clamp(min=1e-6)
        box_aspect = boxes[..., 2:3] / boxes[..., 3:4].clamp(min=1e-6)
        box_depth_prior = _bbox_height_depth_prior(
            boxes=boxes,
            intrinsics=intrinsics,
            image_size=self.image_size,
            human_height_prior_m=self.human_height_prior_m,
        )
        safe_ray_depth = ray_depth.abs().clamp(min=1e-4)
        safe_box_depth = box_depth_prior.clamp(min=1e-4)

        features = torch.cat(
            [
                self.hidden_norm(hidden.float()).to(dtype=hidden.dtype),
                pose6d,
                betas,
                base_transl,
                ray_depth,
                torch.cat([tangent_coord_x, tangent_coord_y], dim=-1),
                box_depth_prior,
                torch.log(safe_ray_depth),
                torch.log(safe_box_depth),
                torch.log(safe_box_depth / safe_ray_depth),
                boxes,
                ray,
                k_features,
                torch.log(box_area),
                torch.log(box_aspect.clamp(min=1e-6)),
            ],
            dim=-1,
        )
        raw_delta = self.mlp(features.float()).to(dtype=hidden.dtype)
        ray_delta = torch.tanh(raw_delta[..., 0:1]) * self.max_ray_delta_m
        tangent_delta = torch.tanh(raw_delta[..., 1:3]) * self.max_tangent_delta_m
        if self.use_log_depth:
            log_depth_delta = torch.tanh(raw_delta[..., 3:4]) * self.max_log_depth_delta
            depth_scale_anchor = ray_depth.abs().clamp(min=1e-4)
            refined_ray_depth = ray_depth + depth_scale_anchor * (torch.exp(log_depth_delta) - 1.0) + ray_delta
        else:
            log_depth_delta = raw_delta.new_zeros(*raw_delta.shape[:2], 1)
            refined_ray_depth = ray_depth + ray_delta
        box_log_depth_delta = torch.tanh(raw_delta[..., 4:5]) * self.max_log_depth_delta
        box_prior_weight = torch.tanh(raw_delta[..., 5:6]) * self.max_box_prior_weight
        refined_box_depth = safe_box_depth * torch.exp(box_log_depth_delta)
        refined_ray_depth = refined_ray_depth + box_prior_weight * (refined_box_depth - safe_ray_depth)
        refined_tangent_x = tangent_coord_x + tangent_delta[..., 0:1]
        refined_tangent_y = tangent_coord_y + tangent_delta[..., 1:2]
        refined_transl = refined_ray_depth * ray + refined_tangent_x * tangent_x + refined_tangent_y * tangent_y
        delta = refined_transl - base_transl
        return {
            "pred_transl_cam": refined_transl,
            "pred_transl_cam_delta": delta,
            "pred_transl_ray_delta": torch.cat(
                [ray_delta, tangent_delta, log_depth_delta, box_log_depth_delta, box_prior_weight],
                dim=-1,
            ),
            "pred_transl_ray_dir": ray,
            "pred_transl_tangent_x": tangent_x,
            "pred_transl_tangent_y": tangent_y,
            "base_pred_transl_ray_depth": ray_depth,
            "base_pred_transl_tangent": torch.cat([tangent_coord_x, tangent_coord_y], dim=-1),
            "pred_transl_ray_depth": refined_ray_depth,
            "pred_transl_tangent": torch.cat([refined_tangent_x, refined_tangent_y], dim=-1),
            "pred_transl_box_depth_prior": box_depth_prior,
            "pred_transl_box_prior_weight": box_prior_weight,
        }


class RayOffsetDepthTranslationDecoder(nn.Module):
    """Geometry-decoded SMPL translation seed.

    The decoder predicts a root position in a camera-ray coordinate system:
    bbox center + intrinsics define the ray and tangent basis; the network only
    predicts a log-depth residual around a bbox-height prior plus two bounded
    tangent offsets.  It does not consume VGGT depth or HSI-scaled depth.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = 512,
        max_log_depth_delta: float = 1.00,
        max_ray_delta_m: float = 1.50,
        max_tangent_offset_m: float = 1.00,
        human_height_prior_m: float = 1.70,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.max_log_depth_delta = float(max_log_depth_delta)
        self.max_ray_delta_m = float(max_ray_delta_m)
        self.max_tangent_offset_m = float(max_tangent_offset_m)
        self.human_height_prior_m = float(human_height_prior_m)
        extra_dim = 144 + 10 + 4 + 3 + 4 + 1 + 2
        self.hidden_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + extra_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        _zero_last_linear(self.mlp)

    def forward(
        self,
        hidden: torch.Tensor,
        pose6d: torch.Tensor,
        betas: torch.Tensor,
        boxes: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or pose6d.ndim != 3 or betas.ndim != 3 or boxes.ndim != 3:
            raise ValueError(
                "RayOffsetDepthTranslationDecoder expects hidden/pose/betas/boxes with "
                f"(N,Q,C)/(N,Q,144)/(N,Q,10)/(N,Q,4), got "
                f"{hidden.shape}/{pose6d.shape}/{betas.shape}/{boxes.shape}"
            )
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics shape (N,3,3), got {intrinsics.shape}")
        if hidden.shape[:2] != boxes.shape[:2] or hidden.shape[:2] != pose6d.shape[:2] or hidden.shape[:2] != betas.shape[:2]:
            raise ValueError("Ray-offset-depth decoder inputs must share (N,Q)")

        boxes = boxes.to(device=hidden.device, dtype=hidden.dtype).clamp(0.0, 1.0)
        intrinsics = intrinsics.to(device=hidden.device, dtype=hidden.dtype)
        pose6d = pose6d.to(device=hidden.device, dtype=hidden.dtype)
        betas = betas.to(device=hidden.device, dtype=hidden.dtype)
        ray, tangent_x, tangent_y, k_features = _camera_ray_basis(boxes, intrinsics, self.image_size)
        box_depth_prior = _bbox_height_depth_prior(
            boxes=boxes,
            intrinsics=intrinsics,
            image_size=self.image_size,
            human_height_prior_m=self.human_height_prior_m,
        )
        box_area = (boxes[..., 2:3] * boxes[..., 3:4]).clamp(min=1e-6)
        box_aspect = boxes[..., 2:3] / boxes[..., 3:4].clamp(min=1e-6)
        features = torch.cat(
            [
                self.hidden_norm(hidden.float()).to(dtype=hidden.dtype),
                pose6d,
                betas,
                boxes,
                ray,
                k_features,
                torch.log(box_depth_prior.clamp(min=1e-4)),
                torch.log(box_area),
                torch.log(box_aspect.clamp(min=1e-6)),
            ],
            dim=-1,
        )
        raw = self.mlp(features.float()).to(dtype=hidden.dtype)
        log_depth_delta = torch.tanh(raw[..., 0:1]) * self.max_log_depth_delta
        ray_delta = torch.tanh(raw[..., 1:2]) * self.max_ray_delta_m
        tangent = torch.tanh(raw[..., 2:4]) * self.max_tangent_offset_m
        ray_depth = box_depth_prior * torch.exp(log_depth_delta) + ray_delta
        transl = ray_depth * ray + tangent[..., 0:1] * tangent_x + tangent[..., 1:2] * tangent_y
        return {
            "pred_transl_cam": transl,
            "seed_pred_transl_cam": transl,
            "pred_transl_ray_delta": torch.cat([ray_delta, tangent, log_depth_delta], dim=-1),
            "pred_transl_ray_dir": ray,
            "pred_transl_tangent_x": tangent_x,
            "pred_transl_tangent_y": tangent_y,
            "pred_transl_ray_depth": ray_depth,
            "pred_transl_tangent": tangent,
            "pred_transl_box_depth_prior": box_depth_prior,
        }


class TrackTemporalTranslationRefiner(nn.Module):
    """Track-aware camera/world translation trajectory refiner.

    It keeps the geometry-decoded per-frame seed as an anchor and predicts a
    bounded velocity residual for each track.  With zero initialization the
    module is nearly identity, so enabling it does not destroy a trained seed.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = 512,
        max_velocity_delta_m: float = 0.25,
        gate_bias: float = 2.5,
        use_world: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        self.max_velocity_delta_m = float(max_velocity_delta_m)
        self.use_world = bool(use_world)
        self.image_size = int(image_size)
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in * 2 + 9, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        _zero_last_linear(self.mlp)
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias[3], float(gate_bias))

    def forward(
        self,
        hidden: torch.Tensor,
        seed_transl_cam: torch.Tensor,
        pose_enc: torch.Tensor,
        track_ids: torch.Tensor | None = None,
        track_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 4 or seed_transl_cam.ndim != 4:
            raise ValueError(f"Expected hidden/seed shapes (B,T,Q,C)/(B,T,Q,3), got {hidden.shape}/{seed_transl_cam.shape}")
        if hidden.shape[:3] != seed_transl_cam.shape[:3] or seed_transl_cam.shape[-1] != 3:
            raise ValueError("Temporal translation hidden and seed_transl_cam must share (B,T,Q)")
        batch_size, num_frames, num_queries, _ = hidden.shape
        if num_frames <= 1:
            zeros = seed_transl_cam.new_zeros(batch_size, num_frames, num_queries, 3)
            gate = seed_transl_cam.new_ones(batch_size, num_frames, num_queries, 1)
            return {
                "pred_transl_cam": seed_transl_cam,
                "pred_transl_temporal_delta": zeros,
                "pred_transl_temporal_velocity_delta": zeros,
                "pred_transl_temporal_gate": gate,
            }

        work_transl, extrinsics = self._to_work_coords(seed_transl_cam, pose_enc)
        norm_hidden = self.norm(hidden.float()).to(dtype=hidden.dtype)
        refined_frames = [work_transl[:, 0]]
        velocity_delta_frames = [work_transl.new_zeros(batch_size, num_queries, 3)]
        gate_frames = [work_transl.new_ones(batch_size, num_queries, 1)]

        for frame_idx in range(1, num_frames):
            prev_index = self._previous_query_indices(track_ids, track_mask, frame_idx, batch_size, num_queries, hidden.device)
            prev_hidden = torch.gather(
                norm_hidden[:, frame_idx - 1],
                dim=1,
                index=prev_index[..., None].expand(-1, -1, norm_hidden.shape[-1]),
            )
            prev_refined = torch.gather(
                refined_frames[-1],
                dim=1,
                index=prev_index[..., None].expand(-1, -1, 3),
            )
            prev_seed = torch.gather(
                work_transl[:, frame_idx - 1],
                dim=1,
                index=prev_index[..., None].expand(-1, -1, 3),
            )
            seed_velocity = work_transl[:, frame_idx] - prev_seed
            mlp_in = torch.cat(
                [
                    norm_hidden[:, frame_idx],
                    prev_hidden,
                    seed_velocity,
                    work_transl[:, frame_idx],
                    prev_seed,
                ],
                dim=-1,
            )
            raw = self.mlp(mlp_in.float()).to(dtype=hidden.dtype)
            velocity_delta = torch.tanh(raw[..., :3]) * self.max_velocity_delta_m
            seed_gate = torch.sigmoid(raw[..., 3:4])
            rollout = prev_refined + seed_velocity + velocity_delta
            refined = seed_gate * work_transl[:, frame_idx] + (1.0 - seed_gate) * rollout
            refined_frames.append(refined)
            velocity_delta_frames.append(velocity_delta)
            gate_frames.append(seed_gate)

        refined_work = torch.stack(refined_frames, dim=1)
        velocity_delta = torch.stack(velocity_delta_frames, dim=1)
        gate = torch.stack(gate_frames, dim=1)
        refined_cam = self._from_work_coords(refined_work, extrinsics)
        outputs = {
            "pred_transl_cam": refined_cam,
            "pred_transl_temporal_delta": refined_cam - seed_transl_cam,
            "pred_transl_temporal_velocity_delta": velocity_delta,
            "pred_transl_temporal_gate": gate,
        }
        if self.use_world:
            outputs["pred_transl_world_seed"] = work_transl
            outputs["pred_transl_world_refined"] = refined_work
        return outputs

    def _to_work_coords(self, transl_cam: torch.Tensor, pose_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_world:
            return transl_cam, None
        extrinsics, _ = encoding_to_camera(pose_enc, image_size_hw=(self.image_size, self.image_size), build_intrinsics=False)
        extrinsics = extrinsics.to(device=transl_cam.device, dtype=transl_cam.dtype)
        rot = extrinsics[..., :3, :3]
        trans = extrinsics[..., :3, 3]
        transl_world = torch.matmul(rot.transpose(-1, -2)[:, :, None], (transl_cam - trans[:, :, None])[..., None]).squeeze(-1)
        return transl_world, extrinsics

    def _from_work_coords(self, transl_work: torch.Tensor, extrinsics: torch.Tensor | None) -> torch.Tensor:
        if not self.use_world:
            return transl_work
        if extrinsics is None:
            raise RuntimeError("Temporal translation world mode lost extrinsics")
        rot = extrinsics[..., :3, :3]
        trans = extrinsics[..., :3, 3]
        return torch.matmul(rot[:, :, None], transl_work[..., None]).squeeze(-1) + trans[:, :, None]

    @staticmethod
    def _previous_query_indices(
        track_ids: torch.Tensor | None,
        track_mask: torch.Tensor | None,
        frame_idx: int,
        batch_size: int,
        num_queries: int,
        device: torch.device,
    ) -> torch.Tensor:
        default = torch.arange(num_queries, device=device, dtype=torch.long).reshape(1, num_queries).expand(batch_size, -1)
        if track_ids is None:
            return default
        prev_ids = track_ids[:, frame_idx - 1].to(device=device)
        curr_ids = track_ids[:, frame_idx].to(device=device)
        prev_ok = torch.ones_like(prev_ids, dtype=torch.bool)
        curr_ok = torch.ones_like(curr_ids, dtype=torch.bool)
        if track_mask is not None:
            prev_ok = track_mask[:, frame_idx - 1].to(device=device).bool()
            curr_ok = track_mask[:, frame_idx].to(device=device).bool()
        out = default.clone()
        for batch_idx in range(batch_size):
            for query_idx in range(num_queries):
                if not bool(curr_ok[batch_idx, query_idx]):
                    continue
                matches = torch.nonzero((prev_ids[batch_idx] == curr_ids[batch_idx, query_idx]) & prev_ok[batch_idx], as_tuple=False).flatten()
                if matches.numel() > 0:
                    out[batch_idx, query_idx] = matches[0]
        return out


class SMPLRegressionHead(nn.Module):
    """SAT-HMR-style residual regression from mean SMPL pose and shape."""

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_joints: int = 24,
        num_betas: int = 10,
        mlp_layers: int = 3,
        mean_pose_6d: torch.Tensor | None = None,
        mean_shape: torch.Tensor | None = None,
        return_aux: bool = False,
        predict_boxes: bool = False,
        bbox_mode: str = "direct",
        predict_id_embed: bool = False,
        id_embed_dim: int = 256,
        translation_output_mode: str = "direct",
        translation_decode_hidden_dim: int = 512,
        translation_decode_max_log_depth_delta: float = 1.00,
        translation_decode_max_ray_delta_m: float = 1.50,
        translation_decode_max_tangent_offset_m: float = 1.00,
        translation_decode_human_height_prior_m: float = 1.70,
        enable_translation_refine: bool = False,
        translation_refine_hidden_dim: int = 512,
        translation_refine_max_ray_delta_m: float = 0.60,
        translation_refine_max_tangent_delta_m: float = 0.35,
        translation_refine_max_log_depth_delta: float = 0.50,
        translation_refine_max_box_prior_weight: float = 0.50,
        translation_refine_human_height_prior_m: float = 1.70,
        translation_refine_use_log_depth: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_layers = num_layers
        self.num_joints = num_joints
        self.num_betas = num_betas
        self.return_aux = return_aux
        self.predict_boxes = predict_boxes
        self.bbox_mode = bbox_mode
        self.predict_id_embed = predict_id_embed
        self.translation_output_mode = str(translation_output_mode or "direct")
        self.enable_translation_refine = enable_translation_refine
        self.image_size = int(image_size)
        if self.bbox_mode not in {"direct", "reference_residual"}:
            raise ValueError(f"Unsupported bbox_mode: {self.bbox_mode}")
        if self.translation_output_mode not in {"direct", "ray_offset_depth"}:
            raise ValueError(f"Unsupported translation_output_mode: {self.translation_output_mode}")
        pose_dim = num_joints * 6

        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.pose_heads = _get_clones(MLP(dim_in, hidden_dim, pose_dim, mlp_layers), num_layers)
        self.shape_heads = _get_clones(MLP(dim_in, hidden_dim, num_betas, mlp_layers), num_layers)
        self.transl_cam_heads = _get_clones(MLP(dim_in, hidden_dim, 3, mlp_layers), num_layers)
        if self.translation_output_mode == "ray_offset_depth":
            self.translation_decode_heads = _get_clones(
                RayOffsetDepthTranslationDecoder(
                    dim_in=dim_in,
                    hidden_dim=translation_decode_hidden_dim,
                    max_log_depth_delta=translation_decode_max_log_depth_delta,
                    max_ray_delta_m=translation_decode_max_ray_delta_m,
                    max_tangent_offset_m=translation_decode_max_tangent_offset_m,
                    human_height_prior_m=translation_decode_human_height_prior_m,
                    image_size=image_size,
                ),
                num_layers,
            )
        else:
            self.translation_decode_heads = None
        self.conf_heads = _get_clones(MLP(dim_in, hidden_dim, 1, mlp_layers), num_layers)
        if predict_boxes:
            if self.bbox_mode == "direct":
                self.box_heads = _get_clones(MLP(dim_in, hidden_dim, 4, mlp_layers), num_layers)
                self.box_delta_heads = None
            else:
                self.box_heads = None
                self.box_delta_heads = _get_clones(MLP(dim_in, hidden_dim, 4, mlp_layers), num_layers)
        else:
            self.box_heads = None
            self.box_delta_heads = None
        if predict_id_embed:
            id_hidden_dim = max(hidden_dim // 2, 1)
            self.id_embed_head = nn.Sequential(
                nn.Linear(dim_in, id_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(id_hidden_dim),
                nn.Linear(id_hidden_dim, id_embed_dim),
            )
        else:
            self.id_embed_head = None
        if enable_translation_refine:
            self.translation_refiner = CameraRayTranslationRefiner(
                dim_in=dim_in,
                hidden_dim=translation_refine_hidden_dim,
                max_ray_delta_m=translation_refine_max_ray_delta_m,
                max_tangent_delta_m=translation_refine_max_tangent_delta_m,
                max_log_depth_delta=translation_refine_max_log_depth_delta,
                max_box_prior_weight=translation_refine_max_box_prior_weight,
                human_height_prior_m=translation_refine_human_height_prior_m,
                use_log_depth=translation_refine_use_log_depth,
                image_size=image_size,
            )
        else:
            self.translation_refiner = None

        if mean_pose_6d is None:
            mean_pose_6d = _identity_pose_6d(num_joints)
        if mean_shape is None:
            mean_shape = torch.zeros(num_betas)
        if mean_pose_6d.numel() != pose_dim:
            raise ValueError(f"mean_pose_6d must contain {pose_dim} values, got {mean_pose_6d.numel()}")
        if mean_shape.numel() != num_betas:
            raise ValueError(f"mean_shape must contain {num_betas} values, got {mean_shape.numel()}")

        self.register_buffer("mean_pose_6d", mean_pose_6d.reshape(1, 1, pose_dim).float(), persistent=False)
        self.register_buffer("mean_shape", mean_shape.reshape(1, 1, num_betas).float(), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_boxes: torch.Tensor | None = None,
        pose_enc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected hidden_states shape (L, B, Q, C), got {hidden_states.shape}")
        num_layers, batch_size, num_queries, _ = hidden_states.shape
        if num_layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden-state layers, got {num_layers}")
        if self.bbox_mode == "reference_residual":
            if reference_boxes is None:
                raise ValueError("bbox_mode='reference_residual' requires reference_boxes")
            if reference_boxes.shape != (batch_size, num_queries, 4):
                raise ValueError(f"Expected reference_boxes shape {(batch_size, num_queries, 4)}, got {reference_boxes.shape}")
            current_reference = reference_boxes.to(device=hidden_states.device, dtype=hidden_states.dtype).clamp(1e-4, 1.0 - 1e-4)
        else:
            current_reference = None

        pose_6d = self.mean_pose_6d.to(dtype=hidden_states.dtype, device=hidden_states.device).expand(batch_size, num_queries, -1)
        shape = self.mean_shape.to(dtype=hidden_states.dtype, device=hidden_states.device).expand(batch_size, num_queries, -1)
        aux_outputs: dict[str, list[torch.Tensor]] = {
            "pred_poses": [],
            "pred_pose_6d": [],
            "pred_betas": [],
            "pred_confs": [],
            "pred_transl_cam": [],
        }
        if self.predict_boxes:
            aux_outputs["pred_boxes"] = []

        pred_transl_cam = None
        pred_confs = None
        pred_poses = None
        pred_boxes = None
        last_hidden = None
        legacy_pred_transl_cam = None
        translation_decode_outputs = None
        intrinsics_for_translation = None
        if self.translation_decode_heads is not None:
            if pose_enc is None:
                raise ValueError("translation_output_mode='ray_offset_depth' requires pose_enc; set model.enable_camera=true")
            intrinsics_for_translation = _flatten_intrinsics_from_pose_enc(pose_enc, image_size=self.image_size)
        for layer_idx in range(self.num_layers):
            hidden = self.norm(hidden_states[layer_idx].float())
            last_hidden = hidden
            pose_6d = pose_6d + self.pose_heads[layer_idx](hidden)
            shape = shape + self.shape_heads[layer_idx](hidden)
            if self.box_heads is not None:
                pred_boxes = torch.sigmoid(self.box_heads[layer_idx](hidden))
            elif self.box_delta_heads is not None:
                if current_reference is None:
                    raise RuntimeError("reference_residual bbox head missing current_reference")
                delta = self.box_delta_heads[layer_idx](hidden)
                pred_boxes = torch.sigmoid(inverse_sigmoid(current_reference) + delta)
                current_reference = pred_boxes.detach().clamp(1e-4, 1.0 - 1e-4)
            legacy_pred_transl_cam = self.transl_cam_heads[layer_idx](hidden)
            if self.translation_decode_heads is not None:
                boxes_for_translation = reference_boxes
                if boxes_for_translation is None and pred_boxes is not None:
                    boxes_for_translation = pred_boxes.detach()
                if boxes_for_translation is None:
                    raise ValueError("translation_output_mode='ray_offset_depth' requires reference_boxes or pred_boxes")
                if intrinsics_for_translation is None:
                    raise RuntimeError("Ray-offset-depth translation missing intrinsics")
                translation_decode_outputs = self.translation_decode_heads[layer_idx](
                    hidden=hidden,
                    pose6d=pose_6d,
                    betas=shape,
                    boxes=boxes_for_translation,
                    intrinsics=intrinsics_for_translation,
                )
                pred_transl_cam = translation_decode_outputs["pred_transl_cam"]
            else:
                pred_transl_cam = legacy_pred_transl_cam
            pred_confs = torch.sigmoid(self.conf_heads[layer_idx](hidden))
            pred_poses = rot6d_to_axis_angle(pose_6d)

            aux_outputs["pred_poses"].append(pred_poses)
            aux_outputs["pred_pose_6d"].append(pose_6d)
            aux_outputs["pred_betas"].append(shape)
            aux_outputs["pred_confs"].append(pred_confs)
            aux_outputs["pred_transl_cam"].append(pred_transl_cam)
            if pred_boxes is not None:
                aux_outputs["pred_boxes"].append(pred_boxes)

        if pred_poses is None or pred_transl_cam is None or pred_confs is None:
            raise RuntimeError("SMPLRegressionHead produced no predictions")

        base_pred_transl_cam = legacy_pred_transl_cam if legacy_pred_transl_cam is not None else pred_transl_cam
        refine_anchor_transl_cam = pred_transl_cam
        translation_refine_outputs = None
        if self.translation_refiner is not None:
            if last_hidden is None:
                raise RuntimeError("Translation refiner needs final hidden states")
            if pose_enc is None:
                raise ValueError("SMPL translation refiner requires pose_enc; set model.enable_camera=true")
            intrinsics = _flatten_intrinsics_from_pose_enc(pose_enc, image_size=self.image_size)
            boxes_for_refine = reference_boxes
            if boxes_for_refine is None and pred_boxes is not None:
                boxes_for_refine = pred_boxes.detach()
            if boxes_for_refine is None:
                raise ValueError("SMPL translation refiner requires reference_boxes or pred_boxes")
            translation_refine_outputs = self.translation_refiner(
                hidden=last_hidden,
                base_transl=refine_anchor_transl_cam,
                pose6d=pose_6d,
                betas=shape,
                boxes=boxes_for_refine,
                intrinsics=intrinsics,
            )
            pred_transl_cam = translation_refine_outputs["pred_transl_cam"]

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "pred_poses": pred_poses,
            "pred_pose_6d": pose_6d,
            "pred_betas": shape,
            "pred_confs": pred_confs,
            "pred_transl_cam": pred_transl_cam,
            # Temporary alias for older callers/checkpoints. New code should use pred_transl_cam.
            "pred_cam": pred_transl_cam,
        }
        if translation_decode_outputs is not None:
            outputs.update(translation_decode_outputs)
            outputs["pred_transl_cam"] = pred_transl_cam
            outputs["pred_cam"] = pred_transl_cam
            outputs["base_pred_transl_cam"] = base_pred_transl_cam
        if translation_refine_outputs is not None:
            outputs.update(translation_refine_outputs)
            outputs["base_pred_transl_cam"] = base_pred_transl_cam
        if pred_boxes is not None:
            outputs["pred_boxes"] = pred_boxes
        if self.id_embed_head is not None:
            if last_hidden is None:
                raise RuntimeError("SMPLRegressionHead produced no hidden states for ID embeddings")
            outputs["pred_id_embed"] = F.normalize(self.id_embed_head(last_hidden), dim=-1)
        if self.return_aux:
            outputs["aux_outputs"] = {key: torch.stack(values, dim=0) for key, values in aux_outputs.items()}
        return outputs


class AggregatorSMPLHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        return_aux: bool = False,
        mean_pose_6d: torch.Tensor | None = None,
        mean_shape: torch.Tensor | None = None,
        predict_boxes: bool = False,
        bbox_mode: str = "direct",
        predict_id_embed: bool = False,
        id_embed_dim: int = 256,
        translation_output_mode: str = "direct",
        translation_decode_hidden_dim: int = 512,
        translation_decode_max_log_depth_delta: float = 1.00,
        translation_decode_max_ray_delta_m: float = 1.50,
        translation_decode_max_tangent_offset_m: float = 1.00,
        translation_decode_human_height_prior_m: float = 1.70,
        enable_translation_refine: bool = False,
        translation_refine_hidden_dim: int = 512,
        translation_refine_max_ray_delta_m: float = 0.60,
        translation_refine_max_tangent_delta_m: float = 0.35,
        translation_refine_max_log_depth_delta: float = 0.50,
        translation_refine_max_box_prior_weight: float = 0.50,
        translation_refine_human_height_prior_m: float = 1.70,
        translation_refine_use_log_depth: bool = True,
        enable_temporal_translation: bool = False,
        temporal_translation_hidden_dim: int = 512,
        temporal_translation_max_velocity_delta_m: float = 0.25,
        temporal_translation_gate_bias: float = 2.5,
        temporal_translation_use_world: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if len(intermediate_layer_idx) != num_layers:
            raise ValueError("intermediate_layer_idx length must match num_layers")
        self.intermediate_layer_idx = intermediate_layer_idx
        self.regression_head = SMPLRegressionHead(
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mean_pose_6d=mean_pose_6d,
            mean_shape=mean_shape,
            return_aux=return_aux,
            predict_boxes=predict_boxes,
            bbox_mode=bbox_mode,
            predict_id_embed=predict_id_embed,
            id_embed_dim=id_embed_dim,
            translation_output_mode=translation_output_mode,
            translation_decode_hidden_dim=translation_decode_hidden_dim,
            translation_decode_max_log_depth_delta=translation_decode_max_log_depth_delta,
            translation_decode_max_ray_delta_m=translation_decode_max_ray_delta_m,
            translation_decode_max_tangent_offset_m=translation_decode_max_tangent_offset_m,
            translation_decode_human_height_prior_m=translation_decode_human_height_prior_m,
            enable_translation_refine=enable_translation_refine,
            translation_refine_hidden_dim=translation_refine_hidden_dim,
            translation_refine_max_ray_delta_m=translation_refine_max_ray_delta_m,
            translation_refine_max_tangent_delta_m=translation_refine_max_tangent_delta_m,
            translation_refine_max_log_depth_delta=translation_refine_max_log_depth_delta,
            translation_refine_max_box_prior_weight=translation_refine_max_box_prior_weight,
            translation_refine_human_height_prior_m=translation_refine_human_height_prior_m,
            translation_refine_use_log_depth=translation_refine_use_log_depth,
            image_size=image_size,
        )
        self.temporal_translation_refiner = (
            TrackTemporalTranslationRefiner(
                dim_in=dim_in,
                hidden_dim=temporal_translation_hidden_dim,
                max_velocity_delta_m=temporal_translation_max_velocity_delta_m,
                gate_bias=temporal_translation_gate_bias,
                use_world=temporal_translation_use_world,
                image_size=image_size,
            )
            if enable_temporal_translation
            else None
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        token_layout: AggregatorTokenLayout,
        reference_boxes: torch.Tensor | None = None,
        pose_enc: torch.Tensor | None = None,
        track_ids: torch.Tensor | None = None,
        track_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if token_layout.num_smpl_queries <= 0:
            raise ValueError("AggregatorSMPLHead requires token_layout with at least one SMPL query")

        hidden_states = []
        batch_size = num_frames = None
        for layer_idx in self.intermediate_layer_idx:
            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which AggregatorSMPLHead needs.")
            batch_size, num_frames, _, dim = tokens.shape
            smpl_tokens = tokens[:, :, token_layout.smpl_start : token_layout.smpl_end]
            hidden_states.append(smpl_tokens.reshape(batch_size * num_frames, token_layout.num_smpl_queries, dim))

        flat_reference_boxes = None
        if reference_boxes is not None:
            flat_reference_boxes = reference_boxes.reshape(batch_size * num_frames, token_layout.num_smpl_queries, 4)
        flat_pose_enc = None
        if pose_enc is not None:
            flat_pose_enc = pose_enc.reshape(batch_size * num_frames, pose_enc.shape[-1])
        regression_outputs = self.regression_head(
            torch.stack(hidden_states, dim=0),
            reference_boxes=flat_reference_boxes,
            pose_enc=flat_pose_enc,
        )
        if batch_size is None or num_frames is None:
            raise RuntimeError("AggregatorSMPLHead produced no hidden states")

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for key, value in regression_outputs.items():
            if key == "aux_outputs":
                outputs[key] = {
                    aux_key: aux_value.reshape(
                        aux_value.shape[0],
                        batch_size,
                        num_frames,
                        *aux_value.shape[2:],
                    )
                    for aux_key, aux_value in value.items()
                }
            else:
                outputs[key] = value.reshape(batch_size, num_frames, *value.shape[1:])
        if self.temporal_translation_refiner is not None:
            if pose_enc is None:
                raise ValueError("Temporal SMPL translation requires pose_enc; set model.enable_camera=true")
            final_hidden = hidden_states[-1].reshape(batch_size, num_frames, token_layout.num_smpl_queries, -1)
            seed_transl = outputs.get("seed_pred_transl_cam", outputs["pred_transl_cam"])
            if not isinstance(seed_transl, torch.Tensor):
                raise RuntimeError("Temporal SMPL translation seed must be a tensor")
            outputs["seed_pred_transl_cam"] = seed_transl
            temporal_outputs = self.temporal_translation_refiner(
                hidden=final_hidden,
                seed_transl_cam=seed_transl,
                pose_enc=pose_enc,
                track_ids=track_ids,
                track_mask=track_mask,
            )
            outputs.update(temporal_outputs)
            outputs["pred_cam"] = outputs["pred_transl_cam"]
            _refresh_translation_ray_components(outputs)
        return outputs


def _refresh_translation_ray_components(outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> None:
    transl = outputs.get("pred_transl_cam")
    ray = outputs.get("pred_transl_ray_dir")
    tangent_x = outputs.get("pred_transl_tangent_x")
    tangent_y = outputs.get("pred_transl_tangent_y")
    if not isinstance(transl, torch.Tensor) or not isinstance(ray, torch.Tensor):
        return
    if not isinstance(tangent_x, torch.Tensor) or not isinstance(tangent_y, torch.Tensor):
        return
    outputs["pred_transl_ray_depth"] = (transl * ray).sum(dim=-1, keepdim=True)
    outputs["pred_transl_tangent"] = torch.cat(
        [
            (transl * tangent_x).sum(dim=-1, keepdim=True),
            (transl * tangent_y).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )


def _identity_pose_6d(num_joints: int) -> torch.Tensor:
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return identity_6d.repeat(num_joints)


def _flatten_intrinsics_from_pose_enc(pose_enc: torch.Tensor, image_size: int) -> torch.Tensor:
    if pose_enc.ndim == 2:
        pose_enc = pose_enc[:, None]
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(image_size, image_size), build_intrinsics=True)
    if intrinsics is None:
        raise RuntimeError("encoding_to_camera did not return intrinsics")
    return intrinsics.reshape(-1, 3, 3)


def _camera_ray_basis(
    boxes: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames, num_queries = boxes.shape[:2]
    center = boxes[..., :2] * float(image_size)
    fx = intrinsics[:, 0, 0].reshape(num_frames, 1).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].reshape(num_frames, 1).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].reshape(num_frames, 1)
    cy = intrinsics[:, 1, 2].reshape(num_frames, 1)
    ray_x = (center[..., 0] - cx) / fx
    ray_y = (center[..., 1] - cy) / fy
    ray = F.normalize(torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1), dim=-1)

    camera_x = boxes.new_tensor([1.0, 0.0, 0.0]).reshape(1, 1, 3).expand(num_frames, num_queries, 3)
    tangent_x = camera_x - (camera_x * ray).sum(dim=-1, keepdim=True) * ray
    fallback_y = boxes.new_tensor([0.0, 1.0, 0.0]).reshape(1, 1, 3).expand_as(tangent_x)
    tangent_x = torch.where(
        torch.linalg.norm(tangent_x, dim=-1, keepdim=True) > 1e-4,
        tangent_x,
        fallback_y - (fallback_y * ray).sum(dim=-1, keepdim=True) * ray,
    )
    tangent_x = F.normalize(tangent_x, dim=-1)
    tangent_y = F.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1)

    k_features = torch.stack(
        [
            fx.expand(-1, num_queries) / float(image_size),
            fy.expand(-1, num_queries) / float(image_size),
            cx.expand(-1, num_queries) / float(image_size),
            cy.expand(-1, num_queries) / float(image_size),
        ],
        dim=-1,
    )
    return ray, tangent_x, tangent_y, k_features


def _bbox_height_depth_prior(
    boxes: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    human_height_prior_m: float,
) -> torch.Tensor:
    num_frames, num_queries = boxes.shape[:2]
    fy = intrinsics[:, 1, 1].reshape(num_frames, 1, 1).clamp(min=1e-6)
    bbox_h_px = (boxes[..., 3:4] * float(image_size)).clamp(min=1.0)
    height = boxes.new_full((num_frames, num_queries, 1), float(human_height_prior_m))
    return fy * height / bbox_h_px


def _zero_last_linear(module: nn.Sequential) -> nn.Sequential:
    last = module[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return module


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))
