from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.models.geometry import SMPLRegionBank
from vggt_omega.models.geometry.smpl_region_bank import DEFAULT_REGION_COUNTS
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class HSIRegionalTranslationRefiner(nn.Module):
    """Translation-only regional surface refiner.

    Each valid person owns an independent region bank. Regions vote for one
    shared camera-space translation correction; pose and betas are read-only.
    """

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 256,
        region_embedding_dim: int = 32,
        num_regions: int = 96,
        representative_vertices: int = 8,
        num_iters: int = 2,
        patch_sizes: tuple[int, ...] = (3, 7),
        min_valid_ratio: float = 0.25,
        max_ray_delta_m: float = 0.35,
        max_tangent_delta_m: float = 0.20,
        max_person_delta_m: float = 0.50,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSIRegionalTranslationRefiner requires smpl_model_dir")
        if int(num_regions) not in {48, 72, 96}:
            raise ValueError("TRSTR num_regions must be one of the ablation budgets: 48, 72, 96")
        patch_sizes = tuple(int(size) for size in patch_sizes)
        if not patch_sizes or any(size <= 0 or size % 2 == 0 for size in patch_sizes):
            raise ValueError("patch_sizes must contain positive odd window sizes")

        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        self.region_bank = SMPLRegionBank(
            self.smpl.layer,
            region_counts=_scaled_region_counts(int(num_regions)),
            representative_vertices=representative_vertices,
        )
        self.num_regions = int(num_regions)
        if self.region_bank.num_regions != self.num_regions:
            raise RuntimeError("TRSTR region bank size does not match num_regions")
        self.num_iters = max(int(num_iters), 1)
        self.patch_sizes = patch_sizes
        self.min_valid_ratio = min(max(float(min_valid_ratio), 0.0), 1.0)
        self.max_ray_delta_m = float(max_ray_delta_m)
        self.max_tangent_delta_m = float(max_tangent_delta_m)
        self.max_person_delta_m = max(float(max_person_delta_m), 1e-4)
        self.image_size = int(image_size)

        self.region_embedding = nn.Embedding(self.num_regions, int(region_embedding_dim))
        feature_dim = 3 + 3 + 3 + 2 * len(self.patch_sizes) + 4 + int(region_embedding_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.vote_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)))
        self.gate_head = _biased_last_linear(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)),
            bias=2.0,
        )
        self.logvar_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.person_gate_head = _biased_last_linear(
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
        person_valid: torch.Tensor | None = None,
        track_ids: torch.Tensor | None = None,
        track_quality: torch.Tensor | None = None,
        track_gap: torch.Tensor | None = None,
        track_memory: object | None = None,
    ) -> dict[str, torch.Tensor]:
        del track_ids, track_quality, track_gap, track_memory
        pose6d = _require(predictions, "pred_pose_6d").float()
        betas = _require(predictions, "pred_betas").float()
        transl = _require(predictions, "pred_transl_cam").float()
        if pose6d.ndim != 4 or betas.ndim != 4 or transl.ndim != 4:
            raise ValueError("TRSTR expects pose6d/betas/transl with [B,S,Q,*] shapes")
        batch_size, num_frames, num_queries = transl.shape[:3]
        if person_valid is None:
            conf = predictions.get("pred_confs")
            person_valid = torch.ones(batch_size, num_frames, num_queries, dtype=torch.bool, device=transl.device)
            if isinstance(conf, torch.Tensor):
                person_valid = conf[..., 0] > 0.0
        person_valid = person_valid.to(device=transl.device).bool()

        depth_hw = _canonical_depth(depth).float()
        if not depth_is_metric:
            scale = predictions.get("hsi_scene_scale")
            bias = predictions.get("hsi_scene_depth_bias")
            if isinstance(scale, torch.Tensor) and isinstance(bias, torch.Tensor):
                depth_hw = depth_hw * _expand_frame_value(scale, depth_hw) + _expand_frame_value(bias, depth_hw)
        height, width = depth_hw.shape[-2:]
        image_size_hw = image_size_hw or (height, width)
        intrinsics = _resolve_intrinsics(
            pose_enc,
            image_size_hw=image_size_hw,
            batch_size=batch_size,
            num_frames=num_frames,
            dtype=transl.dtype,
            device=transl.device,
            override=intrinsics_override,
        )

        current = transl
        iteration_transl = [current]
        last = None
        for _ in range(self.num_iters):
            vertices = self._decode_vertices(pose6d, betas, current)
            pooled, _ = self.region_bank.pool_vertices(vertices)
            representatives = self.region_bank.representative_points(vertices)
            flat_depth = depth_hw.reshape(batch_size * num_frames, height, width).repeat_interleave(num_queries, dim=0)
            flat_intrinsics = intrinsics.repeat_interleave(num_queries, dim=0)
            probe = _probe_regions(
                centers=pooled,
                representatives=representatives,
                depth=flat_depth,
                intrinsics=flat_intrinsics,
                image_size_hw=image_size_hw,
                patch_sizes=self.patch_sizes,
                min_valid_ratio=self.min_valid_ratio,
            )
            region_group = self.region_bank.region_group_ids.to(device=transl.device)
            group_embedding = self.region_embedding(region_group).unsqueeze(0).expand(pooled.shape[0], -1, -1)
            features = torch.cat(
                [
                    pooled.clamp(-20.0, 20.0) / 20.0,
                    representatives.std(dim=2, unbiased=False).clamp(max=5.0) / 5.0,
                    probe["scene_delta"].clamp(-5.0, 5.0) / 5.0,
                    probe["depth_medians"].clamp(-5.0, 5.0) / 5.0,
                    probe["depth_mads"].clamp(min=0.0, max=5.0) / 5.0,
                    probe["valid_ratios"],
                    probe["projected_norm"],
                    group_embedding,
                ],
                dim=-1,
            )
            if features.shape[-1] != self.feature_mlp[0].in_features:
                raise RuntimeError(
                    f"TRSTR feature dimension mismatch: got {features.shape[-1]}, "
                    f"expected {self.feature_mlp[0].in_features}"
                )
            hidden = self.feature_mlp(features)
            vote = self._basis_vote(hidden, pooled)
            gate = torch.sigmoid(self.gate_head(hidden))
            logvar = self.logvar_head(hidden).clamp(-4.0, 4.0)
            valid = probe["region_valid"] & person_valid.reshape(-1, 1)
            weight = valid.to(dtype=vote.dtype) * gate * torch.exp(-logvar)
            denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-5)
            aggregate = (vote * weight).sum(dim=1) / denom
            aggregate = aggregate.clamp(min=-self.max_person_delta_m, max=self.max_person_delta_m)
            person_hidden = (hidden * valid.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
            person_gate = torch.sigmoid(self.person_gate_head(person_hidden))
            person_gate = person_gate * person_valid.reshape(-1, 1).to(person_gate.dtype)
            update = person_gate * aggregate
            current = current.reshape(-1, 3) + update
            current = current.reshape(batch_size, num_frames, num_queries, 3)
            iteration_transl.append(current)
            last = {
                "region_vote": vote,
                "region_gate": gate,
                "region_logvar": logvar,
                "region_valid": valid,
                "person_gate": person_gate.reshape(batch_size, num_frames, num_queries, 1),
                "scene_delta": probe["scene_delta"],
                "depth_medians": probe["depth_medians"],
                "valid_ratios": probe["valid_ratios"],
            }

        if last is None:
            raise RuntimeError("TRSTR produced no refinement iteration")
        outputs = {
            "hsi_trstr_refined_pred_transl_cam": current,
            "hsi_trstr_delta_transl_cam": current - transl,
            "hsi_trstr_iteration_transl": torch.stack(iteration_transl, dim=0),
            "hsi_trstr_region_vote": last["region_vote"].reshape(batch_size, num_frames, num_queries, self.num_regions, 3),
            "hsi_trstr_region_gate": last["region_gate"].reshape(batch_size, num_frames, num_queries, self.num_regions, 1),
            "hsi_trstr_region_logvar": last["region_logvar"].reshape(batch_size, num_frames, num_queries, self.num_regions, 1),
            "hsi_trstr_region_valid": last["region_valid"].reshape(batch_size, num_frames, num_queries, self.num_regions),
            "hsi_trstr_person_gate": last["person_gate"],
            "hsi_trstr_person_uncertainty": torch.exp(last["region_logvar"].mean(dim=1)).reshape(batch_size, num_frames, num_queries, 1),
            "hsi_trstr_region_scene_delta": last["scene_delta"].reshape(batch_size, num_frames, num_queries, self.num_regions, 3),
        }
        return outputs

    def _decode_vertices(self, pose6d: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        flat_pose = pose6d.reshape(-1, 24, 6)
        flat_betas = betas.reshape(-1, betas.shape[-1])
        with torch.no_grad():
            poses = rot6d_to_axis_angle(flat_pose).reshape(-1, 72)
            vertices, _ = self.smpl(poses.float(), flat_betas.float())
        return vertices.to(device=transl.device, dtype=transl.dtype) + transl.reshape(-1, 1, 3)

    def _basis_vote(self, hidden: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.vote_head(hidden))
        ray = F.normalize(centers, dim=-1, eps=1e-5)
        reference = torch.zeros_like(ray)
        reference[..., 1] = 1.0
        tangent_x = F.normalize(torch.cross(ray, reference, dim=-1), dim=-1, eps=1e-5)
        fallback = torch.zeros_like(ray)
        fallback[..., 0] = 1.0
        tangent_x = torch.where(torch.linalg.norm(tangent_x, dim=-1, keepdim=True) < 1e-4, F.normalize(torch.cross(ray, fallback, dim=-1), dim=-1, eps=1e-5), tangent_x)
        tangent_y = F.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1, eps=1e-5)
        coeff = raw * centers.new_tensor([self.max_ray_delta_m, self.max_tangent_delta_m, self.max_tangent_delta_m])
        return coeff[..., :1] * ray + coeff[..., 1:2] * tangent_x + coeff[..., 2:3] * tangent_y


def _probe_regions(
    centers: torch.Tensor,
    representatives: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    patch_sizes: tuple[int, ...],
    min_valid_ratio: float,
) -> dict[str, torch.Tensor]:
    n, num_regions = centers.shape[:2]
    height, width = depth.shape[-2:]
    z = centers[..., 2].clamp(min=1e-5)
    px = intrinsics[:, None, 0, 0] * centers[..., 0] / z + intrinsics[:, None, 0, 2]
    py = intrinsics[:, None, 1, 1] * centers[..., 1] / z + intrinsics[:, None, 1, 2]
    projected_norm = torch.stack([px / max(float(image_size_hw[1] - 1), 1.0), py / max(float(image_size_hw[0] - 1), 1.0)], dim=-1)
    median_depths = []
    mad_depths = []
    valid_ratios = []
    scene_points = []
    for size in patch_sizes:
        radius = size // 2
        offsets = torch.arange(-radius, radius + 1, device=centers.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        xs_raw = px.round().long()[..., None] + ox.reshape(1, 1, -1)
        ys_raw = py.round().long()[..., None] + oy.reshape(1, 1, -1)
        valid = (xs_raw >= 0) & (xs_raw < width) & (ys_raw >= 0) & (ys_raw < height)
        xs = xs_raw.clamp(0, width - 1)
        ys = ys_raw.clamp(0, height - 1)
        frame_idx = torch.arange(n, device=centers.device).view(n, 1, 1).expand_as(xs)
        values = depth[frame_idx, ys, xs]
        valid = valid & torch.isfinite(values) & (values > 1e-5)
        count = valid.sum(dim=-1)
        safe = torch.where(valid, values, torch.full_like(values, float("inf")))
        sorted_values, _ = safe.sort(dim=-1)
        median = sorted_values.gather(-1, ((count.clamp(min=1) - 1) // 2).unsqueeze(-1)).squeeze(-1)
        median = torch.where(count > 0, median, torch.zeros_like(median))
        deviations = torch.where(valid, (values - median.unsqueeze(-1)).abs(), torch.full_like(values, float("inf")))
        sorted_dev, _ = deviations.sort(dim=-1)
        mad = sorted_dev.gather(-1, ((count.clamp(min=1) - 1) // 2).unsqueeze(-1)).squeeze(-1)
        mad = torch.where(count > 0, mad, torch.zeros_like(mad))
        valid_ratio = count.to(dtype=centers.dtype) / float(values.shape[-1])
        scene_x = (px - intrinsics[:, None, 0, 2]) * median / intrinsics[:, None, 0, 0].clamp(min=1e-5)
        scene_y = (py - intrinsics[:, None, 1, 2]) * median / intrinsics[:, None, 1, 1].clamp(min=1e-5)
        scene_points.append(torch.stack([scene_x, scene_y, median], dim=-1))
        median_depths.append((median - z).unsqueeze(-1))
        mad_depths.append(mad.unsqueeze(-1))
        valid_ratios.append(valid_ratio.unsqueeze(-1))
    scene = torch.stack(scene_points, dim=2)
    valid_ratios_tensor = torch.cat(valid_ratios, dim=-1)
    median_tensor = torch.cat(median_depths, dim=-1)
    mad_tensor = torch.cat(mad_depths, dim=-1)
    robust_scene = scene.mean(dim=2)
    scene_delta = robust_scene - centers
    region_valid = (valid_ratios_tensor.min(dim=-1).values >= float(min_valid_ratio)) & torch.isfinite(scene_delta).all(dim=-1) & (z > 1e-5)
    return {
        "scene_delta": torch.nan_to_num(scene_delta),
        "depth_medians": torch.nan_to_num(median_tensor),
        "depth_mads": torch.nan_to_num(mad_tensor),
        "valid_ratios": torch.nan_to_num(valid_ratios_tensor),
        "projected_norm": torch.nan_to_num(projected_norm),
        "region_valid": region_valid,
    }


def _canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported TRSTR depth shape: {tuple(depth.shape)}")


def _resolve_intrinsics(
    pose_enc: torch.Tensor | None,
    image_size_hw: tuple[int, int],
    batch_size: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    override: torch.Tensor | None,
) -> torch.Tensor:
    if override is None:
        if pose_enc is None:
            raise ValueError("TRSTR requires pose_enc when intrinsics_override is absent")
        _, intrinsics = encoding_to_camera(pose_enc.float(), image_size_hw=image_size_hw, build_intrinsics=True)
        return intrinsics.reshape(-1, 3, 3).to(device=device, dtype=dtype)
    intrinsics = override.to(device=device, dtype=dtype)
    if intrinsics.ndim == 4 and tuple(intrinsics.shape[:2]) == (batch_size, num_frames):
        return intrinsics.reshape(-1, 3, 3)
    if intrinsics.ndim == 3 and intrinsics.shape[0] == batch_size * num_frames:
        return intrinsics
    raise ValueError(f"Unsupported TRSTR intrinsics shape: {tuple(intrinsics.shape)}")


def _expand_frame_value(value: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    out = value.to(device=depth.device, dtype=depth.dtype)
    while out.ndim < depth.ndim:
        out = out.unsqueeze(-1)
    return out


def _require(mapping: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"TRSTR predictions missing tensor: {key}")
    return value


def _zero_last_linear(module: nn.Sequential) -> nn.Sequential:
    linear = module[-1]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
    return module


def _biased_last_linear(module: nn.Sequential, bias: float) -> nn.Sequential:
    linear = module[-1]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        nn.init.constant_(linear.bias, float(bias))
    return module


def _scaled_region_counts(num_regions: int) -> dict[str, int]:
    """Scale the 96-region risk budget while preserving every anatomy group."""
    num_regions = int(num_regions)
    if num_regions not in {48, 72, 96}:
        raise ValueError(f"Unsupported TRSTR region budget: {num_regions}")
    names = tuple(DEFAULT_REGION_COUNTS.keys())
    base = torch.tensor([DEFAULT_REGION_COUNTS[name] for name in names], dtype=torch.float32)
    raw = base * (float(num_regions) / 96.0)
    counts = torch.floor(raw).long().clamp(min=1)
    remaining = num_regions - int(counts.sum())
    if remaining > 0:
        order = torch.argsort(raw - counts.float(), descending=True)
        for index in order[:remaining].tolist():
            counts[index] += 1
    elif remaining < 0:
        order = torch.argsort(raw - counts.float(), descending=False)
        for index in order.tolist():
            if remaining == 0:
                break
            if counts[index] > 1:
                counts[index] -= 1
                remaining += 1
    if int(counts.sum()) != num_regions:
        raise RuntimeError(f"Could not allocate region budget {num_regions}: {counts.tolist()}")
    return {name: int(count) for name, count in zip(names, counts.tolist(), strict=True)}
