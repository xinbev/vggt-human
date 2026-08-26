from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionalSceneProbe(nn.Module):
    """Multi-scale regional depth probe with human-overlap rejection."""

    def __init__(
        self,
        token_dim: int = 16,
        fixed_patch_sizes: tuple[int, ...] = (3, 7),
        adaptive_radius_max: int = 5,
        annulus_width: int = 2,
        human_depth_tolerance_m: float = 0.15,
        human_depth_dilation_px: int = 2,
        min_valid_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if any(size <= 0 or size % 2 == 0 for size in fixed_patch_sizes):
            raise ValueError("fixed_patch_sizes must be positive odd values")
        self.fixed_patch_sizes = tuple(int(size) for size in fixed_patch_sizes)
        self.adaptive_radius_max = max(int(adaptive_radius_max), 1)
        self.annulus_width = max(int(annulus_width), 1)
        self.human_depth_tolerance_m = max(float(human_depth_tolerance_m), 1e-4)
        self.human_depth_dilation_px = max(int(human_depth_dilation_px), 0)
        self.min_valid_ratio = min(max(float(min_valid_ratio), 0.0), 1.0)
        self.scale_names = tuple([f"fixed_{size}" for size in self.fixed_patch_sizes] + ["adaptive", "annulus"])
        self.channel_names = ("human", "environment")
        self.num_tokens = len(self.scale_names) * len(self.channel_names)
        self.token_dim = int(token_dim)

        point_dim = 3 + 1 + 2 + 1 + 1 + 1 + 3
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(token_dim, 1)

    def forward(
        self,
        centers: torch.Tensor,
        representatives: torch.Tensor,
        vertices_by_frame: torch.Tensor,
        depth_by_frame: torch.Tensor,
        intrinsics_by_frame: torch.Tensor,
        person_valid: torch.Tensor,
        image_size_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Probe regions.

        Args:
            centers: [F,Q,A,3]
            representatives: [F,Q,A,R,3]
            vertices_by_frame: [F,Q,V,3]
            depth_by_frame: [F,H,W]
            intrinsics_by_frame: [F,3,3]
            person_valid: [F,Q]
        """
        if centers.ndim != 4 or representatives.ndim != 5 or vertices_by_frame.ndim != 4:
            raise ValueError("RegionalSceneProbe received invalid region/vertex ranks")
        frames, queries, regions = centers.shape[:3]
        height, width = depth_by_frame.shape[-2:]
        flat_centers = centers.reshape(frames * queries, regions, 3)
        flat_representatives = representatives.reshape(frames * queries, regions, representatives.shape[3], 3)
        flat_intrinsics = intrinsics_by_frame.repeat_interleave(queries, dim=0)
        flat_depth = depth_by_frame.repeat_interleave(queries, dim=0)

        center_xy = _project_to_depth(flat_centers, flat_intrinsics, image_size_hw, (height, width))
        rep_xy = _project_to_depth(
            flat_representatives.reshape(frames * queries, -1, 3),
            flat_intrinsics,
            image_size_hw,
            (height, width),
        ).reshape(*flat_representatives.shape[:-1], 2)
        rep_radius = (rep_xy - center_xy.unsqueeze(2)).abs().amax(dim=(-1, -2))
        adaptive_radius = torch.ceil(rep_radius + 1.0).long().clamp(1, self.adaptive_radius_max)

        max_radius = max(
            max(size // 2 for size in self.fixed_patch_sizes),
            self.adaptive_radius_max + self.annulus_width,
        )
        offsets = torch.arange(-max_radius, max_radius + 1, device=centers.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        ox = ox.reshape(1, 1, -1)
        oy = oy.reshape(1, 1, -1)
        xs_raw = center_xy[..., 0].round().long().unsqueeze(-1) + ox
        ys_raw = center_xy[..., 1].round().long().unsqueeze(-1) + oy
        image_valid = (xs_raw >= 0) & (xs_raw < width) & (ys_raw >= 0) & (ys_raw < height)
        xs = xs_raw.clamp(0, width - 1)
        ys = ys_raw.clamp(0, height - 1)
        person_frame_idx = torch.arange(frames * queries, device=centers.device).view(-1, 1, 1).expand_as(xs)
        depth_values = flat_depth[person_frame_idx, ys, xs]
        depth_valid = image_valid & torch.isfinite(depth_values) & (depth_values > 1e-5)

        intr = flat_intrinsics
        image_x = xs.to(dtype=centers.dtype) * (float(image_size_hw[1]) / float(width))
        image_y = ys.to(dtype=centers.dtype) * (float(image_size_hw[0]) / float(height))
        scene_x = (image_x - intr[:, None, 0, 2].unsqueeze(-1)) * depth_values / intr[:, None, 0, 0].unsqueeze(-1).clamp(min=1e-5)
        scene_y = (image_y - intr[:, None, 1, 2].unsqueeze(-1)) * depth_values / intr[:, None, 1, 1].unsqueeze(-1).clamp(min=1e-5)
        scene_points = torch.stack([scene_x, scene_y, depth_values], dim=-1)
        scene_delta = scene_points - flat_centers.unsqueeze(2)

        nearest_rep_z, _ = _nearest_representative_depth(
            xs=xs,
            ys=ys,
            representative_xy=rep_xy,
            representative_z=flat_representatives[..., 2],
        )
        human_depth, human_owner = _rasterize_human_depth_and_owner(
            vertices=vertices_by_frame,
            intrinsics=intrinsics_by_frame,
            person_valid=person_valid,
            image_size_hw=image_size_hw,
            depth_hw=(height, width),
            dilation_px=self.human_depth_dilation_px,
        )
        frame_idx = torch.div(torch.arange(frames * queries, device=centers.device), queries, rounding_mode="floor")
        human_samples = human_depth[frame_idx[:, None, None], ys, xs]
        owner_samples = human_owner[frame_idx[:, None, None], ys, xs]
        current_owner = torch.remainder(
            torch.arange(frames * queries, device=centers.device), queries
        ).view(-1, 1, 1)
        any_human = torch.isfinite(human_samples) & ((depth_values - human_samples).abs() <= self.human_depth_tolerance_m)
        owner_is_self = owner_samples == current_owner
        self_surface = depth_valid & any_human & owner_is_self
        other_human = depth_valid & any_human & ~owner_is_self
        environment = depth_valid & ~any_human

        chebyshev = torch.maximum(ox.abs(), oy.abs()).expand(frames * queries, regions, -1)
        scale_masks: list[torch.Tensor] = []
        for size in self.fixed_patch_sizes:
            scale_masks.append(chebyshev <= size // 2)
        adaptive = chebyshev <= adaptive_radius.unsqueeze(-1)
        annulus = (chebyshev > adaptive_radius.unsqueeze(-1)) & (
            chebyshev <= (adaptive_radius + self.annulus_width).unsqueeze(-1)
        )
        scale_masks.extend([adaptive, annulus])

        radial = torch.sqrt(ox.to(centers.dtype).square() + oy.to(centers.dtype).square()).expand_as(depth_values)
        radial = radial / max(float(max_radius), 1.0)
        point_features = torch.cat(
            [
                scene_delta.clamp(-5.0, 5.0) / 5.0,
                (depth_values - flat_centers[..., 2].unsqueeze(-1)).clamp(-5.0, 5.0).unsqueeze(-1) / 5.0,
                ox.to(centers.dtype).expand_as(depth_values).unsqueeze(-1) / max(float(max_radius), 1.0),
                oy.to(centers.dtype).expand_as(depth_values).unsqueeze(-1) / max(float(max_radius), 1.0),
                radial.unsqueeze(-1),
                (depth_values - nearest_rep_z).clamp(-2.0, 2.0).unsqueeze(-1) / 2.0,
                (depth_values - human_samples).nan_to_num(0.0).clamp(-2.0, 2.0).unsqueeze(-1) / 2.0,
                torch.stack(
                    [self_surface.to(centers.dtype), environment.to(centers.dtype), other_human.to(centers.dtype)],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        encoded = self.point_encoder(torch.nan_to_num(point_features))
        logits = self.attention_score(encoded).squeeze(-1)

        tokens = []
        ratios = []
        for scale_mask in scale_masks:
            for channel_mask in (self_surface, environment):
                mask = scale_mask & channel_mask
                token, ratio = _masked_attention_pool(encoded, logits, mask, scale_mask)
                tokens.append(token)
                ratios.append(ratio)
        token_tensor = torch.stack(tokens, dim=2)
        ratio_tensor = torch.stack(ratios, dim=-1)
        person_valid_flat = person_valid.reshape(-1, 1)
        region_valid = (
            (ratio_tensor.max(dim=-1).values >= self.min_valid_ratio)
            & person_valid_flat
            & (flat_centers[..., 2] > 1e-5)
        )
        other_human_ratio = (other_human & image_valid).float().mean(dim=-1)
        return {
            "tokens": token_tensor,
            "valid_ratios": ratio_tensor,
            "region_valid": region_valid,
            "projected_norm": torch.stack(
                [
                    center_xy[..., 0] / max(float(width - 1), 1.0),
                    center_xy[..., 1] / max(float(height - 1), 1.0),
                ],
                dim=-1,
            ).nan_to_num(0.0),
            "other_human_ratio": other_human_ratio,
            "adaptive_radius": adaptive_radius,
            "self_surface_ratio": self_surface.float().mean(dim=-1),
            "environment_ratio": environment.float().mean(dim=-1),
        }


def _masked_attention_pool(
    encoded: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e4))
    weights = torch.softmax(masked_logits, dim=-1) * mask.to(dtype=encoded.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    token = (encoded * weights.unsqueeze(-1)).sum(dim=-2)
    ratio = mask.to(dtype=encoded.dtype).sum(dim=-1) / support.to(dtype=encoded.dtype).sum(dim=-1).clamp(min=1.0)
    return token, ratio


def _nearest_representative_depth(
    xs: torch.Tensor,
    ys: torch.Tensor,
    representative_xy: torch.Tensor,
    representative_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = xs.to(representative_xy.dtype).unsqueeze(-1) - representative_xy[..., 0].unsqueeze(-2)
    dy = ys.to(representative_xy.dtype).unsqueeze(-1) - representative_xy[..., 1].unsqueeze(-2)
    distance = torch.sqrt(dx.square() + dy.square())
    nearest = distance.argmin(dim=-1)
    z = representative_z.gather(-1, nearest)
    dist = distance.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
    return z, dist


def _rasterize_human_depth_and_owner(
    vertices: torch.Tensor,
    intrinsics: torch.Tensor,
    person_valid: torch.Tensor,
    image_size_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    dilation_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames, queries, vertex_count = vertices.shape[:3]
    height, width = depth_hw
    flat_vertices = vertices.reshape(frames, queries * vertex_count, 3)
    xy = _project_to_depth(flat_vertices, intrinsics, image_size_hw, depth_hw)
    px = xy[..., 0].round().long()
    py = xy[..., 1].round().long()
    z = flat_vertices[..., 2]
    valid_person = person_valid[:, :, None].expand(-1, -1, vertex_count).reshape(frames, -1)
    valid = valid_person & torch.isfinite(flat_vertices).all(dim=-1) & (z > 1e-5) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    frame = torch.arange(frames, device=vertices.device).view(-1, 1).expand_as(px)
    index = frame * (height * width) + py.clamp(0, height - 1) * width + px.clamp(0, width - 1)
    output_size = frames * height * width
    output = torch.full((output_size,), float("inf"), device=vertices.device, dtype=vertices.dtype)
    if bool(valid.any()):
        output.scatter_reduce_(0, index[valid], z[valid], reduce="amin", include_self=True)
    owner = torch.full((output_size,), queries, device=vertices.device, dtype=torch.long)
    if bool(valid.any()):
        frontmost = valid & torch.isclose(z, output[index], rtol=1e-5, atol=1e-6)
        query_owner = torch.arange(queries, device=vertices.device).view(1, queries, 1)
        query_owner = query_owner.expand(frames, queries, vertex_count).reshape(frames, -1)
        owner.scatter_reduce_(
            0,
            index[frontmost],
            query_owner[frontmost],
            reduce="amin",
            include_self=True,
        )
    output = output.reshape(frames, height, width)
    owner = owner.reshape(frames, height, width)
    if dilation_px > 0:
        kernel = 2 * int(dilation_px) + 1
        output, source_index = F.max_pool2d(
            -output[:, None],
            kernel_size=kernel,
            stride=1,
            padding=dilation_px,
            return_indices=True,
        )
        output = -output[:, 0]
        owner = owner.reshape(frames, -1).gather(1, source_index[:, 0].reshape(frames, -1))
        owner = owner.reshape(frames, height, width)
    owner = torch.where(torch.isfinite(output), owner, torch.full_like(owner, -1))
    return output, owner


def _project_to_depth(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    depth_hw: tuple[int, int],
) -> torch.Tensor:
    z = points[..., 2].clamp(min=1e-5)
    px = intrinsics[:, None, 0, 0] * points[..., 0] / z + intrinsics[:, None, 0, 2]
    py = intrinsics[:, None, 1, 1] * points[..., 1] / z + intrinsics[:, None, 1, 2]
    scale = points.new_tensor(
        [float(depth_hw[1]) / float(image_size_hw[1]), float(depth_hw[0]) / float(image_size_hw[0])]
    )
    return torch.stack([px, py], dim=-1) * scale
