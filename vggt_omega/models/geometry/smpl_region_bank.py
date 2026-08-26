from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


DEFAULT_REGION_COUNTS = {
    "head_neck": 6,
    "torso": 20,
    "pelvis": 8,
    "upper_limb": 16,
    "hands": 20,
    "lower_limb": 16,
    "feet": 10,
}

JOINT_GROUPS = {
    "head_neck": (12, 15),
    "torso": (0, 3, 6, 9, 13, 14),
    "pelvis": (1, 2),
    "upper_limb": (16, 17, 18, 19),
    "hands": (20, 21, 22, 23),
    "lower_limb": (4, 5, 7, 8),
    "feet": (10, 11),
}


class SMPLRegionBank(nn.Module):
    """Deterministic non-uniform SMPL surface regions.

    Regions are built from dominant LBS joints and template-space farthest-point
    seeds. Runtime pooling uses the fixed vertex-to-region assignment, so every
    person gets the same region layout and every vertex is assigned exactly once.
    """

    def __init__(
        self,
        smpl_layer: nn.Module,
        region_counts: dict[str, int] | None = None,
        representative_vertices: int = 8,
    ) -> None:
        super().__init__()
        template = getattr(smpl_layer, "v_template", None)
        lbs_weights = getattr(smpl_layer, "lbs_weights", None)
        if not isinstance(template, torch.Tensor) or template.ndim != 2 or template.shape[-1] != 3:
            raise ValueError("SMPLRegionBank requires smpl_layer.v_template with shape [V,3]")
        if not isinstance(lbs_weights, torch.Tensor) or lbs_weights.ndim != 2:
            raise ValueError("SMPLRegionBank requires smpl_layer.lbs_weights with shape [V,24]")
        if lbs_weights.shape[0] != template.shape[0]:
            raise ValueError("SMPL v_template and lbs_weights must share the vertex dimension")

        counts = dict(DEFAULT_REGION_COUNTS)
        if region_counts:
            counts.update({str(key): int(value) for key, value in region_counts.items()})
        group_names = tuple(DEFAULT_REGION_COUNTS.keys())
        if any(name not in counts or counts[name] <= 0 for name in group_names):
            raise ValueError(f"region_counts must define positive counts for {group_names}")
        self.group_names = group_names
        self.num_regions = int(sum(counts[name] for name in group_names))
        self.representative_vertices = max(int(representative_vertices), 1)

        dominant_joint = lbs_weights.detach().float().argmax(dim=-1)
        joint_to_group = {joint: name for name, joints in JOINT_GROUPS.items() for joint in joints}
        group_ids = torch.tensor(
            [self.group_names.index(joint_to_group.get(int(joint), "torso")) for joint in dominant_joint],
            dtype=torch.long,
            device=template.device,
        )
        region_ids = torch.full((template.shape[0],), -1, dtype=torch.long, device=template.device)
        region_groups: list[int] = []
        region_seeds: list[int] = []
        next_region = 0
        for group_id, group_name in enumerate(group_names):
            vertices = torch.nonzero(group_ids == group_id, as_tuple=False).reshape(-1)
            if vertices.numel() == 0:
                raise ValueError(f"No SMPL vertices assigned to region group {group_name!r}")
            seeds = _farthest_point_indices(template[vertices].detach().float(), counts[group_name])
            seed_vertices = vertices[seeds]
            distances = torch.cdist(template[vertices].detach().float(), template[seed_vertices].detach().float())
            local_region = distances.argmin(dim=-1)
            region_ids[vertices] = local_region + next_region
            region_groups.extend([group_id] * counts[group_name])
            region_seeds.extend(seed_vertices.detach().cpu().tolist())
            next_region += counts[group_name]

        if bool((region_ids < 0).any()):
            raise RuntimeError("SMPLRegionBank left vertices unassigned")
        if torch.unique(region_ids).numel() != self.num_regions:
            raise RuntimeError("SMPLRegionBank did not create the requested number of non-empty regions")

        representative = []
        for region_id in range(self.num_regions):
            members = torch.nonzero(region_ids == region_id, as_tuple=False).reshape(-1)
            if members.numel() == 0:
                raise RuntimeError(f"Region {region_id} is empty")
            picks = torch.linspace(
                0,
                members.numel() - 1,
                steps=self.representative_vertices,
                device=members.device,
            ).round().long()
            representative.append(members[picks])

        self.register_buffer("vertex_region_ids", region_ids, persistent=True)
        self.register_buffer("region_group_ids", torch.tensor(region_groups, dtype=torch.long), persistent=True)
        self.register_buffer("region_seed_indices", torch.tensor(region_seeds, dtype=torch.long), persistent=True)
        self.register_buffer("representative_indices", torch.stack(representative), persistent=True)
        self.register_buffer("template_vertices", template.detach().float().clone(), persistent=True)

    def pool_vertices(self, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool vertices [N,V,3] into centers [N,A,3] and counts [A]."""
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[1] != self.vertex_region_ids.numel():
            raise ValueError(f"Expected vertices [N,{self.vertex_region_ids.numel()},3], got {tuple(vertices.shape)}")
        ids = self.vertex_region_ids.to(device=vertices.device)
        pooled = vertices.new_zeros(vertices.shape[0], self.num_regions, 3)
        pooled.scatter_add_(1, ids.view(1, -1, 1).expand(vertices.shape[0], -1, 3), vertices)
        counts = torch.bincount(ids, minlength=self.num_regions).to(device=vertices.device, dtype=vertices.dtype)
        pooled = pooled / counts.clamp(min=1).view(1, -1, 1)
        return pooled, counts

    def representative_points(self, vertices: torch.Tensor) -> torch.Tensor:
        if vertices.ndim != 3 or vertices.shape[1] != self.vertex_region_ids.numel():
            raise ValueError("representative_points expects vertices [N,V,3]")
        indices = self.representative_indices.to(device=vertices.device)
        return vertices[:, indices]


def _farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    points = points.reshape(-1, 3)
    count = max(int(count), 1)
    if points.shape[0] < count:
        base = torch.arange(points.shape[0], device=points.device)
        return base.repeat((count + base.numel() - 1) // base.numel())[:count]
    selected = torch.zeros(count, dtype=torch.long, device=points.device)
    distance = torch.full((points.shape[0],), float("inf"), device=points.device, dtype=points.dtype)
    current = 0
    for index in range(count):
        selected[index] = current
        distance = torch.minimum(distance, torch.linalg.norm(points - points[current], dim=-1))
        current = int(distance.argmax().detach().cpu())
    return selected
