from __future__ import annotations

from dataclasses import dataclass
import math

import torch


V4_NOISE_CATEGORIES = ("clean", "ray", "tangent", "combined")
V4_RAY_LEVELS = (-0.15, -0.10, -0.05, -0.03, 0.03, 0.05, 0.10, 0.15)
V4_TANGENT_LEVELS_M = (0.02, 0.05, 0.08)


@dataclass(frozen=True)
class V4NoiseAssignment:
    category_id: int
    ray_ratio: float
    tangent_x_m: float
    tangent_y_m: float


def deterministic_v4_assignment(dataset_index: int, person_slot: int, seed: int, epoch: int) -> V4NoiseAssignment:
    key = int(dataset_index) * 1_000_003 + int(person_slot) * 97_409 + int(seed) * 65_537 + int(epoch) * 8_191
    values = [_hash_uniform(key + offset * 104_729) for offset in range(4)]
    if values[0] < 0.20:
        category_id = 0
    elif values[0] < 0.40:
        category_id = 1
    elif values[0] < 0.60:
        category_id = 2
    else:
        category_id = 3
    ray_ratio = 0.0
    tangent_x = 0.0
    tangent_y = 0.0
    if category_id in {1, 3}:
        ray_ratio = V4_RAY_LEVELS[min(int(values[1] * len(V4_RAY_LEVELS)), len(V4_RAY_LEVELS) - 1)]
    if category_id in {2, 3}:
        magnitude = V4_TANGENT_LEVELS_M[
            min(int(values[2] * len(V4_TANGENT_LEVELS_M)), len(V4_TANGENT_LEVELS_M) - 1)
        ]
        angle = values[3] * 2.0 * math.pi
        tangent_x = float(magnitude * math.cos(angle))
        tangent_y = float(magnitude * math.sin(angle))
    return V4NoiseAssignment(category_id, float(ray_ratio), tangent_x, tangent_y)


def apply_deterministic_v4_noise(
    transl: torch.Tensor,
    valid: torch.Tensor,
    dataset_indices: torch.Tensor,
    *,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if transl.ndim != 4 or transl.shape[-1] != 3:
        raise ValueError(f"V4 translation noise expects [B,S,Q,3], got {tuple(transl.shape)}")
    batch_size, num_frames, num_people = transl.shape[:3]
    if valid.shape != transl.shape[:-1]:
        raise ValueError(f"V4 valid mask shape mismatch: {tuple(valid.shape)} vs {tuple(transl.shape[:-1])}")
    flat_indices = dataset_indices.reshape(-1).detach().cpu().tolist()
    if len(flat_indices) != batch_size:
        raise ValueError(f"Expected {batch_size} dataset indices, got {len(flat_indices)}")

    category = torch.zeros(batch_size, 1, num_people, 1, dtype=torch.long, device=transl.device)
    ray_ratio = transl.new_zeros(batch_size, 1, num_people, 1)
    tangent_coeff = transl.new_zeros(batch_size, 1, num_people, 2)
    for batch_idx, dataset_index in enumerate(flat_indices):
        for person_slot in range(num_people):
            assignment = deterministic_v4_assignment(int(dataset_index), person_slot, int(seed), int(epoch))
            category[batch_idx, 0, person_slot, 0] = assignment.category_id
            ray_ratio[batch_idx, 0, person_slot, 0] = assignment.ray_ratio
            tangent_coeff[batch_idx, 0, person_slot, 0] = assignment.tangent_x_m
            tangent_coeff[batch_idx, 0, person_slot, 1] = assignment.tangent_y_m

    category = category.expand(batch_size, num_frames, num_people, 1)
    ray_ratio = ray_ratio.expand(batch_size, num_frames, num_people, 1)
    tangent_coeff = tangent_coeff.expand(batch_size, num_frames, num_people, 2)
    valid_f = valid.unsqueeze(-1).to(dtype=transl.dtype)
    ray_ratio = ray_ratio * valid_f
    tangent_coeff = tangent_coeff * valid_f
    ray, tangent_x, tangent_y = _camera_basis(transl)
    noisy = transl * (1.0 + ray_ratio)
    noisy = noisy + tangent_coeff[..., :1] * tangent_x + tangent_coeff[..., 1:] * tangent_y
    clean = ((category == 0) | ~valid.unsqueeze(-1)).to(dtype=transl.dtype)
    return noisy, 1.0 + ray_ratio, tangent_coeff, clean, category


def _hash_uniform(value: int) -> float:
    modulus = 2_147_483_647
    mixed = (int(value) * 48_271 + 12_345) % modulus
    mixed = (mixed * 40_692 + 54_321) % modulus
    return float(mixed) / float(modulus)


def _camera_basis(transl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ray = torch.nn.functional.normalize(transl, dim=-1, eps=1e-6)
    x_axis = torch.zeros_like(ray)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(ray)
    y_axis[..., 1] = 1.0
    tangent_x = x_axis - (x_axis * ray).sum(dim=-1, keepdim=True) * ray
    fallback = y_axis - (y_axis * ray).sum(dim=-1, keepdim=True) * ray
    tangent_x = torch.where(torch.linalg.norm(tangent_x, dim=-1, keepdim=True) > 1e-4, tangent_x, fallback)
    tangent_x = torch.nn.functional.normalize(tangent_x, dim=-1, eps=1e-6)
    tangent_y = torch.nn.functional.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1, eps=1e-6)
    return ray, tangent_x, tangent_y
