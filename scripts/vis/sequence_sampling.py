"""Dependency-light deterministic sequence sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar

T = TypeVar("T")
SamplingStrategy = Literal["head", "uniform"]


def uniform_sample_indices(total_count: int, target_count: int) -> list[int]:
    """Return exactly ``target_count`` ordered indices spanning both endpoints."""
    total = max(0, int(total_count))
    target = max(0, min(int(target_count), total))
    if target <= 0:
        return []
    if target == 1:
        return [0]
    if target == total:
        return list(range(total))
    return [index * (total - 1) // (target - 1) for index in range(target)]


def sample_sequence(items: Sequence[T], max_count: int, strategy: SamplingStrategy | str) -> list[T]:
    """Cap a sequence with baseline prefix or full-range uniform sampling."""
    values = list(items)
    limit = int(max_count)
    if limit <= 0 or len(values) <= limit:
        return values
    if strategy == "head":
        return values[:limit]
    if strategy == "uniform":
        return [values[index] for index in uniform_sample_indices(len(values), limit)]
    raise ValueError(f"Unknown sampling strategy: {strategy}")
