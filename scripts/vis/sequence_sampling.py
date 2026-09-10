"""Dependency-light deterministic sequence sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar

T = TypeVar("T")
SamplingStrategy = Literal["head", "uniform"]
LONG_SEQUENCE_FRAME_LIMIT = 500


def select_inclusive_range(items: Sequence[T], start_index: int = 0, end_index: int = -1) -> list[T]:
    """Select an inclusive index range; ``end_index=-1`` means the sequence end."""
    values = list(items)
    start = max(0, int(start_index))
    end = int(end_index)
    if end < -1:
        raise ValueError(f"end_index must be -1 or a non-negative integer; got {end}")
    if end >= 0 and end < start:
        raise ValueError(f"end_index must be >= start_index; got start={start}, end={end}")
    if values and start >= len(values):
        raise ValueError(
            f"start_index is outside the source sequence: start={start}, total_frames={len(values)}"
        )
    stop = None if end == -1 else end + 1
    return values[start:stop]


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


def sample_with_max_frames(
    items: Sequence[T],
    max_frames: int,
    strategy: SamplingStrategy | str,
    long_sequence_limit: int = LONG_SEQUENCE_FRAME_LIMIT,
) -> list[T]:
    """Apply viewer MAX_FRAMES semantics, including -1 long-sequence mode."""
    requested = int(max_frames)
    if requested < -1:
        raise ValueError(f"max_frames must be -1, 0, or a positive integer; got {requested}")
    if requested == -1:
        return sample_sequence(items, max_count=int(long_sequence_limit), strategy="uniform")
    return sample_sequence(items, max_count=requested, strategy=strategy)


def select_frame_candidates(
    items: Sequence[T],
    start_index: int = 0,
    end_index: int = -1,
    frame_stride: int = 1,
    max_frames: int = 0,
    strategy: SamplingStrategy | str = "head",
    long_sequence_limit: int = LONG_SEQUENCE_FRAME_LIMIT,
) -> list[T]:
    """Apply the complete range -> stride -> frame-count selection pipeline."""
    selected_range = select_inclusive_range(items, start_index=start_index, end_index=end_index)
    strided = selected_range[:: max(1, int(frame_stride))]
    return sample_with_max_frames(
        strided,
        max_frames=max_frames,
        strategy=strategy,
        long_sequence_limit=long_sequence_limit,
    )


def select_inference_frame_candidates(
    items: Sequence[T],
    start_index: int = 0,
    end_index: int = -1,
    inference_frames: int = 0,
) -> list[T]:
    """Select a range, then uniformly sample the requested inference count."""
    selected_range = select_inclusive_range(items, start_index=start_index, end_index=end_index)
    target = int(inference_frames)
    if target < 0:
        raise ValueError(f"inference_frames must be 0 or a positive integer; got {target}")
    if target == 0 or target >= len(selected_range):
        return selected_range
    if target == 1:
        return [selected_range[(len(selected_range) - 1) // 2]]
    return [selected_range[index] for index in uniform_sample_indices(len(selected_range), target)]
