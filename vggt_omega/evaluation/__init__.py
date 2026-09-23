"""Evaluation-only utilities for reproducible benchmark scripts."""

from .human_scene_consistency import (
    HumanSceneConsistencyConfig,
    compute_from_visible_points,
    compute_human_scene_consistency,
    render_mesh_silhouette,
)

__all__ = [
    "HumanSceneConsistencyConfig",
    "compute_from_visible_points",
    "compute_human_scene_consistency",
    "render_mesh_silhouette",
]
