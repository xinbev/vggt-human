#!/usr/bin/env python3
"""Dependency-light numerical checks for SHOW HS-V/HS-CF metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.evaluation.human_scene_consistency import (  # noqa: E402
    HumanSceneConsistencyConfig,
    compute_from_visible_points,
    compute_human_scene_consistency,
    directed_nearest_mean,
    filter_depth_percentile,
)


def main() -> None:
    body = torch.tensor(
        [
            [-0.5, -1.0, 2.0],
            [0.5, -1.0, 2.1],
            [-0.5, 1.0, 2.2],
            [0.5, 1.0, 2.3],
        ],
        dtype=torch.float32,
    )
    identical = compute_from_visible_points(body, body, distance_chunk_size=2)
    assert identical["valid"]
    assert abs(float(identical["hs_cf"])) < 1e-7
    assert abs(float(identical["hs_cf_paper"])) < 1e-7
    assert abs(float(identical["hs_v"])) < 1e-7

    translated_scene = body + torch.tensor([0.4, -0.2, 0.0])
    translated = compute_from_visible_points(body, translated_scene, distance_chunk_size=2)
    assert float(translated["hs_cf"]) > 0.0
    assert float(translated["hs_cf_paper"]) > 0.0
    assert abs(float(translated["hs_v"])) < 1e-6, "variance must be translation invariant"

    scaled_scene = body.clone()
    scaled_scene[:, :2] *= 1.8
    scaled = compute_from_visible_points(body, scaled_scene, distance_chunk_size=2)
    assert float(scaled["hs_v"]) > 0.0

    flat_body = body.clone()
    flat_body[:, 2] = 2.0
    degenerate_paper = compute_from_visible_points(flat_body, flat_body, distance_chunk_size=2)
    assert degenerate_paper["valid_show_code"]
    assert not degenerate_paper["valid_paper"]
    assert torch.isnan(torch.tensor(float(degenerate_paper["hs_cf_paper"])))

    depth = torch.full((4, 4), 2.0, dtype=torch.float32)
    intrinsics = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 2.0],
            [0.0, -1.0, 2.2],
            [-1.0, 0.0, 2.4],
            [0.0, 0.0, 2.6],
        ],
        dtype=torch.float32,
    )
    end_to_end = compute_human_scene_consistency(
        vertices,
        depth,
        intrinsics,
        torch.ones_like(depth, dtype=torch.bool),
        config=HumanSceneConsistencyConfig(
            percentile_trims=(0.0,),
            visibility_backend="vertex_zbuffer",
            distance_chunk_size=2,
            min_body_points=1,
            min_scene_points=1,
            min_points_for_percentile=1,
        ),
    )
    assert end_to_end["valid_0"]
    assert torch.isfinite(torch.tensor(float(end_to_end["hs_cf0"])))
    assert end_to_end["chamfer_norm_pct_p0_100"] == end_to_end["hs_cf0"]
    assert end_to_end["xy_scale_mse_p0_100"] == end_to_end["hs_v0"]

    invalid_show = compute_human_scene_consistency(
        torch.empty((0, 3), dtype=torch.float32),
        depth,
        intrinsics,
        torch.ones_like(depth, dtype=torch.bool),
        config=HumanSceneConsistencyConfig(
            percentile_trims=(0.0,),
            visibility_backend="vertex_zbuffer",
            min_body_points=1,
            min_scene_points=1,
            min_points_for_percentile=1,
        ),
    )
    assert not invalid_show["valid"]
    assert invalid_show["hs_cf0"] == 0.0
    assert invalid_show["xy_scale_mse_p0_100"] == 0.0

    try:
        HumanSceneConsistencyConfig(percentile_trims=(50.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid percentile trim must be rejected")

    source = torch.randn(17, 3)
    target = torch.randn(23, 3)
    chunked = directed_nearest_mean(source, target, chunk_size=5)
    direct = torch.cdist(source, target).amin(dim=1).mean()
    assert torch.allclose(chunked, direct, atol=1e-6, rtol=1e-6)

    scene_with_depth_outliers = torch.cat(
        [body, torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 20.0]])], dim=0
    )
    filtered = filter_depth_percentile(scene_with_depth_outliers, trim=20.0, min_points_for_percentile=1)
    assert filtered[:, 2].min() > 0.2
    assert filtered[:, 2].max() < 20.0

    print(
        json.dumps(
            {
                "status": "ok",
                "identical": identical,
                "translated": translated,
                "scaled": scaled,
                "degenerate_paper": degenerate_paper,
                "end_to_end_vertex_zbuffer": end_to_end,
                "invalid_show_zero": invalid_show,
                "chunked_distance": float(chunked),
                "filtered_points": int(filtered.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
