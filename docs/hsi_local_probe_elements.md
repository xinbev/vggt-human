# HSI Local Probe Elements

This script generates architecture-figure assets for:

```text
5. Local Scene Probing
6. Body-Scene Residual
```

The assets are deterministic visual elements, not model outputs. They follow
the current `HSIRefinementHead` logic:

```text
body anchor + camera intrinsics -> projected uv
projected uv -> VGGT depth / patch-token grid
3x3 local scene window -> local scene tokens
(u, v, z_scene) -> scene xyz
offset = scene_point - anchor
distance = ||offset||
depth_residual = scene_point.z - anchor.z
depth gradients -> approximate normal
```

## Code

```text
scripts/vis/create_hsi_local_probe_elements.py
scripts/vis/create_hsi_local_probe_elements.sh
```

Server usage:

```text
bash scripts/vis/create_hsi_local_probe_elements.sh
```

Default output:

```text
outputs/vis/paper_hsi_local_probe_elements/
```

## Outputs

```text
05a_depth_patch_grid_3x3.png
05a_depth_patch_grid_3x3.svg
```

Depth / patch-token grid with a highlighted 3x3 local window.

```text
05b_anchor_project_to_depth.png
05b_anchor_project_to_depth.svg
```

Body anchor projected into the scene/depth grid.

```text
05c_local_scene_probe_composite.png
```

Combined local probing inset: anchor, projection arrow, depth grid, 3x3 local
window, and local scene token points.

```text
06a_body_scene_residual.png
06a_body_scene_residual.svg
```

Body anchor, scene point, offset/distance arrow, scene normal arrow, and
z-residual guide.

## Notes

- PNG files use transparent backgrounds.
- SVG files contain editable geometry and no text.
- Add final labels such as `scene xyz`, `offset / distance`, `normal`, and
  `z residual` in the paper figure editor.
