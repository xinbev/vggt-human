# Multi-cache Viser viewer

`serve_multi_sequence_viewer_cache.sh` replays multiple full viewer caches in
one Viser process. The caches are concatenated in the order supplied by
`CACHE_DIRS`, so the global timeline moves through cache 1, then cache 2, and
so on. No inference or checkpoint loading is performed.

Each cache has an independent transform folder with uniform scale, XYZ
translation, and XYZ Euler rotation. The transform is applied to its complete
display layer: environment points, SMPL meshes, labels, cameras, and camera
trajectory. Existing single-cache viewer commands are unchanged.

Example on the Linux server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CACHE_DIRS=/path/cache_a:/path/cache_b \\
CACHE_NAMES=sequence_a:sequence_b \\
PORT=8080 \\
VIEWER_MODE=Hybrid \\
SMPL_DISPLAY_FRAMES=50 \\
bash scripts/vis/serve_multi_sequence_viewer_cache.sh
```

`CACHE_DIRS` and `CACHE_NAMES` use `:` only as separators; cache paths should
therefore be ordinary Linux paths without a colon. Set `SHOW_CAMERAS=true` to
show per-frame camera frustums and `SHOW_TRAJECTORIES=false` to hide the
trajectory overlays.

## Boundary alignment

Set `ALIGNMENT_PREVIEW=true`. The viewer then hides the normal timeline and
shows only the last frame of the left cache and the first frame of the right
cache. These endpoint layers use cached `hsi_points_full` data, so human depth
points are retained, and they also show the SMPL meshes. The `Alignment Pair`
dropdown selects which adjacent cache pair is being aligned. Each cache's
`Overall Scale (log10)` uses the same logarithmic convention as the original
HSI viewer: `0` means x1, `-1` means x0.1, and `1` means x10. Translation and
Euler rotation remain linear controls.

The `Save Multi-cache Layout` button writes the cache list, transforms, pair,
preview state, and timeline position to `LAYOUT_JSON` (by default
`outputs/vis/multi_cache_alignment.json`). The `Load Multi-cache Layout` button
restores that file immediately. To reopen a saved layout directly:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
LAYOUT_JSON=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/multi_cache_alignment.json \\
bash scripts/vis/serve_multi_sequence_viewer_cache.sh
```

When `CACHE_DIRS` is omitted, cache paths are read from the saved layout. When
explicit `CACHE_DIRS` are supplied, the saved layout is applied only if the
cache list matches exactly.
