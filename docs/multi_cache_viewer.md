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
