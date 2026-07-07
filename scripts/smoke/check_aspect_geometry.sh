#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python - <<'PY'
from vggt_omega.data.geometry import compute_resize_geometry

cases = [
    (408, 612),
    (612, 408),
    (512, 512),
]
for hw in cases:
    geom = compute_resize_geometry(hw, image_resolution=512, patch_size=16, mode="balanced")
    h, w = geom.input_hw
    assert h % 16 == 0 and w % 16 == 0, geom
    print({"orig_hw": hw, "input_hw": geom.input_hw, "grid_hw": (h // 16, w // 16), "crop_xyxy": geom.crop_xyxy})

wide = compute_resize_geometry((408, 612), image_resolution=512, patch_size=16, mode="balanced")
assert wide.input_hw[0] != wide.input_hw[1], wide
print("[aspect-geometry] OK")
PY
