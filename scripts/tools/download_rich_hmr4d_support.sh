#!/usr/bin/env bash
set -euo pipefail

# Minimal RICH support files used by this project. The GVHMR image-feature file
# rich_test_preproc.pt is intentionally omitted: OmegaHSR processes RGB itself.

RICH_ROOT="${RICH_ROOT:-/home/zhw/xyb_space/RICH}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${RICH_ROOT}/hmr4d_support}"
LABELS_URL="https://drive.google.com/file/d/17fbG1IsN6DfF_KwYWR2FbNZFKvs9waVb/view?usp=drive_link"
CAM_PARAMS_URL="https://raw.githubusercontent.com/zju3dv/GVHMR/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/dataset/rich/resource/cam2params.pt"

mkdir -p "${SUPPORT_ROOT}/resource"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed; installing it into the active Python environment..."
  python -m pip install gdown
fi

echo "[download] rich_test_labels.pt"
gdown \
  --fuzzy \
  --continue \
  --output "${SUPPORT_ROOT}/rich_test_labels.pt" \
  "${LABELS_URL}"

echo "[download] resource/cam2params.pt"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 5 \
    --output "${SUPPORT_ROOT}/resource/cam2params.pt" \
    "${CAM_PARAMS_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --continue --tries=5 \
    --output-document="${SUPPORT_ROOT}/resource/cam2params.pt" \
    "${CAM_PARAMS_URL}"
else
  echo "error: curl or wget is required to download cam2params.pt." >&2
  exit 1
fi

python - "${SUPPORT_ROOT}/rich_test_labels.pt" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
try:
    labels = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    labels = torch.load(path, map_location="cpu")

if not isinstance(labels, dict) or not labels:
    raise RuntimeError(f"Unexpected RICH labels payload in {path}")

missing = [key for key, value in labels.items() if "frame_id" not in value]
if missing:
    raise RuntimeError(f"RICH labels missing frame_id for {len(missing)} sequences")

print(f"[verified] {path}: {len(labels)} test sequences")
print("[first sequences]")
for key in sorted(labels)[:5]:
    print(f"  {key}: {len(labels[key]['frame_id'])} frames")
PY

echo
echo "RICH support files are ready:"
ls -lh \
  "${SUPPORT_ROOT}/rich_test_labels.pt" \
  "${SUPPORT_ROOT}/resource/cam2params.pt"
