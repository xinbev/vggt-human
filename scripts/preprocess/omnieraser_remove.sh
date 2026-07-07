#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE_OR_DIR="${1:?Usage: $0 <image_or_image_dir> <mask_or_mask_dir> [extra args...]}"
MASK_OR_DIR="${2:?Usage: $0 <image_or_image_dir> <mask_or_mask_dir> [extra args...]}"
shift 2

if [[ -d "${IMAGE_OR_DIR}" ]]; then
  SOURCE_ARGS=(--image-dir "${IMAGE_OR_DIR}" --mask-dir "${MASK_OR_DIR}")
else
  SOURCE_ARGS=(--image "${IMAGE_OR_DIR}" --mask "${MASK_OR_DIR}")
fi

python scripts/preprocess/omnieraser_remove.py \
  "${SOURCE_ARGS[@]}" \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --output-dir "${OUTPUT_DIR:-outputs/preprocess/omnieraser}" \
  "$@"
