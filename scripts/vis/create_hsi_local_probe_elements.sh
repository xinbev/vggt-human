#!/usr/bin/env bash

set -euo pipefail

# Generate paper-figure elements for HSI local scene probing and body-scene residuals.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_hsi_local_probe_elements}"
SCALE="${SCALE:-3}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI local probe paper elements =========="
echo "Output: ${OUTPUT_DIR}"

python scripts/vis/create_hsi_local_probe_elements.py \
  --output-dir "${OUTPUT_DIR}" \
  --scale "${SCALE}"

echo "========== HSI local probe paper elements finished =========="
