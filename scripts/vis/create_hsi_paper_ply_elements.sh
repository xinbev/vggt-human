#!/usr/bin/env bash

set -euo pipefail

# Generate deterministic geometry-only PLY assets for the HSI paper figure.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_hsi_ply_elements}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI paper PLY elements =========="
echo "Repo   : ${REPO_ROOT}"
echo "Output : ${OUTPUT_DIR}"

python scripts/vis/create_hsi_paper_ply_elements.py \
  --output-dir "${OUTPUT_DIR}"

echo "========== HSI paper PLY elements finished =========="
echo "PLY root: ${OUTPUT_DIR}"
