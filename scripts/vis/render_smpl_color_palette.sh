#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/smpl_color_palette}"
BACKEND="${BACKEND:-xvfb}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "SMPL model: ${REPO_ROOT}/checkpoints/body_models/smpl"
echo "Colors    : ${REPO_ROOT}/assets/smpl_colors.txt"
echo "Output    : ${OUTPUT_DIR}"
echo "Backend   : ${BACKEND}"

if [[ "${BACKEND}" == "xvfb" ]]; then
  # Xvfb provides a software OpenGL context and does not require /dev/dri access.
  env -u PYOPENGL_PLATFORM xvfb-run -a -s "-screen 0 512x512x24" \
    python scripts/vis/render_smpl_color_palette.py \
      --output-dir "${OUTPUT_DIR}" \
      --backend default
else
  PYOPENGL_PLATFORM="${BACKEND}" \
  python scripts/vis/render_smpl_color_palette.py \
    --output-dir "${OUTPUT_DIR}" \
    --backend "${BACKEND}"
fi
