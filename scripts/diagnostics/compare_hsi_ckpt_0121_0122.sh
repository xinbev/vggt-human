#!/usr/bin/env bash

set -euo pipefail

# Compare the last good HSI checkpoint against the first bad resumed checkpoint.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q}"

BEFORE_CKPT="${BEFORE_CKPT:-${OUTPUT_DIR}/checkpoint_epoch_0121.pt}"
AFTER_CKPT="${AFTER_CKPT:-${OUTPUT_DIR}/checkpoint_epoch_0122.pt}"
TOP_K="${TOP_K:-40}"

cd "${REPO_ROOT}"

[[ -f "${BEFORE_CKPT}" ]] || { echo "[ERROR] Missing before checkpoint: ${BEFORE_CKPT}" >&2; exit 1; }
[[ -f "${AFTER_CKPT}" ]] || { echo "[ERROR] Missing after checkpoint: ${AFTER_CKPT}" >&2; exit 1; }
[[ -f "scripts/diagnostics/compare_hsi_checkpoints.py" ]] || {
  echo "[ERROR] Missing diagnostic script: scripts/diagnostics/compare_hsi_checkpoints.py" >&2
  exit 1
}

echo "========== Compare HSI checkpoints =========="
echo "Before : ${BEFORE_CKPT}"
echo "After  : ${AFTER_CKPT}"
echo "Top-K  : ${TOP_K}"

python scripts/diagnostics/compare_hsi_checkpoints.py \
  --before "${BEFORE_CKPT}" \
  --after "${AFTER_CKPT}" \
  --top-k "${TOP_K}"
