#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_vggt_wild_sequence_viewer_smoke/walking}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
MAX_FRAMES="${MAX_FRAMES:-8}"

cd "${REPO_ROOT}"

echo "========== Wild NLF-HSI VGGT sequence viewer smoke =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Stage2 dir  : ${STAGE2_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Max frames  : ${MAX_FRAMES}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"

SMOKE_ONLY=true \
FRAMES_DIR="${FRAMES_DIR}" \
STAGE2_DIR="${STAGE2_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MAX_FRAMES="${MAX_FRAMES}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
bash scripts/vis/serve_nlf_hsi_vggt_wild_sequence_viewer.sh

SUMMARY="${OUTPUT_DIR}/run_summary.json"
[[ -f "${SUMMARY}" ]] || { echo "[ERROR] Missing smoke summary: ${SUMMARY}" >&2; exit 1; }
echo "========== wild sequence viewer smoke passed =========="
echo "Summary: ${SUMMARY}"
