#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl}"
FRAMES_DIR="${FRAMES_DIR:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_vggt_sequence_viewer_smoke}"
QUERY_SOURCE="${QUERY_SOURCE:-bedlam_sidecar}"
MAX_FRAMES="${MAX_FRAMES:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

cd "${REPO_ROOT}"

echo "========== NLF-HSI VGGT sequence viewer smoke =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Query source: ${QUERY_SOURCE}"
echo "Stage2 dir  : ${STAGE2_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Max frames  : ${MAX_FRAMES}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"

SMOKE_ONLY=true \
DATA_ROOT="${DATA_ROOT}" \
BEDLAM_ROOT="${BEDLAM_ROOT}" \
PREPROCESSED_ROOT="${PREPROCESSED_ROOT}" \
STAGE2_DIR="${STAGE2_DIR}" \
FRAMES_DIR="${FRAMES_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
QUERY_SOURCE="${QUERY_SOURCE}" \
MAX_FRAMES="${MAX_FRAMES}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh

SUMMARY="${OUTPUT_DIR}/run_summary.json"
[[ -f "${SUMMARY}" ]] || { echo "[ERROR] Missing smoke summary: ${SUMMARY}" >&2; exit 1; }
echo "========== sequence viewer smoke passed =========="
echo "Summary: ${SUMMARY}"
