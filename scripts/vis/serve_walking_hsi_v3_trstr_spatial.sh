#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
TRSTR_DIR="${TRSTR_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial}"
TRSTR_CHECKPOINT="${TRSTR_CHECKPOINT:-}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/walking_hsi_v3_trstr_spatial}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-20}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing walking frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -d "${TRSTR_DIR}" ]] || { echo "[ERROR] Missing TRSTR output directory: ${TRSTR_DIR}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing HSI scale v3 checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }
[[ -f "${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr.yaml" ]] || {
  echo "[ERROR] Missing TRSTR model config" >&2
  exit 1
}

if [[ -z "${TRSTR_CHECKPOINT}" ]]; then
  TOPK_INDEX="${TRSTR_DIR}/checkpoint_topk_index.json"
  if [[ -f "${TOPK_INDEX}" ]]; then
    TRSTR_CHECKPOINT="$(python - "${TOPK_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
payload = json.loads(index_path.read_text(encoding="utf-8"))
entries = payload.get("entries", [])
print(entries[0].get("path", "") if entries else "")
PY
)"
  fi
  if [[ -z "${TRSTR_CHECKPOINT}" ]]; then
    TRSTR_CHECKPOINT="${TRSTR_DIR}/checkpoint_latest.pt"
  fi
fi

if [[ "${TRSTR_CHECKPOINT}" != /* ]]; then
  TRSTR_CHECKPOINT="${REPO_ROOT}/${TRSTR_CHECKPOINT}"
fi
[[ -f "${TRSTR_CHECKPOINT}" ]] || { echo "[ERROR] Missing TRSTR checkpoint: ${TRSTR_CHECKPOINT}" >&2; exit 1; }

echo "========== Walking inference: VGGT + NLF + HSI scale v3 + TRSTR spatial =========="
echo "Frames            : ${FRAMES_DIR}"
echo "TRSTR checkpoint  : ${TRSTR_CHECKPOINT}"
echo "Scale v3 overlay  : ${SCALE_CHECKPOINT}"
echo "TRSTR temporal    : disabled"
echo "Output            : ${OUTPUT_DIR}"
echo "Viewer            : http://127.0.0.1:${PORT}"
echo "Pipeline          : raw VGGT -> analytic coarse -> v3 residual -> TRSTR spatial"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr.yaml" \
STAGE2_DIR="${TRSTR_DIR}" \
CHECKPOINT="${TRSTR_CHECKPOINT}" \
HSI_OVERLAY_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
FRAME_STRIDE=1 \
MAX_HUMANS="${MAX_HUMANS:-8}" \
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}" \
SMPL_USE_AGGREGATOR_QUERIES=false \
HSI_SCENE_AFFINE_MODE=per_frame \
SCENE_SCALE_PREALIGN=smpl_median \
COARSE_SCALE_MIN="${COARSE_SCALE_MIN:-0.10}" \
COARSE_SCALE_MAX="${COARSE_SCALE_MAX:-10.0}" \
COARSE_ANCHOR_STRIDE="${COARSE_ANCHOR_STRIDE:-8}" \
COARSE_MIN_ANCHOR_PIXELS="${COARSE_MIN_ANCHOR_PIXELS:-32}" \
COARSE_FALLBACK=sequence_median \
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-2}" \
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-80}" \
VIEWER_MODE="${VIEWER_MODE:-hybrid}" \
ENVIRONMENT_DISPLAY="${ENVIRONMENT_DISPLAY:-points}" \
HSI_VISUAL_SCALE=1.0 \
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-12}" \
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-true}" \
POINT_SIZE="${POINT_SIZE:-0.006}" \
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}" \
TRACKING_OVERLAY=none \
SHOW_TRACK_IDS="${SHOW_TRACK_IDS:-true}" \
SMOKE_ONLY="${SMOKE_ONLY}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"

echo "Summary: ${OUTPUT_DIR}/run_summary.json"
