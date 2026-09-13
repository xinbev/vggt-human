#!/usr/bin/env bash
set -euo pipefail

# Stable display path for long 3DPW sequences: one VGGT/NLF forward pass over
# at most 550 uniformly sampled frames, followed by a GT-camera display warp.
# No cross-chunk stitching is performed.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01}"
GT_PKL="${GT_PKL:-/home/zhw/xyb_space/3DPW/sequenceFiles/test/downtown_walkBridge_01.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_uniform550}"
PREDICTED_CACHE="${PREDICTED_CACHE:-${OUTPUT_ROOT}/predicted_viewer_cache}"
GT_CACHE="${GT_CACHE:-${OUTPUT_ROOT}/full_viewer_cache_gt_camera}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-8090}"
MAX_HUMANS="${MAX_HUMANS:-64}"
NLF_DETECTOR_THRESHOLD="${NLF_DETECTOR_THRESHOLD:-0.25}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
COARSE_CONF_THRESHOLD="${COARSE_CONF_THRESHOLD:-0.10}"
COARSE_MAX_PEOPLE="${COARSE_MAX_PEOPLE:-0}"
DISPLAY_PEOPLE="${DISPLAY_PEOPLE:-0}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-50}"
CASCADE_EFFECTIVE_AFFINE_MODE="${CASCADE_EFFECTIVE_AFFINE_MODE:-clip_median}"
POINT_SIZE="${POINT_SIZE:-0.006}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-12}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-true}"
SERVE_VIEWER="${SERVE_VIEWER:-false}"

cd "${REPO_ROOT}"
[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${GT_PKL}" ]] || { echo "[ERROR] Missing 3DPW GT pickle: ${GT_PKL}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

echo "========== 3DPW GT-camera uniform <=550 visualisation =========="
echo "Frames       : ${FRAMES_DIR}"
echo "GT camera    : ${GT_PKL}"
echo "Sampling     : full range, uniform, MAX_FRAMES=550"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Pred cache   : ${PREDICTED_CACHE}"
echo "GT cache     : ${GT_CACHE}"

# Select at most 550 uniformly across the complete source range while keeping
# one single VGGT/NLF forward pass (no cross-chunk stitching).
REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
SCALE_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_ROOT}/inference" \
FULL_VIEWER_CACHE_OUTPUT="${PREDICTED_CACHE}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
START_INDEX=0 \
END_INDEX=-1 \
FRAME_STRIDE=1 \
FRAME_SAMPLING=uniform \
INFERENCE_FRAMES= \
MAX_FRAMES=550 \
MAX_HUMANS="${MAX_HUMANS}" \
NLF_DETECTOR_THRESHOLD="${NLF_DETECTOR_THRESHOLD}" \
CONF_THRESHOLD="${CONF_THRESHOLD}" \
COARSE_CONF_THRESHOLD="${COARSE_CONF_THRESHOLD}" \
COARSE_MAX_PEOPLE="${COARSE_MAX_PEOPLE}" \
DISPLAY_PEOPLE="${DISPLAY_PEOPLE}" \
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES}" \
CASCADE_EFFECTIVE_AFFINE_MODE="${CASCADE_EFFECTIVE_AFFINE_MODE}" \
POINT_SIZE="${POINT_SIZE}" \
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX}" \
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS}" \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh

python scripts/vis/apply_3dpw_gt_camera_to_cache.py \
  --input-dir "${PREDICTED_CACHE}" \
  --gt-pkl "${GT_PKL}" \
  --output-dir "${GT_CACHE}"

if [[ "${SERVE_VIEWER}" == "true" || "${SERVE_VIEWER}" == "1" ]]; then
  CACHE_DIR="${GT_CACHE}" \
  OUTPUT_DIR="${OUTPUT_ROOT}" \
  PORT="${PORT}" \
  VIEWER_MODE="${VIEWER_MODE:-Hybrid}" \
  SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES}" \
  DISPLAY_PEOPLE="${DISPLAY_PEOPLE}" \
  bash scripts/vis/serve_3dpw_gt_camera_full_sequence.sh
fi
