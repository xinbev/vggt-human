#!/usr/bin/env bash
set -euo pipefail

# Full-sequence visualisation for native 3DPW.  The model runs in bounded
# chunks; the merge step places every predicted point/SMPL result into the
# native 3DPW camera world.  No GT SMPL is used.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01}"
GT_PKL="${GT_PKL:-/home/zhw/xyb_space/3DPW/sequenceFiles/test/downtown_walkBridge_01.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full}"
MERGED_CACHE="${MERGED_CACHE:-${OUTPUT_ROOT}/full_viewer_cache_gt_camera}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-8090}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-0}"
TOTAL_FRAMES="${TOTAL_FRAMES:-}"
MAX_HUMANS="${MAX_HUMANS:-8}"
NLF_DETECTOR_THRESHOLD="${NLF_DETECTOR_THRESHOLD:-0.30}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
COARSE_CONF_THRESHOLD="${COARSE_CONF_THRESHOLD:-0.30}"
COARSE_MAX_PEOPLE="${COARSE_MAX_PEOPLE:-1}"
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
(( CHUNK_SIZE > 0 )) || { echo "[ERROR] CHUNK_SIZE must be positive" >&2; exit 1; }
(( CHUNK_OVERLAP >= 0 && CHUNK_OVERLAP < CHUNK_SIZE )) || { echo "[ERROR] CHUNK_OVERLAP must satisfy 0 <= overlap < chunk size" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${TOTAL_FRAMES}" ]]; then
  TOTAL_FRAMES="$(find "${FRAMES_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -printf '.' | wc -c)"
fi
(( TOTAL_FRAMES > 0 )) || { echo "[ERROR] No frames found under ${FRAMES_DIR}" >&2; exit 1; }
echo "========== 3DPW GT-camera full visualisation =========="
echo "Frames       : ${FRAMES_DIR}"
echo "GT camera    : ${GT_PKL}"
echo "Total frames : ${TOTAL_FRAMES}"
echo "Chunk size   : ${CHUNK_SIZE} (overlap=${CHUNK_OVERLAP})"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Merged cache : ${MERGED_CACHE}"

start=0
chunk_id=0
while (( start < TOTAL_FRAMES )); do
  end=$(( start + CHUNK_SIZE - 1 ))
  if (( end >= TOTAL_FRAMES )); then end=$(( TOTAL_FRAMES - 1 )); fi
  chunk_start=$start
  chunk_end=$end
  chunk_output="${OUTPUT_ROOT}/chunk_$(printf '%04d' "${chunk_id}")"
  chunk_cache="${chunk_output}/full_viewer_cache"
  mkdir -p "${chunk_output}"
  echo "[chunk ${chunk_id}] source range=[${chunk_start},${chunk_end}] cache=${chunk_cache}"
  REPO_ROOT="${REPO_ROOT}" \
  FRAMES_DIR="${FRAMES_DIR}" \
  STAGE2_DIR="${STAGE2_DIR}" \
  CHECKPOINT="${CHECKPOINT}" \
  SCALE_CHECKPOINT="${SCALE_CHECKPOINT}" \
  OUTPUT_DIR="${chunk_output}" \
  FULL_VIEWER_CACHE_OUTPUT="${chunk_cache}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  PORT="$(( PORT + chunk_id ))" \
  START_INDEX="${chunk_start}" \
  END_INDEX="${chunk_end}" \
  FRAME_STRIDE=1 \
  INFERENCE_FRAMES=0 \
  MAX_FRAMES=0 \
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
  step=$(( CHUNK_SIZE - CHUNK_OVERLAP ))
  start=$(( start + step ))
  (( chunk_id += 1 ))
done

INPUT_ROOT="${OUTPUT_ROOT}" \
GT_PKL="${GT_PKL}" \
OUTPUT_DIR="${MERGED_CACHE}" \
TOTAL_FRAMES="${TOTAL_FRAMES}" \
bash scripts/vis/merge_3dpw_gt_camera_full_caches.sh

if [[ "${SERVE_VIEWER}" == "true" || "${SERVE_VIEWER}" == "1" ]]; then
  CACHE_DIR="${MERGED_CACHE}" \
  OUTPUT_DIR="${OUTPUT_ROOT}" \
  PORT="${PORT}" \
  bash scripts/vis/serve_3dpw_gt_camera_full_sequence.sh
fi
