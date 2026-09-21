#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-${REPO_ROOT}/benchmarks/emdb2_global/inference_config.yaml}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/vggt_nlf_hsi_scale_fps}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_FRAMES="${NUM_FRAMES:-100}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
MAX_HUMANS="${MAX_HUMANS:-8}"
WARMUP="${WARMUP:-3}"
REPEATS="${REPEATS:-10}"

cd "${REPO_ROOT}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frame directory: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${INFERENCE_CONFIG}" ]] || { echo "[ERROR] Missing inference config: ${INFERENCE_CONFIG}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing HSI scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }

echo "========== Steady-state FPS: VGGT+NLF vs VGGT+NLF+HSI scale =========="
echo "Frames            : ${FRAMES_DIR}"
echo "Frame count       : ${NUM_FRAMES}"
echo "Resolution        : ${IMAGE_RESOLUTION}"
echo "Max humans        : ${MAX_HUMANS}"
echo "Warm-up / repeats : ${WARMUP} / ${REPEATS}"
echo "GPU selector      : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Output            : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
python scripts/eval/benchmark_vggt_nlf_hsi_scale_fps.py \
  --frames-dir "${FRAMES_DIR}" \
  --inference-config "${INFERENCE_CONFIG}" \
  --scale-checkpoint "${SCALE_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --num-frames "${NUM_FRAMES}" \
  --start-index "${START_INDEX}" \
  --frame-stride "${FRAME_STRIDE}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --max-humans "${MAX_HUMANS}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  "$@"

echo "JSON: ${OUTPUT_DIR}/fps_summary.json"
echo "CSV : ${OUTPUT_DIR}/fps_summary.csv"
