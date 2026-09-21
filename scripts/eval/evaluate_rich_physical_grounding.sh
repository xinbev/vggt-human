#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-smoke}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/zhw/xyb_space/RICH/official}"
SUPPORT_ROOT="${SUPPORT_ROOT:-/home/zhw/xyb_space/RICH/hmr4d_support}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${ROOT_DIR}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${ROOT_DIR}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
RICH_SEQUENCE="${RICH_SEQUENCE-}"
WINDOW_SIZE="${WINDOW_SIZE:-${CHUNK_SIZE:-100}}"
MAX_HUMANS="${MAX_HUMANS:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-0}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-}"
RESUME="${RESUME-}"

if [[ "${MODE}" == "smoke" ]]; then
  RICH_SEQUENCE="${RICH_SEQUENCE:-Gym_010_cooking1/cam_01}"
  MAX_FRAMES_PER_SEQUENCE="${SMOKE_FRAMES:-4}"
  RESUME="${RESUME:-false}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/debug/rich_physical_grounding/evaluator_smoke}"
else
  RESUME="${RESUME:-true}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/rich_physical_grounding}"
fi

args=(
  --official-root "${OFFICIAL_ROOT}"
  --support-root "${SUPPORT_ROOT}"
  --train-config "${TRAIN_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --scale-checkpoint "${SCALE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --window-size "${WINDOW_SIZE}"
  --max-humans "${MAX_HUMANS}"
  --frame-stride "${FRAME_STRIDE}"
  --max-frames-per-sequence "${MAX_FRAMES_PER_SEQUENCE}"
  --max-sequences "${MAX_SEQUENCES}"
)

if [[ -n "${RICH_SEQUENCE}" ]]; then
  args+=(--sequence "${RICH_SEQUENCE}")
fi
if [[ -n "${SEQUENCE_MANIFEST}" ]]; then
  args+=(--sequence-manifest "${SEQUENCE_MANIFEST}")
fi
if [[ "${RESUME}" == "true" ]]; then
  args+=(--resume)
fi

echo "RICH physical-grounding metric evaluation"
echo "Mode       : ${MODE}"
echo "Sequence   : ${RICH_SEQUENCE:-<manifest/all>}"
echo "Frames max : ${MAX_FRAMES_PER_SEQUENCE} (0 means all selected label frames)"
echo "Window size: ${WINDOW_SIZE} (non-overlapping; final shorter window is retained)"
echo "Train cfg  : ${TRAIN_CONFIG}"
echo "Stage2 ckpt: ${CHECKPOINT}"
echo "Scale ckpt : ${SCALE_CHECKPOINT}"
echo "NLF slots  : ${MAX_HUMANS} (detector, then RICH-box IoU matching)"
echo "Output     : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
python scripts/eval/evaluate_rich_physical_grounding.py "${args[@]}"
