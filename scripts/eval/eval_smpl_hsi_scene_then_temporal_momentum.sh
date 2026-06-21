#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_after_scene.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_scene_then_momentum/stage2_human_momentum/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_scene_then_temporal_momentum}"

MAX_SAMPLES="${MAX_SAMPLES:-64}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_VIEWS="${NUM_VIEWS:-12}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"
SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
TEMPORAL_NO_WORSE_MARGIN_M="${TEMPORAL_NO_WORSE_MARGIN_M:-0.002}"
TEMPORAL_NO_WORSE_ACCEL_MARGIN_M="${TEMPORAL_NO_WORSE_ACCEL_MARGIN_M:-0.003}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== SMPL HSI scene-then-temporal-momentum metrics =========="
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Views       : ${NUM_VIEWS}"
echo "Samples     : ${MAX_SAMPLES}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "Momentum    : ${MOMENTUM_DECAY}"
echo "Scene affine: ${SCENE_AFFINE_MODE}"
echo "No-worse m  : ${TEMPORAL_NO_WORSE_MARGIN_M} / accel ${TEMPORAL_NO_WORSE_ACCEL_MARGIN_M}"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_hsi_temporal_metrics.py \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --max-samples "${MAX_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --temporal-no-worse-margin-m "${TEMPORAL_NO_WORSE_MARGIN_M}" \
  --temporal-no-worse-accel-margin-m "${TEMPORAL_NO_WORSE_ACCEL_MARGIN_M}" \
  "${PRIOR_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.stride=1" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.freeze_hsi_scene_affine=true" \
  --override "model.train_hsi_transl_only=true" \
  --override "model.hsi_enable_temporal_momentum=true" \
  --override "model.hsi_temporal_momentum_decay=${MOMENTUM_DECAY}" \
  --override "model.hsi_temporal_momentum_detach=true" \
  --override "model.hsi_temporal_momentum_use_track_ids=true" \
  --override "model.hsi_scene_affine_mode=${SCENE_AFFINE_MODE}"

echo "========== SMPL HSI scene-then-temporal-momentum metrics finished =========="
echo "Metrics json: ${OUTPUT_DIR}/hsi_temporal_metrics.json"
