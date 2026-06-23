#!/usr/bin/env bash

set -euo pipefail

# High-information 2D diagnostic panel for the known bad frame. It draws:
# original RGB, GT input boxes, predicted boxes, GT/base/HSI SMPL projections,
# and per-person translation/MPJPE errors.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_bad_frame_after_translation_ray_refine_2d}"

NUM_FRAMES="${NUM_FRAMES:-27}"
STRIDE="${STRIDE:-1}"
TARGET_FRAME_STEM="${TARGET_FRAME_STEM:-seq_000000_0100}"
TARGET_FRAME_INDEX="${TARGET_FRAME_INDEX:--1}"
PANEL_WIDTH="${PANEL_WIDTH:-720}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
USE_TRACK_IDS="${USE_TRACK_IDS:-true}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"
SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
SMPL_ENABLE_TRANSLATION_REFINE="${SMPL_ENABLE_TRANSLATION_REFINE:-true}"
SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M:-1.20}"
SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M:-0.60}"
SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA:-0.85}"
SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT:-1.00}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing HSI checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== HSI bad-frame 2D diagnostics =========="
echo "Image start : ${IMAGE_PATH}"
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Target stem : ${TARGET_FRAME_STEM}"
echo "Target idx  : ${TARGET_FRAME_INDEX}"
echo "Frames      : ${NUM_FRAMES}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "Track ids   : ${USE_TRACK_IDS}"
echo "SMPL ray ref: ${SMPL_ENABLE_TRANSLATION_REFINE}"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

TRACK_ARGS=()
if [[ "${USE_TRACK_IDS}" != "true" ]]; then
  TRACK_ARGS+=(--disable-track-ids)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_bad_frame_2d_diagnostics.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --stride "${STRIDE}" \
  --target-frame-stem "${TARGET_FRAME_STEM}" \
  --target-frame-index "${TARGET_FRAME_INDEX}" \
  --split "${SPLIT}" \
  --panel-width "${PANEL_WIDTH}" \
  "${PRIOR_ARGS[@]}" \
  "${TRACK_ARGS[@]}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.smpl_enable_translation_refine=${SMPL_ENABLE_TRANSLATION_REFINE}" \
  --override "model.smpl_translation_refine_max_ray_delta_m=${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M}" \
  --override "model.smpl_translation_refine_max_tangent_delta_m=${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M}" \
  --override "model.smpl_translation_refine_max_log_depth_delta=${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA}" \
  --override "model.smpl_translation_refine_max_box_prior_weight=${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT}" \
  --override "model.freeze_hsi_scene_affine=true" \
  --override "model.train_hsi_transl_only=true" \
  --override "model.hsi_enable_temporal_momentum=true" \
  --override "model.hsi_temporal_momentum_decay=${MOMENTUM_DECAY}" \
  --override "model.hsi_temporal_momentum_detach=true" \
  --override "model.hsi_temporal_momentum_use_track_ids=${USE_TRACK_IDS}" \
  --override "model.hsi_scene_affine_mode=${SCENE_AFFINE_MODE}"

echo "========== HSI bad-frame 2D diagnostics finished =========="
echo "Output image: ${OUTPUT_DIR}/${TARGET_FRAME_STEM}_bad_frame_2d_diagnostics.png"
echo "Output json : ${OUTPUT_DIR}/${TARGET_FRAME_STEM}_bad_frame_2d_diagnostics.json"
