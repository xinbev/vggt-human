#!/usr/bin/env bash

set -euo pipefail

# Dataset-wide scan for long-tail base SMPL translation failures after the
# validated translation-ray-refine + HSI + no-worse checkpoint.  The default
# NUM_FRAMES=1 makes this a unique input-frame scan, so frames are not counted
# repeatedly through overlapping temporal windows.

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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine}"

NUM_FRAMES="${NUM_FRAMES:-1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
MATCH_SOURCE="${MATCH_SOURCE:-base}"
INTRINSICS_SOURCE="${INTRINSICS_SOURCE:-gt}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
ROI_EXPAND="${ROI_EXPAND:-0.75}"
SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
HSI_ENABLE_TEMPORAL_MOMENTUM="${HSI_ENABLE_TEMPORAL_MOMENTUM:-false}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"
DEDUP_FRAME_PERSON="${DEDUP_FRAME_PERSON:-true}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
TOP_K="${TOP_K:-50}"

BAD_BASE_TRANSL_M="${BAD_BASE_TRANSL_M:-0.50}"
SEVERE_BASE_TRANSL_M="${SEVERE_BASE_TRANSL_M:-0.80}"
BAD_HSI_TRANSL_M="${BAD_HSI_TRANSL_M:-0.50}"
SEVERE_HSI_TRANSL_M="${SEVERE_HSI_TRANSL_M:-0.80}"
BAD_BASE_MPJPE_M="${BAD_BASE_MPJPE_M:-0.50}"
SEVERE_BASE_MPJPE_M="${SEVERE_BASE_MPJPE_M:-0.80}"
BAD_HSI_MPJPE_M="${BAD_HSI_MPJPE_M:-0.50}"
SEVERE_HSI_MPJPE_M="${SEVERE_HSI_MPJPE_M:-0.80}"
HSI_WORSE_MARGIN_M="${HSI_WORSE_MARGIN_M:-0.05}"

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
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== HSI bad translation dataset scan =========="
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "BEDLAM root : ${BEDLAM_ROOT}"
echo "Output      : ${OUTPUT_DIR}"
echo "Split       : ${SPLIT}"
echo "Frames/window: ${NUM_FRAMES}"
echo "Max samples : ${MAX_SAMPLES} (0 means all)"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "Match source: ${MATCH_SOURCE}"
echo "K source    : ${INTRINSICS_SOURCE}"
echo "Scene affine: ${SCENE_AFFINE_MODE}"
echo "Temporal HSI: ${HSI_ENABLE_TEMPORAL_MOMENTUM}"
echo "Bad transl  : base>${BAD_BASE_TRANSL_M}m severe>${SEVERE_BASE_TRANSL_M}m"
echo "SMPL ray ref: ${SMPL_ENABLE_TRANSLATION_REFINE}"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

DEDUP_ARGS=()
if [[ "${DEDUP_FRAME_PERSON}" == "true" ]]; then
  DEDUP_ARGS+=(--dedupe-frame-person)
else
  DEDUP_ARGS+=(--no-dedupe-frame-person)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/scan_hsi_bad_translation_frames.py \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --num-frames "${NUM_FRAMES}" \
  --frame-stride "${FRAME_STRIDE}" \
  --max-samples "${MAX_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --match-source "${MATCH_SOURCE}" \
  --intrinsics-source "${INTRINSICS_SOURCE}" \
  --depth-max-m "${DEPTH_MAX_M}" \
  --roi-expand "${ROI_EXPAND}" \
  --bad-base-transl-m "${BAD_BASE_TRANSL_M}" \
  --severe-base-transl-m "${SEVERE_BASE_TRANSL_M}" \
  --bad-hsi-transl-m "${BAD_HSI_TRANSL_M}" \
  --severe-hsi-transl-m "${SEVERE_HSI_TRANSL_M}" \
  --bad-base-mpjpe-m "${BAD_BASE_MPJPE_M}" \
  --severe-base-mpjpe-m "${SEVERE_BASE_MPJPE_M}" \
  --bad-hsi-mpjpe-m "${BAD_HSI_MPJPE_M}" \
  --severe-hsi-mpjpe-m "${SEVERE_HSI_MPJPE_M}" \
  --hsi-worse-margin-m "${HSI_WORSE_MARGIN_M}" \
  --top-k "${TOP_K}" \
  --log-interval "${LOG_INTERVAL}" \
  "${PRIOR_ARGS[@]}" \
  "${DEDUP_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.sequence_length=${NUM_FRAMES}" \
  --override "data.stride=${FRAME_STRIDE}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
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
  --override "model.hsi_enable_temporal_momentum=${HSI_ENABLE_TEMPORAL_MOMENTUM}" \
  --override "model.hsi_temporal_momentum_decay=${MOMENTUM_DECAY}" \
  --override "model.hsi_temporal_momentum_detach=true" \
  --override "model.hsi_temporal_momentum_use_track_ids=true" \
  --override "model.hsi_scene_affine_mode=${SCENE_AFFINE_MODE}"

echo "========== HSI bad translation dataset scan finished =========="
echo "Summary json : ${OUTPUT_DIR}/hsi_bad_translation_scan_summary.json"
echo "Bad people   : ${OUTPUT_DIR}/bad_frame_person_rows.csv"
echo "Bad frames   : ${OUTPUT_DIR}/bad_frame_summary.csv"
echo "Bad sequences: ${OUTPUT_DIR}/bad_sequence_summary.csv"
echo "All rows     : ${OUTPUT_DIR}/all_frame_person_translation_rows.csv"
