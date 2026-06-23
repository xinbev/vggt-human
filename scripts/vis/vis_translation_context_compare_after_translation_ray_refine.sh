#!/usr/bin/env bash

set -euo pipefail

# Export the same target frames under two contexts:
#   1) single-frame forward, matching the full single-frame scan
#   2) 27-frame clip forward, matching the bad-frame PLY/2D visualizations
#
# The key output is *_translation_only_compare*.ply, where the same predicted
# SMPL mesh is placed at pre-refine, post-refine, HSI, and GT translations.

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
SPLIT="${SPLIT:-Training}"
SEQUENCE_NAME="${SEQUENCE_NAME:-20221013_3_250_batch01hand_orbit_bigOffice_seq_000000}"
CLIP_START_IMAGE="${CLIP_START_IMAGE:-${BEDLAM_ROOT}/${SPLIT}/${SEQUENCE_NAME}/rgb/seq_000000_0000.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/translation_context_compare_after_translation_ray_refine}"

FRAME_STEMS="${FRAME_STEMS:-seq_000000_0085,seq_000000_0100}"
RUN_SINGLE_FRAME="${RUN_SINGLE_FRAME:-1}"
RUN_CLIP_CONTEXT="${RUN_CLIP_CONTEXT:-1}"
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-27}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-20}"
PLY_TOP_K="${PLY_TOP_K:-3}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"

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
[[ -f "${CLIP_START_IMAGE}" ]] || { echo "[ERROR] Missing clip start image: ${CLIP_START_IMAGE}" >&2; exit 1; }

echo "========== Translation context comparison =========="
echo "Checkpoint : ${SMPL_CKPT}"
echo "Sequence   : ${SEQUENCE_NAME}"
echo "Frames     : ${FRAME_STEMS}"
echo "Output     : ${OUTPUT_DIR}"
echo "Single     : ${RUN_SINGLE_FRAME}"
echo "Clip       : ${RUN_CLIP_CONTEXT} (${CLIP_NUM_FRAMES} frames)"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

if [[ "${RUN_SINGLE_FRAME}" == "1" ]]; then
  IFS=',' read -ra STEMS <<< "${FRAME_STEMS}"
  for stem in "${STEMS[@]}"; do
    stem="$(echo "${stem}" | xargs)"
    [[ -n "${stem}" ]] || continue
    image_path="${BEDLAM_ROOT}/${SPLIT}/${SEQUENCE_NAME}/rgb/${stem}.png"
    [[ -f "${image_path}" ]] || { echo "[ERROR] Missing target frame image: ${image_path}" >&2; exit 1; }
    frame_output="${OUTPUT_DIR}/single_frame/${stem}"
    mkdir -p "${frame_output}"
    echo "----- Single-frame ${stem} -----"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_inference.py \
      --image "${image_path}" \
      --checkpoint "${SMPL_CKPT}" \
      --baseline-checkpoint "${VGGT_CKPT}" \
      --smpl-model-dir "${SMPL_MODEL_DIR}" \
      --path-config "${PATH_CONFIG}" \
      --train-config "${TRAIN_CONFIG}" \
      --output-dir "${frame_output}" \
      --conf-threshold "${CONF_THRESHOLD}" \
      --top-k "${TOP_K}" \
      --ply-top-k "${PLY_TOP_K}" \
      --draw-smpl-joints \
      --draw-gt-smpl-joints \
      --export-ply \
      --export-hsi-comparison \
      --export-pre-refine-comparison \
      --export-translation-debug-json \
      --export-translation-only-comparison \
      "${PRIOR_ARGS[@]}" \
      --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
      --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
      --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
      --override "data.sequence_length=1" \
      --override "data.stride=1" \
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
      --override "model.hsi_enable_temporal_momentum=true" \
      --override "model.hsi_temporal_momentum_decay=${MOMENTUM_DECAY}" \
      --override "model.hsi_temporal_momentum_detach=true" \
      --override "model.hsi_temporal_momentum_use_track_ids=false" \
      --override "model.hsi_scene_affine_mode=${SCENE_AFFINE_MODE}"
  done
fi

if [[ "${RUN_CLIP_CONTEXT}" == "1" ]]; then
  echo "----- Clip-context ${FRAME_STEMS} -----"
  IMAGE_PATH="${CLIP_START_IMAGE}" \
  OUTPUT_DIR="${OUTPUT_DIR}/clip_${CLIP_NUM_FRAMES}f" \
  PLY_FRAME_STEMS="${FRAME_STEMS}" \
  NUM_FRAMES="${CLIP_NUM_FRAMES}" \
  STRIDE="${CLIP_STRIDE}" \
  CONF_THRESHOLD="${CONF_THRESHOLD}" \
  TOP_K="${TOP_K}" \
  PLY_TOP_K="${PLY_TOP_K}" \
  USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR}" \
  EXPORT_PRE_REFINE_COMPARISON=true \
  EXPORT_TRANSLATION_DEBUG_JSON=true \
  EXPORT_TRANSLATION_ONLY_COMPARISON=true \
  SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE}" \
  SMPL_CKPT="${SMPL_CKPT}" \
  SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M}" \
  SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M}" \
  SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA}" \
  SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/vis/vis_hsi_bad_frame_ply_after_translation_ray_refine.sh
fi

echo "========== Translation context comparison finished =========="
echo "Single-frame PLY root: ${OUTPUT_DIR}/single_frame"
echo "Clip-context PLY root: ${OUTPUT_DIR}/clip_${CLIP_NUM_FRAMES}f/ply"
