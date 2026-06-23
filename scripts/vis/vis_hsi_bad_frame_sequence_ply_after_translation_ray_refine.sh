#!/usr/bin/env bash

set -euo pipefail

# Export PLY files for every frame in the known bad-frame clip. This is heavier
# than the focused bad-frame export: with the default 27 frames it writes one
# PLY subdirectory per frame, including base SMPL, HSI SMPL, GT SMPL, raw scene,
# SMPL-ratio scene, HSI-affine scene, and combined scenes.

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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_bad_frame_after_translation_ray_refine_sequence_ply}"

NUM_FRAMES="${NUM_FRAMES:-27}"
STRIDE="${STRIDE:-1}"
FPS="${FPS:-6}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-20}"
PLY_TOP_K="${PLY_TOP_K:-3}"
PANEL_SIZE="${PANEL_SIZE:-256}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
DRAW_GT_SMPL_JOINTS="${DRAW_GT_SMPL_JOINTS:-true}"
USE_HSI_REFINED="${USE_HSI_REFINED:-true}"
EXPORT_HSI_COMPARISON="${EXPORT_HSI_COMPARISON:-true}"
EXPORT_PRE_REFINE_COMPARISON="${EXPORT_PRE_REFINE_COMPARISON:-true}"
EXPORT_TRANSLATION_DEBUG_JSON="${EXPORT_TRANSLATION_DEBUG_JSON:-true}"
HSI_ALIGN_SCENE="${HSI_ALIGN_SCENE:-true}"
ALIGN_SCENE_TO_SMPL="${ALIGN_SCENE_TO_SMPL:-true}"
ALIGN_SCALE_MAX="${ALIGN_SCALE_MAX:-20.0}"
ALIGN_USE_GT_SMPL_ANCHORS="${ALIGN_USE_GT_SMPL_ANCHORS:-false}"
INCLUDE_ERROR_PANELS="${INCLUDE_ERROR_PANELS:-true}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"
SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
SMPL_ENABLE_TRANSLATION_REFINE="${SMPL_ENABLE_TRANSLATION_REFINE:-true}"
SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M:-1.20}"
SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M:-0.60}"
SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA:-0.85}"
SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT:-1.00}"

if [[ -z "${PLY_FRAME_INDICES:-}" ]]; then
  PLY_FRAME_INDICES=""
  for ((idx = 0; idx < NUM_FRAMES; idx++)); do
    if [[ -n "${PLY_FRAME_INDICES}" ]]; then
      PLY_FRAME_INDICES+=","
    fi
    PLY_FRAME_INDICES+="${idx}"
  done
fi

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

echo "========== HSI bad-frame full-sequence PLY export =========="
echo "Image start : ${IMAGE_PATH}"
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Frames      : ${NUM_FRAMES}"
echo "Stride      : ${STRIDE}"
echo "PLY indices : ${PLY_FRAME_INDICES}"
echo "PLY Top-K   : ${PLY_TOP_K}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "Use HSI     : ${USE_HSI_REFINED}"
echo "Scene affine: ${SCENE_AFFINE_MODE}"
echo "SMPL ray ref: ${SMPL_ENABLE_TRANSLATION_REFINE}"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

GT_ARGS=()
if [[ "${DRAW_GT_SMPL_JOINTS}" == "true" ]]; then
  GT_ARGS+=(--draw-gt-smpl-joints)
fi

HSI_ARGS=()
if [[ "${USE_HSI_REFINED}" == "true" ]]; then
  HSI_ARGS+=(--use-hsi-refined)
fi
if [[ "${EXPORT_HSI_COMPARISON}" == "true" ]]; then
  HSI_ARGS+=(--export-hsi-comparison)
fi
if [[ "${EXPORT_PRE_REFINE_COMPARISON}" == "true" ]]; then
  HSI_ARGS+=(--export-pre-refine-comparison)
fi
if [[ "${EXPORT_TRANSLATION_DEBUG_JSON}" == "true" ]]; then
  HSI_ARGS+=(--export-translation-debug-json)
fi
if [[ "${HSI_ALIGN_SCENE}" == "true" ]]; then
  HSI_ARGS+=(--hsi-align-scene)
fi

ALIGN_ARGS=()
if [[ "${ALIGN_SCENE_TO_SMPL}" == "true" ]]; then
  ALIGN_ARGS+=(--align-scene-to-smpl --align-scale-max "${ALIGN_SCALE_MAX}")
  if [[ "${ALIGN_USE_GT_SMPL_ANCHORS}" == "true" ]]; then
    ALIGN_ARGS+=(--align-use-gt-smpl-anchors)
  fi
fi

ERROR_ARGS=()
if [[ "${INCLUDE_ERROR_PANELS}" == "true" ]]; then
  ERROR_ARGS+=(--include-error-panels)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_clip_scene_affine_video.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --stride "${STRIDE}" \
  --fps "${FPS}" \
  --panel-size "${PANEL_SIZE}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --ply-top-k "${PLY_TOP_K}" \
  --draw-smpl-joints \
  --export-selected-ply \
  --ply-frame-indices "${PLY_FRAME_INDICES}" \
  --export-ply \
  --export-scene-ply \
  "${GT_ARGS[@]}" \
  "${PRIOR_ARGS[@]}" \
  "${HSI_ARGS[@]}" \
  "${ALIGN_ARGS[@]}" \
  "${ERROR_ARGS[@]}" \
  --depth-max-m "${DEPTH_MAX_M}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
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

echo "========== HSI bad-frame full-sequence PLY export finished =========="
echo "Summary json: ${OUTPUT_DIR}/hsi_clip_scene_affine_video_summary.json"
echo "PLY root    : ${OUTPUT_DIR}/ply"
