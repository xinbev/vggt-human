#!/usr/bin/env bash

set -euo pipefail

# Long-clip inference diagnostic for HSI scene affine:
# compare per-frame HSI s,b against clip-level median/EMA aggregation.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_refine.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_clip_scene_affine_0121}"

NUM_FRAMES="${NUM_FRAMES:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
EMA_ALPHA="${EMA_ALPHA:-0.25}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-20}"
PLY_TOP_K="${PLY_TOP_K:-3}"
EXPORT_STRIDE="${EXPORT_STRIDE:-4}"
MAX_EXPORT_FRAMES="${MAX_EXPORT_FRAMES:-4}"
EXPORT_SCENE_PLY="${EXPORT_SCENE_PLY:-true}"
EXPORT_AFFINE_MODES="${EXPORT_AFFINE_MODES:-per_frame clip_median ema}"
DRAW_OVERLAYS="${DRAW_OVERLAYS:-true}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
USE_HSI_REFINED="${USE_HSI_REFINED:-true}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing HSI checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing start image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== HSI clip scene affine visualization =========="
echo "Start image : ${IMAGE_PATH}"
echo "Frames      : ${NUM_FRAMES}"
echo "Stride      : ${FRAME_STRIDE}"
echo "Checkpoint  : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "Output      : ${OUTPUT_DIR}"
echo "EMA alpha   : ${EMA_ALPHA}"
echo "Export PLY  : ${EXPORT_SCENE_PLY}"
echo "PLY modes   : ${EXPORT_AFFINE_MODES}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"

PLY_ARGS=()
if [[ "${EXPORT_SCENE_PLY}" == "true" ]]; then
  PLY_ARGS+=(--export-scene-ply)
fi

OVERLAY_ARGS=()
if [[ "${DRAW_OVERLAYS}" == "true" ]]; then
  OVERLAY_ARGS+=(--draw-overlays --draw-smpl-joints --draw-gt-smpl-joints)
fi

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

HSI_ARGS=()
if [[ "${USE_HSI_REFINED}" == "true" ]]; then
  HSI_ARGS+=(--use-hsi-refined)
fi

# shellcheck disable=SC2206
AFFINE_MODE_ARGS=(${EXPORT_AFFINE_MODES})

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_clip_scene_affine.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --stride "${FRAME_STRIDE}" \
  --ema-alpha "${EMA_ALPHA}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --ply-top-k "${PLY_TOP_K}" \
  --export-stride "${EXPORT_STRIDE}" \
  --max-export-frames "${MAX_EXPORT_FRAMES}" \
  --export-affine-modes "${AFFINE_MODE_ARGS[@]}" \
  --depth-max-m "${DEPTH_MAX_M}" \
  "${PLY_ARGS[@]}" \
  "${OVERLAY_ARGS[@]}" \
  "${PRIOR_ARGS[@]}" \
  "${HSI_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true"

echo "========== HSI clip scene affine visualization finished =========="
echo "Summary json: ${OUTPUT_DIR}/hsi_clip_scene_affine_summary.json"
