#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710}"
SMPL_CKPT="${SMPL_CKPT:-${PIPELINE_OUTPUT_ROOT}/stage2_anchor_transl/checkpoint_latest.pt}"
VGGT_CKPT="${VGGT_CKPT:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"

SPLIT="${SPLIT:-Training}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
IMAGE_PATH="${IMAGE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/nlf_hsi_depth_flattening}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
ROI_EXPAND="${ROI_EXPAND:-0.65}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
VERTEX_SAMPLE_STRIDE="${VERTEX_SAMPLE_STRIDE:-25}"
OVERLAY_MAX_VERTICES="${OVERLAY_MAX_VERTICES:-600}"

export DATA_ROOT

cd "${REPO_ROOT}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM boxes root: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing HSI checkpoint: ${SMPL_CKPT}" >&2; exit 1; }

ARGS=(
  --checkpoint "${SMPL_CKPT}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --num-samples "${NUM_SAMPLES}"
  --conf-threshold "${CONF_THRESHOLD}"
  --roi-expand "${ROI_EXPAND}"
  --depth-max-m "${DEPTH_MAX_M}"
  --vertex-sample-stride "${VERTEX_SAMPLE_STRIDE}"
  --overlay-max-vertices "${OVERLAY_MAX_VERTICES}"
  --use-gt-box-prior
  --override "datasets.bedlam_root=${BEDLAM_ROOT}"
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
  --override "data.require_depth=true"
  --override "data.require_boxes=true"
  --override "model.smpl_provider=nlf"
  --override "model.nlf_use_detector=false"
  --override "model.nlf_require_boxes=true"
  --override "model.enable_camera=true"
  --override "model.enable_depth=true"
  --override "model.enable_smpl=true"
  --override "model.enable_hsi_refine=true"
)

if [[ -n "${VGGT_CKPT}" ]]; then
  ARGS+=(--override "checkpoints.vggt_baseline=${VGGT_CKPT}")
fi
if [[ -n "${IMAGE_PATH}" ]]; then
  ARGS+=(--image "${IMAGE_PATH}")
fi

echo "========== Debug NLF-HSI depth flattening =========="
echo "Repo        : ${REPO_ROOT}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Image       : ${IMAGE_PATH:-<dataset index ${START_INDEX}>}"
echo "Num samples : ${NUM_SAMPLES}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/diagnostics/debug_nlf_hsi_depth_flattening.py "${ARGS[@]}"

echo "========== Debug finished =========="
echo "Summary: ${OUTPUT_DIR}/summary.json"
