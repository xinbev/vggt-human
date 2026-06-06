#!/usr/bin/env bash

set -euo pipefail

# Visualize the SMPL scene-alignment checkpoint and export PLY geometry.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_scene_align.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_scene_align_20q/checkpoint_latest.pt}"
IMAGE_PATH="${IMAGE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/smpl_scene_align_20q_env_smpl_ply}"

CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
TOP_K="${TOP_K:-20}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
PLY_COORDINATE_FRAME="${PLY_COORDINATE_FRAME:-camera}"
PLY_MAX_DEPTH_POINTS="${PLY_MAX_DEPTH_POINTS:-0}"
PLY_DEPTH_CONF_PERCENTILE="${PLY_DEPTH_CONF_PERCENTILE:-0}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL scene-align checkpoint: ${SMPL_CKPT}" >&2; exit 1; }

if [[ -z "${IMAGE_PATH}" ]]; then
  IMAGE_PATH="$(find "${BEDLAM_ROOT}/Training" -type f -path "*/rgb/*.png" -print -quit)"
fi

[[ -n "${IMAGE_PATH}" ]] || { echo "[ERROR] Could not find a training image under ${BEDLAM_ROOT}/Training" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== SMPL Scene Alignment PLY visualization =========="
echo "Image       : ${IMAGE_PATH}"
echo "SMPL ckpt   : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL model  : ${SMPL_MODEL_DIR}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Confidence  : ${CONF_THRESHOLD}"
echo "Top-K       : ${TOP_K}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "PLY frame   : ${PLY_COORDINATE_FRAME}"

GT_PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  GT_PRIOR_ARGS+=(--use-gt-box-prior)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${PYTHON_BIN}" scripts/vis/visualize_smpl_inference.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --draw-smpl-joints \
  --export-ply \
  --ply-coordinate-frame "${PLY_COORDINATE_FRAME}" \
  --ply-max-depth-points "${PLY_MAX_DEPTH_POINTS}" \
  --ply-depth-conf-percentile "${PLY_DEPTH_CONF_PERCENTILE}" \
  "${GT_PRIOR_ARGS[@]}"

IMAGE_STEM="$(basename "${IMAGE_PATH%.*}")"
echo "========== SMPL Scene Alignment PLY visualization finished =========="
echo "Output image : ${OUTPUT_DIR}/${IMAGE_STEM}_smpl_predictions.jpg"
echo "Output json  : ${OUTPUT_DIR}/${IMAGE_STEM}_smpl_predictions.json"
echo "Env PLY      : ${OUTPUT_DIR}/${IMAGE_STEM}_environment_points_${PLY_COORDINATE_FRAME}.ply"
echo "SMPL PLY     : ${OUTPUT_DIR}/${IMAGE_STEM}_smpl_meshes_${PLY_COORDINATE_FRAME}.ply"
