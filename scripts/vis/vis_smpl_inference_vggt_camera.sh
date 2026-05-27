#!/usr/bin/env bash

set -euo pipefail

# Visualize trained SMPL query predictions with VGGT camera projection.

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VGGT_CKPT="/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
SMPL_MODEL_DIR="/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/smpl/"
SMPL_CKPT="${REPO_ROOT}/outputs/train/smpl_hungarian_20q/checkpoint_latest.pt"
IMAGE_PATH="${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png"
OUTPUT_DIR="${REPO_ROOT}/outputs/vis/smpl_inference_vggt_camera"

CONF_THRESHOLD="0.25"
TOP_K="20"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL training checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== SMPL VGGT-camera visualization =========="
echo "Image       : ${IMAGE_PATH}"
echo "SMPL ckpt   : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL model  : ${SMPL_MODEL_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Confidence  : ${CONF_THRESHOLD}"
echo "Top-K       : ${TOP_K}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_inference.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --draw-smpl-joints

echo "========== SMPL VGGT-camera visualization finished =========="
echo "Output image: ${OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.jpg"
echo "Output json : ${OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.json"
