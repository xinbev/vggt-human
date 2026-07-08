#!/usr/bin/env bash
set -euo pipefail

# Stage 0 geometry check: NLF base SMPL projection on the processed VGGT image plane.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_base_projection_overlay}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
export DATA_ROOT

IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
SPLIT="${SPLIT:-Training}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-8}"

cd "${REPO_ROOT}"
mkdir -p "${VIS_OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== NLF base projection overlay =========="
echo "Image       : ${IMAGE_PATH}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Path config : ${PATH_CONFIG}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "DATA_ROOT   : ${DATA_ROOT}"
echo "Output      : ${VIS_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_nlf_base_projection_overlay.py \
  --image "${IMAGE_PATH}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${VIS_OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.smpl_provider=nlf" \
  --override "model.nlf_use_detector=false" \
  --override "model.nlf_require_boxes=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_smpl=true" \
  --override "model.enable_hsi_refine=false"

echo "========== NLF base projection overlay finished =========="
echo "Overlay: ${VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_nlf_base_projection_overlay.png"
echo "JSON   : ${VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_nlf_base_projection_overlay.json"
