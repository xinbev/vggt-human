#!/usr/bin/env bash

set -euo pipefail

# Visualize trained SMPL query predictions with VGGT camera projection.

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
PREPROCESSED_ROOT="${REPO_ROOT}/outputs/preprocess/bedlam_boxes"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_dab_roi_pool_3d_refine.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VGGT_CKPT="/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
SMPL_MODEL_DIR="/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/"
SMPL_CKPT="${REPO_ROOT}/outputs/train/smpl_dab_roi_pool_3d_refine_20q/checkpoint_latest.pt"
IMAGE_PATH="${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png"
OUTPUT_DIR="${REPO_ROOT}/outputs/vis/smpl_dab_roi_pool_3d_refine_gt_prior_aligned"

CONF_THRESHOLD="0.10"
TOP_K="20"
PLY_TOP_K="3"
ALIGN_SCENE_TO_SMPL="true"
ALIGN_SCALE_MAX="${ALIGN_SCALE_MAX:-20.0}"
ALIGN_USE_GT_SMPL_ANCHORS="${ALIGN_USE_GT_SMPL_ANCHORS:-false}"
USE_GT_BOX_PRIOR="true"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL training checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== SMPL VGGT-camera visualization =========="
echo "Image       : ${IMAGE_PATH}"
echo "SMPL ckpt   : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL model  : ${SMPL_MODEL_DIR}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "Output      : ${OUTPUT_DIR}"
echo "Confidence  : ${CONF_THRESHOLD}"
echo "Top-K       : ${TOP_K}"
echo "Align scene : ${ALIGN_SCENE_TO_SMPL}"
echo "Align max   : ${ALIGN_SCALE_MAX}"
echo "GT anchors  : ${ALIGN_USE_GT_SMPL_ANCHORS}"
echo "GT box prior: ${USE_GT_BOX_PRIOR}"

ALIGN_ARGS=()
if [[ "${ALIGN_SCENE_TO_SMPL}" == "true" ]]; then
  ALIGN_ARGS+=(--export-ply --export-scene-ply --align-scene-to-smpl --ply-top-k "${PLY_TOP_K}" --align-scale-max "${ALIGN_SCALE_MAX}")
  if [[ "${ALIGN_USE_GT_SMPL_ANCHORS}" == "true" ]]; then
    ALIGN_ARGS+=(--align-use-gt-smpl-anchors)
  fi
fi
PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior --draw-gt-smpl-joints)
fi

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
  --draw-smpl-joints \
  "${PRIOR_ARGS[@]}" \
  "${ALIGN_ARGS[@]}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"

echo "========== SMPL VGGT-camera visualization finished =========="
echo "Output image: ${OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.jpg"
echo "Output json : ${OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.json"
