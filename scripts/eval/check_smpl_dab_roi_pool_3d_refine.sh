#!/usr/bin/env bash

set -euo pipefail

# Check SMPL DAB ROI-pool 3D-refine results with bbox metrics, joint visualizations, and PLY exports.

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

EVAL_CLEAN_DIR="${REPO_ROOT}/outputs/eval/smpl_dab_roi_pool_3d_refine_20q_gt_prior"
EVAL_NOISY_DIR="${REPO_ROOT}/outputs/eval/smpl_dab_roi_pool_3d_refine_20q_noisy_prior_c005_s010"
VIS_CLEAN_DIR="${REPO_ROOT}/outputs/vis/smpl_dab_roi_pool_3d_refine_20q_gt_prior"
VIS_NOISY_DIR="${REPO_ROOT}/outputs/vis/smpl_dab_roi_pool_3d_refine_20q_noisy_prior_c005_s010"

MAX_SAMPLES="500"
CONF_THRESHOLD="0.10"
CONF_THRESHOLDS=(0.05 0.10 0.25 0.30 0.50)
TOP_K="20"
PLY_TOP_K="3"
NUM_VIS_IMAGES="5"
NOISY_CENTER="0.05"
NOISY_SIZE="0.10"
NOISY_DROP="0.03"

cd "${REPO_ROOT}"
mkdir -p "${EVAL_CLEAN_DIR}" "${EVAL_NOISY_DIR}" "${VIS_CLEAN_DIR}" "${VIS_NOISY_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL ROI-pool checkpoint: ${SMPL_CKPT}" >&2; exit 1; }

mapfile -t VIS_IMAGES < <(find "${BEDLAM_ROOT}/Training" -path "*/rgb/*.png" | sort | head -n "${NUM_VIS_IMAGES}")
[[ "${#VIS_IMAGES[@]}" -gt 0 ]] || { echo "[ERROR] No BEDLAM RGB images found under ${BEDLAM_ROOT}/Training" >&2; exit 1; }

echo "========== SMPL DAB ROI-Pool 3D-Refine Check =========="
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL models : ${SMPL_MODEL_DIR}"
echo "Max samples : ${MAX_SAMPLES}"
echo "Confidence  : ${CONF_THRESHOLD}"
echo "Top-K       : ${TOP_K}"
echo "PLY Top-K   : ${PLY_TOP_K}"
echo "Vis images  : ${#VIS_IMAGES[@]}"

echo "========== Eval: clean GT box prior =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/eval_smpl_box_metrics.py \
  --checkpoint "${SMPL_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${EVAL_CLEAN_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --conf-thresholds "${CONF_THRESHOLDS[@]}" \
  --use-gt-box-prior \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"

echo "========== Eval: noisy GT box prior =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/eval_smpl_box_metrics.py \
  --checkpoint "${SMPL_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${EVAL_NOISY_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --conf-thresholds "${CONF_THRESHOLDS[@]}" \
  --use-gt-box-prior \
  --gt-box-prior-center-noise "${NOISY_CENTER}" \
  --gt-box-prior-size-noise "${NOISY_SIZE}" \
  --gt-box-prior-drop-prob "${NOISY_DROP}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"

echo "========== Visualization: clean/noisy projected SMPL joints and PLY =========="
for IMAGE_PATH in "${VIS_IMAGES[@]}"; do
  echo "[vis] ${IMAGE_PATH}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_inference.py \
    --image "${IMAGE_PATH}" \
    --checkpoint "${SMPL_CKPT}" \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --output-dir "${VIS_CLEAN_DIR}" \
    --conf-threshold "${CONF_THRESHOLD}" \
    --top-k "${TOP_K}" \
    --use-gt-box-prior \
    --draw-smpl-joints \
    --draw-gt-smpl-joints \
    --export-ply \
    --export-scene-ply \
    --ply-top-k "${PLY_TOP_K}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --smpl-model-dir "${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_inference.py \
    --image "${IMAGE_PATH}" \
    --checkpoint "${SMPL_CKPT}" \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --output-dir "${VIS_NOISY_DIR}" \
    --conf-threshold "${CONF_THRESHOLD}" \
    --top-k "${TOP_K}" \
    --use-gt-box-prior \
    --gt-box-prior-center-noise "${NOISY_CENTER}" \
    --gt-box-prior-size-noise "${NOISY_SIZE}" \
    --gt-box-prior-drop-prob "${NOISY_DROP}" \
    --draw-smpl-joints \
    --draw-gt-smpl-joints \
    --export-ply \
    --export-scene-ply \
    --ply-top-k "${PLY_TOP_K}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --smpl-model-dir "${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
done

echo "========== SMPL DAB ROI-Pool 3D-Refine check finished =========="
echo "Clean metrics: ${EVAL_CLEAN_DIR}/smpl_box_metrics.json"
echo "Noisy metrics: ${EVAL_NOISY_DIR}/smpl_box_metrics.json"
echo "Clean vis    : ${VIS_CLEAN_DIR}"
echo "Noisy vis    : ${VIS_NOISY_DIR}"
