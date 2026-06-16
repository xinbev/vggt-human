#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_121_teacher_plane_contact.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_121_teacher_plane_contact/checkpoint_latest.pt}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/smpl_hsi_121_teacher_plane_contact_gt_prior_aligned}"
DIAG_VIS_OUTPUT_DIR="${DIAG_VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_121_teacher_plane_contact_diagnostics}"
DIAG_EVAL_OUTPUT_DIR="${DIAG_EVAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_121_teacher_plane_contact_diagnostics}"

CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-20}"
PLY_TOP_K="${PLY_TOP_K:-3}"
ALIGN_SCALE_MAX="${ALIGN_SCALE_MAX:-20.0}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
FOOT_SOLE_NUM_VERTICES="${FOOT_SOLE_NUM_VERTICES:-80}"
SUPPORT_PLANE_WINDOW="${SUPPORT_PLANE_WINDOW:-9}"
SUPPORT_PLANE_MIN_POINTS="${SUPPORT_PLANE_MIN_POINTS:-6}"

cd "${REPO_ROOT}"
mkdir -p "${VIS_OUTPUT_DIR}" "${DIAG_VIS_OUTPUT_DIR}" "${DIAG_EVAL_OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== SMPL HSI 121-teacher support-plane visualization =========="
echo "Image       : ${IMAGE_PATH}"
echo "Train config: ${TRAIN_CONFIG}"
echo "HSI ckpt    : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL model  : ${SMPL_MODEL_DIR}"
echo "Vis output  : ${VIS_OUTPUT_DIR}"
echo "Diag output : ${DIAG_EVAL_OUTPUT_DIR}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"

PRIOR_ARGS=()
DIAG_PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior --draw-gt-smpl-joints)
  DIAG_PRIOR_ARGS+=(--use-gt-box-prior)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_inference.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${VIS_OUTPUT_DIR}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --draw-smpl-joints \
  "${PRIOR_ARGS[@]}" \
  --export-ply \
  --export-scene-ply \
  --align-scene-to-smpl \
  --ply-top-k "${PLY_TOP_K}" \
  --align-scale-max "${ALIGN_SCALE_MAX}" \
  --use-hsi-refined \
  --export-hsi-comparison \
  --hsi-align-scene \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"

echo "========== Single-image depth / SMPL / sole-contact diagnostics =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_depth_smpl_diagnostics.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${DIAG_VIS_OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --foot-sole-num-vertices "${FOOT_SOLE_NUM_VERTICES}" \
  --support-plane-window "${SUPPORT_PLANE_WINDOW}" \
  --support-plane-min-points "${SUPPORT_PLANE_MIN_POINTS}" \
  "${DIAG_PRIOR_ARGS[@]}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true"

echo "========== Multi-sample metrics =========="
EVAL_PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  EVAL_PRIOR_ARGS+=(--use-gt-box-prior)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_hsi_refine_metrics.py \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${DIAG_EVAL_OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --max-samples "${MAX_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --foot-sole-num-vertices "${FOOT_SOLE_NUM_VERTICES}" \
  --support-plane-window "${SUPPORT_PLANE_WINDOW}" \
  --support-plane-min-points "${SUPPORT_PLANE_MIN_POINTS}" \
  "${EVAL_PRIOR_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true"

echo "========== 121-teacher support-plane visualization finished =========="
echo "Visualization image: ${VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.jpg"
echo "Visualization json : ${VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_smpl_predictions.json"
echo "Diagnostics json   : ${DIAG_VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_hsi_depth_smpl_diagnostics.json"
echo "Base sole overlay  : ${DIAG_VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_base_sole_contact_overlay.png"
echo "HSI sole overlay   : ${DIAG_VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_hsi_sole_contact_overlay.png"
echo "Metrics json       : ${DIAG_EVAL_OUTPUT_DIR}/hsi_refine_metrics.json"
