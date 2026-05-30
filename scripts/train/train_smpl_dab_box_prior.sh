#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
PREPROCESSED_ROOT="${REPO_ROOT}/outputs/preprocess/bedlam_boxes"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_dab_box_prior.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VGGT_CKPT="/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
SMPL_MODEL_DIR="/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/"
RESUME_CKPT="${REPO_ROOT}/outputs/train/smpl_conf_quality_aux_20q/checkpoint_latest.pt"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/smpl_dab_box_prior_20q"

EPOCHS="40"
LR="2e-5"
MAX_HUMANS="20"
NUM_VIEWS="2"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing resume checkpoint: ${RESUME_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== SMPL DAB Box Prior Query Stage =========="
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "SMPL models : ${SMPL_MODEL_DIR}"
echo "Resume      : ${RESUME_CKPT}"
echo "Output      : ${OUTPUT_DIR}"
echo "Epochs      : ${EPOCHS}"
echo "LR          : ${LR}"
echo "Max humans  : ${MAX_HUMANS}"
echo "Num views   : ${NUM_VIEWS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.resume=${RESUME_CKPT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.val_split=" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.require_boxes=true" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.enable_camera=true" \
  --override "model.freeze_aggregator=true" \
  --override "model.freeze_camera_head=true" \
  --override "model.train_smpl_query_token=true" \
  --override "model.predict_boxes=true" \
  --override "model.smpl_bbox_mode=reference_residual" \
  --override "model.predict_id_embed=false" \
  --override "model.smpl_return_aux=true" \
  --override "model.smpl_query_box_prior=true" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "matching.enabled=true" \
  --override "matching.cost_conf=0.5" \
  --override "matching.cost_bbox=8.0" \
  --override "matching.cost_giou=4.0" \
  --override "matching.cost_kpts=0.0" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.batch_size=1" \
  --override "optim.log_interval=20"

echo "========== SMPL DAB Box Prior Query finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
