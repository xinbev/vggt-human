#!/usr/bin/env bash

set -euo pipefail

# Two-stage temporal fine-tune:
#   Stage 1: freeze human refinement, train only HSI scene scale/bias temporal affine.
#   Stage 2: freeze scene affine, train only HSI human translation residual/gate.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_scene_affine.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_human_after_scene.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
TEACHER_CKPT="${TEACHER_CKPT:-${INIT_CKPT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_two_stage_from0121}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1_scene_affine}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2_human_temporal}"

STAGE1_EXTRA_EPOCHS="${STAGE1_EXTRA_EPOCHS:-2}"
STAGE2_EXTRA_EPOCHS="${STAGE2_EXTRA_EPOCHS:-1}"
STAGE1_LR="${STAGE1_LR:-5e-7}"
STAGE2_LR="${STAGE2_LR:-2e-7}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
DEPTH_ERROR_CLIP_M="${DEPTH_ERROR_CLIP_M:-1.5}"
DEPTH_ROI_EXPAND="${DEPTH_ROI_EXPAND:-0.75}"
DEPTH_MIN_VALID_PIXELS="${DEPTH_MIN_VALID_PIXELS:-2048}"

cd "${REPO_ROOT}"
mkdir -p "${STAGE1_OUTPUT_DIR}" "${STAGE2_OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${STAGE1_CONFIG}" ]] || { echo "[ERROR] Missing stage1 config: ${STAGE1_CONFIG}" >&2; exit 1; }
[[ -f "${STAGE2_CONFIG}" ]] || { echo "[ERROR] Missing stage2 config: ${STAGE2_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -f "${TEACHER_CKPT}" ]] || { echo "[ERROR] Missing teacher checkpoint: ${TEACHER_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

read_epoch() {
  python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0)
PY
}

INIT_EPOCH="$(read_epoch "${INIT_CKPT}")"
STAGE1_TOTAL_EPOCHS=$((INIT_EPOCH + STAGE1_EXTRA_EPOCHS))

echo "========== SMPL HSI temporal two-stage training =========="
echo "Init ckpt          : ${INIT_CKPT}"
echo "Teacher ckpt       : ${TEACHER_CKPT}"
echo "Init epoch         : ${INIT_EPOCH}"
echo "Views              : ${NUM_VIEWS}"
echo "Max humans         : ${MAX_HUMANS}"
echo "Stage1 output      : ${STAGE1_OUTPUT_DIR}"
echo "Stage1 extra epochs: ${STAGE1_EXTRA_EPOCHS}"
echo "Stage1 total epochs: ${STAGE1_TOTAL_EPOCHS}"
echo "Stage1 LR          : ${STAGE1_LR}"
echo "Stage2 output      : ${STAGE2_OUTPUT_DIR}"
echo "Stage2 extra epochs: ${STAGE2_EXTRA_EPOCHS}"
echo "Stage2 LR          : ${STAGE2_LR}"
df -h "${OUTPUT_ROOT}" || true

echo "========== Stage 1/2: scene affine temporal stabilization =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${STAGE1_CONFIG}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.load_vggt_baseline=false" \
  --override "checkpoint.resume=${INIT_CKPT}" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "teacher.enabled=true" \
  --override "teacher.checkpoint=${TEACHER_CKPT}" \
  --override "teacher.strict=false" \
  --override "experiment.output_dir=${STAGE1_OUTPUT_DIR}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.stride=1" \
  --override "data.val_split=" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.freeze_aggregator=true" \
  --override "model.freeze_aggregator_forward=true" \
  --override "model.freeze_camera_head=true" \
  --override "model.freeze_dense_head=true" \
  --override "model.freeze_smpl_head=true" \
  --override "model.freeze_hsi_backbone=true" \
  --override "model.freeze_hsi_scene_affine=false" \
  --override "model.train_hsi_scene_affine_only=true" \
  --override "loss.hsi_depth_teacher_max_m=${DEPTH_MAX_M}" \
  --override "loss.hsi_depth_teacher_error_clip_m=${DEPTH_ERROR_CLIP_M}" \
  --override "loss.hsi_depth_teacher_use_human_roi=true" \
  --override "loss.hsi_depth_teacher_roi_expand=${DEPTH_ROI_EXPAND}" \
  --override "loss.hsi_depth_teacher_min_valid_pixels=${DEPTH_MIN_VALID_PIXELS}" \
  --override "optim.epochs=${STAGE1_TOTAL_EPOCHS}" \
  --override "optim.lr=${STAGE1_LR}" \
  --override "optim.batch_size=1" \
  --override "optim.grad_clip_norm=0.15" \
  --override "optim.log_interval=20"

STAGE1_CKPT="${STAGE1_OUTPUT_DIR}/checkpoint_latest.pt"
[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] Stage1 checkpoint missing: ${STAGE1_CKPT}" >&2; exit 1; }
STAGE1_EPOCH="$(read_epoch "${STAGE1_CKPT}")"
STAGE2_TOTAL_EPOCHS=$((STAGE1_EPOCH + STAGE2_EXTRA_EPOCHS))

echo "========== Stage 2/2: human temporal refinement on fixed scene =========="
echo "Stage1 ckpt        : ${STAGE1_CKPT}"
echo "Stage1 epoch       : ${STAGE1_EPOCH}"
echo "Stage2 total epochs: ${STAGE2_TOTAL_EPOCHS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${STAGE2_CONFIG}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.load_vggt_baseline=false" \
  --override "checkpoint.resume=${STAGE1_CKPT}" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "teacher.enabled=true" \
  --override "teacher.checkpoint=${STAGE1_CKPT}" \
  --override "teacher.strict=false" \
  --override "experiment.output_dir=${STAGE2_OUTPUT_DIR}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.stride=1" \
  --override "data.val_split=" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.freeze_aggregator=true" \
  --override "model.freeze_aggregator_forward=true" \
  --override "model.freeze_camera_head=true" \
  --override "model.freeze_dense_head=true" \
  --override "model.freeze_smpl_head=true" \
  --override "model.freeze_hsi_backbone=true" \
  --override "model.freeze_hsi_scene_affine=true" \
  --override "model.train_hsi_transl_only=true" \
  --override "optim.epochs=${STAGE2_TOTAL_EPOCHS}" \
  --override "optim.lr=${STAGE2_LR}" \
  --override "optim.batch_size=1" \
  --override "optim.grad_clip_norm=0.10" \
  --override "optim.log_interval=20"

echo "========== SMPL HSI temporal two-stage training finished =========="
echo "Stage1 checkpoint: ${STAGE1_CKPT}"
echo "Stage2 checkpoint: ${STAGE2_OUTPUT_DIR}/checkpoint_latest.pt"
