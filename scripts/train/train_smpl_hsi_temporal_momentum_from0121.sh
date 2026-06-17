#!/usr/bin/env bash

set -euo pipefail

# UniCon3R-style temporal momentum fine-tune:
# previous-frame HSI/contact tokens are fused into the current-frame HSI refinement.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_from0121.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
TEACHER_CKPT="${TEACHER_CKPT:-${INIT_CKPT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_momentum_from0121}"

EXTRA_EPOCHS="${EXTRA_EPOCHS:-4}"
LR="${LR:-5e-7}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.7}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -f "${TEACHER_CKPT}" ]] || { echo "[ERROR] Missing teacher checkpoint: ${TEACHER_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

INIT_EPOCH=$(python - "${INIT_CKPT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0)
PY
)
TOTAL_EPOCHS=$((INIT_EPOCH + EXTRA_EPOCHS))

echo "========== SMPL HSI temporal momentum from 0121 =========="
echo "Init ckpt      : ${INIT_CKPT}"
echo "Teacher ckpt   : ${TEACHER_CKPT}"
echo "Init epoch     : ${INIT_EPOCH}"
echo "Extra epochs   : ${EXTRA_EPOCHS}"
echo "Total epochs   : ${TOTAL_EPOCHS}"
echo "Output         : ${OUTPUT_DIR}"
echo "LR             : ${LR}"
echo "Views          : ${NUM_VIEWS}"
echo "Momentum decay : ${MOMENTUM_DECAY}"
df -h "${OUTPUT_DIR}" || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.resume=${INIT_CKPT}" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "teacher.enabled=true" \
  --override "teacher.checkpoint=${TEACHER_CKPT}" \
  --override "teacher.strict=false" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
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
  --override "model.freeze_hsi_backbone=false" \
  --override "model.freeze_hsi_scene_affine=false" \
  --override "model.train_hsi_transl_only=false" \
  --override "model.hsi_probe_mode=projected" \
  --override "model.hsi_affine_probe_mode=projected" \
  --override "model.hsi_probe_blend=1.0" \
  --override "model.hsi_use_delta_gate=true" \
  --override "model.hsi_enable_temporal_momentum=true" \
  --override "model.hsi_temporal_momentum_decay=${MOMENTUM_DECAY}" \
  --override "model.hsi_temporal_momentum_detach=true" \
  --override "model.hsi_temporal_momentum_use_track_ids=true" \
  --override "training_prior.center_noise=0.0" \
  --override "training_prior.size_noise=0.0" \
  --override "training_prior.drop_prob=0.0" \
  --override "loss.hsi_transl_velocity_weight=4.0" \
  --override "loss.hsi_joints_velocity_weight=10.0" \
  --override "loss.hsi_joints_acceleration_weight=6.0" \
  --override "loss.hsi_scene_scale_temporal_weight=1.0" \
  --override "loss.hsi_scene_scale_sequence_weight=0.5" \
  --override "loss.hsi_scene_bias_temporal_weight=0.2" \
  --override "loss.hsi_scene_bias_sequence_weight=0.1" \
  --override "loss.hsi_teacher_transl_weight=2.0" \
  --override "loss.hsi_teacher_joints_weight=3.0" \
  --override "loss.hsi_teacher_vertices_weight=2.0" \
  --override "loss.hsi_teacher_scene_affine_weight=2.5" \
  --override "loss.hsi_depth_teacher_weight=0.0" \
  --override "loss.hsi_anchor_depth_weight=0.0" \
  --override "loss.hsi_contact_weight=0.0" \
  --override "loss.hsi_foot_contact_weight=0.0" \
  --override "loss.hsi_foot_sole_contact_weight=0.0" \
  --override "loss.hsi_support_plane_contact_weight=0.0" \
  --override "optim.epochs=${TOTAL_EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.batch_size=1" \
  --override "optim.grad_clip_norm=0.15" \
  --override "optim.log_interval=20"

echo "========== SMPL HSI temporal momentum finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
