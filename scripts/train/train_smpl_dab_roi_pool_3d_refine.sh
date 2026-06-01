#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
PREPROCESSED_ROOT="${REPO_ROOT}/outputs/preprocess/bedlam_boxes"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_dab_roi_pool_3d_refine.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VGGT_CKPT="/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
SMPL_MODEL_DIR="/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/"
INIT_CKPT="${REPO_ROOT}/outputs/train/smpl_dab_joint_refine_20q/checkpoint_latest.pt"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/smpl_dab_roi_pool_3d_refine_20q"

EXTRA_EPOCHS="40"
LR="1e-5"
MAX_HUMANS="20"
NUM_VIEWS="2"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

INIT_EPOCH=$(python - "${INIT_CKPT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0)
PY
)
TOTAL_EPOCHS=$((INIT_EPOCH + EXTRA_EPOCHS))

echo "========== SMPL DAB ROI-Pool 3D Refinement Stage =========="
echo "BEDLAM       : ${BEDLAM_ROOT}"
echo "Boxes        : ${PREPROCESSED_ROOT}"
echo "SMPL models  : ${SMPL_MODEL_DIR}"
echo "Init ckpt    : ${INIT_CKPT}"
echo "Init epoch   : ${INIT_EPOCH}"
echo "Extra epochs : ${EXTRA_EPOCHS}"
echo "Total epochs : ${TOTAL_EPOCHS}"
echo "Output       : ${OUTPUT_DIR}"
echo "LR           : ${LR}"
echo "Max humans   : ${MAX_HUMANS}"
echo "Num views    : ${NUM_VIEWS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.resume=${INIT_CKPT}" \
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
  --override "model.train_smpl_box_prior_embed=true" \
  --override "model.train_smpl_patch_pool_embed=true" \
  --override "model.predict_boxes=true" \
  --override "model.smpl_bbox_mode=reference_residual" \
  --override "model.predict_id_embed=false" \
  --override "model.smpl_return_aux=true" \
  --override "model.smpl_query_box_prior=true" \
  --override "model.smpl_query_patch_pool=true" \
  --override "model.smpl_query_patch_pool_expand=0.12" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "matching.enabled=true" \
  --override "matching.cost_conf=0.5" \
  --override "matching.cost_bbox=8.0" \
  --override "matching.cost_giou=4.0" \
  --override "matching.cost_kpts=0.0" \
  --override "loss.pose_weight=6.0" \
  --override "loss.betas_weight=0.8" \
  --override "loss.transl_cam_weight=2.0" \
  --override "loss.joints3d_weight=12.0" \
  --override "loss.projected_joints2d_weight=0.25" \
  --override "loss.conf_weight=0.35" \
  --override "loss.bbox_weight=0.5" \
  --override "loss.giou_weight=0.25" \
  --override "loss.projected_bbox_weight=0.05" \
  --override "loss.projected_giou_weight=0.05" \
  --override "loss.duplicate_conf_weight=0.10" \
  --override "loss.aux_weight=0.05" \
  --override "loss.aux_conf_weight=0.2" \
  --override "loss.aux_bbox_weight=0.5" \
  --override "loss.aux_giou_weight=0.25" \
  --override "optim.epochs=${TOTAL_EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.batch_size=1" \
  --override "optim.log_interval=20"

echo "========== SMPL DAB ROI-Pool 3D Refinement finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
