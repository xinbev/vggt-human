#!/usr/bin/env bash

set -euo pipefail

# Body Stage 2: SMPL-style body joints with small 2D reprojection loss.

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-omega"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
CONFIG_PATH="${REPO_ROOT}/configs/human3r_vggt_bedlam.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VGGT_CKPT="checkpoints/vggt_omega_1b_512.pt"
MHMR_CKPT="checkpoints/multiHMR_672_L.pt"
SMPLX_MODEL_DIR="checkpoints/body_models/smplx"
INIT_CKPT="${REPO_ROOT}/outputs/body_stage1_bootstrap/checkpoint_last.pt"
OUTPUT_DIR="${REPO_ROOT}/outputs/body_stage2_j2d1e-4"

EXTRA_EPOCHS="5"
LR="5e-5"
JOINTS2D_WEIGHT="0.0001"
JOINT_LOSS_COUNT="22"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] Missing config: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${MHMR_CKPT}" ]] || { echo "[ERROR] Missing Multi-HMR checkpoint: ${MHMR_CKPT}" >&2; exit 1; }
[[ -d "${SMPLX_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL-X model dir: ${SMPLX_MODEL_DIR}" >&2; exit 1; }

INIT_EPOCH="$(python - <<PY
import torch
ckpt = torch.load('${INIT_CKPT}', map_location='cpu')
print(int(ckpt.get('epoch', 0)))
PY
)"
TOTAL_EPOCHS="$((INIT_EPOCH + EXTRA_EPOCHS))"

echo "========== Body Stage 2: j2d1e-4 =========="
echo "Init ckpt   : ${INIT_CKPT}"
echo "Output      : ${OUTPUT_DIR}"
echo "Init epoch  : ${INIT_EPOCH}"
echo "Extra epochs: ${EXTRA_EPOCHS}"
echo "Total epochs: ${TOTAL_EPOCHS}"
echo "LR          : ${LR}"
echo "joints2d    : ${JOINTS2D_WEIGHT}"
echo "joint count : ${JOINT_LOSS_COUNT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python -m vggt_omega.human3r.train \
  --config "${CONFIG_PATH}" \
  --override "output_dir=${OUTPUT_DIR}" \
  --override "dataset.root=${BEDLAM_ROOT}" \
  --override "dataset.num_views=2" \
  --override "dataset.max_humans=6" \
  --override "dataset.num_workers=4" \
  --override "model.vggt_checkpoint=${VGGT_CKPT}" \
  --override "model.mhmr_checkpoint=${MHMR_CKPT}" \
  --override "model.smplx_model_dir=${SMPLX_MODEL_DIR}" \
  --override "train.batch_size=1" \
  --override "train.epochs=${TOTAL_EPOCHS}" \
  --override "train.lr=${LR}" \
  --override "train.weight_decay=0.05" \
  --override "train.amp=true" \
  --override "train.log_every=20" \
  --override "train.val_every=1" \
  --override "train.save_every=1" \
  --override "train.resume=${INIT_CKPT}" \
  --override "wandb.enabled=true" \
  --override "wandb.project=human3r-vggt-omega" \
  --override "wandb.name=body_stage2_j2d1e-4" \
  --override "loss.smplx_as_smpl=true" \
  --override "loss.joint_loss_count=${JOINT_LOSS_COUNT}" \
  --override "loss.transl_pelvis=1.0" \
  --override "loss.joints3d_abs=1.0" \
  --override "loss.joints2d=${JOINTS2D_WEIGHT}" \
  --override "loss.joints2d_abs3d_threshold=1.0"

echo "========== Body Stage 2 finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_last.pt"
