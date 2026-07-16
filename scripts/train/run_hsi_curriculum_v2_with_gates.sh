#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/hsi_scale_trans_contact_v2}"
GATE_ROOT="${GATE_ROOT:-${REPO_ROOT}/outputs/debug/hsi_curriculum_v2_gates_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"

for mode in smoke overfit distribution; do
  echo "========== Stage2-A ${mode} gate =========="
  GATE_STAGE=2A GATE_MODE="${mode}" \
  OUTPUT_ROOT="${GATE_ROOT}/stage2_${mode}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/smoke/check_hsi_curriculum_v2.sh
done

echo "========== Full Stage2-A/B =========="
RUN_STAGES=2A,2B \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh

STAGE2B_CKPT="${OUTPUT_ROOT}/stage2b_real_bridge/checkpoint_top01.pt"
for mode in smoke overfit distribution; do
  echo "========== Stage3-A1 ${mode} gate =========="
  GATE_STAGE=3A1 GATE_MODE="${mode}" \
  OUTPUT_ROOT="${GATE_ROOT}/stage3_${mode}" \
  STAGE2B_CKPT="${STAGE2B_CKPT}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/smoke/check_hsi_curriculum_v2.sh
done

echo "========== Full Stage3-A1/A2/B =========="
RUN_STAGES=3A1,3A2,3B \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
STAGE2B_CKPT="${STAGE2B_CKPT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh
