#!/usr/bin/env bash
set -euo pipefail

# Single-frame Stage2 visual diagnostics. This reuses the established HSI depth/SMPL
# diagnostic renderer and points it at a Stage2 checkpoint.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
STAGE2_CKPT="${STAGE2_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_abc_transl_refine/stageC_temporal_refine/checkpoint_latest.pt}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_stage2_transl_refine_debug}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

export REPO_ROOT BEDLAM_ROOT PREPROCESSED_ROOT VIS_OUTPUT_DIR IMAGE_PATH CUDA_VISIBLE_DEVICES_VALUE
export SMPL_CKPT="${STAGE2_CKPT}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"

echo "========== HSI Stage2 translation refine visual debug =========="
echo "Stage2 ckpt : ${STAGE2_CKPT}"
echo "Image       : ${IMAGE_PATH}"
echo "Output      : ${VIS_OUTPUT_DIR}"

bash "${REPO_ROOT}/scripts/vis/vis_nlf_hsi_depth_smpl_diagnostics.sh"
