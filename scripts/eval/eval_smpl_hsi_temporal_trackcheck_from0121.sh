#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_temporal_trackcheck_0121}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
NUM_VIEWS="${NUM_VIEWS:-4}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"

export REPO_ROOT
export SMPL_CKPT
export OUTPUT_DIR
export MAX_SAMPLES
export NUM_VIEWS
export USE_GT_BOX_PRIOR

echo "========== SMPL HSI temporal GT-track check =========="
echo "Checkpoint : ${SMPL_CKPT}"
echo "Output     : ${OUTPUT_DIR}"
echo "Views      : ${NUM_VIEWS}"
echo "Samples    : ${MAX_SAMPLES}"
echo "GT prior   : ${USE_GT_BOX_PRIOR}"

bash "${REPO_ROOT}/scripts/eval/eval_smpl_hsi_temporal_from0121.sh"

echo "========== SMPL HSI temporal GT-track check finished =========="
echo "Metrics json: ${OUTPUT_DIR}/hsi_temporal_metrics.json"
