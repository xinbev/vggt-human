#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/long_tum_s1}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/vggt_omega_1b_512.pt}"
PRED_PARENT="${PRED_PARENT:-${REPO_ROOT}/outputs/eval/tum_dynamics_predictions}"

python benchmarks/tum_dynamics_ate/infer_vggt.py \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-parent "${PRED_PARENT}" \
  --lengths "${LENGTHS:-50,100,150,200,300,400,500,600,700,800,900,1000}" \
  --model-name "${MODEL:-vggt}" \
  --image-resolution "${IMAGE_RESOLUTION:-512}" \
  --image-mode "${IMAGE_MODE:-balanced}" \
  --device "${DEVICE:-cuda:0}" \
  --max-sequences "${MAX_SEQUENCES:-0}"

