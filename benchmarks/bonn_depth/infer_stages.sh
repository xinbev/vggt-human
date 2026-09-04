#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:-/home/zhw/xyb_space/vggt_bonn_predictions}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/vggt_omega_1b_512.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:?Set STAGE2_CHECKPOINT to the Stage2 HSI checkpoint}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:?Set SCALE_CHECKPOINT to the HSI coarse-residual checkpoint}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
OVERWRITE_FLAG="${OVERWRITE:+--overwrite}"

python "${REPO_ROOT}/benchmarks/bonn_depth/infer.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${PRED_ROOT}/pure_vggt" \
  --stage pure_vggt \
  --checkpoint "${CHECKPOINT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --device "${DEVICE}" \
  ${OVERWRITE_FLAG}

python "${REPO_ROOT}/benchmarks/bonn_depth/infer.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${PRED_ROOT}/vggt_traditional_hsi_scale" \
  --stage vggt_traditional_hsi_scale \
  --checkpoint "${CHECKPOINT}" \
  --stage2-checkpoint "${STAGE2_CHECKPOINT}" \
  --scale-checkpoint "${SCALE_CHECKPOINT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --device "${DEVICE}" \
  ${OVERWRITE_FLAG}
