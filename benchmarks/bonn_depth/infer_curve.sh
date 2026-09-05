#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:-/home/zhw/xyb_space/vggt_bonn_curve_predictions}"
STAGE="${STAGE:-vggt_traditional_hsi_scale}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/vggt_omega_1b_512.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
RESIZE_MODE="${RESIZE_MODE:-balanced}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-50 100 150 200 250 300 350 400 450 500}"
START_FRAME="${START_FRAME:-30}"
ALLOW_SHORT="${ALLOW_SHORT:-true}"
OVERWRITE_FLAG="${OVERWRITE:+--overwrite}"

if [[ "${STAGE}" == "vggt_traditional_hsi_scale" ]]; then
  [[ -n "${STAGE2_CHECKPOINT}" ]] || { echo "[ERROR] Set STAGE2_CHECKPOINT" >&2; exit 2; }
  [[ -n "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Set SCALE_CHECKPOINT" >&2; exit 2; }
fi

for frames in ${PREFIX_LENGTHS}; do
  args=(
    --dataset-root "${DATASET_ROOT}"
    --output-root "${PRED_ROOT}/${STAGE}/prefix_${frames}"
    --stage "${STAGE}"
    --checkpoint "${CHECKPOINT}"
    --start-frame "${START_FRAME}"
    --num-frames "${frames}"
    --chunk-size 0
    --image-resolution "${IMAGE_RESOLUTION}"
    --resize-mode "${RESIZE_MODE}"
    --device "${DEVICE}"
  )
  if [[ "${STAGE}" == "vggt_traditional_hsi_scale" ]]; then
    args+=(--stage2-checkpoint "${STAGE2_CHECKPOINT}" --scale-checkpoint "${SCALE_CHECKPOINT}")
  fi
  if [[ "${ALLOW_SHORT}" == "true" ]]; then
    args+=(--allow-short)
  fi
  if [[ -n "${OVERWRITE_FLAG}" ]]; then
    args+=("${OVERWRITE_FLAG}")
  fi
  echo "[curve] stage=${STAGE} prefix=${frames} (one complete forward; no chunking)"
  python "${REPO_ROOT}/benchmarks/bonn_depth/infer.py" "${args[@]}"
done
