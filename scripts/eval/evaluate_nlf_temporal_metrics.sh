#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT to the VGGT/NLF experiment checkpoint}"
: "${TEMPORAL_CHECKPOINT:?Set TEMPORAL_CHECKPOINT to a TemporalSMPLRefiner checkpoint}"

DATASETS="${DATASETS:-emdb1 3dpw}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/nlf_temporal}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"

for DATASET in ${DATASETS}; do
  ARGS=(
    --dataset "${DATASET}"
    --checkpoint "${CHECKPOINT}"
    --temporal-checkpoint "${TEMPORAL_CHECKPOINT}"
    --path-config "${PATH_CONFIG:-configs/path.yaml}"
    --train-config "${TRAIN_CONFIG:-configs/eval_nlf_temporal.yaml}"
    --output-dir "${OUT_ROOT}/${DATASET}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE:-1}"
    --num-workers "${NUM_WORKERS:-2}"
    --stride "${STRIDE:-1}"
  )
  if [[ "${MAX_WINDOWS}" != "0" ]]; then
    ARGS+=(--max-windows "${MAX_WINDOWS}")
  fi
  python scripts/eval/evaluate_nlf_temporal_metrics.py "${ARGS[@]}"
done
