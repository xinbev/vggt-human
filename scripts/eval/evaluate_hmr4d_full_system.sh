#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/checkpoint.pt" >&2
  exit 2
fi

DATASETS="${DATASETS:-emdb1 emdb2 3dpw}"
DEVICE="${DEVICE:-cuda}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/hmr4d_smpl_metrics}"
ALLOW_MISSING_METRICS="${ALLOW_MISSING_METRICS:-}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_hsi_full_system_restructure.yaml}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFER_HSI="${PREFER_HSI:-true}"

for DATASET in ${DATASETS}; do
  ARGS=(
    --dataset "${DATASET}"
    --checkpoint "${CHECKPOINT}"
    --path-config "${PATH_CONFIG}"
    --train-config "${TRAIN_CONFIG}"
    --output-dir "${OUT_ROOT}/${DATASET}"
    --device "${DEVICE}"
    --sequence-length "${SEQUENCE_LENGTH}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ "${PREFER_HSI}" == "true" ]]; then
    ARGS+=(--prefer-hsi)
  else
    ARGS+=(--no-prefer-hsi)
  fi
  if [[ "${MAX_WINDOWS}" != "0" ]]; then
    ARGS+=(--max-windows "${MAX_WINDOWS}")
  fi
  if [[ -n "${ALLOW_MISSING_METRICS}" ]]; then
    ARGS+=(--allow-missing-metrics)
  fi
  python scripts/eval/evaluate_hmr4d_smpl_metrics.py "${ARGS[@]}"
done
