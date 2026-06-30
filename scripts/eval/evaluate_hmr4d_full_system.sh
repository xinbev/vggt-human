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

for DATASET in ${DATASETS}; do
  ARGS=(
    --dataset "${DATASET}"
    --checkpoint "${CHECKPOINT}"
    --path-config configs/path.yaml
    --train-config configs/train_smpl_hsi_full_system_restructure.yaml
    --output-dir "${OUT_ROOT}/${DATASET}"
    --device "${DEVICE}"
    --sequence-length "${SEQUENCE_LENGTH}"
    --batch-size 1
    --num-workers 2
    --prefer-hsi
  )
  if [[ "${MAX_WINDOWS}" != "0" ]]; then
    ARGS+=(--max-windows "${MAX_WINDOWS}")
  fi
  if [[ -n "${ALLOW_MISSING_METRICS}" ]]; then
    ARGS+=(--allow-missing-metrics)
  fi
  python scripts/eval/evaluate_hmr4d_smpl_metrics.py "${ARGS[@]}"
done
