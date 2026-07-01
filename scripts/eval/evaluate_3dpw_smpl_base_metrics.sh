#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/checkpoint.pt" >&2
  exit 2
fi

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw.yaml}"
  --output-dir "${OUT_DIR:-outputs/eval/3dpw_smpl_base}"
  --split "${SPLIT:-test}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${LOG_INTERVAL:-}" ]]; then
  ARGS+=(--log-interval "${LOG_INTERVAL}")
fi
if [[ -n "${SUBSET_INDICES_CSV:-}" ]]; then
  ARGS+=(--subset-indices-csv "${SUBSET_INDICES_CSV}")
fi
if [[ -n "${SUBSET_INDEX_COLUMN:-}" ]]; then
  ARGS+=(--subset-index-column "${SUBSET_INDEX_COLUMN}")
fi

python scripts/eval/evaluate_3dpw_smpl_base_metrics.py "${ARGS[@]}"
