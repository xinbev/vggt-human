#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASETS="${DATASETS:-emdb1 3dpw}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

for DATASET in ${DATASETS}; do
  ARGS=(
    --dataset "${DATASET}"
    --path-config "${PATH_CONFIG}"
    --max-sequences "${MAX_SEQUENCES}"
  )
  if [[ -n "${OVERWRITE_FLAG:-}" ]]; then
    ARGS+=(--overwrite)
  fi
  python scripts/preprocess/extract_hmr4d_eval_frames.py "${ARGS[@]}"
done
