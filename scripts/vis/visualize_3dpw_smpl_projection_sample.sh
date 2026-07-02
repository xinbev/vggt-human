#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${DEVICE:-}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  DEVICE=cuda
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/checkpoint.pt" >&2
  exit 2
fi

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_ray_refine.yaml}"
  --output-dir "${OUT_DIR:-outputs/vis/3dpw_smpl_projection_sample}"
  --split "${SPLIT:-test}"
  --top-k "${TOP_K:-4}"
  --start-index "${START_INDEX:-0}"
  --sort-metric "${SORT_METRIC:-transl_l2_mm}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${INDICES:-}" ]]; then
  ARGS+=(--indices "${INDICES}")
fi
if [[ -n "${ROWS_CSV:-}" ]]; then
  ARGS+=(--rows-csv "${ROWS_CSV}")
fi

python scripts/vis/visualize_3dpw_smpl_projection_sample.py "${ARGS[@]}"
