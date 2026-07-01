#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/checkpoint.pt" >&2
  exit 2
fi

SUBSET_CSV="${SUBSET_CSV:-outputs/debug/3dpw_overfit_subset/train_100_indices.csv}"
[[ -f "${SUBSET_CSV}" ]] || {
  python scripts/tools/create_3dpw_overfit_subset.py \
    --output "${SUBSET_CSV}" \
    --start-index "${START_INDEX:-0}" \
    --num-samples "${NUM_SAMPLES:-100}"
}

SPLIT="${SPLIT:-train}" \
SUBSET_INDICES_CSV="${SUBSET_CSV}" \
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_ray_refine_overfit_100.yaml}" \
OUT_DIR="${OUT_DIR:-outputs/eval/3dpw_smpl_base_overfit_100}" \
bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
