#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python scripts/tools/create_3dpw_overfit_subset.py \
  --output "${OUT_CSV:-outputs/debug/3dpw_overfit_subset/train_100_indices.csv}" \
  --start-index "${START_INDEX:-0}" \
  --num-samples "${NUM_SAMPLES:-100}"
