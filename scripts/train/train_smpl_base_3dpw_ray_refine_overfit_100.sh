#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SUBSET_CSV="${SUBSET_CSV:-outputs/debug/3dpw_overfit_subset/train_100_indices.csv}"

python scripts/tools/create_3dpw_overfit_subset.py \
  --output "${SUBSET_CSV}" \
  --start-index "${START_INDEX:-0}" \
  --num-samples "${NUM_SAMPLES:-100}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_ray_refine_overfit_100.yaml}"
  --override "data.subset_indices_csv=${SUBSET_CSV}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  ARGS+=(--override "optim.epochs=${EPOCHS}")
fi
if [[ -n "${LR:-}" ]]; then
  ARGS+=(--override "optim.lr=${LR}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  ARGS+=(--override "experiment.output_dir=${OUT_DIR}")
fi

python scripts/train/train_smpl.py "${ARGS[@]}"
