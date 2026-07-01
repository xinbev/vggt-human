#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_hf_bedlam_ray_refine.yaml}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${MAX_NPZ_FILES:-}" ]]; then
  ARGS+=(--override "data.max_npz_files=${MAX_NPZ_FILES}")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--override "data.max_frames=${MAX_FRAMES}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  ARGS+=(--override "optim.epochs=${EPOCHS}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  ARGS+=(--override "experiment.output_dir=${OUT_DIR}")
fi

python scripts/train/train_smpl.py "${ARGS[@]}"
