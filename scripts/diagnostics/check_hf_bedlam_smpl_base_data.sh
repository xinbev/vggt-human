#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_hf_bedlam_ray_refine.yaml}"
  --batch-size "${BATCH_SIZE:-1}"
  --max-npz-files "${MAX_NPZ_FILES:-1}"
  --max-frames "${MAX_FRAMES:-50}"
)

python scripts/diagnostics/check_hf_bedlam_smpl_base_data.py "${ARGS[@]}"
