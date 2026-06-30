#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw.yaml}"
  --batch-size "${BATCH_SIZE:-1}"
)

if [[ -n "${SPLIT:-}" ]]; then
  ARGS+=(--split "${SPLIT}")
fi

python scripts/diagnostics/check_3dpw_smpl_base_data.py "${ARGS[@]}"
