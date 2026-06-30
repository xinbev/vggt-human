#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_hsi_full_system_restructure.yaml}"
  --batch-size "${BATCH_SIZE:-1}"
)

if [[ -n "${SPLIT:-}" ]]; then
  ARGS+=(--split "${SPLIT}")
fi

python scripts/diagnostics/check_bedlam_full_system_data.py "${ARGS[@]}"
