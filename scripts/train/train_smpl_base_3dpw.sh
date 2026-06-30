#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw.yaml}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

python scripts/train/train_smpl.py "${ARGS[@]}"
