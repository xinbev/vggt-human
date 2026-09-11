#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PLY_FILE="${PLY_FILE:-}"
PLY_FILES="${PLY_FILES:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/ply_smpl_recolor}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${PLY_FILE}" ]]; then
  ARGS+=(--ply "${PLY_FILE}")
fi
if [[ -n "${PLY_FILES}" ]]; then
  IFS=';' read -r -a PLY_FILE_LIST <<< "${PLY_FILES}"
  for PLY_ITEM in "${PLY_FILE_LIST[@]}"; do
    if [[ -n "${PLY_ITEM}" ]]; then
      ARGS+=(--ply "${PLY_ITEM}")
    fi
  done
fi

cd "${REPO_ROOT}"
python scripts/vis/serve_ply_smpl_recolor_viser.py "${ARGS[@]}"
