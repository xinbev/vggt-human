#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-emdb1}"
ARGS=(
  --dataset "${DATASET}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --png-compression "${PNG_COMPRESSION:-3}"
)
if [[ -n "${SUPPORT_ROOT:-}" ]]; then ARGS+=(--support-root "${SUPPORT_ROOT}"); fi
if [[ -n "${FRAMES_ROOT:-}" ]]; then ARGS+=(--frames-root "${FRAMES_ROOT}"); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

python scripts/preprocess/extract_hmr4d_eval_frames.py "${ARGS[@]}"
