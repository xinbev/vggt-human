#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/cache}"
SCALE_FILE="${SCALE_FILE:-${CACHE_DIR}/manual_scales.json}"
PORT="${PORT:-8080}"

PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
python scripts/vis/serve_rich_manual_scale_viewer.py \
  --cache-dir "${CACHE_DIR}" \
  --scale-file "${SCALE_FILE}" \
  --port "${PORT}" \
  --point-size "${POINT_SIZE:-0.008}" \
  --scale-min "${SCALE_MIN:-0.10}" \
  --scale-max "${SCALE_MAX:-10.0}"
