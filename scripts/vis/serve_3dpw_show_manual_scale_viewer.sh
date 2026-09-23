#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
python scripts/vis/serve_3dpw_show_manual_scale_viewer.py \
  --cache-dir "${CACHE_DIR:-outputs/eval/show_3dpw_manual_scale/cache}" \
  --scale-file "${SCALE_FILE:-}" \
  --port "${PORT:-8080}" \
  --point-stride "${POINT_STRIDE:-8}" \
  --point-size "${POINT_SIZE:-0.006}" \
  --initial-sequence "${INITIAL_SEQUENCE:-0}"
