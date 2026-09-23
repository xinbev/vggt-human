#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
python scripts/eval/evaluate_3dpw_show_manual_scale.py \
  --cache-dir "${CACHE_DIR:-outputs/eval/show_3dpw_manual_scale/cache}" \
  --scale-file "${SCALE_FILE:-}" \
  --output-dir "${OUTPUT_DIR:-outputs/eval/show_3dpw_manual_scale/metrics}" \
  --device "${DEVICE:-cuda}" \
  --scale-mode "${SCALE_MODE:-manual}" \
  ${ALLOW_MISSING_MANUAL_SCALES:+--allow-missing-manual-scales}
