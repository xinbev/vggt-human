#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/cache}"
SCALE_FILE="${SCALE_FILE:-${CACHE_DIR}/manual_scales.json}"
SCALE_MODE="${SCALE_MODE:-manual}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/${SCALE_MODE}_metrics}"

args=(
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --mode "${SCALE_MODE}"
  --oracle-scale-min "${ORACLE_SCALE_MIN:-0.25}"
  --oracle-scale-max "${ORACLE_SCALE_MAX:-4.0}"
  --oracle-steps "${ORACLE_STEPS:-41}"
)
if [[ "${SCALE_MODE}" == "manual" ]]; then
  args+=(--scale-file "${SCALE_FILE}")
fi
if [[ "${ALLOW_MISSING_MANUAL_SCALES:-false}" == "true" ]]; then
  args+=(--allow-missing-manual-scales)
fi

PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
python scripts/eval/evaluate_rich_manual_scale.py "${args[@]}"
