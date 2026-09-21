#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}" \
MODE=full \
WINDOW_SIZE="${WINDOW_SIZE:-100}" \
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-0}" \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/base_evaluation}" \
MANUAL_SCALE_CACHE_DIR="${MANUAL_SCALE_CACHE_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/cache}" \
RESUME="${RESUME:-true}" \
bash scripts/eval/evaluate_rich_physical_grounding.sh
