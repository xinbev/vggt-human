#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_emdb_p4_36_long_walk_coarse_residual_v3}"
VIEWER_CACHE_OUTPUT="${VIEWER_CACHE_OUTPUT:-${OUTPUT_DIR}/viewer_cache}"

REPO_ROOT="${REPO_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VIEWER_CACHE_OUTPUT="${VIEWER_CACHE_OUTPUT}" \
SMOKE_ONLY=true \
bash "${REPO_ROOT}/scripts/vis/serve_stage2_emdb_p4_36_long_walk_uniform.sh"
