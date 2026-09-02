#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --config "${CONFIG:-benchmarks/emdb2_global/config.yaml}"
  --output "${OUTPUT:-outputs/debug/emdb2_global_data_check/summary.json}"
)
if [[ -n "${EMDB_ROOT:-}" ]]; then ARGS+=(--emdb-root "${EMDB_ROOT}"); fi
python benchmarks/emdb2_global/check_data.py "${ARGS[@]}"

