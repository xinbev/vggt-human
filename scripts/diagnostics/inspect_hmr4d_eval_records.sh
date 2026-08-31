#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="${DATASET:-3dpw}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
FILTER="${FILTER:-}"
ARGS=(--dataset "${DATASET}" --path-config "${PATH_CONFIG}")
if [[ -n "${FRAMES_ROOT}" ]]; then ARGS+=(--frames-root "${FRAMES_ROOT}"); fi
if [[ -n "${FILTER}" ]]; then ARGS+=(--filter "${FILTER}"); fi
python scripts/diagnostics/inspect_hmr4d_eval_records.py "${ARGS[@]}"
