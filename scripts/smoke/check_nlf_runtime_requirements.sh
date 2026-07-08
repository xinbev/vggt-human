#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/nlf_runtime_requirements}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
export DATA_ROOT

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }

python scripts/diagnostics/check_nlf_runtime_requirements.py \
  --path-config "${PATH_CONFIG}" \
  --output-dir "${OUTPUT_DIR}"
