#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SEQUENCE_DIR="${SEQUENCE_DIR:-/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/20241213_1_250_rome_tracking_seq_000002}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/bedlam_coordinate_hypothesis_audit}"
PORT="${PORT:-8090}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smpl}"
FRAME_COUNT="${FRAME_COUNT:-3}"
FRAME_STEP="${FRAME_STEP:-5}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] Missing repository: ${REPO_ROOT}" >&2; exit 1; }
[[ -d "${SEQUENCE_DIR}" ]] || { echo "[ERROR] Missing processed sequence: ${SEQUENCE_DIR}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model directory: ${SMPL_MODEL_DIR}" >&2; exit 1; }

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
ARGS=(
  --sequence-dir "${SEQUENCE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --port "${PORT}"
  --smpl-model-dir "${SMPL_MODEL_DIR}"
  --frame-count "${FRAME_COUNT}"
  --frame-step "${FRAME_STEP}"
)
if [[ "${SMOKE_ONLY}" == "1" || "${SMOKE_ONLY}" == "true" || "${SMOKE_ONLY}" == "TRUE" ]]; then
  ARGS+=(--smoke-only)
fi

exec python3 scripts/vis/serve_bedlam_coordinate_hypothesis_viser.py "${ARGS[@]}" "$@"
