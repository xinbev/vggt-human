#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${FRAMES_DIR:-}" ]]; then
  echo "[ERROR] Set FRAMES_DIR=/path/to/frame_folder" >&2
  exit 1
fi

REQUESTED_DEVICE="${DEVICE:-cuda}"
RUN_DEVICE="${REQUESTED_DEVICE}"
if [[ "${REQUESTED_DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  RUN_DEVICE="cuda"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE_NAME="${SOURCE_NAME:-$(basename "${FRAMES_DIR}")}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vis/vggt_omega_folder_depth_ply/${SOURCE_NAME}}"

ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --path-config "${PATH_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${RUN_DEVICE}"
  --resize-mode "${RESIZE_MODE:-balanced}"
  --image-resolution "${IMAGE_RESOLUTION:-512}"
  --patch-size "${PATCH_SIZE:-16}"
  --sequence-length "${SEQUENCE_LENGTH:-0}"
  --coordinate-frame "${COORDINATE_FRAME:-world}"
  --depth-point-stride "${DEPTH_POINT_STRIDE:-2}"
  --max-depth "${MAX_DEPTH:-30.0}"
  --min-depth-conf "${MIN_DEPTH_CONF:-0.0}"
  --log-interval "${LOG_INTERVAL:-10}"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${FRAME_STRIDE:-}" ]]; then
  ARGS+=(--frame-stride "${FRAME_STRIDE}")
fi
if [[ "${STRICT_LOAD:-0}" == "1" ]]; then
  ARGS+=(--strict-load)
fi

echo "========== Native VGGT-Omega depth PLY export (no SMPL) =========="
echo "Frames dir       : ${FRAMES_DIR}"
echo "Source name      : ${SOURCE_NAME}"
echo "Path config      : ${PATH_CONFIG}"
echo "Checkpoint       : ${CHECKPOINT:-configs/path.yaml:checkpoints.vggt_baseline}"
echo "Output dir       : ${OUTPUT_DIR}"
echo "Device           : ${RUN_DEVICE}"
echo "Sequence length  : ${SEQUENCE_LENGTH:-0} (<=0 means all selected frames in one VGGT forward)"
echo "Coordinate frame : ${COORDINATE_FRAME:-world}"

python scripts/vis/export_vggt_omega_folder_depth_ply.py "${ARGS[@]}"

echo "========== Done =========="
