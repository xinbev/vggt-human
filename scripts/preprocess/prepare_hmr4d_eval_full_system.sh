#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASETS="${DATASETS:-emdb1 emdb2 rich 3dpw}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"
SAM2_FLAG="${SAM2_FLAG:---enable}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
MAX_HUMANS="${MAX_HUMANS:-20}"

for DATASET in ${DATASETS}; do
  EXTRACT_ARGS=(
    --dataset "${DATASET}"
    --path-config configs/path.yaml
    --max-sequences "${MAX_SEQUENCES}"
  )
  if [[ -n "${OVERWRITE_FLAG}" ]]; then
    EXTRACT_ARGS+=(--overwrite)
  fi
  python scripts/preprocess/extract_hmr4d_eval_frames.py "${EXTRACT_ARGS[@]}"

  TRACK_ARGS=(
    --dataset "${DATASET}"
    --path-config configs/path.yaml
    --max-sequences "${MAX_SEQUENCES}"
    --device "${DEVICE}"
    --detector-image-size 640
    --det-conf 0.25
    --det-iou 0.70
    --max-age 90
    --min-hits 1
    --aspect-ratio-thresh 10.0
    --stitch-max-gap 30
    --stitch-center-thresh 1.25
    --stitch-size-log-thresh 0.70
    --stitch-min-score 0.25
  )
  if [[ "${MAX_FRAMES}" != "0" ]]; then
    TRACK_ARGS+=(--max-frames "${MAX_FRAMES}")
  fi
  if [[ -n "${OVERWRITE_FLAG}" ]]; then
    TRACK_ARGS+=(--overwrite)
  fi
  if [[ "${SAM2_FLAG}" == "--disable" ]]; then
    TRACK_ARGS+=(--no-sam2-masks)
  fi
  python scripts/preprocess/prepare_hmr4d_eval_tracks.py "${TRACK_ARGS[@]}"

  CHECK_ARGS=(
    --dataset "${DATASET}"
    --path-config configs/path.yaml
    --sequence-length "${SEQUENCE_LENGTH}"
    --image-resolution "${IMAGE_RESOLUTION}"
    --max-humans "${MAX_HUMANS}"
  )
  if [[ -n "${IMAGE_SIZE:-}" ]]; then
    CHECK_ARGS+=(--image-size "${IMAGE_SIZE}")
  fi
  python scripts/diagnostics/check_hmr4d_eval_data_interface.py "${CHECK_ARGS[@]}"
done
