#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SPLIT="${SPLIT:-Training}"
START_INDEX="${START_INDEX:-0}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/preprocess/video_tracks}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"
SAM2_FLAG="${SAM2_FLAG:---enable}"

ARGS=(
  --path-config configs/path.yaml
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --max-sequences "${MAX_SEQUENCES}"
  --output-root "${OUT_ROOT}"
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
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${OVERWRITE_FLAG}" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SAM2_FLAG}" == "--disable" ]]; then
  ARGS+=(--no-sam2-masks)
fi
if [[ -n "${SEQUENCE:-}" ]]; then
  ARGS+=(--sequence "${SEQUENCE}")
fi

python scripts/preprocess/prepare_bedlam_full_system_tracks.py "${ARGS[@]}"
