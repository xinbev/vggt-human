#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

VIDEO_OR_FRAMES="${1:?Usage: $0 <video_or_frames_dir> [extra args...]}"
shift || true

if [[ -d "${VIDEO_OR_FRAMES}" ]]; then
  SOURCE_ARGS=(--frames-dir "${VIDEO_OR_FRAMES}")
else
  SOURCE_ARGS=(--video "${VIDEO_OR_FRAMES}")
fi

python scripts/preprocess/prepare_video_person_tracks.py \
  "${SOURCE_ARGS[@]}" \
  --path-config configs/path.yaml \
  --output-root outputs/preprocess/video_tracks \
  --overwrite \
  --detector-image-size 640 \
  --det-conf 0.25 \
  --det-iou 0.70 \
  --max-age 90 \
  --min-hits 1 \
  --aspect-ratio-thresh 10.0 \
  "$@"
