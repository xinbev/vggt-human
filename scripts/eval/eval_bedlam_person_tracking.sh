#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SPLIT="${SPLIT:-Training}"
RUN_TRACKER="${RUN_TRACKER:-1}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-0}"
OUT_DIR="${OUT_DIR:-outputs/eval/bedlam_person_tracking}"

ARGS=(
  --split "${SPLIT}"
  --path-config configs/path.yaml
  --tracking-root outputs/preprocess/video_tracks
  --output-dir "${OUT_DIR}"
  --stitch-max-gap "${STITCH_MAX_GAP:-30}"
  --stitch-center-thresh "${STITCH_CENTER_THRESH:-1.25}"
  --stitch-size-log-thresh "${STITCH_SIZE_LOG_THRESH:-0.70}"
  --stitch-min-score "${STITCH_MIN_SCORE:-0.25}"
)

if [[ "${RUN_TRACKER}" == "1" ]]; then
  ARGS+=(--run-tracker --overwrite-tracks)
fi
if [[ "${MAX_SEQUENCES}" != "0" ]]; then
  ARGS+=(--max-sequences "${MAX_SEQUENCES}")
fi
if [[ "${MAX_FRAMES_PER_SEQUENCE}" != "0" ]]; then
  ARGS+=(--max-frames-per-sequence "${MAX_FRAMES_PER_SEQUENCE}")
fi

python scripts/eval/evaluate_bedlam_person_tracking.py "${ARGS[@]}" "$@"
