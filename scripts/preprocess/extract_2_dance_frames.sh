#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# The default path is the location on the Linux server described in AGENTS.md.
# Override it with the first positional argument when processing another copy.
VIDEO_PATH="${VIDEO_PATH:-/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/video/2_dance.mp4}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  VIDEO_PATH="$1"
  shift
fi

# Frames are written to outputs/vis/2_dance_frames/ by default.  The generic
# extractor also writes manifest.json with video metadata and frame count.
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vis}"
SOURCE_NAME="${SOURCE_NAME:-2_dance_frames}"

bash scripts/preprocess/extract_video_frames.sh \
  "${VIDEO_PATH}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-name "${SOURCE_NAME}" \
  "$@"
