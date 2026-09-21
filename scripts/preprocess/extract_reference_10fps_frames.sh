#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VIDEO_PATH="${VIDEO_PATH:-/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/video/reference_10fps.mp4}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  VIDEO_PATH="$1"
  shift
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vis}"
SOURCE_NAME="${SOURCE_NAME:-reference_10fps_frames}"

bash scripts/preprocess/extract_video_frames.sh \
  "${VIDEO_PATH}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-name "${SOURCE_NAME}" \
  --target-fps 10 \
  "$@"
