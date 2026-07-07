#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VIDEO="${1:?Usage: $0 <video> [extra args...]}"
shift || true

python scripts/preprocess/extract_video_frames.py \
  "${VIDEO}" \
  --output-dir "${OUTPUT_DIR:-outputs/preprocess/video_frames}" \
  "$@"
