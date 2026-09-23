#!/usr/bin/env bash
set -euo pipefail


CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/image/teaser/outputs/full_viewer_cache \
POINT_SIZE=0.006 \
FILTER_HUMAN_POINTS=false \
PORT="${PORT:-8087}" \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
