#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEQ_DIR="${1:-/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/20241213_1_250_rome_tracking_seq_000002}"
shift $(( $# > 0 ? 1 : 0 )) || true

exec python3 "${SCRIPT_DIR}/serve_bedlam2_single_frame_viser.py" \
  --sequence-dir "${SEQ_DIR}" \
  --output-dir "${ROOT_DIR}/outputs/vis/bedlam2_single_frame_viser" \
  "$@"
