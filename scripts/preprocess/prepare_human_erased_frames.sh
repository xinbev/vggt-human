#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FRAMES_DIR="${1:?Usage: $0 <frames_dir> [extra args...]}"
shift || true

python scripts/preprocess/prepare_human_erased_frames.py \
  "${FRAMES_DIR}" \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --output-root "${OUTPUT_ROOT:-outputs/preprocess/human_erasure}" \
  "$@"
