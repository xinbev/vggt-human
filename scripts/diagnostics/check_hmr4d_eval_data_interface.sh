#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-emdb1}"

python scripts/diagnostics/check_hmr4d_eval_data_interface.py \
  --dataset "${DATASET}" \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --sequence-length "${SEQUENCE_LENGTH:-16}" \
  --stride "${STRIDE:-1}" \
  --image-size "${IMAGE_SIZE:-518}" \
  --max-humans "${MAX_HUMANS:-20}" \
  --batch-size "${BATCH_SIZE:-1}"
