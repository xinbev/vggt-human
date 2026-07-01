#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python scripts/diagnostics/inspect_hf_bedlam_npz.py \
  --npz-root "${NPZ_ROOT:-/home/zhw/xyb_space/bedlam/all_npz_12_training}" \
  --file "${NPZ_FILE:-}" \
  --max-files "${MAX_FILES:-1}" \
  --preview-keys "${PREVIEW_KEYS:-16}"
