#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --npz-root "${NPZ_ROOT:-/home/zhw/xyb_space/bedlam/all_npz_12_training}"
  --images-root "${IMAGES_ROOT:-/home/zhw/xyb_space/bedlam/hf_bedlam/training_images}"
  --output-dir "${OUT_DIR:-outputs/vis/hf_bedlam_box_samples}"
  --num-samples "${NUM_SAMPLES:-10}"
  --start-index "${START_INDEX:-0}"
)

if [[ -n "${NPZ_FILE:-}" ]]; then
  ARGS+=(--npz-file "${NPZ_FILE}")
fi

python scripts/vis/visualize_hf_bedlam_boxes.py "${ARGS[@]}"
