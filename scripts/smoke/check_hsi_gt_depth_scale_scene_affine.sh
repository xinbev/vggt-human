#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python scripts/smoke/check_hsi_gt_depth_scale_scene_affine.py
python scripts/smoke/check_hsi_gt_depth_scale_boxfree_data.py \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml}" \
  --num-batches "${NUM_DATA_SMOKE_BATCHES:-8}"
