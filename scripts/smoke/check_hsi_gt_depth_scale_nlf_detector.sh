#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

python scripts/diagnostics/check_nlf_runtime_requirements.py \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --output-dir "${OUTPUT_DIR:-outputs/debug/hsi_gt_depth_scale_nlf_detector_smoke}" \
  --require-detector
