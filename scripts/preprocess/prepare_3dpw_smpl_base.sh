#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --splits ${SPLITS:-train validation test}
  --min-keypoint-conf "${MIN_KEYPOINT_CONF:-0.50}"
  --min-valid-keypoints "${MIN_VALID_KEYPOINTS:-4}"
  --bbox-expand "${BBOX_EXPAND:-0.15}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${MAX_SEQUENCES:-}" ]]; then
  ARGS+=(--max-sequences "${MAX_SEQUENCES}")
fi
if [[ -n "${THREEDPW_ROOT:-}" ]]; then
  ARGS+=(--root "${THREEDPW_ROOT}")
fi
if [[ -n "${OUT_ROOT:-}" ]]; then
  ARGS+=(--output-root "${OUT_ROOT}")
fi

python scripts/preprocess/prepare_3dpw_smpl_base.py "${ARGS[@]}"
