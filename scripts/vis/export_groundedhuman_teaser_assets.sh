#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
args=(--config "${CONFIG:-configs/vis_groundedhuman_teaser_sofa.yaml}")
[[ -z "${IMAGE:-}" ]] || args+=(--image "${IMAGE}")
[[ -z "${OUTPUT_DIR:-}" ]] || args+=(--output-dir "${OUTPUT_DIR}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "${CHECKPOINT}")
[[ -z "${SCALE_CHECKPOINT:-}" ]] || args+=(--scale-checkpoint "${SCALE_CHECKPOINT}")
[[ -z "${DEVICE:-}" ]] || args+=(--device "${DEVICE}")
[[ -z "${PERSON_RANK:-}" ]] || args+=(--person-rank "${PERSON_RANK}")
[[ "${PREFLIGHT_ONLY:-false}" != true ]] || args+=(--preflight-only)
echo "[teaser] input default: ${REPO_ROOT}/assets/image/teaser/sofa.png"
echo "[teaser] outputs default: ${REPO_ROOT}/outputs/vis/groundedhuman_teaser/sofa"
"${PYTHON_BIN:-python}" scripts/vis/export_groundedhuman_teaser_assets.py "${args[@]}" "$@"
