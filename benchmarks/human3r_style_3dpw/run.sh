#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${TEMPORAL_CHECKPOINT:?Set V2 checkpoint path.}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --config "${CONFIG:-benchmarks/human3r_style_3dpw/config.yaml}"
  --threedpw-root "${THREEDPW_ROOT:-}"
  --temporal-checkpoint "${TEMPORAL_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR:-outputs/eval/human3r_style_3dpw}"
  --sequence-filter "${SEQUENCE_FILTER:-}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --max-frames "${MAX_FRAMES:-0}"
  --device "${DEVICE:-cuda:0}"
)
if [[ "${COMPONENT_DIAGNOSTICS:-false}" == "true" ]]; then ARGS+=(--component-diagnostics); fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0}" python benchmarks/human3r_style_3dpw/evaluate.py "${ARGS[@]}"
