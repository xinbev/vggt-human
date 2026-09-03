#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${PREDICTIONS_ROOT:?Set PREDICTIONS_ROOT to EMDB-2 sequence NPZ archives}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --config "${CONFIG:-benchmarks/emdb2_global/config.yaml}"
  --predictions-root "${PREDICTIONS_ROOT}"
  --output-dir "${OUTPUT_DIR:-outputs/eval/emdb2_global}"
  --device "${DEVICE:-cuda:0}"
  --chunk-length "${CHUNK_LENGTH:-0}"
  --subsample-stride "${SUBSAMPLE_STRIDE:-1}"
  --root-index "${ROOT_INDEX:-0}"
  --smpl-batch-size "${SMPL_BATCH_SIZE:-512}"
  --sequence-filter "${SEQUENCE_FILTER:-}"
  --max-sequences "${MAX_SEQUENCES:-0}"
)
if [[ -n "${EMDB_ROOT:-}" ]]; then ARGS+=(--emdb-root "${EMDB_ROOT}"); fi
if [[ "${REQUIRE_ALL_SEQUENCES:-true}" == "true" ]]; then
  ARGS+=(--require-all-sequences)
else
  ARGS+=(--no-require-all-sequences)
fi
if [[ "${METRICS_ONLY_OUTPUT:-false}" == "true" ]]; then
  ARGS+=(--metrics-only-output)
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

python benchmarks/emdb2_global/evaluate.py "${ARGS[@]}"
