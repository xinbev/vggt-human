#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/emdb2_s7_nlf_gt_camera}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${OUTPUT_ROOT}/predictions}"
METRICS_ROOT="${METRICS_ROOT:-${OUTPUT_ROOT}/metrics}"

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

COMMON_ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --subsample-stride 7
  --sequence-filter "${SEQUENCE_FILTER:-}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --device "${DEVICE:-cuda:0}"
)
if [[ -n "${EMDB_ROOT:-}" ]]; then COMMON_ARGS+=(--emdb-root "${EMDB_ROOT}"); fi

echo "========== EMDB-2-S7 NLF + GT camera oracle =========="
echo "Camera intrinsics : EMDB GT K, transformed into the processed image plane"
echo "Camera extrinsics : EMDB GT T_w2c (inverted to T_c2w)"
echo "Human prediction  : NLF pose, shape, and metric camera translation"
echo "Output            : ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

python benchmarks/emdb2_global/export_gt_camera_nlf.py \
  "${COMMON_ARGS[@]}" \
  --inference-config "${INFERENCE_CONFIG:-benchmarks/emdb2_global/inference_config.yaml}" \
  --output-dir "${PREDICTIONS_ROOT}" \
  --max-input-frames "${MAX_INPUT_FRAMES:-500}" \
  --max-humans "${MAX_HUMANS:-8}" \
  --conf-threshold "${CONF_THRESHOLD:-0.05}" \
  --match-iou-threshold "${MATCH_IOU_THRESHOLD:-0.05}"

python benchmarks/emdb2_global/evaluate_gt_camera_nlf.py \
  "${COMMON_ARGS[@]}" \
  --config "${CONFIG:-benchmarks/emdb2_global/config.yaml}" \
  --predictions-root "${PREDICTIONS_ROOT}" \
  --output-dir "${METRICS_ROOT}" \
  --chunk-length 14 \
  --root-index "${ROOT_INDEX:-0}" \
  --smpl-batch-size "${SMPL_BATCH_SIZE:-512}"

echo "Summary: ${METRICS_ROOT}/summary.json"
