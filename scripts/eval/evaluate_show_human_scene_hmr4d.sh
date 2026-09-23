#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/stage2-human-scene-align-checkpoint.pt" >&2
  exit 2
fi
if [[ -z "${SCALE_CHECKPOINT:-}" ]]; then
  echo "Please set SCALE_CHECKPOINT=/path/to/coarse-residual-scale-checkpoint.pt" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
if [[ "${DEVICE:-}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  DEVICE=cuda
fi

DATASET="${DATASET:-emdb1}"
ARGS=(
  --dataset "${DATASET}"
  --checkpoint "${CHECKPOINT}"
  --scale-checkpoint "${SCALE_CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --model-config "${MODEL_CONFIG:-configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml}"
  --eval-config "${EVAL_CONFIG:-configs/eval_show_human_scene_hmr4d.yaml}"
  --output-dir "${OUTPUT_DIR:-outputs/eval/show_human_scene_${DATASET}}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
  --sequence-length "${WINDOW_SIZE:-${SEQUENCE_LENGTH:-100}}"
  --max-humans "${MAX_HUMANS:-8}"
  --stride "${STRIDE:-1}"
  --max-windows "${MAX_WINDOWS:-0}"
  --log-interval "${LOG_INTERVAL:-10}"
)

if [[ -n "${DEVICE:-}" ]]; then ARGS+=(--device "${DEVICE}"); fi
if [[ -n "${SUPPORT_ROOT:-}" ]]; then ARGS+=(--support-root "${SUPPORT_ROOT}"); fi
if [[ -n "${FRAMES_ROOT:-}" ]]; then ARGS+=(--frames-root "${FRAMES_ROOT}"); fi
if [[ -n "${VISIBILITY_BACKEND:-}" ]]; then
  ARGS+=(--override "human_scene_evaluation.visibility_backend=${VISIBILITY_BACKEND}")
fi

python scripts/eval/evaluate_show_human_scene_hmr4d.py "${ARGS[@]}"
