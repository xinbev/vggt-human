#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${DEVICE:-}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  DEVICE=cuda
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/a/checkpoint-containing-all-enabled-HSI-TRSTR-heads.pt" >&2
  exit 2
fi

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --model-config "${MODEL_CONFIG:-configs/infer_smpl_hsi_v3_trstr_spatial.yaml}"
  --eval-config "${EVAL_CONFIG:-configs/eval_show_human_scene_3dpw.yaml}"
  --output-dir "${OUTPUT_DIR:-outputs/eval/show_human_scene_3dpw}"
  --split "${SPLIT:-test}"
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
  --start-index "${START_INDEX:-0}"
  --max-samples "${MAX_SAMPLES:-0}"
  --log-interval "${LOG_INTERVAL:-10}"
)

if [[ -n "${DEVICE:-}" ]]; then ARGS+=(--device "${DEVICE}"); fi
if [[ -n "${SUBSET_INDICES_CSV:-}" ]]; then ARGS+=(--subset-indices-csv "${SUBSET_INDICES_CSV}"); fi
if [[ -n "${SUBSET_INDEX_COLUMN:-}" ]]; then ARGS+=(--subset-index-column "${SUBSET_INDEX_COLUMN}"); fi
if [[ -n "${SAM2_PATCH_MASKS_ROOT:-}" ]]; then
  ARGS+=(--override "data.sam2_patch_masks_root=${SAM2_PATCH_MASKS_ROOT}")
fi
if [[ -n "${VISIBILITY_BACKEND:-}" ]]; then
  ARGS+=(--override "human_scene_evaluation.visibility_backend=${VISIBILITY_BACKEND}")
fi
if [[ -n "${BRANCHES:-}" ]]; then
  ARGS+=(--override "human_scene_evaluation.branches=${BRANCHES}")
fi

python scripts/eval/evaluate_show_human_scene_3dpw.py "${ARGS[@]}"
