#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/stage2-checkpoint.pt}"
: "${SCALE_CHECKPOINT:?Set SCALE_CHECKPOINT=/path/to/coarse-scale-checkpoint.pt}"

ARGS=(
  --dataset emdb1
  --checkpoint "${CHECKPOINT}"
  --scale-checkpoint "${SCALE_CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --model-config "${MODEL_CONFIG:-configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml}"
  --eval-config "${EVAL_CONFIG:-configs/eval_show_human_scene_hmr4d.yaml}"
  --output-dir "${OUTPUT_DIR:-outputs/eval/show_emdb1_manual_scale}"
  --sequence-length "${WINDOW_SIZE:-200}"
  --max-humans "${MAX_HUMANS:-8}"
  --target-min-iou "${TARGET_MIN_IOU:-0.30}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --num-workers "${NUM_WORKERS:-2}"
  --support-root "${SUPPORT_ROOT:-}"
  --frames-root "${FRAMES_ROOT:-}"
  --device "${DEVICE:-cuda}"
)

python scripts/eval/prepare_3dpw_show_manual_scale_cache.py "${ARGS[@]}"
