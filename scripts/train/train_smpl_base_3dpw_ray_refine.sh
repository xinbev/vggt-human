#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_ray_refine.yaml}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  ARGS+=(--override "optim.epochs=${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--override "optim.batch_size=${BATCH_SIZE}")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  ARGS+=(--override "data.num_workers=${NUM_WORKERS}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  ARGS+=(--override "experiment.output_dir=${OUT_DIR}")
fi
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--override "checkpoint.resume=${RESUME}")
fi
if [[ -n "${RESUME_OPTIMIZER:-}" ]]; then
  ARGS+=(--override "checkpoint.resume_optimizer=${RESUME_OPTIMIZER}")
fi
if [[ -n "${RESET_EPOCH:-}" ]]; then
  ARGS+=(--override "checkpoint.reset_epoch=${RESET_EPOCH}")
fi
if [[ -n "${SAVE_TOP_K:-}" ]]; then
  ARGS+=(--override "checkpoint.save_top_k=${SAVE_TOP_K}")
fi
if [[ -n "${CHECKPOINT_MONITOR:-}" ]]; then
  ARGS+=(--override "checkpoint.monitor=${CHECKPOINT_MONITOR}")
fi
if [[ -n "${CHECKPOINT_MONITOR_MODE:-}" ]]; then
  ARGS+=(--override "checkpoint.monitor_mode=${CHECKPOINT_MONITOR_MODE}")
fi

python scripts/train/train_smpl.py "${ARGS[@]}"
