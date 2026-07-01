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

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_base_hf_bedlam_ray_refine.yaml}"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${MAX_NPZ_FILES:-}" ]]; then
  ARGS+=(--override "data.max_npz_files=${MAX_NPZ_FILES}")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--override "data.max_frames=${MAX_FRAMES}")
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
if [[ -n "${PREFETCH_FACTOR:-}" ]]; then
  ARGS+=(--override "data.prefetch_factor=${PREFETCH_FACTOR}")
fi
if [[ -n "${PERSISTENT_WORKERS:-}" ]]; then
  ARGS+=(--override "data.persistent_workers=${PERSISTENT_WORKERS}")
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
if [[ -n "${SAVE_LATEST:-}" ]]; then
  ARGS+=(--override "checkpoint.save_latest=${SAVE_LATEST}")
fi
if [[ -n "${SAVE_FINAL:-}" ]]; then
  ARGS+=(--override "checkpoint.save_final=${SAVE_FINAL}")
fi

python scripts/train/train_smpl.py "${ARGS[@]}"
