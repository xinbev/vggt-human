#!/usr/bin/env bash

set -euo pipefail

# Stage A reuses or trains the clip scene affine checkpoint.
# Stage B trains human temporal momentum with temporal no-worse protection.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

export STAGE2_CONFIG="${STAGE2_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_scene_then_momentum}"
export STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2_human_momentum_no_worse}"
export STAGE1_EXTRA_EPOCHS="${STAGE1_EXTRA_EPOCHS:-0}"
export SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
export SCENE_AFFINE_EMA_ALPHA="${SCENE_AFFINE_EMA_ALPHA:-0.25}"
export TEMPORAL_NO_WORSE_WEIGHT="${TEMPORAL_NO_WORSE_WEIGHT:-20.0}"
export TEMPORAL_NO_WORSE_MARGIN_M="${TEMPORAL_NO_WORSE_MARGIN_M:-0.002}"
export TEMPORAL_NO_WORSE_ACCEL_MARGIN_M="${TEMPORAL_NO_WORSE_ACCEL_MARGIN_M:-0.003}"

exec bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_scene_then_temporal_momentum_from0121.sh"
