#!/usr/bin/env bash

set -euo pipefail

# Main-line temporal continuation after the validated base SMPL camera-ray
# translation repair. This starts from the HSI reconnect checkpoint rather than
# the older 0121 HSI baseline.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

export INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt}"
export STAGE1_TEACHER_CKPT="${STAGE1_TEACHER_CKPT:-${INIT_CKPT}}"
export STAGE2_CONFIG="${STAGE2_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine}"
export STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1_scene_affine}"
export STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2_human_momentum_no_worse}"

export STAGE1_EXTRA_EPOCHS="${STAGE1_EXTRA_EPOCHS:-3}"
export STAGE2_EXTRA_EPOCHS="${STAGE2_EXTRA_EPOCHS:-4}"
export SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
export SCENE_AFFINE_EMA_ALPHA="${SCENE_AFFINE_EMA_ALPHA:-0.25}"
export SMPL_ENABLE_TRANSLATION_REFINE="${SMPL_ENABLE_TRANSLATION_REFINE:-true}"
export SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M:-1.20}"
export SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M:-0.60}"
export SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA:-0.85}"
export SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT:-1.00}"
export TEMPORAL_NO_WORSE_WEIGHT="${TEMPORAL_NO_WORSE_WEIGHT:-20.0}"
export TEMPORAL_NO_WORSE_MARGIN_M="${TEMPORAL_NO_WORSE_MARGIN_M:-0.002}"
export TEMPORAL_NO_WORSE_ACCEL_MARGIN_M="${TEMPORAL_NO_WORSE_ACCEL_MARGIN_M:-0.003}"

exec bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_scene_then_temporal_momentum_from0121.sh"
