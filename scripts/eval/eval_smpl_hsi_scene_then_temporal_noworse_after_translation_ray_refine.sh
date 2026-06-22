#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

export TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
export SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_temporal_after_translation_ray_refine_noworse}"
export SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"
export SMPL_ENABLE_TRANSLATION_REFINE="${SMPL_ENABLE_TRANSLATION_REFINE:-true}"
export SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M:-1.20}"
export SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M:-0.60}"
export SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA:-0.85}"
export SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT:-1.00}"

exec bash "${REPO_ROOT}/scripts/eval/eval_smpl_hsi_scene_then_temporal_momentum.sh"
