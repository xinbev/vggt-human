#!/usr/bin/env bash

set -euo pipefail

# Focused bad-frame PLY export for the hard-tail repaired translation refiner.
# This delegates to the existing after-translation-ray-refine visualizer but
# points it at the merged full-HSI tail-hard checkpoint and matching refiner
# output ranges.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_translation_ray_refine_tail_hard_from_after_hsi/merged_hsi_tail_translation/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_bad_frame_tail_hard_after_translation_ray_refine_ply}"

SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M:-1.20}"
SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M:-0.60}"
SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA:-0.85}"
SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT:-1.00}"
EXPORT_PRE_REFINE_COMPARISON="${EXPORT_PRE_REFINE_COMPARISON:-true}"
EXPORT_TRANSLATION_DEBUG_JSON="${EXPORT_TRANSLATION_DEBUG_JSON:-true}"

export REPO_ROOT
export SMPL_CKPT
export OUTPUT_DIR
export SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M
export SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M
export SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA
export SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT
export EXPORT_PRE_REFINE_COMPARISON
export EXPORT_TRANSLATION_DEBUG_JSON

bash "${REPO_ROOT}/scripts/vis/vis_hsi_bad_frame_ply_after_translation_ray_refine.sh"
