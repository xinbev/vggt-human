#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

export TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
export SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_scene_then_momentum/stage2_human_momentum_no_worse/checkpoint_latest.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_scene_then_temporal_noworse_clip}"
export SCENE_AFFINE_MODE="${SCENE_AFFINE_MODE:-clip_median}"

exec bash "${REPO_ROOT}/scripts/vis/vis_smpl_hsi_scene_then_temporal_momentum_clip.sh"
