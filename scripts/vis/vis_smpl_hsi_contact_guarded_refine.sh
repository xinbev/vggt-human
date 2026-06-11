#!/usr/bin/env bash

set -euo pipefail

# Visualize and diagnose the contact-guarded HSI refinement checkpoint.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

export TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_contact_guarded_refine.yaml}"
export SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q_contact_guarded_from_3d_guarded/checkpoint_latest.pt}"
export VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/smpl_hsi_contact_guarded_gt_prior_aligned}"
export DIAG_VIS_OUTPUT_DIR="${DIAG_VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_contact_guarded_depth_smpl_diagnostics}"
export DIAG_EVAL_OUTPUT_DIR="${DIAG_EVAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_contact_guarded_depth_smpl_diagnostics}"

cd "${REPO_ROOT}"
bash scripts/vis/vis_smpl_hsi_3d_guarded_refine.sh
