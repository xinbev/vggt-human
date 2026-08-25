#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/xyb_space/bedlam/processed_bedlam/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree/checkpoint_top_train_epoch_0005_loss_total_0.010738.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_coarse_scale_cascade}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
cd "${REPO_ROOT}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames directory: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing HSI checkpoint: ${CHECKPOINT}" >&2; exit 1; }

python scripts/eval/evaluate_hsi_coarse_scale_cascade.py \
  --frames-dir "${FRAMES_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --path-config "${PATH_CONFIG:-configs/path.yaml}" \
  --train-config "${TRAIN_CONFIG:-configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-frames "${MAX_FRAMES:-32}" \
  --start-index "${START_INDEX:-0}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --max-humans "${MAX_HUMANS:-20}" \
  --conf-threshold "${CONF_THRESHOLD:-0.10}" \
  --scale-min "${SCALE_MIN:-0.10}" \
  --scale-max "${SCALE_MAX:-10.0}" \
  --anchor-stride "${ANCHOR_STRIDE:-8}" \
  --min-anchor-pixels "${MIN_ANCHOR_PIXELS:-32}"

echo "[ok] coarse-scale cascade summary: ${OUTPUT_DIR}/summary.json"
