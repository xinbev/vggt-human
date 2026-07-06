#!/usr/bin/env bash

set -euo pipefail

# Generate real-data PLY layers for the HSI 24-anchor projection figure.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
IMAGE="${IMAGE:-${REPO_ROOT}/assets/image/f2/f2.jpg}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_hsi_after_translation_ray_refine.yaml}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_hsi_anchor_projection_ply_elements}"
CHECKPOINT="${CHECKPOINT:-}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
PERSON_INDEX="${PERSON_INDEX:--1}"
PERSON_SELECT="${PERSON_SELECT:-all}"
TOP_K="${TOP_K:-2}"
AUTO_TOP_K="${AUTO_TOP_K:-2}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
DETECTOR_IMAGE_SIZE="${DETECTOR_IMAGE_SIZE:-640}"
DEPTH_SOURCE="${DEPTH_SOURCE:-hsi}"
SMPL_STAGE="${SMPL_STAGE:-base}"
DEPTH_UPSAMPLE="${DEPTH_UPSAMPLE:-2}"
DEPTH_STRIDE="${DEPTH_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
ANCHOR_RADIUS_SCALE="${ANCHOR_RADIUS_SCALE:-0.018}"
PROJECTION_RADIUS_SCALE="${PROJECTION_RADIUS_SCALE:-0.0035}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI anchor projection PLY elements =========="
echo "Repo        : ${REPO_ROOT}"
echo "Image       : ${IMAGE}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Depth source: ${DEPTH_SOURCE}"
echo "SMPL stage  : ${SMPL_STAGE}"
echo "Output      : ${OUTPUT_DIR}"

args=(
  --image "${IMAGE}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --person-index "${PERSON_INDEX}"
  --person-select "${PERSON_SELECT}"
  --top-k "${TOP_K}"
  --auto-top-k "${AUTO_TOP_K}"
  --conf-threshold "${CONF_THRESHOLD}"
  --detector-image-size "${DETECTOR_IMAGE_SIZE}"
  --depth-source "${DEPTH_SOURCE}"
  --smpl-stage "${SMPL_STAGE}"
  --depth-upsample "${DEPTH_UPSAMPLE}"
  --depth-stride "${DEPTH_STRIDE}"
  --max-scene-depth "${MAX_SCENE_DEPTH}"
  --anchor-radius-scale "${ANCHOR_RADIUS_SCALE}"
  --projection-radius-scale "${PROJECTION_RADIUS_SCALE}"
  --auto-person-prior
)

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BASELINE_CHECKPOINT}" ]]; then
  args+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
fi

python scripts/vis/create_hsi_anchor_projection_ply_elements.py "${args[@]}"

echo "========== HSI anchor projection PLY elements finished =========="
echo "PLY root: ${OUTPUT_DIR}"
