#!/usr/bin/env bash

set -euo pipefail

# Generate real-data paper-figure elements for HSI local scene probing.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
IMAGE="${IMAGE:-${REPO_ROOT}/assets/image/f2/f2.jpg}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_hsi_after_translation_ray_refine.yaml}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_hsi_local_probe_real_elements}"
CHECKPOINT="${CHECKPOINT:-}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-1}"
AUTO_TOP_K="${AUTO_TOP_K:-2}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
ANCHOR_INDEX="${ANCHOR_INDEX:--1}"
PERSON_INDEX="${PERSON_INDEX:--1}"
PERSON_SELECT="${PERSON_SELECT:-rightmost}"
ANCHOR_MODE="${ANCHOR_MODE:-foot}"
DETECTOR_IMAGE_SIZE="${DETECTOR_IMAGE_SIZE:-640}"
PLY_SCENE_STRIDE="${PLY_SCENE_STRIDE:-4}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== Real HSI local probe paper elements =========="
echo "Image: ${IMAGE}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Output: ${OUTPUT_DIR}"

args=(
  --image "${IMAGE}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --top-k "${TOP_K}"
  --auto-top-k "${AUTO_TOP_K}"
  --conf-threshold "${CONF_THRESHOLD}"
  --anchor-index "${ANCHOR_INDEX}"
  --anchor-mode "${ANCHOR_MODE}"
  --person-index "${PERSON_INDEX}"
  --person-select "${PERSON_SELECT}"
  --detector-image-size "${DETECTOR_IMAGE_SIZE}"
  --ply-scene-stride "${PLY_SCENE_STRIDE}"
  --auto-person-prior
)

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BASELINE_CHECKPOINT}" ]]; then
  args+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
fi

python scripts/vis/create_hsi_local_probe_real_elements.py "${args[@]}"

echo "========== Real HSI local probe paper elements finished =========="
