#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
IMAGE="${IMAGE:-${REPO_ROOT}/assets/image/f2/f2.jpg}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/infer_smpl_hsi_v3_trstr_spatial.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_refine/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_figure_assets/f2}"
DEVICE="${DEVICE:-cuda:0}"
TOP_K="${TOP_K:-2}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
TEMPORAL_FRAMES_DIR="${TEMPORAL_FRAMES_DIR:-${REPO_ROOT}/assets/image/f2}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${IMAGE}" ]]; then
  echo "Missing input image: ${IMAGE}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing HSI/TRSTR checkpoint: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT to the accepted server checkpoint." >&2
  exit 2
fi

args=(
  --image "${IMAGE}"
  --path-config "${PATH_CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --top-k "${TOP_K}"
  --conf-threshold "${CONF_THRESHOLD}"
  --temporal-frames-dir "${TEMPORAL_FRAMES_DIR}"
)

if [[ -n "${SCALE_CHECKPOINT}" ]]; then
  args+=(--scale-checkpoint "${SCALE_CHECKPOINT}")
fi

echo "========== Paper architecture real-asset export =========="
echo "Image      : ${IMAGE}"
echo "Config     : ${MODEL_CONFIG}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Output     : ${OUTPUT_DIR}"

python scripts/vis/export_paper_architecture_real_assets.py "${args[@]}"

echo "========== Export complete =========="
echo "Manifest: ${OUTPUT_DIR}/manifest.json"

