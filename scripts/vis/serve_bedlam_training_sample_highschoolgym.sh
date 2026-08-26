#!/usr/bin/env bash
set -euo pipefail

# This launcher is deliberately self-contained.  It invokes the companion
# viewer directly instead of delegating to another visualization script.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl.yaml}"
SEQUENCE_DIR="${SEQUENCE_DIR:-/home/zhw/xyb_space/bedlam/processed_bedlam/Training/20221013_3-10_500_batch01hand_static_highSchoolGym_seq_000000}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/bedlam_training_sample_highschoolgym}"
PORT="${PORT:-8091}"
WINDOW_INDEX="${WINDOW_INDEX:-0}"
FRAME_OFFSET="${FRAME_OFFSET:-0}"
DEVICE="${DEVICE:-cuda}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] Missing repository: ${REPO_ROOT}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing training config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${SEQUENCE_DIR}/rgb" ]] || { echo "[ERROR] Missing target sequence RGB directory: ${SEQUENCE_DIR}/rgb" >&2; exit 1; }

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== BEDLAM training-sample viewer =========="
echo "Repository   : ${REPO_ROOT}"
echo "Training cfg : ${TRAIN_CONFIG}"
echo "Sequence     : ${SEQUENCE_DIR}"
echo "Window index : ${WINDOW_INDEX}"
echo "Frame offset : ${FRAME_OFFSET} (0 selects the middle frame)"
echo "Output       : ${OUTPUT_DIR}"
echo "Viser port   : ${PORT}"

ARGS=(
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --sequence-dir "${SEQUENCE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --window-index "${WINDOW_INDEX}"
  --frame-offset "${FRAME_OFFSET}"
  --port "${PORT}"
  --device "${DEVICE}"
)

if [[ "${SMOKE_ONLY}" == "1" || "${SMOKE_ONLY}" == "true" || "${SMOKE_ONLY}" == "TRUE" ]]; then
  ARGS+=(--smoke-only)
fi

exec python3 scripts/vis/serve_bedlam_training_sample.py "${ARGS[@]}" "$@"
