#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
SEQUENCE="${SEQUENCE:-20221013_3_250_batch01hand_orbit_bigOffice_seq_000000}"
FRAMES_DIR="${FRAMES_DIR:-${BEDLAM_ROOT}/Training/${SEQUENCE}/rgb}"
ID_CHECKPOINT="${ID_CHECKPOINT:-${REPO_ROOT}/outputs/train/nlf_roi_id_tracking_v2_gpu5/checkpoint_epoch_0014.pt}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_nlf_roi_id_tracking_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_roi_id_tracking_v2_gpu5/epoch14_${SEQUENCE}}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-5}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-32}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
ID_WEIGHT="${ID_WEIGHT:-0.10}"
MAX_ID_DISTANCE="${MAX_ID_DISTANCE:-2.0}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM box sidecars: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${ID_CHECKPOINT}" ]] || { echo "[ERROR] Missing ID checkpoint: ${ID_CHECKPOINT}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }

echo "========== NLF ROI ID tracking V2 Viser =========="
echo "GPU          : physical GPU ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Frames       : ${FRAMES_DIR}"
echo "ID checkpoint: ${ID_CHECKPOINT}"
echo "ID weight    : ${ID_WEIGHT}"
echo "Max ID dist  : ${MAX_ID_DISTANCE}"
echo "Max frames   : ${MAX_FRAMES}"
echo "Viewer       : http://127.0.0.1:${PORT}"
echo "Output       : ${OUTPUT_DIR}"

ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --id-checkpoint "${ID_CHECKPOINT}"
  --preprocessed-root "${PREPROCESSED_ROOT}"
  --bedlam-root "${BEDLAM_ROOT}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
  --port "${PORT}"
  --max-frames "${MAX_FRAMES}"
  --start-index "${START_INDEX}"
  --frame-stride "${FRAME_STRIDE}"
  --max-humans "${MAX_HUMANS}"
  --conf-threshold "${CONF_THRESHOLD}"
  --id-weight "${ID_WEIGHT}"
  --max-id-distance "${MAX_ID_DISTANCE}"
  --depth-point-stride "${DEPTH_POINT_STRIDE}"
  --max-scene-depth "${MAX_SCENE_DEPTH}"
)
if [[ "${SMOKE_ONLY}" == "1" || "${SMOKE_ONLY}" == "true" || "${SMOKE_ONLY}" == "TRUE" ]]; then
  ARGS+=(--smoke-only)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python -u scripts/vis/serve_nlf_roi_id_tracking_v2_viewer.py "${ARGS[@]}"
