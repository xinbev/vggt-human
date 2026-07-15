#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl}"
FRAMES_DIR="${FRAMES_DIR:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb}"
QUERY_SOURCE="${QUERY_SOURCE:-bedlam_sidecar}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_vggt_sequence_viewer}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-32}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
POINT_SIZE="${POINT_SIZE:-0.012}"
CAMERA_FRUSTUM_SCALE="${CAMERA_FRUSTUM_SCALE:-0.20}"
ALIGNMENT_VERTEX_STRIDE="${ALIGNMENT_VERTEX_STRIDE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames dir: ${FRAMES_DIR}" >&2; exit 1; }
[[ -d "${STAGE2_DIR}" ]] || { echo "[ERROR] Missing stage2 dir: ${STAGE2_DIR}" >&2; exit 1; }
if [[ "${QUERY_SOURCE}" == "bedlam_sidecar" ]]; then
  [[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
  [[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed sidecars: ${PREPROCESSED_ROOT}" >&2; exit 1; }
fi

echo "========== NLF-HSI VGGT sequence Viser viewer =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Query source: ${QUERY_SOURCE}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Sidecars    : ${PREPROCESSED_ROOT}"
echo "Stage2 dir  : ${STAGE2_DIR}"
echo "Checkpoint  : ${CHECKPOINT:-<rank1 from checkpoint_topk_index.json>}"
echo "Output      : ${OUTPUT_DIR}"
echo "Port        : ${PORT}"
echo "Max frames  : ${MAX_FRAMES}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Smoke only  : ${SMOKE_ONLY}"

ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --query-source "${QUERY_SOURCE}"
  --preprocessed-root "${PREPROCESSED_ROOT}"
  --bedlam-root "${BEDLAM_ROOT}"
  --stage2-dir "${STAGE2_DIR}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --port "${PORT}"
  --max-frames "${MAX_FRAMES}"
  --start-index "${START_INDEX}"
  --frame-stride "${FRAME_STRIDE}"
  --max-humans "${MAX_HUMANS}"
  --conf-threshold "${CONF_THRESHOLD}"
  --depth-point-stride "${DEPTH_POINT_STRIDE}"
  --max-scene-depth "${MAX_SCENE_DEPTH}"
  --point-size "${POINT_SIZE}"
  --camera-frustum-scale "${CAMERA_FRUSTUM_SCALE}"
  --alignment-vertex-stride "${ALIGNMENT_VERTEX_STRIDE}"
  --image-size "${IMAGE_SIZE}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BASELINE_CHECKPOINT}" ]]; then
  ARGS+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
fi
if [[ -n "${SMPL_MODEL_DIR}" ]]; then
  ARGS+=(--smpl-model-dir "${SMPL_MODEL_DIR}")
fi
if [[ "${SMOKE_ONLY}" == "1" || "${SMOKE_ONLY}" == "true" || "${SMOKE_ONLY}" == "TRUE" ]]; then
  ARGS+=(--smoke-only)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py "${ARGS[@]}"
