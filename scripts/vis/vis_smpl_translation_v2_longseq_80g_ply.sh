#!/usr/bin/env bash

set -euo pipefail

# Export PLY diagnostics for the SMPL Translation V2 80GB checkpoint.
# Colors in the comparison PLYs:
#   orange = old/base_pred_transl_cam
#   cyan   = ray/depth seed_pred_transl_cam
#   red    = final temporal pred_transl_cam
#   green  = BEDLAM GT SMPL/transl

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_v2_longseq.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_translation_v2_longseq_80g/stageC_temporal_27f_polish/checkpoint_latest.pt}"
PERSON_CSV="${PERSON_CSV:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/all_windows_27f/smpl_translation_person_metrics.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vis/smpl_translation_v2_longseq_80g_ply}"

IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000045/rgb/seq_000045_0025.png}"
TARGET_FRAME_OFFSET="${TARGET_FRAME_OFFSET:-4}"
NUM_FRAMES="${NUM_FRAMES:-27}"
STRIDE="${STRIDE:-1}"
MAX_PEOPLE="${MAX_PEOPLE:-3}"
QUERY_INDICES="${QUERY_INDICES:-}"
GT_INDICES="${GT_INDICES:-}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TOP_K="${TOP_K:-20}"
DRAW_SMPL_JOINTS="${DRAW_SMPL_JOINTS:-true}"
DRAW_GT_SMPL_JOINTS="${DRAW_GT_SMPL_JOINTS:-true}"

cd "${REPO_ROOT}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing image: ${IMAGE_PATH}" >&2; exit 1; }

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(basename "${IMAGE_PATH%.*}")_offset${TARGET_FRAME_OFFSET}}"
mkdir -p "${OUTPUT_DIR}"

DRAW_ARGS=()
if [[ "${DRAW_SMPL_JOINTS}" == "true" ]]; then
  DRAW_ARGS+=(--draw-smpl-joints)
fi
if [[ "${DRAW_GT_SMPL_JOINTS}" == "true" ]]; then
  DRAW_ARGS+=(--draw-gt-smpl-joints)
fi

MANUAL_ARGS=()
if [[ -n "${QUERY_INDICES}" ]]; then
  MANUAL_ARGS+=(--query-indices "${QUERY_INDICES}")
fi
if [[ -n "${GT_INDICES}" ]]; then
  MANUAL_ARGS+=(--gt-indices "${GT_INDICES}")
fi

echo "========== SMPL Translation V2 PLY export =========="
echo "Image      : ${IMAGE_PATH}"
echo "Offset     : ${TARGET_FRAME_OFFSET}"
echo "Frames     : ${NUM_FRAMES}"
echo "Checkpoint : ${SMPL_CKPT}"
echo "Person CSV : ${PERSON_CSV}"
echo "Output     : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_smpl_translation_v2_longseq_ply.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --stride "${STRIDE}" \
  --target-frame-offset "${TARGET_FRAME_OFFSET}" \
  --person-csv "${PERSON_CSV}" \
  --max-people "${MAX_PEOPLE}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --top-k "${TOP_K}" \
  "${DRAW_ARGS[@]}" \
  "${MANUAL_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.sequence_length=${NUM_FRAMES}" \
  --override "data.stride=${STRIDE}" \
  --override "data.max_humans=20" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=false" \
  --override "model.enable_hsi_refine=false" \
  --override "model.smpl_translation_output_mode=ray_offset_depth" \
  --override "model.smpl_enable_temporal_translation=true" \
  --override "model.smpl_temporal_translation_use_world=true" \
  --override "model.smpl_enable_translation_refine=false"

echo "========== SMPL Translation V2 PLY export finished =========="
echo "Output: ${OUTPUT_DIR}"
