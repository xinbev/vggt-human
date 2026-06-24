#!/usr/bin/env bash

set -euo pipefail

# Diagnose whether the current top bad Translation V2 frames are true
# translation failures or query/GT binding failures.  It runs the same model
# twice on the same top-10 frame list:
#   1. hungarian: current DETR-style box/conf matching
#   2. slot:      force q_i to match gt_i, useful when GT boxes drive queries

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

DEDUP_FRAME_CSV="${DEDUP_FRAME_CSV:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_translation_summary.csv}"
DEDUP_PERSON_CSV="${DEDUP_PERSON_CSV:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_person_translation_metrics.csv}"
BAD_TOP10_DIR="${BAD_TOP10_DIR:-${REPO_ROOT}/outputs/vis/smpl_translation_v2_bad_input_images_top10}"
BAD_TOP10_CSV="${BAD_TOP10_CSV:-${BAD_TOP10_DIR}/bad_translation_top10.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/top10_match_diagnostics}"

NUM_FRAMES="${NUM_FRAMES:-27}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-Training}"
TOP_K="${TOP_K:-10}"
TOP_WORST="${TOP_WORST:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"

cd "${REPO_ROOT}"
mkdir -p "${BAD_TOP10_DIR}" "${OUTPUT_ROOT}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

if [[ ! -f "${BAD_TOP10_CSV}" ]]; then
  [[ -f "${DEDUP_FRAME_CSV}" ]] || { echo "[ERROR] Missing dedup frame CSV: ${DEDUP_FRAME_CSV}" >&2; exit 1; }
  [[ -f "${DEDUP_PERSON_CSV}" ]] || { echo "[ERROR] Missing dedup person CSV: ${DEDUP_PERSON_CSV}" >&2; exit 1; }
  echo "========== Build top-${TOP_K} bad-frame list =========="
  python scripts/vis/collect_bad_translation_input_images.py \
    --frame-csv "${DEDUP_FRAME_CSV}" \
    --person-csv "${DEDUP_PERSON_CSV}" \
    --path-config "${PATH_CONFIG}" \
    --bedlam-root "${BEDLAM_ROOT}" \
    --split "${SPLIT}" \
    --output-dir "${BAD_TOP10_DIR}" \
    --top-k "${TOP_K}" \
    --sort-key refined_max_transl_l2_m \
    --no-copy-images \
    --no-contact-sheet
fi

[[ -f "${BAD_TOP10_CSV}" ]] || { echo "[ERROR] Missing top-10 CSV: ${BAD_TOP10_CSV}" >&2; exit 1; }

COMMON_ARGS=(
  --checkpoint "${SMPL_CKPT}"
  --baseline-checkpoint "${VGGT_CKPT}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --split "${SPLIT}"
  --max-samples 0
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --top-worst "${TOP_WORST}"
  --log-interval "${LOG_INTERVAL}"
  --use-gt-box-prior
  --subset-frame-csv "${BAD_TOP10_CSV}"
  --subset-sequence-column sequence_name
  --subset-frame-column frame_name
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}"
  --override "datasets.bedlam_root=${BEDLAM_ROOT}"
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
  --override "data.sequence_length=${NUM_FRAMES}"
  --override "data.stride=${FRAME_STRIDE}"
  --override "data.max_humans=${MAX_HUMANS}"
  --override "data.require_boxes=true"
  --override "data.require_smpl=true"
  --override "data.require_depth=false"
  --override "model.enable_camera=true"
  --override "model.enable_depth=false"
  --override "model.enable_hsi_refine=false"
  --override "model.smpl_translation_output_mode=ray_offset_depth"
  --override "model.smpl_enable_temporal_translation=true"
  --override "model.smpl_temporal_translation_use_world=true"
  --override "model.smpl_enable_translation_refine=false"
)

echo "========== SMPL Translation V2 top-10 match diagnostics =========="
echo "Checkpoint : ${SMPL_CKPT}"
echo "Top10 CSV  : ${BAD_TOP10_CSV}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Frames     : ${NUM_FRAMES}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES_VALUE}"

echo "========== Match mode: hungarian =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
  "${COMMON_ARGS[@]}" \
  --match-mode hungarian \
  --output-dir "${OUTPUT_ROOT}/hungarian_${NUM_FRAMES}f"

echo "========== Match mode: slot =========="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
  "${COMMON_ARGS[@]}" \
  --match-mode slot \
  --output-dir "${OUTPUT_ROOT}/slot_${NUM_FRAMES}f"

echo "========== Top-10 match diagnostics finished =========="
echo "Hungarian: ${OUTPUT_ROOT}/hungarian_${NUM_FRAMES}f/smpl_translation_person_metrics.csv"
echo "Slot     : ${OUTPUT_ROOT}/slot_${NUM_FRAMES}f/smpl_translation_person_metrics.csv"
