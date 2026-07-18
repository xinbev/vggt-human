#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PHASE="${PHASE:-gate500}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_grounding_real.yaml}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/val_sequences.txt}"
STAGE2_CKPT="${STAGE2_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-128}"

case "${PHASE}" in
  gate500)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_real_gate500}"
    GROUNDING_CKPT="${GROUNDING_CKPT:-${REPO_ROOT}/outputs/train/hsi_stage3_grounding_gt_full/checkpoint_top01.pt}"
    EPOCHS="${EPOCHS:-1}"
    LR="${LR:-2e-6}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-500}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-100}"
    ;;
  full)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/hsi_stage3_grounding_real_full}"
    GROUNDING_CKPT="${GROUNDING_CKPT:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_real_gate500/checkpoint_top01.pt}"
    EPOCHS="${EPOCHS:-3}"
    LR="${LR:-1e-6}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"
    ;;
  *) echo "[ERROR] PHASE must be gate500 or full" >&2; exit 1 ;;
esac

for path in "${TRAIN_CONFIG}" "${TRAIN_SEQUENCE_MANIFEST}" "${VAL_SEQUENCE_MANIFEST}" "${STAGE2_CKPT}" "${GROUNDING_CKPT}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || { echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "========== HSI real-inference grounding: ${PHASE} =========="
echo "Stage2 frozen ckpt : ${STAGE2_CKPT}"
echo "Grounding overlay  : ${GROUNDING_CKPT}"
echo "Geometry           : scaled VGGT depth + VGGT K + NLF SMPL"
echo "Supervision only   : GT SMPL/contact teacher"
echo "Trainable          : hsi_grounding_head only"
echo "Output             : ${OUTPUT_DIR}"
echo "GPU/batch          : ${CUDA_VISIBLE_DEVICES_VALUE} / ${BATCH_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config configs/path.yaml \
  --train-config "${TRAIN_CONFIG}" \
  --device cuda \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${BOXES_ROOT}" \
  --override "data.contact_teacher_root=${CONTACT_TEACHER_ROOT}" \
  --override "data.train_sequence_manifest=${TRAIN_SEQUENCE_MANIFEST}" \
  --override "data.val_sequence_manifest=${VAL_SEQUENCE_MANIFEST}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.nlf_internal_batch_size=${NLF_INTERNAL_BATCH_SIZE}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${STAGE2_CKPT}" \
  --override "checkpoint.overlay=${GROUNDING_CKPT}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.max_val_steps=${MAX_VAL_STEPS}"

python scripts/smoke/check_hsi_grounding_metrics.py --output-dir "${OUTPUT_DIR}" --mode real
echo "========== ${PHASE} passed =========="
echo "Combined checkpoint: ${OUTPUT_DIR}/checkpoint_top01.pt"
