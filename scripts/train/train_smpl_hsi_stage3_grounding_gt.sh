#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PHASE="${PHASE:-overfit}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_grounding_gt.yaml}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/val_sequences.txt}"
OVERFIT_SUBSET="${OVERFIT_SUBSET:-${SPLIT_ROOT}/overfit64_indices.csv}"
G0_METRICS="${G0_METRICS:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g0/g0_metrics.json}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"

case "${PHASE}" in
  overfit)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g1_overfit64}"
    EPOCHS="${EPOCHS:-1}"
    LR="${LR:-1e-5}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1000}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-8}"
    SUBSET_INDICES_CSV="${OVERFIT_SUBSET}"
    SUBSET_REPEAT="${SUBSET_REPEAT:-400}"
    SUBSET_APPLY_TO_VAL=true
    VAL_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST}"
    RESUME_CKPT="${RESUME_CKPT:-}"
    CHECK_MODE=overfit
    ;;
  gate500)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g2_gate500}"
    EPOCHS="${EPOCHS:-1}"
    LR="${LR:-5e-6}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-500}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-100}"
    SUBSET_INDICES_CSV=""
    SUBSET_REPEAT=1
    SUBSET_APPLY_TO_VAL=false
    RESUME_CKPT="${RESUME_CKPT:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g1_overfit64/checkpoint_top01.pt}"
    CHECK_MODE=distribution
    ;;
  full)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/hsi_stage3_grounding_gt_full}"
    EPOCHS="${EPOCHS:-3}"
    LR="${LR:-2e-6}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"
    SUBSET_INDICES_CSV=""
    SUBSET_REPEAT=1
    SUBSET_APPLY_TO_VAL=false
    RESUME_CKPT="${RESUME_CKPT:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g2_gate500/checkpoint_top01.pt}"
    CHECK_MODE=distribution
    ;;
  *) echo "[ERROR] PHASE must be overfit, gate500, or full" >&2; exit 1 ;;
esac

for path in "${TRAIN_CONFIG}" "${TRAIN_SEQUENCE_MANIFEST}" "${VAL_SEQUENCE_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || { echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2; exit 1; }
if [[ -n "${SUBSET_INDICES_CSV}" ]]; then
  [[ -f "${SUBSET_INDICES_CSV}" ]] || { echo "[ERROR] Missing subset: ${SUBSET_INDICES_CSV}" >&2; exit 1; }
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  [[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing grounding checkpoint: ${RESUME_CKPT}" >&2; exit 1; }
fi
if [[ "${PHASE}" == "overfit" ]]; then
  python "${REPO_ROOT}/scripts/smoke/check_hsi_grounding_g0.py" --metrics "${G0_METRICS}"
fi
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "========== HSI geometry-first grounding: ${PHASE} =========="
echo "Output          : ${OUTPUT_DIR}"
echo "Geometry        : GT depth + GT K + contact-root perturbed GT SMPL"
echo "Trainable       : hsi_grounding_head only"
echo "Resume grounding: ${RESUME_CKPT:-none}"
echo "GPU/batch       : ${CUDA_VISIBLE_DEVICES_VALUE} / ${BATCH_SIZE}"
echo "Epochs/max step : ${EPOCHS} / ${MAX_STEPS_PER_EPOCH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config configs/path.yaml \
  --train-config "${TRAIN_CONFIG}" \
  --device cuda \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${BOXES_ROOT}" \
  --override "data.contact_teacher_root=${CONTACT_TEACHER_ROOT}" \
  --override "data.train_sequence_manifest=${TRAIN_SEQUENCE_MANIFEST}" \
  --override "data.val_sequence_manifest=${VAL_SEQUENCE_MANIFEST}" \
  --override "data.subset_indices_csv=${SUBSET_INDICES_CSV}" \
  --override "data.subset_repeat=${SUBSET_REPEAT}" \
  --override "data.subset_apply_to_val=${SUBSET_APPLY_TO_VAL}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${RESUME_CKPT}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.max_val_steps=${MAX_VAL_STEPS}"

python scripts/smoke/check_hsi_grounding_metrics.py --output-dir "${OUTPUT_DIR}" --mode "${CHECK_MODE}"
echo "========== ${PHASE} passed =========="
echo "Top checkpoint: ${OUTPUT_DIR}/checkpoint_top01.pt"
