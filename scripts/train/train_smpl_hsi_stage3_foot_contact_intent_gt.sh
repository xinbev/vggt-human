#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PHASE="${PHASE:-smoke}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_foot_contact_intent_gt.yaml}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/val_sequences.txt}"
OVERFIT_SUBSET="${OVERFIT_SUBSET:-${SPLIT_ROOT}/overfit64_indices.csv}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_contact_intent_ci_a_smoke}"
OVERFIT_OUTPUT_DIR="${OVERFIT_OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_contact_intent_ci_a_overfit64}"
GATE_OUTPUT_DIR="${GATE_OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_contact_intent_ci_a_gate500}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-false}"

case "${PHASE}" in
  smoke)
    OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-2}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-2}"
    SUBSET_REPEAT="${SUBSET_REPEAT:-2}"
    SUBSET_INDICES_CSV="${OVERFIT_SUBSET}"
    SUBSET_APPLY_TO_VAL=true
    ACTIVE_VAL_MANIFEST="${TRAIN_SEQUENCE_MANIFEST}"
    RESUME_CKPT=""
    LR="${LR:-5e-5}"
    CHECK_MODE=smoke
    ;;
  overfit)
    OUTPUT_DIR="${OUTPUT_DIR:-${OVERFIT_OUTPUT_DIR}}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1000}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-8}"
    SUBSET_REPEAT="${SUBSET_REPEAT:-400}"
    SUBSET_INDICES_CSV="${OVERFIT_SUBSET}"
    SUBSET_APPLY_TO_VAL=true
    ACTIVE_VAL_MANIFEST="${TRAIN_SEQUENCE_MANIFEST}"
    RESUME_CKPT=""
    LR="${LR:-5e-5}"
    CHECK_MODE=overfit
    python "${REPO_ROOT}/scripts/smoke/check_hsi_foot_contact_intent_metrics.py" \
      --output-dir "${SMOKE_OUTPUT_DIR}" --mode smoke
    ;;
  gate500)
    OUTPUT_DIR="${OUTPUT_DIR:-${GATE_OUTPUT_DIR}}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-500}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-100}"
    SUBSET_REPEAT=1
    SUBSET_INDICES_CSV=""
    SUBSET_APPLY_TO_VAL=false
    ACTIVE_VAL_MANIFEST="${VAL_SEQUENCE_MANIFEST}"
    RESUME_CKPT="${RESUME_CKPT:-${OVERFIT_OUTPUT_DIR}/checkpoint_top01.pt}"
    LR="${LR:-1e-5}"
    CHECK_MODE=distribution
    python "${REPO_ROOT}/scripts/smoke/check_hsi_foot_contact_intent_metrics.py" \
      --output-dir "${OVERFIT_OUTPUT_DIR}" --mode overfit
    ;;
  *)
    echo "[ERROR] PHASE must be smoke, overfit, or gate500" >&2
    exit 1
    ;;
esac

for path in "${TRAIN_CONFIG}" "${TRAIN_SEQUENCE_MANIFEST}" "${ACTIVE_VAL_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
if [[ -n "${SUBSET_INDICES_CSV}" ]]; then
  [[ -f "${SUBSET_INDICES_CSV}" ]] || { echo "[ERROR] Missing subset: ${SUBSET_INDICES_CSV}" >&2; exit 1; }
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  [[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing contact-intent checkpoint: ${RESUME_CKPT}" >&2; exit 1; }
fi
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || {
  echo "[ERROR] Missing V3 strict contact teachers: ${CONTACT_TEACHER_ROOT}" >&2
  exit 1
}
if [[ "${ALLOW_EXISTING_OUTPUT}" != "true" && -e "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
  echo "[ERROR] Refusing to overwrite existing run: ${OUTPUT_DIR}" >&2
  echo "[ERROR] Set a new OUTPUT_DIR; existing checkpoints are intentionally protected." >&2
  exit 1
fi

cd "${REPO_ROOT}"
python scripts/smoke/check_hsi_foot_contact_intent_config.py --config "${TRAIN_CONFIG}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI foot contact intent CI-A: ${PHASE} =========="
echo "Output          : ${OUTPUT_DIR}"
echo "Input           : clean GT SMPL pose + track-aligned 3-frame motion"
echo "Teacher         : existing V3 strict per-foot contact labels"
echo "Translation     : read for temporal differences only; never modified"
echo "Trainable       : hsi_foot_contact_intent_head only"
echo "Resume          : ${RESUME_CKPT:-none}"
echo "GPU/batch       : ${CUDA_VISIBLE_DEVICES_VALUE} / ${BATCH_SIZE}"
echo "Steps/val steps : ${MAX_STEPS_PER_EPOCH} / ${MAX_VAL_STEPS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config configs/path.yaml \
  --train-config "${TRAIN_CONFIG}" \
  --device cuda \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${BOXES_ROOT}" \
  --override "data.contact_teacher_root=${CONTACT_TEACHER_ROOT}" \
  --override "data.train_sequence_manifest=${TRAIN_SEQUENCE_MANIFEST}" \
  --override "data.val_sequence_manifest=${ACTIVE_VAL_MANIFEST}" \
  --override "data.subset_indices_csv=${SUBSET_INDICES_CSV}" \
  --override "data.subset_repeat=${SUBSET_REPEAT}" \
  --override "data.subset_apply_to_val=${SUBSET_APPLY_TO_VAL}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${RESUME_CKPT}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=1" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.max_val_steps=${MAX_VAL_STEPS}"

python scripts/smoke/check_hsi_foot_contact_intent_metrics.py \
  --output-dir "${OUTPUT_DIR}" --mode "${CHECK_MODE}"

echo "========== ${PHASE} passed =========="
echo "Metrics    : ${OUTPUT_DIR}/metrics_latest.json"
echo "Gate report: ${OUTPUT_DIR}/contact_intent_gate_${CHECK_MODE}.json"
echo "Checkpoint : ${OUTPUT_DIR}/checkpoint_top01.pt"
