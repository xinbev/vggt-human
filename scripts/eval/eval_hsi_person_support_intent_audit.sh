#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_person_support_intent_gt.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/train/hsi_stage3_person_support_intent_v3_joint5_full/checkpoint_top01.pt}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_stage3_person_support_intent_v3_threshold_audit}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-false}"

for path in "${TRAIN_CONFIG}" "${CHECKPOINT}" "${VAL_SEQUENCE_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || {
  echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2
  exit 1
}
if [[ "${ALLOW_EXISTING_OUTPUT}" != "true" && -e "${OUTPUT_DIR}/intent_audit.json" ]]; then
  echo "[ERROR] Refusing to overwrite existing audit: ${OUTPUT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI person-support threshold audit =========="
echo "Checkpoint       : ${CHECKPOINT}"
echo "Validation       : full manifest"
echo "Temporal context : required before applying support intent"
echo "Output           : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/eval_hsi_person_support_intent_audit.py \
  --path-config configs/path.yaml \
  --train-config "${TRAIN_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --max-batches "${MAX_BATCHES}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${BOXES_ROOT}" \
  --override "data.contact_teacher_root=${CONTACT_TEACHER_ROOT}" \
  --override "data.val_sequence_manifest=${VAL_SEQUENCE_MANIFEST}" \
  --override "data.subset_indices_csv=" \
  --override "data.subset_repeat=1" \
  --override "data.subset_apply_to_val=false" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "optim.batch_size=${BATCH_SIZE}"

echo "========== Intent audit finished =========="
echo "Summary          : ${OUTPUT_DIR}/intent_audit.json"
echo "Per-person rows  : ${OUTPUT_DIR}/intent_people.jsonl"
echo "High negatives   : ${OUTPUT_DIR}/highest_confidence_negatives.json"
echo "Low positives    : ${OUTPUT_DIR}/lowest_confidence_positives.json"
