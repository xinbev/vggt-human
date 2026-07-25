#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_grounding_severe_float_gt.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g2_severe_float_gate500/checkpoint_top01.pt}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_gate_audit_g2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
MAX_BATCHES="${MAX_BATCHES:-100}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-false}"

for path in "${TRAIN_CONFIG}" "${CHECKPOINT}" "${VAL_SEQUENCE_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || {
  echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2
  exit 1
}
if [[ "${ALLOW_EXISTING_OUTPUT}" != "true" && -e "${OUTPUT_DIR}/gate_audit.json" ]]; then
  echo "[ERROR] Refusing to overwrite existing audit: ${OUTPUT_DIR}" >&2
  echo "[ERROR] Set a new OUTPUT_DIR, or explicitly set ALLOW_EXISTING_OUTPUT=true." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI Grounding Gate person audit =========="
echo "Checkpoint       : ${CHECKPOINT}"
echo "Validation       : ${VAL_SEQUENCE_MANIFEST}"
echo "Batches / batch  : ${MAX_BATCHES} / ${BATCH_SIZE}"
echo "Threshold sweep  : 0.50 to 0.95"
echo "Output           : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/eval_hsi_grounding_gate_audit.py \
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

echo "========== Gate audit finished =========="
echo "Summary          : ${OUTPUT_DIR}/gate_audit.json"
echo "Per-person rows  : ${OUTPUT_DIR}/gate_people.jsonl"
echo "False positives  : ${OUTPUT_DIR}/false_positives_top.json"
echo "False negatives  : ${OUTPUT_DIR}/false_negatives_top.json"
