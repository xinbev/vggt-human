#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PHASE="${PHASE:-smoke}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_grounding_severe_float_gt.yaml}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
OVERFIT_SUBSET="${OVERFIT_SUBSET:-${SPLIT_ROOT}/overfit64_indices.csv}"
G0_METRICS="${G0_METRICS:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g0/g0_metrics.json}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g1_severe_float_smoke}"
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
    CHECK_MODE=severe_float_smoke
    ;;
  overfit)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g1_severe_float_overfit64}"
    MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1000}"
    MAX_VAL_STEPS="${MAX_VAL_STEPS:-8}"
    SUBSET_REPEAT="${SUBSET_REPEAT:-400}"
    CHECK_MODE=severe_float_overfit
    python "${REPO_ROOT}/scripts/smoke/check_hsi_grounding_metrics.py" \
      --output-dir "${SMOKE_OUTPUT_DIR}" \
      --mode severe_float_smoke
    ;;
  *) echo "[ERROR] PHASE must be smoke or overfit" >&2; exit 1 ;;
esac

for path in "${TRAIN_CONFIG}" "${TRAIN_SEQUENCE_MANIFEST}" "${OVERFIT_SUBSET}" "${G0_METRICS}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
done
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || {
  echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2
  exit 1
}
if [[ "${ALLOW_EXISTING_OUTPUT}" != "true" && -e "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
  echo "[ERROR] Refusing to overwrite existing run: ${OUTPUT_DIR}" >&2
  echo "[ERROR] Set a new OUTPUT_DIR, or explicitly set ALLOW_EXISTING_OUTPUT=true." >&2
  exit 1
fi

cd "${REPO_ROOT}"
python scripts/smoke/check_hsi_grounding_severe_float_config.py --config "${TRAIN_CONFIG}"
python scripts/smoke/check_hsi_grounding_g0.py --metrics "${G0_METRICS}"
mkdir -p "${OUTPUT_DIR}"

echo "========== HSI severe-float grounding Gate: ${PHASE} =========="
echo "Output          : ${OUTPUT_DIR}"
echo "Geometry        : GT depth + GT K + contact-root perturbed GT SMPL"
echo "Positive target : valid float candidate > 4 cm and better than GT base"
echo "Hard gate       : train/eval threshold 0.70"
echo "Loss            : grounding gate BCE only"
echo "Resume          : none (fresh Gate)"
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
  --override "data.val_sequence_manifest=${TRAIN_SEQUENCE_MANIFEST}" \
  --override "data.subset_indices_csv=${OVERFIT_SUBSET}" \
  --override "data.subset_repeat=${SUBSET_REPEAT}" \
  --override "data.subset_apply_to_val=true" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=1" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.max_val_steps=${MAX_VAL_STEPS}"

python scripts/smoke/check_hsi_grounding_metrics.py \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${CHECK_MODE}"

echo "========== ${PHASE} passed =========="
echo "Metrics: ${OUTPUT_DIR}/metrics_latest.json"
echo "Gate report: ${OUTPUT_DIR}/grounding_gate_${CHECK_MODE}.json"
echo "Top checkpoint: ${OUTPUT_DIR}/checkpoint_top01.pt"
