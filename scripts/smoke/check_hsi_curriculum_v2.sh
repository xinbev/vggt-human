#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
GATE_STAGE="${GATE_STAGE:-2A}"
GATE_MODE="${GATE_MODE:-smoke}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug/hsi_curriculum_v2_${GATE_STAGE}_${GATE_MODE}}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt}"

case "${GATE_STAGE}" in
  2A) METRIC_STAGE=stage2 ;;
  3A1) METRIC_STAGE=stage3 ;;
  *) echo "[ERROR] GATE_STAGE must be 2A or 3A1" >&2; exit 1 ;;
esac

if [[ "${GATE_MODE}" == "overfit" ]]; then
  MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1000}"
  MAX_VAL_STEPS="${MAX_VAL_STEPS:-8}"
  SUBSET_INDICES_CSV="${SUBSET_INDICES_CSV:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/overfit64_indices.csv}"
  SUBSET_REPEAT="${SUBSET_REPEAT:-400}"
  SUBSET_MAX_SAMPLES="${SUBSET_MAX_SAMPLES:-0}"
  SUBSET_APPLY_TO_VAL=true
  VAL_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST}"
  [[ -f "${SUBSET_INDICES_CSV}" ]] || { echo "[ERROR] Missing overfit subset: ${SUBSET_INDICES_CSV}" >&2; exit 1; }
elif [[ "${GATE_MODE}" == "distribution" ]]; then
  MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-500}"
  MAX_VAL_STEPS="${MAX_VAL_STEPS:-32}"
  SUBSET_INDICES_CSV=""
  SUBSET_REPEAT=1
  SUBSET_MAX_SAMPLES=0
  SUBSET_APPLY_TO_VAL=false
else
  MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-2}"
  MAX_VAL_STEPS="${MAX_VAL_STEPS:-2}"
  SUBSET_INDICES_CSV=""
  SUBSET_REPEAT=1
  SUBSET_MAX_SAMPLES=0
  SUBSET_APPLY_TO_VAL=false
fi

cd "${REPO_ROOT}"
RUN_STAGES="${GATE_STAGE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH}" \
MAX_VAL_STEPS="${MAX_VAL_STEPS}" \
SUBSET_INDICES_CSV="${SUBSET_INDICES_CSV}" \
SUBSET_REPEAT="${SUBSET_REPEAT}" \
SUBSET_MAX_SAMPLES="${SUBSET_MAX_SAMPLES}" \
SUBSET_APPLY_TO_VAL="${SUBSET_APPLY_TO_VAL}" \
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST}" \
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST}" \
EPOCHS_2A=1 EPOCHS_3A1=1 \
bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh

case "${GATE_STAGE}" in
  2A) STAGE_DIR="${OUTPUT_ROOT}/stage2a_gt_transl" ;;
  3A1) STAGE_DIR="${OUTPUT_ROOT}/stage3a1_root_contact" ;;
esac
python scripts/smoke/check_hsi_curriculum_metrics.py --output-dir "${STAGE_DIR}" --stage "${METRIC_STAGE}" --mode "${GATE_MODE}"
