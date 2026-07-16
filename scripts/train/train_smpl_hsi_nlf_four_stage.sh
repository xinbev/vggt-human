#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_HSI_FOUR_STAGE:-false}" != "true" ]]; then
  echo "[ERROR] This legacy pipeline still uses anchor_transl/contact_detail and is disabled by default." >&2
  echo "Use: bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh" >&2
  echo "Set ALLOW_LEGACY_HSI_FOUR_STAGE=true only to reproduce the historical baseline." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf}"
STAGE1_DIR="${STAGE1_DIR:-${PIPELINE_OUTPUT_ROOT}/stage1_gt_smpl_scale}"
STAGE2_DIR="${STAGE2_DIR:-${PIPELINE_OUTPUT_ROOT}/stage2_anchor_transl}"
STAGE3_DIR="${STAGE3_DIR:-${PIPELINE_OUTPUT_ROOT}/stage3_contact_detail}"
STAGE4_DIR="${STAGE4_DIR:-${PIPELINE_OUTPUT_ROOT}/stage4_temporal_tracks}"

RUN_STAGE1="${RUN_STAGE1:-true}"
RUN_STAGE2="${RUN_STAGE2:-true}"
RUN_STAGE3="${RUN_STAGE3:-true}"
RUN_STAGE4="${RUN_STAGE4:-true}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-8}"
STAGE4_EPOCHS="${STAGE4_EPOCHS:-8}"

STAGE1_LR="${STAGE1_LR:-5e-6}"
STAGE2_LR="${STAGE2_LR:-3e-6}"
STAGE3_LR="${STAGE3_LR:-2e-6}"
STAGE4_LR="${STAGE4_LR:-1e-6}"

STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-20}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-12}"
STAGE4_BATCH_SIZE="${STAGE4_BATCH_SIZE:-8}"

STAGE1_NUM_VIEWS="${STAGE1_NUM_VIEWS:-2}"
STAGE2_NUM_VIEWS="${STAGE2_NUM_VIEWS:-2}"
STAGE3_NUM_VIEWS="${STAGE3_NUM_VIEWS:-2}"
STAGE4_NUM_VIEWS="${STAGE4_NUM_VIEWS:-4}"

NUM_WORKERS="${NUM_WORKERS:-16}"
PIN_MEMORY="${PIN_MEMORY:-true}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

export REPO_ROOT BEDLAM_ROOT DATA_ROOT PREPROCESSED_ROOT PATH_CONFIG TRAIN_CONFIG CUDA_VISIBLE_DEVICES_VALUE
export NUM_WORKERS PIN_MEMORY MAX_HUMANS NLF_INTERNAL_BATCH_SIZE SAVE_TOP_K

stage_checkpoint() {
  local stage_dir="$1"
  echo "${stage_dir}/checkpoint_latest.pt"
}

require_checkpoint() {
  local label="$1"
  local ckpt="$2"
  [[ -f "${ckpt}" ]] || { echo "[ERROR] Missing ${label} checkpoint: ${ckpt}" >&2; exit 1; }
}

run_stage1() {
  echo "========== NLF-HSI four-stage pipeline: Stage 1 GT-SMPL scale teacher =========="
  OUTPUT_DIR="${STAGE1_DIR}" \
  EPOCHS="${STAGE1_EPOCHS}" \
  LR="${STAGE1_LR}" \
  BATCH_SIZE="${STAGE1_BATCH_SIZE}" \
  NUM_VIEWS="${STAGE1_NUM_VIEWS}" \
  RESET_EPOCH=false \
  RESUME_CKPT="" \
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale.sh"
  require_checkpoint "Stage1" "$(stage_checkpoint "${STAGE1_DIR}")"
}

run_stage2() {
  local stage1_ckpt
  stage1_ckpt="$(stage_checkpoint "${STAGE1_DIR}")"
  require_checkpoint "Stage1" "${stage1_ckpt}"
  echo "========== NLF-HSI four-stage pipeline: Stage 2 anchor translation =========="
  STAGE1_DIR="${STAGE1_DIR}" \
  OUTPUT_DIR="${STAGE2_DIR}" \
  RESUME_CKPT="${stage1_ckpt}" \
  EPOCHS="${STAGE2_EPOCHS}" \
  LR="${STAGE2_LR}" \
  BATCH_SIZE="${STAGE2_BATCH_SIZE}" \
  NUM_VIEWS="${STAGE2_NUM_VIEWS}" \
  RESET_EPOCH=true \
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage2_anchor_transl.sh"
  require_checkpoint "Stage2" "$(stage_checkpoint "${STAGE2_DIR}")"
}

run_stage3() {
  local stage2_ckpt
  stage2_ckpt="$(stage_checkpoint "${STAGE2_DIR}")"
  require_checkpoint "Stage2" "${stage2_ckpt}"
  echo "========== NLF-HSI four-stage pipeline: Stage 3 contact detail =========="
  STAGE2_DIR="${STAGE2_DIR}" \
  OUTPUT_DIR="${STAGE3_DIR}" \
  RESUME_CKPT="${stage2_ckpt}" \
  EPOCHS="${STAGE3_EPOCHS}" \
  LR="${STAGE3_LR}" \
  BATCH_SIZE="${STAGE3_BATCH_SIZE}" \
  NUM_VIEWS="${STAGE3_NUM_VIEWS}" \
  RESET_EPOCH=true \
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage3_contact_detail.sh"
  require_checkpoint "Stage3" "$(stage_checkpoint "${STAGE3_DIR}")"
}

run_stage4() {
  local stage3_ckpt
  stage3_ckpt="$(stage_checkpoint "${STAGE3_DIR}")"
  require_checkpoint "Stage3" "${stage3_ckpt}"
  echo "========== NLF-HSI four-stage pipeline: Stage 4 temporal tracks =========="
  STAGE3_DIR="${STAGE3_DIR}" \
  OUTPUT_DIR="${STAGE4_DIR}" \
  RESUME_CKPT="${stage3_ckpt}" \
  EPOCHS="${STAGE4_EPOCHS}" \
  LR="${STAGE4_LR}" \
  BATCH_SIZE="${STAGE4_BATCH_SIZE}" \
  NUM_VIEWS="${STAGE4_NUM_VIEWS}" \
  RESET_EPOCH=true \
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage4_temporal_tracks.sh"
  require_checkpoint "Stage4" "$(stage_checkpoint "${STAGE4_DIR}")"
}

echo "========== NLF-HSI four-stage training =========="
echo "Repo        : ${REPO_ROOT}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "DATA_ROOT   : ${DATA_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "Output root : ${PIPELINE_OUTPUT_ROOT}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Stages      : 1=${RUN_STAGE1} 2=${RUN_STAGE2} 3=${RUN_STAGE3} 4=${RUN_STAGE4}"
echo "Epochs      : ${STAGE1_EPOCHS}/${STAGE2_EPOCHS}/${STAGE3_EPOCHS}/${STAGE4_EPOCHS}"
echo "Batch sizes : ${STAGE1_BATCH_SIZE}/${STAGE2_BATCH_SIZE}/${STAGE3_BATCH_SIZE}/${STAGE4_BATCH_SIZE}"
echo "Num views   : ${STAGE1_NUM_VIEWS}/${STAGE2_NUM_VIEWS}/${STAGE3_NUM_VIEWS}/${STAGE4_NUM_VIEWS}"

[[ "${RUN_STAGE1}" == "true" ]] && run_stage1
[[ "${RUN_STAGE2}" == "true" ]] && run_stage2
[[ "${RUN_STAGE3}" == "true" ]] && run_stage3
[[ "${RUN_STAGE4}" == "true" ]] && run_stage4

echo "========== NLF-HSI four-stage training finished =========="
echo "Stage1 latest: $(stage_checkpoint "${STAGE1_DIR}")"
echo "Stage2 latest: $(stage_checkpoint "${STAGE2_DIR}")"
echo "Stage3 latest: $(stage_checkpoint "${STAGE3_DIR}")"
echo "Stage4 latest: $(stage_checkpoint "${STAGE4_DIR}")"
