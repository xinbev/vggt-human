#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
LOG_SCALE_STD_SCHEDULE="${LOG_SCALE_STD_SCHEDULE:-0.30}"
LOG10_SCALE_STD_SCHEDULE="${LOG10_SCALE_STD_SCHEDULE:-}"
NOISE_MODE="${NOISE_MODE:-lognormal}"
NOISE_UNIT="${NOISE_UNIT:-sequence}"
CLEAN_PROB="${CLEAN_PROB:-0.0}"
HSI_SCALE_TRAINING_MODE="${HSI_SCALE_TRAINING_MODE:-direct_perturb}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-human}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-smpl_hsi_gt_depth_scale_scene_affine_boxfree}"
WANDB_GROUP="${WANDB_GROUP:-hsi_scene_scale_finetune}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"
SAVE_LATEST="${SAVE_LATEST:-true}"
HSI_SCENE_LOG_SCALE_MIN="${HSI_SCENE_LOG_SCALE_MIN:--5.0}"
HSI_SCENE_LOG_SCALE_MAX="${HSI_SCENE_LOG_SCALE_MAX:-5.0}"
SMPL_SCALE_TEACHER_WEIGHT="${SMPL_SCALE_TEACHER_WEIGHT:-4.0}"
DEPTH_TEACHER_WEIGHT="${DEPTH_TEACHER_WEIGHT:-0.10}"

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }

VGGT_CKPT="${VGGT_CKPT:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("checkpoints", {}).get("vggt_baseline", ""))
PY
)}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("assets", {}).get("smpl_model_dir", ""))
PY
)}"

[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== HSI GT-depth scale scene-affine training =========="
echo "Repo        : ${REPO_ROOT}"
echo "Config      : ${TRAIN_CONFIG}"
echo "Path config : ${PATH_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Finetune ckpt: ${INIT_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "SMPL models : ${SMPL_MODEL_DIR}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Person input: box-free GT SMPL with online depth visibility"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "Batch/views : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Epochs/lr   : ${EPOCHS} / ${LR}"
echo "Scale train : ${HSI_SCALE_TRAINING_MODE}"
if [[ -n "${LOG10_SCALE_STD_SCHEDULE}" ]]; then
  NOISE_SCHEDULE_OVERRIDE="training_prior.hsi_gt_depth_log10_scale_std_schedule=${LOG10_SCALE_STD_SCHEDULE}"
  echo "Noise       : log10_std=${LOG10_SCALE_STD_SCHEDULE}, mode=${NOISE_MODE}, unit=${NOISE_UNIT}, clean=${CLEAN_PROB}"
else
  NOISE_SCHEDULE_OVERRIDE="training_prior.hsi_gt_depth_log_scale_std_schedule=${LOG_SCALE_STD_SCHEDULE}"
  echo "Noise       : ln_std=${LOG_SCALE_STD_SCHEDULE}, mode=${NOISE_MODE}, unit=${NOISE_UNIT}, clean=${CLEAN_PROB}"
fi
echo "W&B         : enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, name=${WANDB_RUN_NAME}, mode=${WANDB_MODE}"
echo "Checkpoints : latest=${SAVE_LATEST}, top_k=${SAVE_TOP_K}"

python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${INIT_CKPT}" \
  --override "checkpoint.reset_epoch=true" \
  --override "checkpoint.save_top_k=${SAVE_TOP_K}" \
  --override "checkpoint.save_latest=${SAVE_LATEST}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.hsi_scene_log_scale_min=${HSI_SCENE_LOG_SCALE_MIN}" \
  --override "model.hsi_scene_log_scale_max=${HSI_SCENE_LOG_SCALE_MAX}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "${NOISE_SCHEDULE_OVERRIDE}" \
  --override "training_prior.hsi_gt_depth_scale_noise_mode=${NOISE_MODE}" \
  --override "training_prior.hsi_gt_depth_scale_noise_unit=${NOISE_UNIT}" \
  --override "training_prior.hsi_gt_depth_scale_clean_prob=${CLEAN_PROB}" \
  --override "training_prior.hsi_scale_training_mode=${HSI_SCALE_TRAINING_MODE}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.entity=${WANDB_ENTITY}" \
  --override "logging.wandb.name=${WANDB_RUN_NAME}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.mode=${WANDB_MODE}" \
  --override "loss.hsi_smpl_scale_teacher_weight=${SMPL_SCALE_TEACHER_WEIGHT}" \
  --override "loss.hsi_depth_teacher_weight=${DEPTH_TEACHER_WEIGHT}"

echo "========== HSI GT-depth scale scene-affine training finished =========="
if [[ "${SAVE_LATEST}" == "true" ]]; then
  echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
fi
if [[ "${SAVE_TOP_K}" -gt 0 ]]; then
  echo "Best checkpoint  : see ${OUTPUT_DIR}/checkpoint_topk_index.json"
fi
