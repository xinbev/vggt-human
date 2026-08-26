#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_translation_only}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree/checkpoint_latest.pt}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_REGIONS="${NUM_REGIONS:-96}"
NUM_VIEWS="${NUM_VIEWS:-1}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
RAY_NOISE="${RAY_NOISE:-0.25}"
TANGENT_NOISE="${TANGENT_NOISE:-0.10}"
CLEAN_PROB="${CLEAN_PROB:-0.15}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-human}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-smpl_hsi_stage2_trstr_translation_only}"
WANDB_GROUP="${WANDB_GROUP:-hsi_stage2_trstr}"
WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing Stage1 scale checkpoint: ${INIT_CKPT}" >&2; exit 1; }
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

echo "========== HSI Stage2 TRSTR translation-only training =========="
echo "Init scale ckpt: ${INIT_CKPT}"
echo "Output         : ${OUTPUT_DIR}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "Batch/views    : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Region budget  : ${NUM_REGIONS} (ablation options: 48/72/96)"
echo "Epochs/lr      : ${EPOCHS} / ${LR}"
echo "Translation noise: ray=${RAY_NOISE}, tangent=${TANGENT_NOISE}, clean=${CLEAN_PROB}"
echo "W&B            : enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"

python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${INIT_CKPT}" \
  --override "checkpoint.reset_epoch=true" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.hsi_trstr_num_regions=${NUM_REGIONS}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "training_prior.smpl_transl_ray_noise_schedule=${RAY_NOISE}" \
  --override "training_prior.smpl_transl_tangent_noise_schedule_m=${TANGENT_NOISE}" \
  --override "training_prior.smpl_transl_ray_noise_clean_prob=${CLEAN_PROB}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.entity=${WANDB_ENTITY}" \
  --override "logging.wandb.name=${WANDB_RUN_NAME}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
echo "Best checkpoint index: ${OUTPUT_DIR}/checkpoint_topk_index.json"
