#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr_v3_refine.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
V2_DIR="${V2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v2_strong}"
V2_CHECKPOINT="${V2_CHECKPOINT:-}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_refine}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_HUMANS="${MAX_HUMANS:-20}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-6}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-human}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-smpl_hsi_stage2_trstr_v3_refine}"
WANDB_GROUP="${WANDB_GROUP:-hsi_stage2_trstr_v3_refine}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${V2_CHECKPOINT}" && -f "${V2_DIR}/checkpoint_topk_index.json" ]]; then
  V2_CHECKPOINT="$(python - "${V2_DIR}/checkpoint_topk_index.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entries = payload.get("entries", [])
print(entries[0].get("path", "") if entries else "")
PY
)"
fi
if [[ -z "${V2_CHECKPOINT}" ]]; then
  V2_CHECKPOINT="${V2_DIR}/checkpoint_latest.pt"
fi
if [[ "${V2_CHECKPOINT}" != /* ]]; then
  V2_CHECKPOINT="${REPO_ROOT}/${V2_CHECKPOINT}"
fi

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing v3 config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${V2_CHECKPOINT}" ]] || { echo "[ERROR] Missing v2 checkpoint: ${V2_CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing HSI scale v3 checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "========== TRSTR v3 refinement =========="
echo "V2 source        : ${V2_CHECKPOINT}"
echo "Scale v3 overlay : ${SCALE_CHECKPOINT}"
echo "Independent output: ${OUTPUT_DIR}"
echo "Epochs/lr        : ${EPOCHS} / ${LR}"
echo "Batch/workers    : ${BATCH_SIZE} / ${NUM_WORKERS}"
echo "Fixed eval       : Training indices 90000..90255, deterministic four-case cycle"
echo "Loss isolation   : all non-TRSTR weights explicitly zero"
echo "Extra objectives : strong=2 clean=2 no-worse=2 monotonic=3"
echo "Temporal         : false"

python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${V2_CHECKPOINT}" \
  --override "checkpoint.overlay=${SCALE_CHECKPOINT}" \
  --override "checkpoint.reset_epoch=true" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.hsi_trstr_enable_temporal=false" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.max_val_steps=${MAX_VAL_STEPS}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.entity=${WANDB_ENTITY}" \
  --override "logging.wandb.name=${WANDB_RUN_NAME}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
echo "Top-k index      : ${OUTPUT_DIR}/checkpoint_topk_index.json"
