#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr_v2_strong.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
V1_DIR="${V1_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial}"
V1_CHECKPOINT="${V1_CHECKPOINT:-}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v2_strong}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_REGIONS="${NUM_REGIONS:-96}"
EPOCHS="${EPOCHS:-7}"
LR="${LR:-5e-6}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
ALIGNMENT_CASE_PROBABILITIES="${ALIGNMENT_CASE_PROBABILITIES:-[0.15,0.40,0.25,0.20]}"
ALIGNMENT_SCALE_LOG_STD="${ALIGNMENT_SCALE_LOG_STD:-0.08}"
ALIGNMENT_SCALE_MIN="${ALIGNMENT_SCALE_MIN:-0.85}"
ALIGNMENT_SCALE_MAX="${ALIGNMENT_SCALE_MAX:-1.15}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-human}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-smpl_hsi_stage2_trstr_v2_strong}"
WANDB_GROUP="${WANDB_GROUP:-hsi_stage2_trstr_v2_strong}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${V1_CHECKPOINT}" && -f "${V1_DIR}/checkpoint_topk_index.json" ]]; then
  V1_CHECKPOINT="$(python - "${V1_DIR}/checkpoint_topk_index.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entries = payload.get("entries", [])
print(entries[0].get("path", "") if entries else "")
PY
)"
fi
if [[ -z "${V1_CHECKPOINT}" ]]; then
  V1_CHECKPOINT="${V1_DIR}/checkpoint_latest.pt"
fi
if [[ "${V1_CHECKPOINT}" != /* ]]; then
  V1_CHECKPOINT="${REPO_ROOT}/${V1_CHECKPOINT}"
fi

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing v2 config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${V1_CHECKPOINT}" ]] || { echo "[ERROR] Missing frozen v1 checkpoint: ${V1_CHECKPOINT}" >&2; exit 1; }
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

echo "========== TRSTR v2 strong spatial continuation =========="
echo "Frozen v1 source : ${V1_CHECKPOINT}"
echo "Scale v3 overlay : ${SCALE_CHECKPOINT}"
echo "Independent output: ${OUTPUT_DIR}"
echo "GPU              : ${CUDA_VISIBLE_DEVICES}"
echo "Batch/regions    : ${BATCH_SIZE} / ${NUM_REGIONS}"
echo "Epochs/lr        : ${EPOCHS} / ${LR}"
echo "Ray mixture      : p=[0.30,0.40,0.30] std=[0.06,0.16,0.30]m clip=[0.15,0.35,0.60]m"
echo "XY mixture       : p=[0.20,0.35,0.45] std=[0.12,0.55,1.10]m norm_clip=[0.35,1.20,2.00]m"
echo "Case mix         : ${ALIGNMENT_CASE_PROBABILITIES} (clean,scale,nlf,mixed)"
echo "Scale residual   : log_std=${ALIGNMENT_SCALE_LOG_STD} clamp=[${ALIGNMENT_SCALE_MIN},${ALIGNMENT_SCALE_MAX}], sole-anchor target"
echo "Correction bounds: ray_vote=1.00m tangent_vote=1.25m person_step=1.25m, iterations=2"
echo "Temporal         : false"
echo "W&B              : ${WANDB_ENABLED} / ${WANDB_GROUP}"

python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${V1_CHECKPOINT}" \
  --override "checkpoint.overlay=${SCALE_CHECKPOINT}" \
  --override "checkpoint.reset_epoch=true" \
  --override "data.sequence_length=1" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.hsi_trstr_num_regions=${NUM_REGIONS}" \
  --override "model.hsi_trstr_enable_temporal=false" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "training_prior.trstr_alignment_case_probabilities=${ALIGNMENT_CASE_PROBABILITIES}" \
  --override "training_prior.trstr_alignment_scale_log_std=${ALIGNMENT_SCALE_LOG_STD}" \
  --override "training_prior.trstr_alignment_scale_min=${ALIGNMENT_SCALE_MIN}" \
  --override "training_prior.trstr_alignment_scale_max=${ALIGNMENT_SCALE_MAX}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.entity=${WANDB_ENTITY}" \
  --override "logging.wandb.name=${WANDB_RUN_NAME}" \
  --override "logging.wandb.group=${WANDB_GROUP}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
echo "Top-k index      : ${OUTPUT_DIR}/checkpoint_topk_index.json"
