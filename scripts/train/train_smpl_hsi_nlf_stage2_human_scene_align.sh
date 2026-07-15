#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-128}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-6}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
HSI_ALIGN_POINT_WEIGHT="${HSI_ALIGN_POINT_WEIGHT:-2.0}"
HSI_TRANSL_WEIGHT="${HSI_TRANSL_WEIGHT:-4.0}"
HSI_JOINTS3D_WEIGHT="${HSI_JOINTS3D_WEIGHT:-2.0}"
HSI_VERTICES_WEIGHT="${HSI_VERTICES_WEIGHT:-0.5}"
HSI_ALIGN_MAX_RAY_DELTA_M="${HSI_ALIGN_MAX_RAY_DELTA_M:-0.35}"
HSI_ALIGN_MAX_TANGENT_DELTA_M="${HSI_ALIGN_MAX_TANGENT_DELTA_M:-0.12}"
HSI_ALIGN_LOCAL_WINDOW="${HSI_ALIGN_LOCAL_WINDOW:-7}"
HSI_ALIGN_NUM_SAMPLE_VERTICES="${HSI_ALIGN_NUM_SAMPLE_VERTICES:-96}"

echo "========== HSI Stage2 human-scene translation alignment =========="
echo "Repo        : ${REPO_ROOT}"
echo "Config      : ${CONFIG_PATH}"
echo "Path config : ${PATH_CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Stage1 ckpt : ${STAGE1_CKPT}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "Batch/views : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "NLF batch   : ${NLF_INTERNAL_BATCH_SIZE}"
echo "Loss weights: align=${HSI_ALIGN_POINT_WEIGHT} transl=${HSI_TRANSL_WEIGHT} j3d=${HSI_JOINTS3D_WEIGHT} verts=${HSI_VERTICES_WEIGHT}"

CMD=(
  python scripts/train/train_smpl.py
  --train-config "${CONFIG_PATH}"
  --path-config "${PATH_CONFIG}"
  --override "experiment.output_dir=${OUTPUT_DIR}"
  --override "checkpoint.resume=${STAGE1_CKPT}"
  --override "checkpoint.reset_epoch=true"
  --override "optim.batch_size=${BATCH_SIZE}"
  --override "optim.epochs=${EPOCHS}"
  --override "optim.lr=${LR}"
  --override "optim.log_interval=${LOG_INTERVAL}"
  --override "data.num_workers=${NUM_WORKERS}"
  --override "data.max_humans=${MAX_HUMANS}"
  --override "data.sequence_length=${NUM_VIEWS}"
  --override "model.num_smpl_queries=${MAX_HUMANS}"
  --override "model.nlf_internal_batch_size=${NLF_INTERNAL_BATCH_SIZE}"
  --override "model.hsi_align_max_ray_delta_m=${HSI_ALIGN_MAX_RAY_DELTA_M}"
  --override "model.hsi_align_max_tangent_delta_m=${HSI_ALIGN_MAX_TANGENT_DELTA_M}"
  --override "model.hsi_align_local_window=${HSI_ALIGN_LOCAL_WINDOW}"
  --override "model.hsi_align_num_sample_vertices=${HSI_ALIGN_NUM_SAMPLE_VERTICES}"
  --override "loss.hsi_align_point_weight=${HSI_ALIGN_POINT_WEIGHT}"
  --override "loss.hsi_transl_cam_weight=${HSI_TRANSL_WEIGHT}"
  --override "loss.hsi_joints3d_weight=${HSI_JOINTS3D_WEIGHT}"
  --override "loss.hsi_vertices_weight=${HSI_VERTICES_WEIGHT}"
)

if [[ -n "${MAX_STEPS_PER_EPOCH}" ]]; then
  CMD+=(--override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}")
fi

"${CMD[@]}"

echo "========== HSI Stage2 human-scene translation alignment finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
