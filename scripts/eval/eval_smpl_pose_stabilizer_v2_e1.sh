#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/eval_smpl_pose_stabilizer_v2_e1.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
THREEDPW_ROOT="${THREEDPW_ROOT:-/home/zhw/xyb_space/3DPW/sequenceFiles/train}"
EMDB_ROOT="${EMDB_ROOT:-/home/zhw/xyb_space/emdb}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/debug/smpl_temporal_stabilizer_v2_pose_e0/checkpoint_e0.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/smpl_temporal_stabilizer_v2_pose_e1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

[[ -f "${CONFIG}" ]] || { echo "[ERROR] Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing E0 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -d "${THREEDPW_ROOT}" ]] || { echo "[ERROR] Missing 3DPW root: ${THREEDPW_ROOT}" >&2; exit 1; }
[[ -d "${EMDB_ROOT}" ]] || { echo "[ERROR] Missing EMDB root: ${EMDB_ROOT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
SOURCES_JSON="[{\"name\":\"3dpw\",\"root\":\"${THREEDPW_ROOT}\"},{\"name\":\"emdb\",\"root\":\"${EMDB_ROOT}\"}]"

python scripts/eval/eval_smpl_pose_stabilizer_v2_e1.py \
  --config "${CONFIG}" \
  --path-config "${PATH_CONFIG}" \
  --override "data.sources=${SOURCES_JSON}" \
  --override "checkpoint.path=${CHECKPOINT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Summary: ${OUTPUT_DIR}/e1_summary.json"
