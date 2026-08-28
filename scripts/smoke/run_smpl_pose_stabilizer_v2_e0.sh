#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/overfit_smpl_pose_stabilizer_v2.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
THREEDPW_ROOT="${THREEDPW_ROOT:-/home/zhw/xyb_space/3DPW/sequenceFiles/train}"
EMDB_ROOT="${EMDB_ROOT:-/home/zhw/xyb_space/emdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/smpl_temporal_stabilizer_v2_pose_e0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
STEPS="${STEPS:-1000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

[[ -f "${CONFIG}" ]] || { echo "[ERROR] Missing config: ${CONFIG}" >&2; exit 1; }
[[ -d "${THREEDPW_ROOT}" ]] || { echo "[ERROR] Missing 3DPW root: ${THREEDPW_ROOT}" >&2; exit 1; }
[[ -d "${EMDB_ROOT}" ]] || { echo "[ERROR] Missing EMDB root: ${EMDB_ROOT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
SOURCES_JSON="[{\"name\":\"3dpw\",\"root\":\"${THREEDPW_ROOT}\"},{\"name\":\"emdb\",\"root\":\"${EMDB_ROOT}\"}]"

echo "========== V2 E0: fixed SO(3) pose stabilizer overfit =========="
python scripts/train/overfit_smpl_pose_stabilizer_v2.py \
  --config "${CONFIG}" \
  --path-config "${PATH_CONFIG}" \
  --override "data.sources=${SOURCES_JSON}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "optim.steps=${STEPS}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Summary   : ${OUTPUT_DIR}/e0_summary.json"
echo "Checkpoint: ${OUTPUT_DIR}/checkpoint_e0.pt"
