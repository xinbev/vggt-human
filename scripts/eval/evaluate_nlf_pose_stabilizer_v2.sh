#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CHECKPOINT:?Set CHECKPOINT to the VGGT checkpoint used to obtain runtime camera for NLF.}"
: "${TEMPORAL_CHECKPOINT:?Set TEMPORAL_CHECKPOINT to the V2 pose mixture or hard-finetune checkpoint.}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/eval_nlf_pose_stabilizer_v2.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
DATASETS="${DATASETS:-3dpw emdb1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/nlf_pose_stabilizer_v2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"

[[ -f "${CONFIG}" ]] || { echo "[ERROR] Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${TEMPORAL_CHECKPOINT}" ]] || { echo "[ERROR] Missing V2 pose checkpoint: ${TEMPORAL_CHECKPOINT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "========== RGB -> VGGT -> NLF -> PoseTemporalStabilizer V2 =========="
echo "Datasets : ${DATASETS}"
echo "VGGT ckpt: ${CHECKPOINT}"
echo "V2 ckpt  : ${TEMPORAL_CHECKPOINT}"
echo "Output   : ${OUT_ROOT}"
echo "Protocol : unique 9-frame centre evaluation; HSI/TRSTR disabled"

for DATASET in ${DATASETS}; do
  ARGS=(
    --dataset "${DATASET}"
    --checkpoint "${CHECKPOINT}"
    --temporal-checkpoint "${TEMPORAL_CHECKPOINT}"
    --path-config "${PATH_CONFIG}"
    --train-config "${CONFIG}"
    --output-dir "${OUT_ROOT}/${DATASET}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ "${MAX_WINDOWS}" != "0" ]]; then
    ARGS+=(--max-windows "${MAX_WINDOWS}")
  fi
  python scripts/eval/evaluate_nlf_pose_stabilizer_v2_metrics.py "${ARGS[@]}"
done

python scripts/eval/summarize_nlf_pose_stabilizer_v2_metrics.py --root "${OUT_ROOT}"
echo "Table  : ${OUT_ROOT}/summary.md"
echo "Details: ${OUT_ROOT}/{3dpw,emdb1}/*_metrics.json"
