#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRSTR_DIR="${TRSTR_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_refine}"
TRSTR_CHECKPOINT="${TRSTR_CHECKPOINT:-}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${REPO_ROOT}/outputs/eval/emdb2_global_stride7/predictions}"

if [[ -z "${TRSTR_CHECKPOINT}" && -f "${TRSTR_DIR}/checkpoint_topk_index.json" ]]; then
  TRSTR_CHECKPOINT="$(python - "${TRSTR_DIR}/checkpoint_topk_index.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entries = payload.get("entries", [])
print(entries[0].get("path", "") if entries else "")
PY
)"
fi
if [[ -z "${TRSTR_CHECKPOINT}" ]]; then
  TRSTR_CHECKPOINT="${TRSTR_DIR}/checkpoint_latest.pt"
fi
if [[ "${TRSTR_CHECKPOINT}" != /* ]]; then
  TRSTR_CHECKPOINT="${REPO_ROOT}/${TRSTR_CHECKPOINT}"
fi

[[ -f "${TRSTR_CHECKPOINT}" ]] || { echo "[ERROR] Missing TRSTR checkpoint: ${TRSTR_CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }

ARGS=(
  --checkpoint "${TRSTR_CHECKPOINT}"
  --scale-checkpoint "${SCALE_CHECKPOINT}"
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --inference-config "${INFERENCE_CONFIG:-benchmarks/emdb2_global/inference_config.yaml}"
  --output-dir "${PREDICTIONS_ROOT}"
  --subsample-stride "${SUBSAMPLE_STRIDE:-7}"
  --max-input-frames "${MAX_INPUT_FRAMES:-500}"
  --max-humans "${MAX_HUMANS:-8}"
  --conf-threshold "${CONF_THRESHOLD:-0.05}"
  --match-iou-threshold "${MATCH_IOU_THRESHOLD:-0.05}"
  --sequence-filter "${SEQUENCE_FILTER:-}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --device "${DEVICE:-cuda:0}"
)
if [[ -n "${EMDB_ROOT:-}" ]]; then ARGS+=(--emdb-root "${EMDB_ROOT}"); fi

echo "========== EMDB-2-S7 unchunked two-pass prediction export =========="
echo "TRSTR checkpoint : ${TRSTR_CHECKPOINT}"
echo "Scale checkpoint : ${SCALE_CHECKPOINT}"
echo "Predictions      : ${PREDICTIONS_ROOT}"
echo "Stride/limit     : ${SUBSAMPLE_STRIDE:-7} / ${MAX_INPUT_FRAMES:-500}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

# Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPU and usually remaps it
# to index 0. Only override it when the caller explicitly requests a value.
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

python benchmarks/emdb2_global/export_stride7.py "${ARGS[@]}"
