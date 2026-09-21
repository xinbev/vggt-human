#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/rich_global_chunk100}"
mkdir -p "${OUTPUT_DIR}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smpl}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smplx}"
SMPLX_TO_SMPL="${SMPLX_TO_SMPL:-${REPO_ROOT}/checkpoints/utils/smplx2smpl.pkl}"
for gender in NEUTRAL MALE FEMALE; do
  if [[ ! -f "${SMPL_MODEL_DIR}/smpl/SMPL_${gender}.pkl" && ! -f "${SMPL_MODEL_DIR}/smpl/SMPL_${gender}.npz" ]]; then
    echo "Missing SMPL_${gender} model in ${SMPL_MODEL_DIR}/smpl; set SMPL_MODEL_DIR to its parent." >&2
    exit 1
  fi
done
for gender in MALE FEMALE; do
  if [[ ! -f "${SMPLX_MODEL_DIR}/smplx/SMPLX_${gender}.npz" && ! -f "${SMPLX_MODEL_DIR}/smplx/SMPLX_${gender}.pkl" ]]; then
    echo "Missing SMPLX_${gender} model in ${SMPLX_MODEL_DIR}/smplx; set SMPLX_MODEL_DIR to its parent." >&2
    exit 1
  fi
done
[[ -f "${SMPLX_TO_SMPL}" ]] || { echo "Missing conversion matrix: ${SMPLX_TO_SMPL}; set SMPLX_TO_SMPL." >&2; exit 1; }
RICH_SUPPORT_ROOT="${RICH_SUPPORT_ROOT:-/home/zhw/xyb_space/RICH/hmr4d_support}"
for name in rich_test_labels.pt rich_test_preproc.pt cam2params.pt; do
  [[ -f "${RICH_SUPPORT_ROOT}/${name}" ]] || { echo "Missing RICH support: ${RICH_SUPPORT_ROOT}/${name}" >&2; exit 1; }
done

args=(
  --smpl-model-dir "${SMPL_MODEL_DIR}"
  --smplx-model-dir "${SMPLX_MODEL_DIR}"
  --smplx-to-smpl "${SMPLX_TO_SMPL}"
  --output-dir "${OUTPUT_DIR}"
  --max-sequences "${MAX_SEQUENCES:-0}"
  --max-frames-per-sequence "${MAX_FRAMES_PER_SEQUENCE:-0}"
  --max-humans "${MAX_HUMANS:-8}"
  --device "${DEVICE:-cuda:0}"
)
if [[ -n "${SEQUENCE:-}" ]]; then args+=(--sequence "${SEQUENCE}"); fi
if [[ -n "${RICH_OFFICIAL_ROOT:-}" ]]; then args+=(--official-root "${RICH_OFFICIAL_ROOT}"); fi
args+=(--support-root "${RICH_SUPPORT_ROOT}")

if ! python benchmarks/rich_global/chunk100.py "${args[@]}" >"${OUTPUT_DIR}/run.log" 2>&1; then
  tail -n 100 "${OUTPUT_DIR}/run.log" >&2
  exit 1
fi
cat "${OUTPUT_DIR}/sequence_metrics.csv"
tail -n 1 "${OUTPUT_DIR}/run.log"
