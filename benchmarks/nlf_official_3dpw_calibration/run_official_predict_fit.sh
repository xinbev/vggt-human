#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/third_party/nlf:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${NLF_DATA_ROOT:?Set NLF_DATA_ROOT such that NLF_DATA_ROOT/3dpw exists (lowercase path required by official NLF scripts).}"
: "${NLF_PROJDIR:?Set NLF_PROJDIR containing canonical_verts/smpl.npy, canonical_joints/smpl.npy and smpl_faces.npy.}"
: "${NLF_MODEL_PATH:?Set NLF_MODEL_PATH to the released TorchScript checkpoint.}"

RAW_OUTPUT="${RAW_OUTPUT:-${REPO_ROOT}/outputs/benchmarks/nlf_official_3dpw/raw_predictions}"
FITTED_OUTPUT="${FITTED_OUTPUT:-${REPO_ROOT}/outputs/benchmarks/nlf_official_3dpw/fitted_smpl}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
NLF_BATCH_SIZE="${NLF_BATCH_SIZE:-64}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-64}"
NLF_NUM_AUG="${NLF_NUM_AUG:-5}"

export DATA_ROOT="${NLF_DATA_ROOT}"
export PROJDIR="${NLF_PROJDIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

python benchmarks/nlf_official_3dpw_calibration/check_environment.py \
  --data-root "${DATA_ROOT}" \
  --projdir "${PROJDIR}" \
  --nlf-root "${REPO_ROOT}/third_party/nlf" \
  --model-path "${NLF_MODEL_PATH}"

python third_party/nlf/nlf/pt/inference_scripts/predict_tdpw.py \
  --model-path "${NLF_MODEL_PATH}" \
  --output-path "${RAW_OUTPUT}" \
  --testset-only \
  --real-intrinsics \
  --gtassoc \
  --batch-size "${NLF_BATCH_SIZE}" \
  --internal-batch-size "${NLF_INTERNAL_BATCH_SIZE}" \
  --num-aug "${NLF_NUM_AUG}"

python third_party/nlf/nlf/tf/inference_scripts/fit_tdpw.py \
  --in-pred-path "${RAW_OUTPUT}" \
  --out-pred-path "${FITTED_OUTPUT}" \
  --testset-only \
  --batch-size 256 \
  --num-iter 3 \
  --l2-regul 1.0

python benchmarks/nlf_official_3dpw_calibration/evaluate_fitted.py \
  --fitted-root "${FITTED_OUTPUT}" \
  --threedpw-root "${DATA_ROOT}/3dpw" \
  --output "${REPO_ROOT}/outputs/benchmarks/nlf_official_3dpw/calibration.json" \
  --device "cuda:0"
