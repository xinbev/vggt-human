#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/tum_dynamics_raw}"
PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/data/long_tum_s1}"
LENGTHS="${LENGTHS:-50,100,150,200,300,400,500,600,700,800,900,1000}"

python benchmarks/tum_dynamics_ate/prepare_tum_dynamics.py \
  --raw-root "${RAW_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --lengths "${LENGTHS}" \
  --sample-interval "${SAMPLE_INTERVAL:-1}" \
  --association-tolerance "${ASSOCIATION_TOLERANCE:-0.02}"

