#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m unittest benchmarks.emdb2_global.test_metrics -v
python -m py_compile \
  benchmarks/emdb2_global/__init__.py \
  benchmarks/emdb2_global/data.py \
  benchmarks/emdb2_global/metrics.py \
  benchmarks/emdb2_global/evaluate.py \
  benchmarks/emdb2_global/test_metrics.py

echo "[ok] EMDB-2 Human3R global metric protocol checks passed"
