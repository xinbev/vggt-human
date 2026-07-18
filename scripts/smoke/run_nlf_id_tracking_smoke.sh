#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
python -u scripts/diagnostics/check_nlf_id_tracking_smoke.py
