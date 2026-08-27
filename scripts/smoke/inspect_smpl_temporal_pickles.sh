#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
THREEDPW_ROOT="${THREEDPW_ROOT:-/home/zhw/xyb_space/3DPW/sequenceFiles/train}"
EMDB_ROOT="${EMDB_ROOT:-/home/zhw/xyb_space/emdb}"
WINDOW_SIZE="${WINDOW_SIZE:-9}"

[[ -d "${THREEDPW_ROOT}" ]] || { echo "[ERROR] Missing 3DPW root: ${THREEDPW_ROOT}" >&2; exit 1; }
[[ -d "${EMDB_ROOT}" ]] || { echo "[ERROR] Missing EMDB root: ${EMDB_ROOT}" >&2; exit 1; }
python scripts/smoke/inspect_smpl_temporal_pickles.py \
  --threedpw-root "${THREEDPW_ROOT}" \
  --emdb-root "${EMDB_ROOT}" \
  --window-size "${WINDOW_SIZE}"
