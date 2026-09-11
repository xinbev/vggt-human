#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"

python scripts/smoke/check_emdb_camera_trajectory_plot.py
if python -c "import matplotlib" >/dev/null 2>&1; then
  python scripts/smoke/check_emdb_camera_trajectory_plot_e2e.py
else
  echo "[skip] matplotlib unavailable; skipped end-to-end PNG/PDF export check"
fi
