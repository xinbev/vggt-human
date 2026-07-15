#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

cd "${REPO_ROOT}"
mkdir -p outputs/debug/hsi_stage2_transl_perturb_smoke

echo "========== HSI Stage2 transl perturbation interface smoke =========="
echo "Repo: ${REPO_ROOT}"

python scripts/diagnostics/check_hsi_stage2_transl_perturb_interface.py

echo "========== HSI Stage2 transl perturbation interface smoke passed =========="
