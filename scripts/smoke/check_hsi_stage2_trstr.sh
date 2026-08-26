#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage2_trstr_smoke}"
DEVICE="${DEVICE:-cuda}"
CHECK_TEMPORAL="${CHECK_TEMPORAL:-false}"
cd "${REPO_ROOT}"

SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("assets", {}).get("smpl_model_dir", ""))
PY
)}"

[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

ARGS=(
  scripts/smoke/check_hsi_stage2_trstr.py
  --smpl-model-dir "${SMPL_MODEL_DIR}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${CHECK_TEMPORAL}" == "true" ]]; then
  ARGS+=(--check-temporal)
fi

python "${ARGS[@]}"
