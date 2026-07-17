#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the completed stage output directory}"
GATE_STAGE="${GATE_STAGE:-stage2}"
GATE_MODE="${GATE_MODE:-overfit}"

cd "${REPO_ROOT}"

[[ -f "${OUTPUT_DIR}/metrics_latest.json" ]] || {
  echo "[ERROR] Missing metrics: ${OUTPUT_DIR}/metrics_latest.json" >&2
  exit 1
}
[[ -f "${OUTPUT_DIR}/resolved_config.json" ]] || {
  echo "[ERROR] Missing resolved config: ${OUTPUT_DIR}/resolved_config.json" >&2
  exit 1
}

python scripts/smoke/check_hsi_curriculum_metrics.py \
  --output-dir "${OUTPUT_DIR}" \
  --stage "${GATE_STAGE}" \
  --mode "${GATE_MODE}"
