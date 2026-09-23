#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Please set CHECKPOINT=/path/to/a/checkpoint-containing-all-enabled-HSI-TRSTR-heads.pt" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/show_human_scene_table3}"
for DATASET_NAME in 3dpw emdb1; do
  DATASET="${DATASET_NAME}" \
  CHECKPOINT="${CHECKPOINT}" \
  DEVICE="${DEVICE:-cuda:0}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}" \
  MAX_WINDOWS="${MAX_WINDOWS:-0}" \
  bash scripts/eval/evaluate_show_human_scene_hmr4d.sh
done

python scripts/eval/summarize_show_human_scene_table3.py \
  --threedpw "${OUTPUT_ROOT}/3dpw/3dpw_show_human_scene_summary.json" \
  --emdb1 "${OUTPUT_ROOT}/emdb1/emdb1_show_human_scene_summary.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --method "${METHOD_NAME:-VGGT-Omega}"
