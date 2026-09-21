#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/outputs/eval/rich_manual_scale/cache}"
SCALE_FILE="${SCALE_FILE:-${CACHE_DIR}/manual_scales.json}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/outputs/eval/rich_manual_scale}"

case "${CACHE_DIR}" in
  "${ROOT_DIR}/outputs/eval/rich_manual_scale/"*|"${ROOT_DIR}/outputs/eval/rich_manual_scale/cache") ;;
  *) echo "[ERROR] Refusing unexpected CACHE_DIR: ${CACHE_DIR}" >&2; exit 1 ;;
esac
case "${RESULT_ROOT}" in
  "${ROOT_DIR}/outputs/eval/rich_manual_scale") ;;
  *) echo "[ERROR] Refusing unexpected RESULT_ROOT: ${RESULT_ROOT}" >&2; exit 1 ;;
esac

echo "[keep] inference cache: ${CACHE_DIR}"
echo "[remove] manual scale file: ${SCALE_FILE}"
echo "[remove] manual/oracle metric outputs under: ${RESULT_ROOT}"

if [[ -f "${SCALE_FILE}" ]]; then
  rm -f -- "${SCALE_FILE}"
fi
rm -rf -- \
  "${RESULT_ROOT}/manual_metrics" \
  "${RESULT_ROOT}/oracle_grid_metrics"

echo "[done] Manual-scale experiment reset. Cached inference was preserved."
