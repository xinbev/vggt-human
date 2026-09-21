#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/zhw/xyb_space/RICH/official}"
SUPPORT_ROOT="${SUPPORT_ROOT:-/home/zhw/xyb_space/RICH/hmr4d_support}"
OUTPUT="${OUTPUT:-outputs/debug/rich_physical_grounding/assets_report.json}"

python scripts/diagnostics/check_rich_physical_grounding_assets.py \
  --official-root "${OFFICIAL_ROOT}" \
  --support-root "${SUPPORT_ROOT}" \
  --output "${OUTPUT}"
