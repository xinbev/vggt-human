#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
args=(--output-dir "${OUTPUT_DIR:-outputs/debug/groundedhuman_teaser_smoke}")
[[ "${WITH_TORCH:-true}" != true ]] || args+=(--with-torch)
"${PYTHON_BIN:-python}" scripts/smoke/check_groundedhuman_teaser_assets.py "${args[@]}" "$@"
