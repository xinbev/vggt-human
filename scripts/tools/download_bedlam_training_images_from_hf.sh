#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

ARGS=(
  --repo-id "${HF_REPO_ID:-nguyenquivinhquang/BEDLAM}"
  --repo-type dataset
  --endpoint "${HF_ENDPOINT}"
  --manifest "${MANIFEST:-outputs/debug/hf_bedlam_training_images/selected.txt}"
  --local-dir "${LOCAL_DIR:-/home/zhw/xyb_space/bedlam/hf_bedlam}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  ARGS+=(--token "${HF_TOKEN}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python scripts/tools/download_hf_manifest.py "${ARGS[@]}"
