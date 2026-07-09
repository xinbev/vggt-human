#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Please run this script on a CUDA/NVIDIA GPU server." >&2
  exit 1
fi

nvidia-smi
