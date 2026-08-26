#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

exec python scripts/smoke/bedlam2_sample_load/check_bedlam2_sample_load.py "$@"
