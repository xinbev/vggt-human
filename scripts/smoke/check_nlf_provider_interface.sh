#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/debug/nlf_provider_interface_smoke
python scripts/diagnostics/check_nlf_provider_interface.py
