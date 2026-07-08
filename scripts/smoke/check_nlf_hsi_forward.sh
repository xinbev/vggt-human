#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/nlf_hsi_forward_smoke}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }

NLF_ROOT="$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("third_party", {}).get("nlf_root", "third_party/nlf"))
PY
)"
NLF_CKPT="$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("checkpoints", {}).get("nlf_smpl", ""))
PY
)"
SMPL_MODEL_DIR="$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("assets", {}).get("smpl_model_dir", ""))
PY
)"

[[ -d "${NLF_ROOT}" ]] || { echo "[ERROR] Missing NLF source directory: ${NLF_ROOT}" >&2; exit 1; }
[[ -f "${NLF_CKPT}" ]] || { echo "[ERROR] Missing NLF checkpoint: ${NLF_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== NLF -> VGGT-Human -> HSI forward smoke =========="
echo "Repo        : ${REPO_ROOT}"
echo "Path config : ${PATH_CONFIG}"
echo "Train config: ${TRAIN_CONFIG}"
echo "NLF root    : ${NLF_ROOT}"
echo "NLF ckpt    : ${NLF_CKPT}"
echo "SMPL models : ${SMPL_MODEL_DIR}"
echo "Output      : ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/diagnostics/check_nlf_hsi_forward.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-batches 1 \
  --override "data.num_workers=0" \
  --override "data.pin_memory=false" \
  --override "optim.batch_size=1"
