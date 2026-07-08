#!/usr/bin/env bash
set -euo pipefail

# Single-image processed-plane diagnostics for the NLF-provider HSI path.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_depth_smpl_diagnostics}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_provider_stage1/checkpoint_latest.pt}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
SPLIT="${SPLIT:-Training}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"

cd "${REPO_ROOT}"
mkdir -p "${VIS_OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }

VGGT_CKPT="${VGGT_CKPT:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("checkpoints", {}).get("vggt_baseline", ""))
PY
)}"
NLF_CKPT="${NLF_CKPT:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("checkpoints", {}).get("nlf_smpl", ""))
PY
)}"
NLF_ROOT="${NLF_ROOT:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("third_party", {}).get("nlf_root", "third_party/nlf"))
PY
)}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-$(python - "${PATH_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg.get("assets", {}).get("smpl_model_dir", ""))
PY
)}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${NLF_CKPT}" ]] || { echo "[ERROR] Missing NLF checkpoint: ${NLF_CKPT}" >&2; exit 1; }
[[ -d "${NLF_ROOT}" ]] || { echo "[ERROR] Missing NLF source directory: ${NLF_ROOT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing NLF-HSI checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing input image: ${IMAGE_PATH}" >&2; exit 1; }

echo "========== NLF HSI single-image visual diagnostics =========="
echo "Image       : ${IMAGE_PATH}"
echo "Train config: ${TRAIN_CONFIG}"
echo "HSI ckpt    : ${SMPL_CKPT}"
echo "VGGT ckpt   : ${VGGT_CKPT}"
echo "NLF ckpt    : ${NLF_CKPT}"
echo "NLF root    : ${NLF_ROOT}"
echo "SMPL models : ${SMPL_MODEL_DIR}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "Output      : ${VIS_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_depth_smpl_diagnostics.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${VIS_OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --use-gt-box-prior \
  --override "checkpoints.nlf_smpl=${NLF_CKPT}" \
  --override "third_party.nlf_root=${NLF_ROOT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.smpl_provider=nlf" \
  --override "model.nlf_use_detector=false" \
  --override "model.nlf_require_boxes=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true"

echo "========== NLF HSI visual diagnostics finished =========="
echo "JSON: ${VIS_OUTPUT_DIR}/$(basename "${IMAGE_PATH%.*}")_hsi_depth_smpl_diagnostics.json"
