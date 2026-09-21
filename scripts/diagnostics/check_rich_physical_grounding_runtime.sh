#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${ROOT_DIR}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-${ROOT_DIR}/checkpoints/vggt_omega_1b_512.pt}"
NLF_CHECKPOINT="${NLF_CHECKPOINT:-${ROOT_DIR}/third_party/weights/nlf/nlf_l_multi_0.3.2.torchscript}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${ROOT_DIR}/checkpoints/body_models/smpl}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${ROOT_DIR}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${ROOT_DIR}/configs/path.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/debug/rich_physical_grounding}"
REPORT="${OUTPUT_DIR}/runtime_report.txt"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${REPORT}") 2>&1

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[missing] ${label}: ${path}"
    return 1
  fi
  echo "[ok] ${label}: ${path} ($(du -h "${path}" | cut -f1))"
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[missing] ${label}: ${path}"
    return 1
  fi
  echo "[ok] ${label}: ${path}"
}

echo "RICH physical-grounding runtime check"
echo "Repository: ${ROOT_DIR}"
echo

status=0
require_file "${CHECKPOINT}" "evaluation checkpoint" || status=1
require_file "${SCALE_CHECKPOINT}" "coarse residual scale checkpoint" || status=1
require_file "${VGGT_CHECKPOINT}" "VGGT baseline checkpoint" || status=1
require_file "${NLF_CHECKPOINT}" "NLF TorchScript checkpoint" || status=1
require_file "${TRAIN_CONFIG}" "Stage-2 train config" || status=1
require_file "${PATH_CONFIG}" "path config" || status=1
require_dir "${SMPL_MODEL_DIR}" "SMPL model directory" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Runtime assets are incomplete. Fix the missing path or override it when running this script."
  echo "Example: CHECKPOINT=/absolute/path/checkpoint.pt bash scripts/diagnostics/check_rich_physical_grounding_runtime.sh"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python - \
  "${SMPL_MODEL_DIR}" "${CHECKPOINT}" "${SCALE_CHECKPOINT}" <<'PY'
import sys

import torch

from vggt_omega.models.smpl_layer import SMPLLayer


model_dir, stage2_path, scale_path = sys.argv[1:]
print(f"[python] torch={torch.__version__}")
print(f"[python] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[python] cuda_device={torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("CUDA is unavailable for the requested visible device")


def load_state_dict(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"]
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    if isinstance(payload, dict) and payload and all(isinstance(key, str) for key in payload):
        return payload
    raise RuntimeError(f"Checkpoint has no model state_dict: {path}")


stage2_state = load_state_dict(stage2_path)
scale_state = load_state_dict(scale_path)
stage2_align = [key for key in stage2_state if key.removeprefix("module.").startswith("hsi_human_scene_align_head.")]
scale_refine = [key for key in scale_state if key.removeprefix("module.").startswith("hsi_refinement_head.")]
if not stage2_align:
    raise RuntimeError("Stage-2 checkpoint contains no hsi_human_scene_align_head tensors")
if not scale_refine:
    raise RuntimeError("Scale checkpoint contains no hsi_refinement_head tensors")
print(f"[ok] Stage-2 alignment tensors={len(stage2_align)}")
print(f"[ok] scale overlay tensors={len(scale_refine)}")

layer = SMPLLayer(model_dir, gender="neutral").eval()
with torch.no_grad():
    vertices, joints = layer(torch.zeros(1, 72), torch.zeros(1, 10))

if vertices.ndim != 3 or vertices.shape[-1] != 3:
    raise RuntimeError(f"Unexpected SMPL vertices shape: {tuple(vertices.shape)}")
if joints.ndim != 3 or joints.shape[-1] != 3:
    raise RuntimeError(f"Unexpected SMPL joints shape: {tuple(joints.shape)}")

print(f"[ok] SMPL vertices shape={tuple(vertices.shape)}")
print(f"[ok] SMPL joints shape={tuple(joints.shape)}")
PY

echo
echo "Runtime prerequisites passed."
echo "Report: ${REPORT}"
