#!/usr/bin/env bash

set -euo pipefail

# Resume the GRAFT-style HSI refinement stage from the latest readable HSI checkpoint.

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
BEDLAM_ROOT="/home/zhw/xyb_space/bedlam/processed_bedlam"
PREPROCESSED_ROOT="${REPO_ROOT}/outputs/preprocess/bedlam_boxes"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_refine.yaml"
CUDA_VISIBLE_DEVICES_VALUE="6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SMPL_MODEL_DIR="/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q"

EXTRA_EPOCHS="20"
LR="5e-6"
MAX_HUMANS="20"
NUM_VIEWS="2"
RESUME_OPTIMIZER="true"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

RESUME_INFO=$(python - "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
candidates = []
latest = output_dir / "checkpoint_latest.pt"
if latest.exists():
    candidates.append(latest)
candidates.extend(
    sorted(
        output_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
)

seen = set()
for path in candidates:
    if path in seen:
        continue
    seen.add(path)
    try:
        checkpoint = torch.load(path, map_location="cpu")
        epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
        if epoch <= 0:
            print(f"[skip] checkpoint has no valid epoch: {path}", file=sys.stderr)
            continue
        print(f"{path} {epoch}")
        raise SystemExit(0)
    except Exception as exc:
        print(f"[skip] unreadable checkpoint: {path} ({exc})", file=sys.stderr)

raise SystemExit(f"No readable checkpoint found in {output_dir}")
PY
)

RESUME_CKPT="$(echo "${RESUME_INFO}" | awk '{print $1}')"
RESUME_EPOCH="$(echo "${RESUME_INFO}" | awk '{print $2}')"
TOTAL_EPOCHS=$((RESUME_EPOCH + EXTRA_EPOCHS))

echo "========== Resume SMPL HSI GRAFT-style Refinement =========="
echo "BEDLAM          : ${BEDLAM_ROOT}"
echo "Boxes           : ${PREPROCESSED_ROOT}"
echo "SMPL models     : ${SMPL_MODEL_DIR}"
echo "Resume ckpt     : ${RESUME_CKPT}"
echo "Resume epoch    : ${RESUME_EPOCH}"
echo "Extra epochs    : ${EXTRA_EPOCHS}"
echo "Total epochs    : ${TOTAL_EPOCHS}"
echo "Output          : ${OUTPUT_DIR}"
echo "LR              : ${LR}"
echo "Max humans      : ${MAX_HUMANS}"
echo "Num views       : ${NUM_VIEWS}"
echo "Resume optimizer: ${RESUME_OPTIMIZER}"
df -h "${OUTPUT_DIR}" || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "checkpoint.load_vggt_baseline=false" \
  --override "checkpoint.resume=${RESUME_CKPT}" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=${RESUME_OPTIMIZER}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.val_split=" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.freeze_aggregator=true" \
  --override "model.freeze_camera_head=true" \
  --override "model.freeze_dense_head=true" \
  --override "model.train_smpl_query_token=true" \
  --override "model.train_smpl_box_prior_embed=true" \
  --override "model.train_smpl_patch_pool_embed=true" \
  --override "model.predict_boxes=true" \
  --override "model.smpl_bbox_mode=reference_residual" \
  --override "model.smpl_return_aux=true" \
  --override "model.smpl_query_box_prior=true" \
  --override "model.smpl_query_patch_pool=true" \
  --override "model.smpl_query_patch_pool_expand=0.12" \
  --override "model.hsi_hidden_dim=512" \
  --override "model.hsi_num_layers=5" \
  --override "model.hsi_num_heads=8" \
  --override "model.hsi_num_iters=3" \
  --override "model.hsi_scene_window=3" \
  --override "loss.hsi_pose_weight=6.0" \
  --override "loss.hsi_betas_weight=0.8" \
  --override "loss.hsi_transl_cam_weight=3.0" \
  --override "loss.hsi_joints3d_weight=16.0" \
  --override "loss.hsi_projected_joints2d_weight=0.35" \
  --override "loss.hsi_depth_teacher_weight=0.20" \
  --override "loss.hsi_anchor_depth_weight=0.10" \
  --override "loss.hsi_contact_weight=0.05" \
  --override "optim.epochs=${TOTAL_EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.batch_size=1" \
  --override "optim.log_interval=20"

echo "========== Resume SMPL HSI refinement finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
