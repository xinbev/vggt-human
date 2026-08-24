#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree}"
CHECKPOINT="${CHECKPOINT:-${STAGE_DIR}/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_gt_depth_scale_nlf_detector}"

[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing scale checkpoint: ${CHECKPOINT}" >&2; exit 1; }

echo "========== HSI GT-depth scale / NLF-detector inference =========="
echo "Repo       : ${REPO_ROOT}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Human input: NLF internal detector (no sidecar boxes)"
echo "Output     : ${OUTPUT_DIR}"

REPO_ROOT="${REPO_ROOT}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml" \
STAGE2_DIR="${STAGE_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
