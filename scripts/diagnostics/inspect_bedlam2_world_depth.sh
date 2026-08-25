#!/usr/bin/env bash
set -euo pipefail

# Read-only validation of BEDLAM2's 4-component WorldDepth EXR payload.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SCENE="${SCENE:-20241213_1_250_rome_tracking}"
LABELS="${LABELS:-/home/zhw/xyb_space/bedlam2/bedlam_data/labels_smpl_6fps/${SCENE}.npz}"
DEPTH_ROOT="${DEPTH_ROOT:-/home/zhw/xyb_space/bedlam2/hf_raw/BEDLAM2-depth/${SCENE}}"
IMGNAME="${IMGNAME:-seq_000002/seq_000002_0000.png}"
EXR_CHANNEL="${EXR_CHANNEL:-FinalImageMovieRenderQueue_WorldDepth}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/debug/bedlam2_world_depth/${SCENE}_coordinate_check.json}"

cd "${REPO_ROOT}"
[[ -f "${LABELS}" ]] || { echo "[ERROR] Missing labels: ${LABELS}" >&2; exit 1; }
[[ -d "${DEPTH_ROOT}/exr_depth" ]] || { echo "[ERROR] Missing EXR depth tree: ${DEPTH_ROOT}/exr_depth" >&2; exit 1; }

mkdir -p "$(dirname "${OUTPUT}")"
python scripts/diagnostics/inspect_bedlam2_world_depth.py \
  --labels "${LABELS}" \
  --depth-root "${DEPTH_ROOT}" \
  --imgname "${IMGNAME}" \
  --exr-channel "${EXR_CHANNEL}" \
  | tee "${OUTPUT}"

echo "Coordinate check: ${OUTPUT}"
