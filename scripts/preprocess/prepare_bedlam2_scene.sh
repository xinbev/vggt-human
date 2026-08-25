#!/usr/bin/env bash
set -euo pipefail

# Materialize BEDLAM2 into the established BedlamDataset per-frame layout.
# First run with INSPECT_ONLY=true and confirm DEPTH_SCALE from its EXR report.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SCENE="${SCENE:-20241213_1_250_rome_tracking}"
RGB_ROOT="${RGB_ROOT:-/home/zhw/xyb_space/bedlam2/hf_raw/BEDLAM2/${SCENE}}"
DEPTH_ROOT="${DEPTH_ROOT:-/home/zhw/xyb_space/bedlam2/hf_raw/BEDLAM2-depth/${SCENE}}"
LABELS="${LABELS:-/home/zhw/xyb_space/bedlam2/bedlam_data/labels_smpl_6fps/${SCENE}.npz}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/outputs/preprocess/bedlam2_processed}"
DEPTH_SCALE="${DEPTH_SCALE:-}"
COPY_MODE="${COPY_MODE:-hardlink}"
INSPECT_ONLY="${INSPECT_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
OVERWRITE="${OVERWRITE:-false}"
SEQUENCE="${SEQUENCE:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"
EXR_CHANNEL="${EXR_CHANNEL:-}"

cd "${REPO_ROOT}"

[[ -d "${RGB_ROOT}/png" ]] || { echo "[ERROR] Missing RGB png directory: ${RGB_ROOT}/png" >&2; exit 1; }
[[ -d "${DEPTH_ROOT}/exr_depth" ]] || { echo "[ERROR] Missing depth EXR directory: ${DEPTH_ROOT}/exr_depth" >&2; exit 1; }
[[ -f "${LABELS}" ]] || { echo "[ERROR] Missing SMPL labels: ${LABELS}" >&2; exit 1; }

ARGS=(
  --rgb-root "${RGB_ROOT}"
  --depth-root "${DEPTH_ROOT}"
  --labels "${LABELS}"
  --outdir "${OUTDIR}"
  --scene "${SCENE}"
  --copy-mode "${COPY_MODE}"
)

if [[ -n "${SEQUENCE}" ]]; then ARGS+=(--sequence "${SEQUENCE}"); fi
if [[ "${MAX_FRAMES}" != "0" ]]; then ARGS+=(--max-frames "${MAX_FRAMES}"); fi
if [[ "${OVERWRITE}" == "true" ]]; then ARGS+=(--overwrite); fi
if [[ "${DRY_RUN}" == "true" ]]; then ARGS+=(--dry-run); fi
if [[ -n "${EXR_CHANNEL}" ]]; then ARGS+=(--exr-channel "${EXR_CHANNEL}"); fi

if [[ "${INSPECT_ONLY}" == "true" ]]; then
  echo "========== BEDLAM2 EXR inspection =========="
  python scripts/preprocess/prepare_bedlam2_scene.py "${ARGS[@]}" --depth-scale 1.0 --inspect-only
  echo "Inspect raw_depth_positive_median. Set DEPTH_SCALE=0.01 only after confirming the EXR values are centimetres."
  exit 0
fi

[[ -n "${DEPTH_SCALE}" ]] || {
  echo "[ERROR] DEPTH_SCALE is required for conversion. Run INSPECT_ONLY=true first, then set (for example) DEPTH_SCALE=0.01." >&2
  exit 1
}

echo "========== Prepare BEDLAM2 scene =========="
echo "Scene       : ${SCENE}"
echo "RGB root    : ${RGB_ROOT}"
echo "Depth root  : ${DEPTH_ROOT}"
echo "SMPL labels : ${LABELS}"
echo "Output      : ${OUTDIR}"
echo "Depth scale : ${DEPTH_SCALE} (raw EXR -> metres)"
echo "Copy mode   : ${COPY_MODE}"

python scripts/preprocess/prepare_bedlam2_scene.py "${ARGS[@]}" --depth-scale "${DEPTH_SCALE}"

echo "========== BEDLAM2 conversion complete =========="
echo "Processed root: ${OUTDIR}"
echo "Next: BEDLAM_ROOT=${OUTDIR} bash scripts/preprocess/prepare_bedlam_boxes.sh"
