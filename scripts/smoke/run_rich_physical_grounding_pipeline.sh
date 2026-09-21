#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
RICH_ROOT="${RICH_ROOT:-/home/zhw/xyb_space/RICH/official}"
RICH_SEQUENCE="${RICH_SEQUENCE:-Gym_010_cooking1}"
RICH_CAMERA="${RICH_CAMERA:-}"

if [[ -z "${FRAMES_DIR:-}" ]]; then
  RICH_SEQUENCE_DIR="${RICH_ROOT}/test/${RICH_SEQUENCE}"
  [[ -d "${RICH_SEQUENCE_DIR}" ]] || {
    echo "[ERROR] Missing RICH sequence: ${RICH_SEQUENCE_DIR}" >&2
    exit 1
  }
  if [[ -z "${RICH_CAMERA}" ]]; then
    shopt -s nullglob
    camera_dirs=("${RICH_SEQUENCE_DIR}"/cam_*)
    shopt -u nullglob
    if [[ "${#camera_dirs[@]}" -eq 0 ]]; then
      echo "[ERROR] No cam_* directories under: ${RICH_SEQUENCE_DIR}" >&2
      exit 1
    fi
    mapfile -t camera_dirs < <(printf '%s\n' "${camera_dirs[@]}" | sort)
    FRAMES_DIR="${camera_dirs[0]}"
    RICH_CAMERA="$(basename "${FRAMES_DIR}")"
  else
    FRAMES_DIR="${RICH_SEQUENCE_DIR}/${RICH_CAMERA}"
  fi
else
  FRAMES_DIR="${FRAMES_DIR}"
  if [[ -z "${RICH_CAMERA}" ]]; then
    RICH_CAMERA="$(basename "${FRAMES_DIR%/}")"
  fi
fi

STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
SMOKE_FRAMES="${SMOKE_FRAMES:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/rich_physical_grounding/pipeline_smoke/${RICH_SEQUENCE}_${RICH_CAMERA}}"
FULL_VIEWER_CACHE_OUTPUT="${FULL_VIEWER_CACHE_OUTPUT:-${OUTPUT_DIR}/full_viewer_cache}"
GEOMETRY_REPORT="${GEOMETRY_REPORT:-${OUTPUT_DIR}/smoke_geometry_report.json}"

cd "${REPO_ROOT}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing RICH frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage-2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }
if [[ "${SMOKE_FRAMES}" -lt 1 ]]; then
  echo "[ERROR] SMOKE_FRAMES must be at least 1" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "RICH sequence : ${RICH_SEQUENCE}/${RICH_CAMERA}"
echo "Frames        : ${FRAMES_DIR}"
echo "Smoke frames  : ${SMOKE_FRAMES}"
echo "Output        : ${OUTPUT_DIR}"

FRAMES_DIR="${FRAMES_DIR}" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
SCALE_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FULL_VIEWER_CACHE_OUTPUT="${FULL_VIEWER_CACHE_OUTPUT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
START_INDEX=0 \
END_INDEX="$((SMOKE_FRAMES - 1))" \
FRAME_STRIDE=1 \
INFERENCE_FRAMES="${SMOKE_FRAMES}" \
MAX_FRAMES="${SMOKE_FRAMES}" \
FRAME_SAMPLING=head \
HUMAN_MASK_DILATION_PX=18 \
SMPL_DISPLAY_FRAMES=0 \
DISPLAY_PEOPLE=4 \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh

python scripts/diagnostics/inspect_rich_physical_grounding_smoke.py \
  --cache-dir "${FULL_VIEWER_CACHE_OUTPUT}" \
  --output "${GEOMETRY_REPORT}"

echo "[ok] RICH end-to-end pipeline smoke passed"
