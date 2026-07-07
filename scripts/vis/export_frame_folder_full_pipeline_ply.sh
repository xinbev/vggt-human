#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${FRAMES_DIR:-}" ]]; then
  echo "[ERROR] Set FRAMES_DIR=/path/to/frame_folder" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "[ERROR] Set CHECKPOINT=/path/to/sam2_3dpw_smpl_checkpoint.pt" >&2
  exit 1
fi

REQUESTED_DEVICE="${DEVICE:-cuda}"
RUN_DEVICE="${REQUESTED_DEVICE}"
if [[ "${REQUESTED_DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  RUN_DEVICE="cuda"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE_NAME="${SOURCE_NAME:-$(basename "${FRAMES_DIR}")}"
PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.yaml}"
TRACK_ROOT="${TRACK_ROOT:-outputs/preprocess/full_pipeline_frame_tracks}"
TRACK_SUBDIR="${TRACK_SUBDIR:-${SOURCE_NAME}}"
SIDECAR_ROOT="${SIDECAR_ROOT:-${TRACK_ROOT}/${TRACK_SUBDIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vis/full_pipeline_frame_folder_ply/${SOURCE_NAME}}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
OVERWRITE_TRACKS="${OVERWRITE_TRACKS:-0}"
OVERWRITE_TRACKS_FLAG=()
if [[ "${OVERWRITE_TRACKS}" == "1" ]]; then
  OVERWRITE_TRACKS_FLAG+=(--overwrite)
fi

echo "========== Full frame-folder VGGT-Omega pipeline PLY export =========="
echo "Frames dir      : ${FRAMES_DIR}"
echo "Source name     : ${SOURCE_NAME}"
echo "Train config    : ${TRAIN_CONFIG}"
echo "Checkpoint      : ${CHECKPOINT}"
echo "Tracking root   : ${SIDECAR_ROOT}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "Device          : ${RUN_DEVICE}"

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  echo "========== Stage 1: YOLO + BoostTrack + SAM2 sidecar preprocessing =========="
  python scripts/preprocess/prepare_video_person_tracks.py \
    --frames-dir "${FRAMES_DIR}" \
    --source-name "${SOURCE_NAME}" \
    --output-root "${TRACK_ROOT}" \
    --output-subdir "${TRACK_SUBDIR}" \
    --path-config "${PATH_CONFIG}" \
    --device "${RUN_DEVICE}" \
    --detector-image-size "${DETECTOR_IMAGE_SIZE:-640}" \
    --det-conf "${DET_CONF:-0.25}" \
    --det-iou "${DET_IOU:-0.70}" \
    --max-age "${MAX_AGE:-90}" \
    --min-hits "${MIN_HITS:-1}" \
    --aspect-ratio-thresh "${ASPECT_RATIO_THRESH:-10.0}" \
    --stitch-max-gap "${STITCH_MAX_GAP:-30}" \
    --stitch-center-thresh "${STITCH_CENTER_THRESH:-1.25}" \
    --stitch-size-log-thresh "${STITCH_SIZE_LOG_THRESH:-0.70}" \
    --stitch-min-score "${STITCH_MIN_SCORE:-0.25}" \
    --frame-log-interval "${TRACK_LOG_INTERVAL:-50}" \
    --enable-sam2-masks \
    "${OVERWRITE_TRACKS_FLAG[@]}"
fi

echo "========== Stage 2: one sequence VGGT/SMPL forward + per-frame PLY export =========="
ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --sidecar-root "${SIDECAR_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${RUN_DEVICE}"
  --max-export-people "${MAX_EXPORT_PEOPLE:-10}"
  --conf-threshold "${CONF_THRESHOLD:-0.05}"
  --depth-point-stride "${DEPTH_POINT_STRIDE:-2}"
  --max-scene-depth "${MAX_SCENE_DEPTH:-30.0}"
  --coordinate-frame "${COORDINATE_FRAME:-world}"
  --log-interval "${EXPORT_LOG_INTERVAL:-20}"
)

if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${FRAME_STRIDE:-}" ]]; then
  ARGS+=(--frame-stride "${FRAME_STRIDE}")
fi
if [[ -n "${IMAGE_SIZE:-}" ]]; then
  ARGS+=(--image-size "${IMAGE_SIZE}")
fi
if [[ "${EXPORT_COMBINED_FRAME:-1}" == "1" ]]; then
  ARGS+=(--export-combined-frame)
fi
if [[ "${USE_HSI_REFINED:-0}" == "1" ]]; then
  ARGS+=(--use-hsi-refined)
fi
if [[ -n "${SMPL_MODEL_DIR:-}" ]]; then
  ARGS+=(--smpl-model-dir "${SMPL_MODEL_DIR}")
fi
if [[ -n "${BASELINE_CHECKPOINT:-}" ]]; then
  ARGS+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
fi
if [[ -n "${OVERRIDE:-}" ]]; then
  ARGS+=(--override "${OVERRIDE}")
fi

python scripts/vis/export_frame_folder_full_pipeline_ply.py "${ARGS[@]}"

echo "========== Done =========="
echo "Manifest: ${OUTPUT_DIR}/manifest.json"
