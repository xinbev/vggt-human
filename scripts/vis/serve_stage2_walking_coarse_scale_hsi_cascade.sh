#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3}"
VIEWER_CACHE_OUTPUT="${VIEWER_CACHE_OUTPUT:-}"
FULL_VIEWER_CACHE_OUTPUT="${FULL_VIEWER_CACHE_OUTPUT:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-20}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
INFERENCE_FRAMES="${INFERENCE_FRAMES:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FRAME_SAMPLING="${FRAME_SAMPLING:-head}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-0}"
DISPLAY_PEOPLE="${DISPLAY_PEOPLE:-0}"
MAX_HUMANS="${MAX_HUMANS:-8}"
NLF_DETECTOR_THRESHOLD="${NLF_DETECTOR_THRESHOLD:-0.3}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
COARSE_CONF_THRESHOLD="${COARSE_CONF_THRESHOLD:-}"
COARSE_MAX_PEOPLE="${COARSE_MAX_PEOPLE:-0}"
CASCADE_EFFECTIVE_AFFINE_MODE="${CASCADE_EFFECTIVE_AFFINE_MODE:-clip_median}"
POINT_SIZE="${POINT_SIZE:-0.006}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-12}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-true}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing walking frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing current scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }

echo "========== Sequence Viser: analytic coarse -> current HSI -> Stage2 align =========="
echo "Frames            : ${FRAMES_DIR}"
echo "Stage2 checkpoint : ${CHECKPOINT}"
echo "Scale overlay     : ${SCALE_CHECKPOINT}"
echo "Output            : ${OUTPUT_DIR}"
echo "Viewer cache      : ${VIEWER_CACHE_OUTPUT:-<disabled>}"
echo "Full UI cache     : ${FULL_VIEWER_CACHE_OUTPUT:-<disabled>}"
echo "Port              : ${PORT}"
echo "Frame range       : [${START_INDEX}, ${END_INDEX}] inclusive"
echo "Inference frames  : ${INFERENCE_FRAMES:-<legacy sampling>}"
echo "Frame sampling    : ${FRAME_SAMPLING}, max=${MAX_FRAMES}, stride=${FRAME_STRIDE}"
echo "Accumulated SMPL  : ${SMPL_DISPLAY_FRAMES} (0 means all)"
echo "Displayed people  : ${DISPLAY_PEOPLE} per frame (0 means all)"
echo "Maximum candidates: ${MAX_HUMANS} per frame"
echo "NLF detector conf : ${NLF_DETECTOR_THRESHOLD}"
echo "Result conf       : ${CONF_THRESHOLD}"
echo "Coarse people     : conf=${COARSE_CONF_THRESHOLD:-${CONF_THRESHOLD}}, max=${COARSE_MAX_PEOPLE}"
echo "Scale application : ${CASCADE_EFFECTIVE_AFFINE_MODE}"
echo "Point size        : ${POINT_SIZE}"
echo "Human filter      : ${FILTER_HUMAN_POINTS}, dilation=${HUMAN_MASK_DILATION_PX}px"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
HSI_OVERLAY_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VIEWER_CACHE_OUTPUT="${VIEWER_CACHE_OUTPUT}" \
FULL_VIEWER_CACHE_OUTPUT="${FULL_VIEWER_CACHE_OUTPUT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
START_INDEX="${START_INDEX}" \
END_INDEX="${END_INDEX}" \
INFERENCE_FRAMES="${INFERENCE_FRAMES}" \
FRAME_STRIDE="${FRAME_STRIDE}" \
FRAME_SAMPLING="${FRAME_SAMPLING}" \
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES}" \
DISPLAY_PEOPLE="${DISPLAY_PEOPLE}" \
MAX_HUMANS="${MAX_HUMANS}" \
NLF_DETECTOR_THRESHOLD="${NLF_DETECTOR_THRESHOLD}" \
CONF_THRESHOLD="${CONF_THRESHOLD}" \
SMPL_USE_AGGREGATOR_QUERIES=false \
HSI_SCENE_AFFINE_MODE=per_frame \
CASCADE_EFFECTIVE_AFFINE_MODE="${CASCADE_EFFECTIVE_AFFINE_MODE}" \
SCENE_SCALE_PREALIGN=smpl_median \
COARSE_SCALE_MIN="${COARSE_SCALE_MIN:-0.10}" \
COARSE_SCALE_MAX="${COARSE_SCALE_MAX:-10.0}" \
COARSE_CONF_THRESHOLD="${COARSE_CONF_THRESHOLD}" \
COARSE_MAX_PEOPLE="${COARSE_MAX_PEOPLE}" \
COARSE_ANCHOR_STRIDE="${COARSE_ANCHOR_STRIDE:-8}" \
COARSE_MIN_ANCHOR_PIXELS="${COARSE_MIN_ANCHOR_PIXELS:-32}" \
COARSE_FALLBACK=sequence_median \
DEPTH_POINT_STRIDE=2 \
MAX_SCENE_DEPTH=80 \
VIEWER_MODE="${VIEWER_MODE:-hybrid}" \
ENVIRONMENT_DISPLAY="${ENVIRONMENT_DISPLAY:-points}" \
HSI_VISUAL_SCALE=1.0 \
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX}" \
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS}" \
POINT_SIZE="${POINT_SIZE}" \
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}" \
HSI_ALIGN_FEATURE_VERSION=legacy_scale_bias_v0 \
TRACKING_OVERLAY=base_smpl \
SHOW_TRACK_IDS="${SHOW_TRACK_IDS:-true}" \
TRACK_MAX_AGE=90 \
TRACK_MIN_QUALITY=0.25 \
TRACK_MAX_CENTER_DISTANCE=0.25 \
TRACK_MAX_TRANSL_DISTANCE=1.50 \
TRACK_MAX_BETA_L1=0.30 \
SMOKE_ONLY="${SMOKE_ONLY:-false}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
