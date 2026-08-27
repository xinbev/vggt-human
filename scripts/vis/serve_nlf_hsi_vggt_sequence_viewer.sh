#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl}"
FRAMES_DIR="${FRAMES_DIR:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb}"
QUERY_SOURCE="${QUERY_SOURCE:-bedlam_sidecar}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_vggt_sequence_viewer}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-32}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
TRACKING_OVERLAY="${TRACKING_OVERLAY:-none}"
TRACK_MAX_AGE="${TRACK_MAX_AGE:-90}"
TRACK_MIN_QUALITY="${TRACK_MIN_QUALITY:-0.25}"
TRACK_MAX_CENTER_DISTANCE="${TRACK_MAX_CENTER_DISTANCE:-0.25}"
TRACK_MAX_TRANSL_DISTANCE="${TRACK_MAX_TRANSL_DISTANCE:-1.50}"
TRACK_MAX_BETA_L1="${TRACK_MAX_BETA_L1:-0.30}"
SHOW_TRACK_IDS="${SHOW_TRACK_IDS:-true}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
VIEWER_MODE="${VIEWER_MODE:-4d}"
ENVIRONMENT_DISPLAY="${ENVIRONMENT_DISPLAY:-points}"
HSI_VISUAL_SCALE="${HSI_VISUAL_SCALE:-1.0}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-5}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-true}"
ENV_MESH_DEPTH_EDGE_RTOL="${ENV_MESH_DEPTH_EDGE_RTOL:-0.15}"
ENV_MESH_COLOR_GROUPS="${ENV_MESH_COLOR_GROUPS:-216}"
ENV_MESH_COLOR_MODE="${ENV_MESH_COLOR_MODE:-point_overlay}"
ENV_MESH_OVERLAY_POINT_SIZE_SCALE="${ENV_MESH_OVERLAY_POINT_SIZE_SCALE:-0.75}"
POINT_SIZE="${POINT_SIZE:-0.012}"
CAMERA_FRUSTUM_SCALE="${CAMERA_FRUSTUM_SCALE:-0.20}"
ALIGNMENT_VERTEX_STRIDE="${ALIGNMENT_VERTEX_STRIDE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-}"
HSI_ALIGN_FEATURE_VERSION="${HSI_ALIGN_FEATURE_VERSION:-}"
HSI_OVERLAY_CHECKPOINT="${HSI_OVERLAY_CHECKPOINT:-}"
SCENE_SCALE_PREALIGN="${SCENE_SCALE_PREALIGN:-none}"
COARSE_SCALE_MIN="${COARSE_SCALE_MIN:-0.10}"
COARSE_SCALE_MAX="${COARSE_SCALE_MAX:-10.0}"
COARSE_ANCHOR_STRIDE="${COARSE_ANCHOR_STRIDE:-8}"
COARSE_MIN_ANCHOR_PIXELS="${COARSE_MIN_ANCHOR_PIXELS:-32}"
COARSE_FALLBACK="${COARSE_FALLBACK:-unit}"
CASCADE_EFFECTIVE_AFFINE_MODE="${CASCADE_EFFECTIVE_AFFINE_MODE:-per_frame}"
SMPL_USE_AGGREGATOR_QUERIES="${SMPL_USE_AGGREGATOR_QUERIES:-}"
HSI_SCENE_AFFINE_MODE="${HSI_SCENE_AFFINE_MODE:-}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames dir: ${FRAMES_DIR}" >&2; exit 1; }
[[ -d "${STAGE2_DIR}" ]] || { echo "[ERROR] Missing stage2 dir: ${STAGE2_DIR}" >&2; exit 1; }
case "${ENVIRONMENT_DISPLAY}" in
  points|mesh|both) ;;
  *) echo "[ERROR] ENVIRONMENT_DISPLAY must be one of: points, mesh, both. Got: ${ENVIRONMENT_DISPLAY}" >&2; exit 1 ;;
esac
case "${VIEWER_MODE}" in
  4d|"4D current frame") VIEWER_MODE_ARG="4D current frame" ;;
  3d|"3D accumulate") VIEWER_MODE_ARG="3D accumulate" ;;
  hybrid|Hybrid) VIEWER_MODE_ARG="Hybrid" ;;
  *) echo "[ERROR] VIEWER_MODE must be one of: 4d, 3d, hybrid. Got: ${VIEWER_MODE}" >&2; exit 1 ;;
esac
if [[ "${QUERY_SOURCE}" == "bedlam_sidecar" ]]; then
  [[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
  [[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed sidecars: ${PREPROCESSED_ROOT}" >&2; exit 1; }
fi

echo "========== NLF-HSI VGGT sequence Viser viewer =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Query source: ${QUERY_SOURCE}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Sidecars    : ${PREPROCESSED_ROOT}"
echo "Stage2 dir  : ${STAGE2_DIR}"
echo "Checkpoint  : ${CHECKPOINT:-<rank1 from checkpoint_topk_index.json>}"
echo "Output      : ${OUTPUT_DIR}"
echo "Port        : ${PORT}"
echo "Max frames  : ${MAX_FRAMES}"
echo "ID overlay  : ${TRACKING_OVERLAY} (post-HSI display only)"
echo "Show IDs    : ${SHOW_TRACK_IDS} (initial GUI state)"
echo "Align compat: ${HSI_ALIGN_FEATURE_VERSION:-<config default>}"
echo "Depth stride: ${DEPTH_POINT_STRIDE} (can be changed in Viser GUI)"
echo "Max depth   : ${MAX_SCENE_DEPTH} (0 disables clipping; GUI adjustable)"
echo "Viewer mode : ${VIEWER_MODE_ARG} (initial GUI state)"
echo "Env display : ${ENVIRONMENT_DISPLAY} (points|mesh|both; GUI adjustable)"
echo "HSI vis scale: ${HSI_VISUAL_SCALE} (viewer-only; GUI adjustable)"
echo "Scale prealign: ${SCENE_SCALE_PREALIGN}"
if [[ "${SCENE_SCALE_PREALIGN}" == "smpl_median" ]]; then
  echo "Coarse scale : range=[${COARSE_SCALE_MIN},${COARSE_SCALE_MAX}] stride=${COARSE_ANCHOR_STRIDE} min_pixels=${COARSE_MIN_ANCHOR_PIXELS}"
  echo "Coarse fallback: ${COARSE_FALLBACK}"
fi
echo "HSI overlay : ${HSI_OVERLAY_CHECKPOINT:-<none>}"
echo "Human mask  : projected SMPL triangles + ${HUMAN_MASK_DILATION_PX}px dilation (unconditional removal)"
echo "Filter human: ${FILTER_HUMAN_POINTS} (initial GUI state)"
if [[ "${ENVIRONMENT_DISPLAY}" != "points" ]]; then
  echo "Mesh edge   : ${ENV_MESH_DEPTH_EDGE_RTOL} (relative depth jump cutoff)"
  echo "Mesh colors : ${ENV_MESH_COLOR_GROUPS} simple-mesh color groups"
  echo "Mesh color  : ${ENV_MESH_COLOR_MODE} (point_overlay matches Human3R point colors)"
fi
echo "Point size  : ${POINT_SIZE}"
echo "SMPL edits  : ${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Smoke only  : ${SMOKE_ONLY}"

ARGS=(
  --frames-dir "${FRAMES_DIR}"
  --query-source "${QUERY_SOURCE}"
  --preprocessed-root "${PREPROCESSED_ROOT}"
  --bedlam-root "${BEDLAM_ROOT}"
  --stage2-dir "${STAGE2_DIR}"
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --port "${PORT}"
  --max-frames "${MAX_FRAMES}"
  --start-index "${START_INDEX}"
  --frame-stride "${FRAME_STRIDE}"
  --max-humans "${MAX_HUMANS}"
  --conf-threshold "${CONF_THRESHOLD}"
  --tracking-overlay "${TRACKING_OVERLAY}"
  --track-max-age "${TRACK_MAX_AGE}"
  --track-min-quality "${TRACK_MIN_QUALITY}"
  --track-max-center-distance "${TRACK_MAX_CENTER_DISTANCE}"
  --track-max-transl-distance "${TRACK_MAX_TRANSL_DISTANCE}"
  --track-max-beta-l1 "${TRACK_MAX_BETA_L1}"
  --depth-point-stride "${DEPTH_POINT_STRIDE}"
  --max-scene-depth "${MAX_SCENE_DEPTH}"
  --viewer-mode "${VIEWER_MODE_ARG}"
  --environment-display "${ENVIRONMENT_DISPLAY}"
  --hsi-visual-scale "${HSI_VISUAL_SCALE}"
  --scene-scale-prealign "${SCENE_SCALE_PREALIGN}"
  --coarse-scale-min "${COARSE_SCALE_MIN}"
  --coarse-scale-max "${COARSE_SCALE_MAX}"
  --coarse-anchor-stride "${COARSE_ANCHOR_STRIDE}"
  --coarse-min-anchor-pixels "${COARSE_MIN_ANCHOR_PIXELS}"
  --coarse-fallback "${COARSE_FALLBACK}"
  --cascade-effective-affine-mode "${CASCADE_EFFECTIVE_AFFINE_MODE}"
  --human-mask-dilation-px "${HUMAN_MASK_DILATION_PX}"
  --env-mesh-depth-edge-rtol "${ENV_MESH_DEPTH_EDGE_RTOL}"
  --env-mesh-color-groups "${ENV_MESH_COLOR_GROUPS}"
  --env-mesh-color-mode "${ENV_MESH_COLOR_MODE}"
  --env-mesh-overlay-point-size-scale "${ENV_MESH_OVERLAY_POINT_SIZE_SCALE}"
  --point-size "${POINT_SIZE}"
  --camera-frustum-scale "${CAMERA_FRUSTUM_SCALE}"
  --alignment-vertex-stride "${ALIGNMENT_VERTEX_STRIDE}"
  --image-size "${IMAGE_SIZE}"
)

case "${SHOW_TRACK_IDS}" in
  0|false|FALSE|False|no|NO|No|off|OFF|Off)
    ARGS+=(--no-show-track-ids)
    ;;
  *)
    ARGS+=(--show-track-ids)
    ;;
esac

case "${FILTER_HUMAN_POINTS}" in
  0|false|FALSE|False|no|NO|No|off|OFF|Off)
    ARGS+=(--no-filter-human-points)
    ;;
  *)
    ARGS+=(--filter-human-points)
    ;;
esac

if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BASELINE_CHECKPOINT}" ]]; then
  ARGS+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
fi
if [[ -n "${HSI_OVERLAY_CHECKPOINT}" ]]; then
  ARGS+=(--hsi-overlay-checkpoint "${HSI_OVERLAY_CHECKPOINT}")
fi
if [[ -n "${SMPL_MODEL_DIR}" ]]; then
  ARGS+=(--smpl-model-dir "${SMPL_MODEL_DIR}")
fi
if [[ -n "${SMPL_EDIT_OUTPUT}" ]]; then
  ARGS+=(--smpl-edit-output "${SMPL_EDIT_OUTPUT}")
fi
if [[ -n "${HSI_ALIGN_FEATURE_VERSION}" ]]; then
  ARGS+=(--override "model.hsi_align_feature_version=${HSI_ALIGN_FEATURE_VERSION}")
fi
if [[ -n "${SMPL_USE_AGGREGATOR_QUERIES}" ]]; then
  ARGS+=(--override "model.smpl_use_aggregator_queries=${SMPL_USE_AGGREGATOR_QUERIES}")
fi
if [[ -n "${HSI_SCENE_AFFINE_MODE}" ]]; then
  ARGS+=(--override "model.hsi_scene_affine_mode=${HSI_SCENE_AFFINE_MODE}")
fi
if [[ "${SMOKE_ONLY}" == "1" || "${SMOKE_ONLY}" == "true" || "${SMOKE_ONLY}" == "TRUE" ]]; then
  ARGS+=(--smoke-only)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py "${ARGS[@]}"
