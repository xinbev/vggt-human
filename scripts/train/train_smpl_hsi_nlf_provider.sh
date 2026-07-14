#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_provider_stage1_roi_depth}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
PIN_MEMORY="${PIN_MEMORY:-true}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-64}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESET_EPOCH="${RESET_EPOCH:-false}"

SAVE_SCOPE="${SAVE_SCOPE:-hsi}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
SAVE_LATEST="${SAVE_LATEST:-true}"
SAVE_EPOCH_CHECKPOINT="${SAVE_EPOCH_CHECKPOINT:-false}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-false}"
MONITOR="${MONITOR:-loss_total}"
MONITOR_MODE="${MONITOR_MODE:-min}"

DEPTH_TEACHER_WEIGHT="${DEPTH_TEACHER_WEIGHT:-0.15}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
DEPTH_ERROR_CLIP_M="${DEPTH_ERROR_CLIP_M:-2.0}"
DEPTH_USE_HUMAN_ROI="${DEPTH_USE_HUMAN_ROI:-true}"
DEPTH_ROI_EXPAND="${DEPTH_ROI_EXPAND:-0.85}"
DEPTH_MIN_VALID_PIXELS="${DEPTH_MIN_VALID_PIXELS:-2048}"
ANCHOR_DEPTH_WEIGHT="${ANCHOR_DEPTH_WEIGHT:-0.05}"
ANCHOR_SCENE_XYZ_WEIGHT="${ANCHOR_SCENE_XYZ_WEIGHT:-0.0}"
HSI_POSE_WEIGHT="${HSI_POSE_WEIGHT:-6.0}"
HSI_BETAS_WEIGHT="${HSI_BETAS_WEIGHT:-0.8}"
HSI_TRANSL_WEIGHT="${HSI_TRANSL_WEIGHT:-3.0}"
HSI_JOINTS3D_WEIGHT="${HSI_JOINTS3D_WEIGHT:-16.0}"
HSI_VERTICES_WEIGHT="${HSI_VERTICES_WEIGHT:-4.0}"
HSI_PROJECTED_J2D_WEIGHT="${HSI_PROJECTED_J2D_WEIGHT:-0.35}"
HSI_DELTA_REG_WEIGHT="${HSI_DELTA_REG_WEIGHT:-0.50}"
HSI_NO_WORSE_WEIGHT="${HSI_NO_WORSE_WEIGHT:-1.0}"
HSI_NO_WORSE_MARGIN_M="${HSI_NO_WORSE_MARGIN_M:-0.02}"
HSI_GATE_REG_WEIGHT="${HSI_GATE_REG_WEIGHT:-0.01}"
HSI_CONTACT_WEIGHT="${HSI_CONTACT_WEIGHT:-0.0}"
HSI_FOOT_CONTACT_WEIGHT="${HSI_FOOT_CONTACT_WEIGHT:-0.0}"
HSI_FOOT_SOLE_CONTACT_WEIGHT="${HSI_FOOT_SOLE_CONTACT_WEIGHT:-0.0}"
HSI_SUPPORT_PLANE_CONTACT_WEIGHT="${HSI_SUPPORT_PLANE_CONTACT_WEIGHT:-0.0}"
HSI_SUPPORT_PLANE_WINDOW="${HSI_SUPPORT_PLANE_WINDOW:-9}"
HSI_SUPPORT_PLANE_MIN_POINTS="${HSI_SUPPORT_PLANE_MIN_POINTS:-6}"
HSI_SUPPORT_PLANE_FLOAT_WEIGHT="${HSI_SUPPORT_PLANE_FLOAT_WEIGHT:-0.70}"
HSI_SUPPORT_PLANE_PENETRATION_WEIGHT="${HSI_SUPPORT_PLANE_PENETRATION_WEIGHT:-3.0}"
HSI_TRANSL_VELOCITY_WEIGHT="${HSI_TRANSL_VELOCITY_WEIGHT:-0.0}"
HSI_JOINTS_VELOCITY_WEIGHT="${HSI_JOINTS_VELOCITY_WEIGHT:-0.0}"
HSI_JOINTS_ACCELERATION_WEIGHT="${HSI_JOINTS_ACCELERATION_WEIGHT:-0.0}"
HSI_TEMPORAL_NO_WORSE_WEIGHT="${HSI_TEMPORAL_NO_WORSE_WEIGHT:-0.0}"
HSI_FOOT_SLIDING_WEIGHT="${HSI_FOOT_SLIDING_WEIGHT:-0.0}"
HSI_SCENE_SCALE_TEMPORAL_WEIGHT="${HSI_SCENE_SCALE_TEMPORAL_WEIGHT:-0.0}"
HSI_SCENE_BIAS_TEMPORAL_WEIGHT="${HSI_SCENE_BIAS_TEMPORAL_WEIGHT:-0.0}"
HSI_ENABLE_TEMPORAL_MOMENTUM="${HSI_ENABLE_TEMPORAL_MOMENTUM:-false}"
HSI_TEMPORAL_MOMENTUM_USE_TRACK_IDS="${HSI_TEMPORAL_MOMENTUM_USE_TRACK_IDS:-true}"
HSI_SCENE_AFFINE_MODE="${HSI_SCENE_AFFINE_MODE:-per_frame}"
TRAIN_HSI_SCENE_AFFINE_ONLY="${TRAIN_HSI_SCENE_AFFINE_ONLY:-false}"
FREEZE_HSI_SCENE_AFFINE="${FREEZE_HSI_SCENE_AFFINE:-false}"
HSI_SCENE_LOG_SCALE_MIN="${HSI_SCENE_LOG_SCALE_MIN:--5.0}"
HSI_SCENE_LOG_SCALE_MAX="${HSI_SCENE_LOG_SCALE_MAX:-5.0}"
SMPL_TRACK_ASSIGNMENT_MODE="${SMPL_TRACK_ASSIGNMENT_MODE:-gt}"
SMPL_USE_EXTERNAL_TRACK_PRIOR="${SMPL_USE_EXTERNAL_TRACK_PRIOR:-false}"
PREDICT_ID_EMBED="${PREDICT_ID_EMBED:-false}"
ID_WEIGHT="${ID_WEIGHT:-0.0}"
HSI_SMPL_SCALE_TEACHER_WEIGHT="${HSI_SMPL_SCALE_TEACHER_WEIGHT:-0.0}"
HSI_SMPL_SCALE_TEACHER_SOURCE="${HSI_SMPL_SCALE_TEACHER_SOURCE:-vertices}"
HSI_SMPL_SCALE_TEACHER_USE_BIAS="${HSI_SMPL_SCALE_TEACHER_USE_BIAS:-false}"
HSI_SMPL_SCALE_TEACHER_VIS_TOL_M="${HSI_SMPL_SCALE_TEACHER_VIS_TOL_M:-0.20}"
HSI_SMPL_SCALE_TEACHER_WINDOW="${HSI_SMPL_SCALE_TEACHER_WINDOW:-3}"
HSI_SMPL_SCALE_TEACHER_MAX_POINTS_PER_PERSON="${HSI_SMPL_SCALE_TEACHER_MAX_POINTS_PER_PERSON:-512}"
HSI_SMPL_SCALE_TEACHER_MIN_POINTS_PER_PERSON="${HSI_SMPL_SCALE_TEACHER_MIN_POINTS_PER_PERSON:-32}"
HSI_SMPL_SCALE_TEACHER_MIN_VISIBLE_POINTS="${HSI_SMPL_SCALE_TEACHER_MIN_VISIBLE_POINTS:-128}"
HSI_SMPL_SCALE_TEACHER_MAD_MULT="${HSI_SMPL_SCALE_TEACHER_MAD_MULT:-2.5}"
HSI_SMPL_SCALE_TEACHER_LOG_LOSS="${HSI_SMPL_SCALE_TEACHER_LOG_LOSS:-true}"
HSI_SMPL_SCALE_TEACHER_BIAS_REG_WEIGHT="${HSI_SMPL_SCALE_TEACHER_BIAS_REG_WEIGHT:-0.05}"
HSI_SMPL_SCALE_TEACHER_MAX_Z_M="${HSI_SMPL_SCALE_TEACHER_MAX_Z_M:-0.0}"
PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS:-}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }

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

[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${NLF_CKPT}" ]] || { echo "[ERROR] Missing NLF checkpoint: ${NLF_CKPT}" >&2; exit 1; }
[[ -d "${NLF_ROOT}" ]] || { echo "[ERROR] Missing NLF source directory: ${NLF_ROOT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

echo "========== Frozen NLF + HSI refinement Stage 1 =========="
echo "BEDLAM       : ${BEDLAM_ROOT}"
echo "Boxes        : ${PREPROCESSED_ROOT}"
echo "VGGT ckpt    : ${VGGT_CKPT}"
echo "NLF ckpt     : ${NLF_CKPT}"
echo "NLF root     : ${NLF_ROOT}"
echo "SMPL models  : ${SMPL_MODEL_DIR}"
echo "Output       : ${OUTPUT_DIR}"
echo "Epochs       : ${EPOCHS}"
echo "LR           : ${LR}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Max steps/ep : ${MAX_STEPS_PER_EPOCH}"
echo "Max humans   : ${MAX_HUMANS}"
echo "Num views    : ${NUM_VIEWS}"
echo "Workers      : ${NUM_WORKERS}"
echo "GPU visible  : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "NLF int batch: ${NLF_INTERNAL_BATCH_SIZE}"
echo "Resume ckpt  : ${RESUME_CKPT:-<none>}"
echo "Save scope   : ${SAVE_SCOPE}, top-k=${SAVE_TOP_K}, monitor=${MONITOR}/${MONITOR_MODE}"
echo "Depth ROI    : use=${DEPTH_USE_HUMAN_ROI}, expand=${DEPTH_ROI_EXPAND}, max_m=${DEPTH_MAX_M}, clip=${DEPTH_ERROR_CLIP_M}"
echo "HSI weights  : depth=${DEPTH_TEACHER_WEIGHT}, anchorD=${ANCHOR_DEPTH_WEIGHT}, anchorXYZ=${ANCHOR_SCENE_XYZ_WEIGHT}, transl=${HSI_TRANSL_WEIGHT}"
echo "HSI scale rng: log=[${HSI_SCENE_LOG_SCALE_MIN}, ${HSI_SCENE_LOG_SCALE_MAX}] scale=[exp(min), exp(max)]"
echo "SMPL scale T : weight=${HSI_SMPL_SCALE_TEACHER_WEIGHT}, source=${HSI_SMPL_SCALE_TEACHER_SOURCE}, window=${HSI_SMPL_SCALE_TEACHER_WINDOW}, vis_tol=${HSI_SMPL_SCALE_TEACHER_VIS_TOL_M}, max_z=${HSI_SMPL_SCALE_TEACHER_MAX_Z_M}, log_loss=${HSI_SMPL_SCALE_TEACHER_LOG_LOSS}"
echo "Contact      : foot=${HSI_FOOT_CONTACT_WEIGHT}, sole=${HSI_FOOT_SOLE_CONTACT_WEIGHT}, plane=${HSI_SUPPORT_PLANE_CONTACT_WEIGHT}"
echo "Temporal     : views=${NUM_VIEWS}, momentum=${HSI_ENABLE_TEMPORAL_MOMENTUM}, track_mode=${SMPL_TRACK_ASSIGNMENT_MODE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "checkpoints.nlf_smpl=${NLF_CKPT}" \
  --override "third_party.nlf_root=${NLF_ROOT}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "checkpoint.resume=${RESUME_CKPT}" \
  --override "checkpoint.reset_epoch=${RESET_EPOCH}" \
  --override "checkpoint.save_scope=${SAVE_SCOPE}" \
  --override "checkpoint.save_optimizer=${SAVE_OPTIMIZER}" \
  --override "checkpoint.save_epoch_checkpoint=${SAVE_EPOCH_CHECKPOINT}" \
  --override "checkpoint.save_latest=${SAVE_LATEST}" \
  --override "checkpoint.save_top_k=${SAVE_TOP_K}" \
  --override "checkpoint.save_top_k_from_train=true" \
  --override "checkpoint.monitor=${MONITOR}" \
  --override "checkpoint.monitor_mode=${MONITOR_MODE}" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.val_split=" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "data.pin_memory=${PIN_MEMORY}" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.smpl_provider=nlf" \
  --override "model.nlf_use_detector=false" \
  --override "model.nlf_require_boxes=true" \
  --override "model.nlf_internal_batch_size=${NLF_INTERNAL_BATCH_SIZE}" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_smpl=true" \
  --override "model.enable_hsi_refine=true" \
  --override "model.freeze_aggregator=true" \
  --override "model.freeze_camera_head=true" \
  --override "model.freeze_dense_head=true" \
  --override "model.freeze_aggregator_forward=true" \
  --override "model.smpl_query_box_prior=true" \
  --override "model.smpl_track_assignment_mode=${SMPL_TRACK_ASSIGNMENT_MODE}" \
  --override "model.smpl_use_external_track_prior=${SMPL_USE_EXTERNAL_TRACK_PRIOR}" \
  --override "model.predict_id_embed=${PREDICT_ID_EMBED}" \
  --override "model.hsi_enable_temporal_momentum=${HSI_ENABLE_TEMPORAL_MOMENTUM}" \
  --override "model.hsi_temporal_momentum_use_track_ids=${HSI_TEMPORAL_MOMENTUM_USE_TRACK_IDS}" \
  --override "model.hsi_scene_affine_mode=${HSI_SCENE_AFFINE_MODE}" \
  --override "model.train_hsi_scene_affine_only=${TRAIN_HSI_SCENE_AFFINE_ONLY}" \
  --override "model.freeze_hsi_scene_affine=${FREEZE_HSI_SCENE_AFFINE}" \
  --override "model.hsi_scene_log_scale_min=${HSI_SCENE_LOG_SCALE_MIN}" \
  --override "model.hsi_scene_log_scale_max=${HSI_SCENE_LOG_SCALE_MAX}" \
  --override "loss.hsi_pose_weight=${HSI_POSE_WEIGHT}" \
  --override "loss.hsi_betas_weight=${HSI_BETAS_WEIGHT}" \
  --override "loss.hsi_transl_cam_weight=${HSI_TRANSL_WEIGHT}" \
  --override "loss.hsi_joints3d_weight=${HSI_JOINTS3D_WEIGHT}" \
  --override "loss.hsi_vertices_weight=${HSI_VERTICES_WEIGHT}" \
  --override "loss.hsi_projected_joints2d_weight=${HSI_PROJECTED_J2D_WEIGHT}" \
  --override "loss.hsi_depth_teacher_weight=${DEPTH_TEACHER_WEIGHT}" \
  --override "loss.hsi_depth_teacher_max_m=${DEPTH_MAX_M}" \
  --override "loss.hsi_depth_teacher_error_clip_m=${DEPTH_ERROR_CLIP_M}" \
  --override "loss.hsi_depth_teacher_use_human_roi=${DEPTH_USE_HUMAN_ROI}" \
  --override "loss.hsi_depth_teacher_roi_expand=${DEPTH_ROI_EXPAND}" \
  --override "loss.hsi_depth_teacher_min_valid_pixels=${DEPTH_MIN_VALID_PIXELS}" \
  --override "loss.hsi_smpl_scale_teacher_weight=${HSI_SMPL_SCALE_TEACHER_WEIGHT}" \
  --override "loss.hsi_smpl_scale_teacher_source=${HSI_SMPL_SCALE_TEACHER_SOURCE}" \
  --override "loss.hsi_smpl_scale_teacher_use_bias=${HSI_SMPL_SCALE_TEACHER_USE_BIAS}" \
  --override "loss.hsi_smpl_scale_teacher_visibility_tolerance_m=${HSI_SMPL_SCALE_TEACHER_VIS_TOL_M}" \
  --override "loss.hsi_smpl_scale_teacher_window=${HSI_SMPL_SCALE_TEACHER_WINDOW}" \
  --override "loss.hsi_smpl_scale_teacher_max_points_per_person=${HSI_SMPL_SCALE_TEACHER_MAX_POINTS_PER_PERSON}" \
  --override "loss.hsi_smpl_scale_teacher_min_points_per_person=${HSI_SMPL_SCALE_TEACHER_MIN_POINTS_PER_PERSON}" \
  --override "loss.hsi_smpl_scale_teacher_min_visible_points=${HSI_SMPL_SCALE_TEACHER_MIN_VISIBLE_POINTS}" \
  --override "loss.hsi_smpl_scale_teacher_mad_multiplier=${HSI_SMPL_SCALE_TEACHER_MAD_MULT}" \
  --override "loss.hsi_smpl_scale_teacher_log_loss=${HSI_SMPL_SCALE_TEACHER_LOG_LOSS}" \
  --override "loss.hsi_smpl_scale_teacher_bias_reg_weight=${HSI_SMPL_SCALE_TEACHER_BIAS_REG_WEIGHT}" \
  --override "loss.hsi_smpl_scale_teacher_max_z_m=${HSI_SMPL_SCALE_TEACHER_MAX_Z_M}" \
  --override "loss.hsi_anchor_depth_weight=${ANCHOR_DEPTH_WEIGHT}" \
  --override "loss.hsi_anchor_scene_xyz_weight=${ANCHOR_SCENE_XYZ_WEIGHT}" \
  --override "loss.hsi_delta_reg_weight=${HSI_DELTA_REG_WEIGHT}" \
  --override "loss.hsi_no_worse_weight=${HSI_NO_WORSE_WEIGHT}" \
  --override "loss.hsi_no_worse_margin_m=${HSI_NO_WORSE_MARGIN_M}" \
  --override "loss.hsi_gate_reg_weight=${HSI_GATE_REG_WEIGHT}" \
  --override "loss.hsi_contact_weight=${HSI_CONTACT_WEIGHT}" \
  --override "loss.hsi_foot_contact_weight=${HSI_FOOT_CONTACT_WEIGHT}" \
  --override "loss.hsi_foot_sole_contact_weight=${HSI_FOOT_SOLE_CONTACT_WEIGHT}" \
  --override "loss.hsi_support_plane_contact_weight=${HSI_SUPPORT_PLANE_CONTACT_WEIGHT}" \
  --override "loss.hsi_support_plane_window=${HSI_SUPPORT_PLANE_WINDOW}" \
  --override "loss.hsi_support_plane_min_points=${HSI_SUPPORT_PLANE_MIN_POINTS}" \
  --override "loss.hsi_support_plane_float_weight=${HSI_SUPPORT_PLANE_FLOAT_WEIGHT}" \
  --override "loss.hsi_support_plane_penetration_weight=${HSI_SUPPORT_PLANE_PENETRATION_WEIGHT}" \
  --override "loss.hsi_transl_velocity_weight=${HSI_TRANSL_VELOCITY_WEIGHT}" \
  --override "loss.hsi_joints_velocity_weight=${HSI_JOINTS_VELOCITY_WEIGHT}" \
  --override "loss.hsi_joints_acceleration_weight=${HSI_JOINTS_ACCELERATION_WEIGHT}" \
  --override "loss.hsi_temporal_no_worse_weight=${HSI_TEMPORAL_NO_WORSE_WEIGHT}" \
  --override "loss.hsi_foot_sliding_weight=${HSI_FOOT_SLIDING_WEIGHT}" \
  --override "loss.hsi_scene_scale_temporal_weight=${HSI_SCENE_SCALE_TEMPORAL_WEIGHT}" \
  --override "loss.hsi_scene_bias_temporal_weight=${HSI_SCENE_BIAS_TEMPORAL_WEIGHT}" \
  --override "loss.id_weight=${ID_WEIGHT}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}" \
  --override "optim.log_style=progress" \
  --override "optim.progress_log_keys=${PROGRESS_LOG_KEYS}" \
  --override "optim.batch_size=${BATCH_SIZE}"

echo "========== Frozen NLF + HSI refinement finished =========="
echo "Last checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
