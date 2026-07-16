#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
export REPO_ROOT

STAGE2_CKPT="${STAGE2_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt}"
export RESUME_CKPT="${RESUME_CKPT:-${STAGE2_CKPT}}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_stage3_contact_refine.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage3_contact_refine}"
export RESET_EPOCH="${RESET_EPOCH:-true}"

export CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-128}"
export MAX_HUMANS="${MAX_HUMANS:-20}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export EPOCHS="${EPOCHS:-3}"
export LR="${LR:-2e-6}"
export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

# Keep the learned metric scene scale fixed and only allow light human/contact
# refinement.  The last two HSI blocks are enough capacity for contact cleanup
# without unfreezing the full HSI scene encoder.
export FREEZE_HSI_SCENE_AFFINE="${FREEZE_HSI_SCENE_AFFINE:-true}"
export FREEZE_HSI_BACKBONE="${FREEZE_HSI_BACKBONE:-true}"
export TRAIN_HSI_LAST_BLOCKS="${TRAIN_HSI_LAST_BLOCKS:-2}"
export FREEZE_HSI_BETAS_DELTA="${FREEZE_HSI_BETAS_DELTA:-true}"
export FREEZE_HSI_HUMAN_SCENE_ALIGN="${FREEZE_HSI_HUMAN_SCENE_ALIGN:-false}"
export HSI_SCENE_AFFINE_MODE="${HSI_SCENE_AFFINE_MODE:-clip_median}"
export HSI_USE_AFFINE_DEPTH_FOR_TRANSL="${HSI_USE_AFFINE_DEPTH_FOR_TRANSL:-true}"
export HSI_AFFINE_DEPTH_DETACH="${HSI_AFFINE_DEPTH_DETACH:-true}"
export HSI_TRANSL_DELTA_SCALE="${HSI_TRANSL_DELTA_SCALE:-0.05}"

export DEPTH_TEACHER_WEIGHT="${DEPTH_TEACHER_WEIGHT:-0.0}"
export ANCHOR_DEPTH_WEIGHT="${ANCHOR_DEPTH_WEIGHT:-0.0}"
export ANCHOR_SCENE_XYZ_WEIGHT="${ANCHOR_SCENE_XYZ_WEIGHT:-0.0}"
export HSI_SMPL_SCALE_TEACHER_WEIGHT="${HSI_SMPL_SCALE_TEACHER_WEIGHT:-0.0}"

export HSI_POSE_WEIGHT="${HSI_POSE_WEIGHT:-0.10}"
export HSI_BETAS_WEIGHT="${HSI_BETAS_WEIGHT:-0.0}"
export HSI_TRANSL_WEIGHT="${HSI_TRANSL_WEIGHT:-2.5}"
export HSI_JOINTS3D_WEIGHT="${HSI_JOINTS3D_WEIGHT:-2.0}"
export HSI_VERTICES_WEIGHT="${HSI_VERTICES_WEIGHT:-0.75}"
export HSI_PROJECTED_J2D_WEIGHT="${HSI_PROJECTED_J2D_WEIGHT:-0.02}"
export HSI_DELTA_REG_WEIGHT="${HSI_DELTA_REG_WEIGHT:-0.20}"
export HSI_NO_WORSE_WEIGHT="${HSI_NO_WORSE_WEIGHT:-8.0}"
export HSI_NO_WORSE_MARGIN_M="${HSI_NO_WORSE_MARGIN_M:-0.006}"

export HSI_ALIGN_POINT_WEIGHT="${HSI_ALIGN_POINT_WEIGHT:-1.0}"
export HSI_ALIGN_DELTA_REG_WEIGHT="${HSI_ALIGN_DELTA_REG_WEIGHT:-0.08}"
export HSI_ALIGN_NO_WORSE_WEIGHT="${HSI_ALIGN_NO_WORSE_WEIGHT:-2.0}"

export HSI_CONTACT_WEIGHT="${HSI_CONTACT_WEIGHT:-0.03}"
export HSI_CONTACT_TEACHER_CAMERA_SOURCE="${HSI_CONTACT_TEACHER_CAMERA_SOURCE:-gt}"
export HSI_FOOT_CONTACT_WEIGHT="${HSI_FOOT_CONTACT_WEIGHT:-0.0}"
export HSI_FOOT_SOLE_CONTACT_WEIGHT="${HSI_FOOT_SOLE_CONTACT_WEIGHT:-0.15}"
export HSI_SUPPORT_PLANE_CONTACT_WEIGHT="${HSI_SUPPORT_PLANE_CONTACT_WEIGHT:-0.35}"
export HSI_SUPPORT_PLANE_WINDOW="${HSI_SUPPORT_PLANE_WINDOW:-9}"
export HSI_SUPPORT_PLANE_MIN_POINTS="${HSI_SUPPORT_PLANE_MIN_POINTS:-6}"
export HSI_SUPPORT_PLANE_FLOAT_WEIGHT="${HSI_SUPPORT_PLANE_FLOAT_WEIGHT:-0.50}"
export HSI_SUPPORT_PLANE_PENETRATION_WEIGHT="${HSI_SUPPORT_PLANE_PENETRATION_WEIGHT:-4.0}"

export HSI_ENABLE_TEMPORAL_MOMENTUM="${HSI_ENABLE_TEMPORAL_MOMENTUM:-false}"
export HSI_TRANSL_VELOCITY_WEIGHT="${HSI_TRANSL_VELOCITY_WEIGHT:-0.0}"
export HSI_JOINTS_VELOCITY_WEIGHT="${HSI_JOINTS_VELOCITY_WEIGHT:-0.0}"
export HSI_JOINTS_ACCELERATION_WEIGHT="${HSI_JOINTS_ACCELERATION_WEIGHT:-0.0}"
export HSI_TEMPORAL_NO_WORSE_WEIGHT="${HSI_TEMPORAL_NO_WORSE_WEIGHT:-0.0}"
export HSI_FOOT_SLIDING_WEIGHT="${HSI_FOOT_SLIDING_WEIGHT:-0.0}"

export SAVE_SCOPE="${SAVE_SCOPE:-hsi}"
export SAVE_TOP_K="${SAVE_TOP_K:-3}"
export SAVE_EPOCH_CHECKPOINT="${SAVE_EPOCH_CHECKPOINT:-false}"
export SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-false}"
export MONITOR="${MONITOR:-loss_total}"
export PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS:-loss_total,loss_hsi_support_plane_contact,metric_hsi_support_plane_penetration_m,metric_hsi_support_plane_float_m,metric_hsi_support_plane_contact_count,loss_hsi_foot_sole_contact,metric_hsi_foot_sole_penetration_m,metric_hsi_foot_sole_float_m,loss_hsi_transl_cam,metric_hsi_transl_l1_delta,metric_hsi_joint_error_delta,metric_hsi_align_point_l1_delta}"

[[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing Stage2 checkpoint: ${RESUME_CKPT}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }

echo "========== HSI Stage3-A contact refinement =========="
echo "Resume ckpt : ${RESUME_CKPT}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Batch/views : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Contact     : sole=${HSI_FOOT_SOLE_CONTACT_WEIGHT}, plane=${HSI_SUPPORT_PLANE_CONTACT_WEIGHT}"

bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_provider.sh"
