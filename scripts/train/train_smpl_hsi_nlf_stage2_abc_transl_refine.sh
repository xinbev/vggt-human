#!/usr/bin/env bash
set -euo pipefail

# Stage2 ABC translation refinement:
# A) GT SMPL + GT K + ray-depth translation perturbation.
# B) Mixed GT-perturbed / NLF base bridge.
# C) NLF base + VGGT K with light pose/beta and temporal smoothing.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
export REPO_ROOT

PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_abc_transl_refine}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt}"
RUN_STAGES="${RUN_STAGES:-A,B,C}"
USER_PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS:-}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}"
MAX_HUMANS="${MAX_HUMANS:-20}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

BATCH_SIZE_A="${BATCH_SIZE_A:-20}"
BATCH_SIZE_B="${BATCH_SIZE_B:-16}"
BATCH_SIZE_C="${BATCH_SIZE_C:-12}"
NUM_VIEWS_A="${NUM_VIEWS_A:-2}"
NUM_VIEWS_B="${NUM_VIEWS_B:-2}"
NUM_VIEWS_C="${NUM_VIEWS_C:-4}"
EPOCHS_A="${EPOCHS_A:-3}"
EPOCHS_B="${EPOCHS_B:-3}"
EPOCHS_C="${EPOCHS_C:-3}"
LR_A="${LR_A:-5e-5}"
LR_B="${LR_B:-3e-5}"
LR_C="${LR_C:-1e-5}"
HSI_TRANSL_DELTA_SCALE_A="${HSI_TRANSL_DELTA_SCALE_A:-0.5}"
HSI_TRANSL_DELTA_SCALE_B="${HSI_TRANSL_DELTA_SCALE_B:-0.5}"
HSI_TRANSL_DELTA_SCALE_C="${HSI_TRANSL_DELTA_SCALE_C:-0.25}"
HSI_TRANSL_DELTA_MODE_A="${HSI_TRANSL_DELTA_MODE_A:-ray}"
HSI_TRANSL_DELTA_MODE_B="${HSI_TRANSL_DELTA_MODE_B:-ray}"
HSI_TRANSL_DELTA_MODE_C="${HSI_TRANSL_DELTA_MODE_C:-ray}"

NOISE_SCHEDULE_A="${NOISE_SCHEDULE_A:-0.03,0.05,0.10}"
NOISE_SCHEDULE_B="${NOISE_SCHEDULE_B:-0.05,0.10,0.15,0.20}"
CLEAN_PROB_A="${CLEAN_PROB_A:-0.20}"
CLEAN_PROB_B="${CLEAN_PROB_B:-0.20}"
GT_BASE_PROB_B="${GT_BASE_PROB_B:-0.60}"

STAGE_A_DIR="${STAGE_A_DIR:-${PIPELINE_OUTPUT_ROOT}/stageA_gt_ray_transl}"
STAGE_B_DIR="${STAGE_B_DIR:-${PIPELINE_OUTPUT_ROOT}/stageB_mixed_nlf_bridge}"
STAGE_C_DIR="${STAGE_C_DIR:-${PIPELINE_OUTPUT_ROOT}/stageC_temporal_refine}"

cd "${REPO_ROOT}"
mkdir -p "${PIPELINE_OUTPUT_ROOT}"
[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] Missing Stage1 checkpoint: ${STAGE1_CKPT}" >&2; exit 1; }

contains_stage() {
  case ",${RUN_STAGES}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_stage() {
  local label="$1"
  echo "========== Stage2 ${label} =========="
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_provider.sh"
}

COMMON_PROGRESS_KEYS="loss_total,loss_hsi_ray_delta,metric_hsi_ray_delta_base_l1,metric_hsi_ray_delta_refined_l1,metric_hsi_ray_delta_l1_delta,metric_hsi_ray_delta_sign_acc,loss_hsi_transl_cam,metric_hsi_base_transl_l1,metric_hsi_refined_transl_l1,metric_hsi_transl_l1_delta"

if contains_stage "A"; then
  export OUTPUT_DIR="${STAGE_A_DIR}"
  export RESUME_CKPT="${STAGE1_CKPT}"
  export RESET_EPOCH=true
  export CUDA_VISIBLE_DEVICES_VALUE NUM_WORKERS NLF_INTERNAL_BATCH_SIZE MAX_HUMANS MAX_STEPS_PER_EPOCH
  export BATCH_SIZE="${BATCH_SIZE_A}"
  export NUM_VIEWS="${NUM_VIEWS_A}"
  export EPOCHS="${EPOCHS_A}"
  export LR="${LR_A}"
  export SMPL_PROVIDER=gt_perturbed
  export HSI_CAMERA_SOURCE=gt
  export SMPL_TRANSL_RAY_NOISE_SCHEDULE="${NOISE_SCHEDULE_A}"
  export SMPL_TRANSL_RAY_NOISE_CLEAN_PROB="${CLEAN_PROB_A}"
  export SMPL_TRANSL_RAY_NOISE_MODE=uniform
  export SMPL_GT_OVERRIDE_PROB=0.0
  export TRAIN_HSI_TRANSL_ONLY=true
  export TRAIN_HSI_SCENE_AFFINE_ONLY=false
  export FREEZE_HSI_BACKBONE=true
  export FREEZE_HSI_SCENE_AFFINE=true
  export FREEZE_HSI_BETAS_DELTA=true
  export HSI_ENABLE_TEMPORAL_MOMENTUM=false
  export HSI_TRANSL_DELTA_SCALE="${HSI_TRANSL_DELTA_SCALE_A}"
  export HSI_TRANSL_DELTA_MODE="${HSI_TRANSL_DELTA_MODE_A}"
  export HSI_USE_AFFINE_DEPTH_FOR_TRANSL=true
  export HSI_AFFINE_DEPTH_DETACH=true
  export HSI_RAY_DELTA_WEIGHT=20.0
  export HSI_TRANSL_WEIGHT=2.0
  export HSI_JOINTS3D_WEIGHT=0.5
  export HSI_VERTICES_WEIGHT=0.25
  export HSI_NO_WORSE_WEIGHT=1.0
  export HSI_DELTA_REG_WEIGHT=0.005
  export HSI_POSE_WEIGHT=0.0
  export HSI_BETAS_WEIGHT=0.0
  export HSI_PROJECTED_J2D_WEIGHT=0.0
  export DEPTH_TEACHER_WEIGHT=0.0
  export ANCHOR_DEPTH_WEIGHT=0.0
  export ANCHOR_SCENE_XYZ_WEIGHT=0.0
  export HSI_CONTACT_WEIGHT=0.0
  export HSI_SMPL_SCALE_TEACHER_WEIGHT=0.0
  export HSI_TRANSL_VELOCITY_WEIGHT=0.0
  export HSI_JOINTS_VELOCITY_WEIGHT=0.0
  export HSI_TEMPORAL_NO_WORSE_WEIGHT=0.0
  export SAVE_SCOPE=hsi SAVE_TOP_K=3 SAVE_OPTIMIZER=false SAVE_EPOCH_CHECKPOINT=false
  export PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS_A:-${USER_PROGRESS_LOG_KEYS:-${COMMON_PROGRESS_KEYS}}}"
  run_stage "A / GT ray-depth transl denoising"
fi

if contains_stage "B"; then
  export OUTPUT_DIR="${STAGE_B_DIR}"
  export RESUME_CKPT="${STAGE_A_DIR}/checkpoint_latest.pt"
  [[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing Stage A checkpoint for Stage B: ${RESUME_CKPT}" >&2; exit 1; }
  export RESET_EPOCH=true
  export CUDA_VISIBLE_DEVICES_VALUE NUM_WORKERS NLF_INTERNAL_BATCH_SIZE MAX_HUMANS MAX_STEPS_PER_EPOCH
  export BATCH_SIZE="${BATCH_SIZE_B}"
  export NUM_VIEWS="${NUM_VIEWS_B}"
  export EPOCHS="${EPOCHS_B}"
  export LR="${LR_B}"
  export SMPL_PROVIDER=nlf
  export HSI_CAMERA_SOURCE=mixed
  export SMPL_TRANSL_RAY_NOISE_SCHEDULE="${NOISE_SCHEDULE_B}"
  export SMPL_TRANSL_RAY_NOISE_CLEAN_PROB="${CLEAN_PROB_B}"
  export SMPL_TRANSL_RAY_NOISE_MODE=uniform
  export SMPL_GT_OVERRIDE_PROB="${GT_BASE_PROB_B}"
  export TRAIN_HSI_TRANSL_ONLY=true
  export TRAIN_HSI_SCENE_AFFINE_ONLY=false
  export FREEZE_HSI_BACKBONE=true
  export FREEZE_HSI_SCENE_AFFINE=true
  export FREEZE_HSI_BETAS_DELTA=true
  export HSI_ENABLE_TEMPORAL_MOMENTUM=false
  export HSI_TRANSL_DELTA_SCALE="${HSI_TRANSL_DELTA_SCALE_B}"
  export HSI_TRANSL_DELTA_MODE="${HSI_TRANSL_DELTA_MODE_B}"
  export HSI_USE_AFFINE_DEPTH_FOR_TRANSL=true
  export HSI_AFFINE_DEPTH_DETACH=true
  export HSI_RAY_DELTA_WEIGHT=10.0
  export HSI_TRANSL_WEIGHT=4.0
  export HSI_JOINTS3D_WEIGHT=1.0
  export HSI_VERTICES_WEIGHT=0.5
  export HSI_PROJECTED_J2D_WEIGHT=0.05
  export HSI_NO_WORSE_WEIGHT=5.0
  export HSI_DELTA_REG_WEIGHT=0.03
  export HSI_POSE_WEIGHT=0.0
  export HSI_BETAS_WEIGHT=0.0
  export DEPTH_TEACHER_WEIGHT=0.0
  export ANCHOR_DEPTH_WEIGHT=0.0
  export ANCHOR_SCENE_XYZ_WEIGHT=0.0
  export HSI_CONTACT_WEIGHT=0.0
  export HSI_SMPL_SCALE_TEACHER_WEIGHT=0.05
  export HSI_SMPL_SCALE_TEACHER_LOG_LOSS=false
  export HSI_SMPL_SCALE_TEACHER_MAX_Z_M=20.0
  export HSI_TRANSL_VELOCITY_WEIGHT=0.0
  export HSI_JOINTS_VELOCITY_WEIGHT=0.0
  export HSI_TEMPORAL_NO_WORSE_WEIGHT=0.0
  export SAVE_SCOPE=hsi SAVE_TOP_K=3 SAVE_OPTIMIZER=false SAVE_EPOCH_CHECKPOINT=false
  export PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS_B:-${USER_PROGRESS_LOG_KEYS:-${COMMON_PROGRESS_KEYS}}}"
  run_stage "B / mixed GT-noisy + NLF bridge"
fi

if contains_stage "C"; then
  export OUTPUT_DIR="${STAGE_C_DIR}"
  export RESUME_CKPT="${STAGE_B_DIR}/checkpoint_latest.pt"
  [[ -f "${RESUME_CKPT}" ]] || { echo "[ERROR] Missing Stage B checkpoint for Stage C: ${RESUME_CKPT}" >&2; exit 1; }
  export RESET_EPOCH=true
  export CUDA_VISIBLE_DEVICES_VALUE NUM_WORKERS NLF_INTERNAL_BATCH_SIZE MAX_HUMANS MAX_STEPS_PER_EPOCH
  export BATCH_SIZE="${BATCH_SIZE_C}"
  export NUM_VIEWS="${NUM_VIEWS_C}"
  export EPOCHS="${EPOCHS_C}"
  export LR="${LR_C}"
  export SMPL_PROVIDER=nlf
  export HSI_CAMERA_SOURCE=vggt
  export SMPL_TRANSL_RAY_NOISE_SCHEDULE=0.0
  export SMPL_TRANSL_RAY_NOISE_CLEAN_PROB=0.0
  export SMPL_GT_OVERRIDE_PROB=0.0
  export TRAIN_HSI_TRANSL_ONLY=false
  export TRAIN_HSI_SCENE_AFFINE_ONLY=false
  export FREEZE_HSI_BACKBONE=true
  export FREEZE_HSI_SCENE_AFFINE=true
  export FREEZE_HSI_BETAS_DELTA=false
  export HSI_ENABLE_TEMPORAL_MOMENTUM=true
  export HSI_TEMPORAL_MOMENTUM_USE_TRACK_IDS=true
  export HSI_TRANSL_DELTA_SCALE="${HSI_TRANSL_DELTA_SCALE_C}"
  export HSI_TRANSL_DELTA_MODE="${HSI_TRANSL_DELTA_MODE_C}"
  export HSI_USE_AFFINE_DEPTH_FOR_TRANSL=true
  export HSI_AFFINE_DEPTH_DETACH=true
  export SMPL_TRACK_ASSIGNMENT_MODE=gt
  export HSI_TRANSL_WEIGHT=8.0
  export HSI_RAY_DELTA_WEIGHT=2.0
  export HSI_JOINTS3D_WEIGHT=2.0
  export HSI_VERTICES_WEIGHT=1.0
  export HSI_POSE_WEIGHT=0.2
  export HSI_BETAS_WEIGHT=0.1
  export HSI_PROJECTED_J2D_WEIGHT=0.05
  export HSI_TRANSL_VELOCITY_WEIGHT=1.0
  export HSI_JOINTS_VELOCITY_WEIGHT=0.5
  export HSI_TEMPORAL_NO_WORSE_WEIGHT=2.0
  export HSI_NO_WORSE_WEIGHT=5.0
  export HSI_DELTA_REG_WEIGHT=0.05
  export DEPTH_TEACHER_WEIGHT=0.0
  export ANCHOR_DEPTH_WEIGHT=0.0
  export ANCHOR_SCENE_XYZ_WEIGHT=0.0
  export HSI_CONTACT_WEIGHT=0.0
  export HSI_SMPL_SCALE_TEACHER_WEIGHT=0.0
  export SAVE_SCOPE=hsi SAVE_TOP_K=3 SAVE_OPTIMIZER=false SAVE_EPOCH_CHECKPOINT=false
  export PROGRESS_LOG_KEYS="${PROGRESS_LOG_KEYS_C:-${USER_PROGRESS_LOG_KEYS:-loss_total,loss_hsi_ray_delta,metric_hsi_ray_delta_l1_delta,loss_hsi_transl_cam,metric_hsi_base_transl_l1,metric_hsi_refined_transl_l1,metric_hsi_transl_l1_delta,loss_hsi_pose,loss_hsi_betas,loss_hsi_transl_velocity}}"
  run_stage "C / light pose-beta temporal refine"
fi

echo "========== Stage2 ABC finished =========="
echo "Stage A: ${STAGE_A_DIR}/checkpoint_latest.pt"
echo "Stage B: ${STAGE_B_DIR}/checkpoint_latest.pt"
echo "Stage C: ${STAGE_C_DIR}/checkpoint_latest.pt"
