#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/val_sequences.txt}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_top_train_epoch_0003_loss_total_0.171740.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/hsi_scale_trans_contact_v2}"
RUN_STAGES="${RUN_STAGES:-2A,2B,3A1,3A2,3B}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-false}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}"
MAX_HUMANS="${MAX_HUMANS:-20}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

BATCH_SIZE_2A="${BATCH_SIZE_2A:-24}"
BATCH_SIZE_2B="${BATCH_SIZE_2B:-24}"
BATCH_SIZE_3A1="${BATCH_SIZE_3A1:-16}"
BATCH_SIZE_3A2="${BATCH_SIZE_3A2:-12}"
BATCH_SIZE_3B="${BATCH_SIZE_3B:-12}"
EPOCHS_2A="${EPOCHS_2A:-3}"
EPOCHS_2B="${EPOCHS_2B:-3}"
EPOCHS_3A1="${EPOCHS_3A1:-2}"
EPOCHS_3A2="${EPOCHS_3A2:-2}"
EPOCHS_3B="${EPOCHS_3B:-3}"
LR_2A="${LR_2A:-2e-5}"
LR_2B="${LR_2B:-5e-6}"
LR_3A1="${LR_3A1:-1e-5}"
LR_3A2="${LR_3A2:-5e-6}"
LR_3B="${LR_3B:-2e-6}"

STAGE2A_DIR="${OUTPUT_ROOT}/stage2a_gt_transl"
STAGE2B_DIR="${OUTPUT_ROOT}/stage2b_real_bridge"
STAGE3A1_DIR="${OUTPUT_ROOT}/stage3a1_root_contact"
STAGE3A2_DIR="${OUTPUT_ROOT}/stage3a2_leg_contact"
STAGE3B_DIR="${OUTPUT_ROOT}/stage3b_real_contact"
STAGE2B_CKPT="${STAGE2B_CKPT:-}"

contains_stage() { case ",${RUN_STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
require_file() { [[ -f "$2" ]] || { echo "[ERROR] Missing $1: $2" >&2; exit 1; }; }
top_ckpt() { echo "$1/checkpoint_top01.pt"; }

[[ -n "${STAGE2B_CKPT}" ]] || STAGE2B_CKPT="$(top_ckpt "${STAGE2B_DIR}")"

require_file "Stage1 checkpoint" "${STAGE1_CKPT}"
require_file "train sequence manifest" "${TRAIN_SEQUENCE_MANIFEST}"
require_file "val sequence manifest" "${VAL_SEQUENCE_MANIFEST}"
if [[ "${RUN_STAGES}" == *"3"* ]]; then
  [[ -f "${CONTACT_TEACHER_ROOT}/summary.json" ]] || {
    echo "[ERROR] Contact teachers are missing under ${CONTACT_TEACHER_ROOT}." >&2
    echo "Run: bash scripts/preprocess/prepare_hsi_contact_teachers.sh" >&2
    exit 1
  }
fi

export REPO_ROOT DATA_ROOT BEDLAM_ROOT PREPROCESSED_ROOT CUDA_VISIBLE_DEVICES_VALUE NUM_WORKERS NLF_INTERNAL_BATCH_SIZE MAX_HUMANS
export TRAIN_SEQUENCE_MANIFEST VAL_SEQUENCE_MANIFEST CONTACT_TEACHER_ROOT MAX_STEPS_PER_EPOCH
export VAL_SPLIT=Training SAVE_SCOPE=hsi SAVE_TOP_K=3 SAVE_OPTIMIZER=false SAVE_EPOCH_CHECKPOINT=false
export SAVE_TOP_K_FROM_TRAIN=false TOPK_CREATE_STABLE_COPIES=true RESET_EPOCH=true
export DEPTH_TEACHER_WEIGHT=0 ANCHOR_DEPTH_WEIGHT=0 ANCHOR_SCENE_XYZ_WEIGHT=0
export HSI_SMPL_SCALE_TEACHER_WEIGHT=0 HSI_BETAS_WEIGHT=0 HSI_GATE_REG_WEIGHT=0
export HSI_ENABLE_TEMPORAL_MOMENTUM=false HSI_TRANSL_VELOCITY_WEIGHT=0 HSI_JOINTS_VELOCITY_WEIGHT=0
export HSI_JOINTS_ACCELERATION_WEIGHT=0 HSI_TEMPORAL_NO_WORSE_WEIGHT=0 HSI_FOOT_SLIDING_WEIGHT=0

run_provider() {
  if [[ -f "${OUTPUT_DIR}/checkpoint_topk_index.json" && "${ALLOW_EXISTING_OUTPUT}" != "true" ]]; then
    echo "[ERROR] Refusing to mix a new run with existing top-k state: ${OUTPUT_DIR}" >&2
    echo "Choose a new OUTPUT_ROOT, or set ALLOW_EXISTING_OUTPUT=true intentionally." >&2
    exit 1
  fi
  bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_provider.sh"
  require_file "top validation checkpoint" "$(top_ckpt "${OUTPUT_DIR}")"
}

if contains_stage 2A; then
  echo "========== V2 Stage2-A: GT metric translation denoising =========="
  (
    export TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage2_transl_curriculum_v2.yaml"
    export OUTPUT_DIR="${STAGE2A_DIR}" RESUME_CKPT="${STAGE1_CKPT}"
    export BATCH_SIZE="${BATCH_SIZE_2A}" NUM_VIEWS=2 EPOCHS="${EPOCHS_2A}" LR="${LR_2A}"
    export SMPL_PROVIDER=gt_perturbed HSI_GEOMETRY_MODE=gt_metric SMPL_PERTURB_MODE=translation
    export SMPL_TRANSL_RAY_NOISE_SCHEDULE=0.05,0.10,0.15
    export SMPL_TRANSL_TANGENT_NOISE_SCHEDULE_M=0.02,0.05,0.08 SMPL_TRANSL_RAY_NOISE_CLEAN_PROB=0.20
    export HSI_TRANSL_WEIGHT=8 HSI_RAY_DELTA_WEIGHT=4 HSI_TANGENT_DELTA_WEIGHT=4
    export RESUME_REQUIRED_PREFIXES=hsi_refinement_head. FROZEN_HASH_PREFIXES=hsi_refinement_head.
    export HSI_JOINTS3D_WEIGHT=1 HSI_VERTICES_WEIGHT=0 HSI_PROJECTED_J2D_WEIGHT=0
    export HSI_DELTA_REG_WEIGHT=0 HSI_NO_WORSE_WEIGHT=2 HSI_ALIGN_POINT_WEIGHT=0.25
    export HSI_ALIGN_DELTA_REG_WEIGHT=0.02 HSI_ALIGN_NO_WORSE_WEIGHT=2 MONITOR=metric_stage2_selection
    run_provider
  )
fi

if contains_stage 2B; then
  require_file "Stage2-A top checkpoint" "$(top_ckpt "${STAGE2A_DIR}")"
  echo "========== V2 Stage2-B: coherent GT/real bridge =========="
  (
    export TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage2_transl_curriculum_v2.yaml"
    export OUTPUT_DIR="${STAGE2B_DIR}" RESUME_CKPT="$(top_ckpt "${STAGE2A_DIR}")"
    export BATCH_SIZE="${BATCH_SIZE_2B}" NUM_VIEWS=2 EPOCHS="${EPOCHS_2B}" LR="${LR_2B}"
    export SMPL_PROVIDER=nlf HSI_GEOMETRY_MODE=mixed SMPL_PERTURB_MODE=translation
    export SMPL_GT_OVERRIDE_PROB_SCHEDULE=0.75,0.50,0.25
    export SMPL_TRANSL_RAY_NOISE_SCHEDULE=0.10,0.15,0.15
    export SMPL_TRANSL_TANGENT_NOISE_SCHEDULE_M=0.05,0.08,0.08 SMPL_TRANSL_RAY_NOISE_CLEAN_PROB=0.20
    export HSI_TRANSL_WEIGHT=6 HSI_RAY_DELTA_WEIGHT=2 HSI_TANGENT_DELTA_WEIGHT=2
    export RESUME_REQUIRED_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head. FROZEN_HASH_PREFIXES=hsi_refinement_head.
    export HSI_JOINTS3D_WEIGHT=1 HSI_VERTICES_WEIGHT=0.25 HSI_PROJECTED_J2D_WEIGHT=0.05
    export HSI_DELTA_REG_WEIGHT=0 HSI_NO_WORSE_WEIGHT=2
    export HSI_ALIGN_POINT_WEIGHT=0.50 HSI_ALIGN_DELTA_REG_WEIGHT=0.05 HSI_ALIGN_NO_WORSE_WEIGHT=5
    export MONITOR=metric_stage2_selection
    run_provider
  )
fi

if contains_stage 3A1; then
  require_file "Stage2-B top checkpoint" "${STAGE2B_CKPT}"
  echo "========== V2 Stage3-A1: GT root contact denoising =========="
  (
    export TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage3_contact_curriculum_v2.yaml"
    export OUTPUT_DIR="${STAGE3A1_DIR}" RESUME_CKPT="${STAGE2B_CKPT}"
    export BATCH_SIZE="${BATCH_SIZE_3A1}" NUM_VIEWS=3 EPOCHS="${EPOCHS_3A1}" LR="${LR_3A1}"
    export SMPL_PROVIDER=gt_perturbed HSI_GEOMETRY_MODE=gt_metric SMPL_PERTURB_MODE=contact_root
    export ENABLE_HSI_CONTACT_REFINE=true TRAIN_HSI_CONTACT_REFINE_ONLY=true REQUIRE_CONTACT_TEACHER=true
    export FREEZE_HSI_HUMAN_SCENE_ALIGN=true FREEZE_HSI_CONTACT_POSE_BRANCH=true FREEZE_HSI_CONTACT_ROOT_BRANCH=false
    export HSI_POSE_WEIGHT=0 HSI_TRANSL_WEIGHT=4 HSI_JOINTS3D_WEIGHT=2 HSI_VERTICES_WEIGHT=0.5
    export HSI_PROJECTED_J2D_WEIGHT=0 HSI_DELTA_REG_WEIGHT=0.05 HSI_NO_WORSE_WEIGHT=8
    export RESUME_REQUIRED_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head. FROZEN_HASH_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head.
    export HSI_CONTACT_REFINE_PLANE_WEIGHT=6 HSI_CONTACT_REFINE_POSE_WEIGHT=0
    export HSI_CONTACT_REFINE_CLASS_WEIGHT=0.2 HSI_CONTACT_REFINE_NO_WORSE_WEIGHT=8
    export HSI_CONTACT_REFINE_SWING_NO_PULL_WEIGHT=5
    export MONITOR=metric_stage3_selection
    run_provider
  )
fi

if contains_stage 3A2; then
  require_file "Stage3-A1 top checkpoint" "$(top_ckpt "${STAGE3A1_DIR}")"
  echo "========== V2 Stage3-A2: GT lower-leg contact denoising =========="
  (
    export TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage3_contact_curriculum_v2.yaml"
    export OUTPUT_DIR="${STAGE3A2_DIR}" RESUME_CKPT="$(top_ckpt "${STAGE3A1_DIR}")"
    export BATCH_SIZE="${BATCH_SIZE_3A2}" NUM_VIEWS=3 EPOCHS="${EPOCHS_3A2}" LR="${LR_3A2}"
    export SMPL_PROVIDER=gt_perturbed HSI_GEOMETRY_MODE=gt_metric SMPL_PERTURB_MODE=contact_pose
    export ENABLE_HSI_CONTACT_REFINE=true TRAIN_HSI_CONTACT_REFINE_ONLY=true REQUIRE_CONTACT_TEACHER=true
    export FREEZE_HSI_HUMAN_SCENE_ALIGN=true FREEZE_HSI_CONTACT_POSE_BRANCH=false FREEZE_HSI_CONTACT_ROOT_BRANCH=false
    export HSI_POSE_WEIGHT=1 HSI_TRANSL_WEIGHT=3 HSI_JOINTS3D_WEIGHT=3 HSI_VERTICES_WEIGHT=1
    export HSI_PROJECTED_J2D_WEIGHT=0 HSI_DELTA_REG_WEIGHT=0.05 HSI_NO_WORSE_WEIGHT=8
    export RESUME_REQUIRED_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head.,hsi_contact_refine_head. FROZEN_HASH_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head.
    export HSI_CONTACT_REFINE_PLANE_WEIGHT=4 HSI_CONTACT_REFINE_POSE_WEIGHT=1
    export HSI_CONTACT_REFINE_CLASS_WEIGHT=0.2 HSI_CONTACT_REFINE_NO_WORSE_WEIGHT=8
    export HSI_CONTACT_REFINE_SWING_NO_PULL_WEIGHT=5
    export MONITOR=metric_stage3_selection
    run_provider
  )
fi

if contains_stage 3B; then
  require_file "Stage3-A2 top checkpoint" "$(top_ckpt "${STAGE3A2_DIR}")"
  echo "========== V2 Stage3-B: coherent GT/real contact bridge =========="
  (
    export TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage3_contact_curriculum_v2.yaml"
    export OUTPUT_DIR="${STAGE3B_DIR}" RESUME_CKPT="$(top_ckpt "${STAGE3A2_DIR}")"
    export BATCH_SIZE="${BATCH_SIZE_3B}" NUM_VIEWS=3 EPOCHS="${EPOCHS_3B}" LR="${LR_3B}"
    export SMPL_PROVIDER=nlf HSI_GEOMETRY_MODE=mixed SMPL_PERTURB_MODE=contact_pose
    export SMPL_GT_OVERRIDE_PROB_SCHEDULE=0.50,0.25,0.0
    export ENABLE_HSI_CONTACT_REFINE=true TRAIN_HSI_CONTACT_REFINE_ONLY=true REQUIRE_CONTACT_TEACHER=true
    export FREEZE_HSI_HUMAN_SCENE_ALIGN=true FREEZE_HSI_CONTACT_POSE_BRANCH=false FREEZE_HSI_CONTACT_ROOT_BRANCH=false
    export HSI_POSE_WEIGHT=0.5 HSI_TRANSL_WEIGHT=2 HSI_JOINTS3D_WEIGHT=2 HSI_VERTICES_WEIGHT=0.5
    export HSI_DELTA_REG_WEIGHT=0.05 HSI_NO_WORSE_WEIGHT=8
    export RESUME_REQUIRED_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head.,hsi_contact_refine_head. FROZEN_HASH_PREFIXES=hsi_refinement_head.,hsi_human_scene_align_head.
    export HSI_PROJECTED_J2D_WEIGHT=0.05 HSI_CONTACT_REFINE_PLANE_WEIGHT=3 HSI_CONTACT_REFINE_POSE_WEIGHT=0.5
    export HSI_CONTACT_REFINE_CLASS_WEIGHT=0.2 HSI_CONTACT_REFINE_NO_WORSE_WEIGHT=10
    export HSI_CONTACT_REFINE_SWING_NO_PULL_WEIGHT=8
    export MONITOR=metric_stage3_selection
    run_provider
  )
fi

echo "========== HSI scale/trans/contact curriculum V2 finished =========="
echo "Stage2-A: $(top_ckpt "${STAGE2A_DIR}")"
echo "Stage2-B: $(top_ckpt "${STAGE2B_DIR}")"
echo "Stage3-A1: $(top_ckpt "${STAGE3A1_DIR}")"
echo "Stage3-A2: $(top_ckpt "${STAGE3A2_DIR}")"
echo "Stage3-B: $(top_ckpt "${STAGE3B_DIR}")"
