#!/usr/bin/env bash

set -euo pipefail

# Full base SMPL translation repair from the verified 0121 checkpoint.
#
# T0: evaluate the original 0121 base translation.
# T1: train only the camera-ray translation refiner, no raw depth, no HSI.
# T2: train original transl_cam_heads + ray refiner, still no raw depth/HSI.
# T3: merge the translation repair keys back into the full 0121 HSI checkpoint.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_ray_refine.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
INIT_HSI_CKPT="${INIT_HSI_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_translation_ray_refine_full_from0121}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/eval/smpl_translation_ray_refine_full_from0121}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1_ray_refiner}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/stage2_transl_heads_ray_refiner}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${OUTPUT_ROOT}/merged_hsi_translation}"
MERGED_CKPT="${MERGED_CKPT:-${MERGED_OUTPUT_DIR}/checkpoint_latest.pt}"

RUN_T0_EVAL="${RUN_T0_EVAL:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE_EVAL="${RUN_STAGE_EVAL:-1}"
RUN_MERGE="${RUN_MERGE:-1}"

STAGE1_EXTRA_EPOCHS="${STAGE1_EXTRA_EPOCHS:-12}"
STAGE2_EXTRA_EPOCHS="${STAGE2_EXTRA_EPOCHS:-8}"
STAGE1_LR="${STAGE1_LR:-2e-5}"
STAGE2_LR="${STAGE2_LR:-8e-6}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${STAGE1_OUTPUT_DIR}" "${STAGE2_OUTPUT_DIR}" "${MERGED_OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${INIT_HSI_CKPT}" ]] || { echo "[ERROR] Missing init HSI checkpoint: ${INIT_HSI_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

read_epoch() {
  python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0)
PY
}

run_eval() {
  local checkpoint_path="$1"
  local output_dir="$2"
  local label="$3"
  mkdir -p "${output_dir}"
  echo "========== Eval ${label} =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --checkpoint "${checkpoint_path}" \
    --output-dir "${output_dir}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --use-gt-box-prior \
    --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
    --override "data.sequence_length=${NUM_VIEWS}" \
    --override "data.max_humans=${MAX_HUMANS}" \
    --override "model.num_smpl_queries=${MAX_HUMANS}" \
    --override "model.enable_camera=true" \
    --override "model.enable_depth=false" \
    --override "model.enable_hsi_refine=false" \
    --override "model.smpl_enable_translation_refine=true" \
    --override "model.smpl_translation_refine_max_ray_delta_m=1.20" \
    --override "model.smpl_translation_refine_max_tangent_delta_m=0.60" \
    --override "model.smpl_translation_refine_max_log_depth_delta=0.85" \
    --override "model.smpl_translation_refine_max_box_prior_weight=1.00"
}

run_translation_stage() {
  local stage_label="$1"
  local init_ckpt="$2"
  local output_dir="$3"
  local extra_epochs="$4"
  local lr="$5"
  local train_translation_heads="$6"
  local max_ray_delta="$7"
  local max_tangent_delta="$8"
  local max_log_depth_delta="$9"
  local max_box_prior_weight="${10}"
  local delta_reg_weight="${11}"
  local ray_depth_weight="${12}"
  local tangent_weight="${13}"
  local init_epoch
  local total_epochs

  init_epoch="$(read_epoch "${init_ckpt}")"
  total_epochs=$((init_epoch + extra_epochs))
  echo "========== ${stage_label} =========="
  echo "Init ckpt       : ${init_ckpt}"
  echo "Init epoch      : ${init_epoch}"
  echo "Extra epochs    : ${extra_epochs}"
  echo "Total epochs    : ${total_epochs}"
  echo "Output          : ${output_dir}"
  echo "LR              : ${lr}"
  echo "Train transl MLP: ${train_translation_heads}"
  echo "Max ray/tangent : ${max_ray_delta} / ${max_tangent_delta}"
  echo "No raw depth, no HSI."

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
    --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
    --override "checkpoint.resume=${init_ckpt}" \
    --override "checkpoint.resume_strict=false" \
    --override "checkpoint.resume_optimizer=false" \
    --override "checkpoint.save_final=true" \
    --override "checkpoint.save_epoch_checkpoint=false" \
    --override "checkpoint.save_latest=true" \
    --override "experiment.output_dir=${output_dir}" \
    --override "data.sequence_length=${NUM_VIEWS}" \
    --override "data.val_split=" \
    --override "data.max_humans=${MAX_HUMANS}" \
    --override "data.num_workers=${NUM_WORKERS}" \
    --override "data.require_boxes=true" \
    --override "data.require_depth=false" \
    --override "model.num_smpl_queries=${MAX_HUMANS}" \
    --override "model.enable_camera=true" \
    --override "model.enable_depth=false" \
    --override "model.enable_hsi_refine=false" \
    --override "model.freeze_aggregator=true" \
    --override "model.freeze_aggregator_forward=true" \
    --override "model.freeze_camera_head=true" \
    --override "model.freeze_smpl_head=true" \
    --override "model.train_smpl_translation_heads=${train_translation_heads}" \
    --override "model.train_smpl_box_heads=false" \
    --override "model.train_smpl_translation_refiner=true" \
    --override "model.train_smpl_query_token=false" \
    --override "model.train_smpl_box_prior_embed=false" \
    --override "model.train_smpl_patch_pool_embed=false" \
    --override "model.predict_boxes=true" \
    --override "model.smpl_bbox_mode=reference_residual" \
    --override "model.smpl_return_aux=true" \
    --override "model.smpl_query_box_prior=true" \
    --override "model.smpl_query_patch_pool=true" \
    --override "model.smpl_query_patch_pool_expand=0.12" \
    --override "model.smpl_enable_translation_refine=true" \
    --override "model.smpl_translation_refine_hidden_dim=512" \
    --override "model.smpl_translation_refine_max_ray_delta_m=${max_ray_delta}" \
    --override "model.smpl_translation_refine_max_tangent_delta_m=${max_tangent_delta}" \
    --override "model.smpl_translation_refine_max_log_depth_delta=${max_log_depth_delta}" \
    --override "model.smpl_translation_refine_max_box_prior_weight=${max_box_prior_weight}" \
    --override "model.smpl_translation_refine_human_height_prior_m=1.70" \
    --override "model.smpl_translation_refine_use_log_depth=true" \
    --override "matching.cost_conf=0.5" \
    --override "matching.cost_bbox=8.0" \
    --override "matching.cost_giou=4.0" \
    --override "matching.cost_kpts=0.0" \
    --override "loss.pose_weight=0.0" \
    --override "loss.betas_weight=0.0" \
    --override "loss.transl_cam_weight=6.0" \
    --override "loss.joints3d_weight=16.0" \
    --override "loss.projected_joints2d_weight=0.05" \
    --override "loss.transl_refine_delta_reg_weight=${delta_reg_weight}" \
    --override "loss.transl_refine_ray_depth_weight=${ray_depth_weight}" \
    --override "loss.transl_refine_tangent_weight=${tangent_weight}" \
    --override "loss.projected_bbox_weight=0.02" \
    --override "loss.projected_giou_weight=0.02" \
    --override "loss.conf_weight=0.0" \
    --override "loss.bbox_weight=0.0" \
    --override "loss.giou_weight=0.0" \
    --override "loss.duplicate_conf_weight=0.0" \
    --override "loss.aux_weight=0.0" \
    --override "optim.epochs=${total_epochs}" \
    --override "optim.lr=${lr}" \
    --override "optim.batch_size=1" \
    --override "optim.save_interval=0" \
    --override "optim.grad_clip_norm=1.0" \
    --override "optim.log_interval=${LOG_INTERVAL}" \
    --override "optim.log_style=progress" \
    --override "optim.progress_log_keys=loss_total,loss_transl_cam,loss_joints3d,loss_transl_refine_ray_depth,loss_transl_refine_tangent,loss_transl_refine_delta_reg,metric_base_transl_l1,metric_refined_transl_l1,metric_transl_refine_l1_delta,metric_transl_ray_depth_l1_delta,metric_transl_tangent_l1_delta,metric_transl_box_prior_weight_abs"
}

echo "========== Full SMPL Translation Ray Refine =========="
echo "Repo             : ${REPO_ROOT}"
echo "Init HSI ckpt    : ${INIT_HSI_CKPT}"
echo "Output root      : ${OUTPUT_ROOT}"
echo "Eval root        : ${EVAL_ROOT}"
echo "CUDA devices     : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Views/max humans : ${NUM_VIEWS}/${MAX_HUMANS}"

if [[ "${RUN_T0_EVAL}" == "1" ]]; then
  run_eval "${INIT_HSI_CKPT}" "${EVAL_ROOT}/t0_init_0121" "T0 original 0121"
fi

STAGE1_CKPT="${STAGE1_OUTPUT_DIR}/checkpoint_latest.pt"
if [[ "${RUN_STAGE1}" == "1" ]]; then
  if [[ "${STAGE1_EXTRA_EPOCHS}" == "0" ]]; then
    [[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] STAGE1_EXTRA_EPOCHS=0 but missing ${STAGE1_CKPT}" >&2; exit 1; }
  else
    run_translation_stage \
      "T1/3 ray refiner only" \
      "${INIT_HSI_CKPT}" \
      "${STAGE1_OUTPUT_DIR}" \
      "${STAGE1_EXTRA_EPOCHS}" \
      "${STAGE1_LR}" \
      "false" \
      "1.20" \
      "0.60" \
      "0.85" \
      "1.00" \
      "0.03" \
      "1.25" \
      "0.60"
  fi
fi
[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] Missing stage1 checkpoint: ${STAGE1_CKPT}" >&2; exit 1; }
if [[ "${RUN_STAGE_EVAL}" == "1" ]]; then
  run_eval "${STAGE1_CKPT}" "${EVAL_ROOT}/t1_ray_refiner" "T1 ray refiner"
fi

STAGE2_CKPT="${STAGE2_OUTPUT_DIR}/checkpoint_latest.pt"
FINAL_TRANSLATION_CKPT="${STAGE1_CKPT}"
if [[ "${RUN_STAGE2}" == "1" ]]; then
  if [[ "${STAGE2_EXTRA_EPOCHS}" == "0" ]]; then
    FINAL_TRANSLATION_CKPT="${STAGE1_CKPT}"
  else
    run_translation_stage \
      "T2/3 transl heads + ray refiner" \
      "${STAGE1_CKPT}" \
      "${STAGE2_OUTPUT_DIR}" \
      "${STAGE2_EXTRA_EPOCHS}" \
      "${STAGE2_LR}" \
      "true" \
      "1.20" \
      "0.60" \
      "0.85" \
      "1.00" \
      "0.02" \
      "1.50" \
      "0.75"
    FINAL_TRANSLATION_CKPT="${STAGE2_CKPT}"
  fi
fi
[[ -f "${FINAL_TRANSLATION_CKPT}" ]] || { echo "[ERROR] Missing final translation checkpoint: ${FINAL_TRANSLATION_CKPT}" >&2; exit 1; }
if [[ "${RUN_STAGE_EVAL}" == "1" ]]; then
  run_eval "${FINAL_TRANSLATION_CKPT}" "${EVAL_ROOT}/t2_final_translation" "T2 final translation"
fi

if [[ "${RUN_MERGE}" == "1" ]]; then
  echo "========== T3/3 merge translation repair into full HSI checkpoint =========="
  CUDA_VISIBLE_DEVICES="" python scripts/diagnostics/merge_translation_refiner_into_hsi.py \
    --hsi-checkpoint "${INIT_HSI_CKPT}" \
    --translation-checkpoint "${FINAL_TRANSLATION_CKPT}" \
    --output "${MERGED_CKPT}" \
    --include-translation-heads \
    --epoch-source translation
fi

echo "========== Full SMPL Translation Ray Refine finished =========="
echo "Stage1 checkpoint       : ${STAGE1_CKPT}"
echo "Final translation ckpt  : ${FINAL_TRANSLATION_CKPT}"
echo "Merged HSI+translation  : ${MERGED_CKPT}"
echo "Eval metrics root       : ${EVAL_ROOT}"
