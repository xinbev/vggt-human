#!/usr/bin/env bash

set -euo pipefail

# Hard-tail repair for remaining single-frame SMPL translation failures.
#
# Stage 1 trains only smpl_head.regression_head.translation_refiner with
# camera-ray coordinates, GT box priors, no raw depth, and no HSI coupling.
# Stage 2 merges the repaired refiner back into a full HSI checkpoint.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_ray_refine_tail_hard.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
DEFAULT_TEMPORAL_HSI_CKPT="${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt"
DEFAULT_SINGLE_HSI_CKPT="${REPO_ROOT}/outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt"
INIT_HSI_CKPT="${INIT_HSI_CKPT:-${DEFAULT_TEMPORAL_HSI_CKPT}}"
if [[ ! -f "${INIT_HSI_CKPT}" && -f "${DEFAULT_SINGLE_HSI_CKPT}" ]]; then
  INIT_HSI_CKPT="${DEFAULT_SINGLE_HSI_CKPT}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/smpl_translation_ray_refine_tail_hard_from_after_hsi}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${OUTPUT_ROOT}/stage1_tail_refiner}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${OUTPUT_ROOT}/merged_hsi_tail_translation}"
MERGED_CKPT="${MERGED_CKPT:-${MERGED_OUTPUT_DIR}/checkpoint_latest.pt}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/eval/smpl_translation_ray_refine_tail_hard_from_after_hsi}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_TRANSLATION_EVAL="${RUN_TRANSLATION_EVAL:-1}"
RUN_HSI_EVAL="${RUN_HSI_EVAL:-0}"
RUN_BAD_SCAN="${RUN_BAD_SCAN:-0}"

EXTRA_EPOCHS="${EXTRA_EPOCHS:-6}"
LR="${LR:-5e-6}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"
DEFAULT_HARD_SUBSET_CSV="${REPO_ROOT}/outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine/bad_frame_person_rows.csv"
USE_HARD_SUBSET="${USE_HARD_SUBSET:-auto}"
HARD_SUBSET_CSV="${HARD_SUBSET_CSV:-}"
if [[ "${USE_HARD_SUBSET}" == "auto" && -z "${HARD_SUBSET_CSV}" && -f "${DEFAULT_HARD_SUBSET_CSV}" ]]; then
  HARD_SUBSET_CSV="${DEFAULT_HARD_SUBSET_CSV}"
fi
if [[ "${USE_HARD_SUBSET}" == "false" || "${USE_HARD_SUBSET}" == "0" ]]; then
  HARD_SUBSET_CSV=""
fi
HARD_SUBSET_INDEX_COLUMN="${HARD_SUBSET_INDEX_COLUMN:-dataset_index}"
HARD_SUBSET_UNIQUE="${HARD_SUBSET_UNIQUE:-false}"
HARD_SUBSET_REPEAT="${HARD_SUBSET_REPEAT:-3}"
HARD_SUBSET_MAX_SAMPLES="${HARD_SUBSET_MAX_SAMPLES:-0}"

TRANSL_CAM_WEIGHT="${TRANSL_CAM_WEIGHT:-4.0}"
JOINTS3D_WEIGHT="${JOINTS3D_WEIGHT:-12.0}"
PROJECTED_JOINTS2D_WEIGHT="${PROJECTED_JOINTS2D_WEIGHT:-0.05}"
HARD_TOPK_WEIGHT="${HARD_TOPK_WEIGHT:-8.0}"
HARD_SEVERE_WEIGHT="${HARD_SEVERE_WEIGHT:-6.0}"
HARD_TOPK_FRACTION="${HARD_TOPK_FRACTION:-0.35}"
HARD_MIN_K="${HARD_MIN_K:-3}"
HARD_ERROR_THRESHOLD_M="${HARD_ERROR_THRESHOLD_M:-0.12}"
DELTA_REG_WEIGHT="${DELTA_REG_WEIGHT:-0.02}"
RAY_DEPTH_WEIGHT="${RAY_DEPTH_WEIGHT:-1.50}"
TANGENT_WEIGHT="${TANGENT_WEIGHT:-0.75}"
MAX_RAY_DELTA_M="${MAX_RAY_DELTA_M:-1.20}"
MAX_TANGENT_DELTA_M="${MAX_TANGENT_DELTA_M:-0.60}"
MAX_LOG_DEPTH_DELTA="${MAX_LOG_DEPTH_DELTA:-0.85}"
MAX_BOX_PRIOR_WEIGHT="${MAX_BOX_PRIOR_WEIGHT:-1.00}"

MERGE_INCLUDE_TRANSLATION_HEADS="${MERGE_INCLUDE_TRANSLATION_HEADS:-false}"
HSI_EVAL_CONFIG="${HSI_EVAL_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_after_translation_ray_refine.yaml}"
BAD_SCAN_CONFIG="${BAD_SCAN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml}"
BAD_SCAN_OUTPUT_DIR="${BAD_SCAN_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_bad_translation_scan_tail_hard_from_after_hsi}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${STAGE1_OUTPUT_DIR}" "${MERGED_OUTPUT_DIR}" "${EVAL_ROOT}"

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

INIT_EPOCH="$(read_epoch "${INIT_HSI_CKPT}")"
TOTAL_EPOCHS=$((INIT_EPOCH + EXTRA_EPOCHS))
STAGE1_CKPT="${STAGE1_OUTPUT_DIR}/checkpoint_latest.pt"

echo "========== SMPL translation ray-refine hard-tail repair =========="
echo "Repo              : ${REPO_ROOT}"
echo "Init full HSI ckpt : ${INIT_HSI_CKPT}"
echo "Init epoch         : ${INIT_EPOCH}"
echo "Extra epochs       : ${EXTRA_EPOCHS}"
echo "Total epochs       : ${TOTAL_EPOCHS}"
echo "Stage output       : ${STAGE1_OUTPUT_DIR}"
echo "Merged HSI output  : ${MERGED_CKPT}"
echo "CUDA devices       : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Views/max humans   : ${NUM_VIEWS}/${MAX_HUMANS}"
echo "Hard tail          : topk_weight=${HARD_TOPK_WEIGHT} severe_weight=${HARD_SEVERE_WEIGHT} threshold=${HARD_ERROR_THRESHOLD_M}m"
echo "Use hard subset    : ${USE_HARD_SUBSET}"
echo "Hard subset CSV    : ${HARD_SUBSET_CSV:-<none>}"

SUBSET_ARGS=()
if [[ "${USE_HARD_SUBSET}" == "true" && -z "${HARD_SUBSET_CSV}" ]]; then
  echo "[ERROR] USE_HARD_SUBSET=true requires HARD_SUBSET_CSV=/path/to/bad_frame_person_rows.csv" >&2
  exit 1
fi
if [[ "${USE_HARD_SUBSET}" != "false" && "${USE_HARD_SUBSET}" != "0" && -n "${HARD_SUBSET_CSV}" ]]; then
  [[ -f "${HARD_SUBSET_CSV}" ]] || { echo "[ERROR] Missing hard subset CSV: ${HARD_SUBSET_CSV}" >&2; exit 1; }
  SUBSET_ARGS+=(--override "data.subset_indices_csv=${HARD_SUBSET_CSV}")
  SUBSET_ARGS+=(--override "data.subset_index_column=${HARD_SUBSET_INDEX_COLUMN}")
  SUBSET_ARGS+=(--override "data.subset_unique=${HARD_SUBSET_UNIQUE}")
  SUBSET_ARGS+=(--override "data.subset_repeat=${HARD_SUBSET_REPEAT}")
  SUBSET_ARGS+=(--override "data.subset_max_samples=${HARD_SUBSET_MAX_SAMPLES}")
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
    --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
    --override "checkpoint.resume=${INIT_HSI_CKPT}" \
    --override "checkpoint.resume_strict=false" \
    --override "checkpoint.resume_optimizer=false" \
    --override "checkpoint.save_final=true" \
    --override "checkpoint.save_epoch_checkpoint=false" \
    --override "checkpoint.save_latest=true" \
    --override "experiment.output_dir=${STAGE1_OUTPUT_DIR}" \
    --override "data.sequence_length=${NUM_VIEWS}" \
    --override "data.stride=1" \
    --override "data.val_split=" \
    --override "data.max_humans=${MAX_HUMANS}" \
    --override "data.num_workers=${NUM_WORKERS}" \
    --override "data.require_boxes=true" \
    --override "data.require_depth=false" \
    "${SUBSET_ARGS[@]}" \
    --override "model.num_smpl_queries=${MAX_HUMANS}" \
    --override "model.enable_camera=true" \
    --override "model.enable_depth=false" \
    --override "model.enable_hsi_refine=false" \
    --override "model.freeze_aggregator=true" \
    --override "model.freeze_aggregator_forward=true" \
    --override "model.freeze_camera_head=true" \
    --override "model.freeze_smpl_head=true" \
    --override "model.train_smpl_translation_heads=false" \
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
    --override "model.smpl_translation_refine_max_ray_delta_m=${MAX_RAY_DELTA_M}" \
    --override "model.smpl_translation_refine_max_tangent_delta_m=${MAX_TANGENT_DELTA_M}" \
    --override "model.smpl_translation_refine_max_log_depth_delta=${MAX_LOG_DEPTH_DELTA}" \
    --override "model.smpl_translation_refine_max_box_prior_weight=${MAX_BOX_PRIOR_WEIGHT}" \
    --override "model.smpl_translation_refine_human_height_prior_m=1.70" \
    --override "model.smpl_translation_refine_use_log_depth=true" \
    --override "matching.cost_conf=0.5" \
    --override "matching.cost_bbox=8.0" \
    --override "matching.cost_giou=4.0" \
    --override "matching.cost_kpts=0.0" \
    --override "loss.pose_weight=0.0" \
    --override "loss.betas_weight=0.0" \
    --override "loss.transl_cam_weight=${TRANSL_CAM_WEIGHT}" \
    --override "loss.joints3d_weight=${JOINTS3D_WEIGHT}" \
    --override "loss.projected_joints2d_weight=${PROJECTED_JOINTS2D_WEIGHT}" \
    --override "loss.transl_refine_delta_reg_weight=${DELTA_REG_WEIGHT}" \
    --override "loss.transl_refine_ray_depth_weight=${RAY_DEPTH_WEIGHT}" \
    --override "loss.transl_refine_tangent_weight=${TANGENT_WEIGHT}" \
    --override "loss.transl_hard_topk_weight=${HARD_TOPK_WEIGHT}" \
    --override "loss.transl_hard_severe_weight=${HARD_SEVERE_WEIGHT}" \
    --override "loss.transl_hard_topk_fraction=${HARD_TOPK_FRACTION}" \
    --override "loss.transl_hard_min_k=${HARD_MIN_K}" \
    --override "loss.transl_hard_error_threshold_m=${HARD_ERROR_THRESHOLD_M}" \
    --override "loss.projected_bbox_weight=0.02" \
    --override "loss.projected_giou_weight=0.02" \
    --override "loss.conf_weight=0.0" \
    --override "loss.bbox_weight=0.0" \
    --override "loss.giou_weight=0.0" \
    --override "loss.duplicate_conf_weight=0.0" \
    --override "loss.aux_weight=0.0" \
    --override "optim.epochs=${TOTAL_EPOCHS}" \
    --override "optim.lr=${LR}" \
    --override "optim.batch_size=1" \
    --override "optim.save_interval=0" \
    --override "optim.grad_clip_norm=1.0" \
    --override "optim.log_interval=${LOG_INTERVAL}" \
    --override "optim.log_style=progress" \
    --override "optim.progress_log_keys=loss_total,loss_transl_cam,loss_transl_hard_topk,loss_transl_hard_severe,loss_joints3d,loss_transl_refine_ray_depth,loss_transl_refine_tangent,metric_transl_l2_mean,metric_transl_l2_topk_mean,metric_transl_l2_max,metric_transl_l2_over_threshold_rate,metric_transl_hard_count,metric_base_transl_l1,metric_refined_transl_l1,metric_transl_refine_l1_delta,metric_transl_ray_depth_l1_delta,metric_transl_tangent_l1_delta,metric_transl_box_prior_weight_abs"
fi

[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] Missing stage1 checkpoint: ${STAGE1_CKPT}" >&2; exit 1; }

if [[ "${RUN_MERGE}" == "1" ]]; then
  MERGE_ARGS=()
  if [[ "${MERGE_INCLUDE_TRANSLATION_HEADS}" == "true" ]]; then
    MERGE_ARGS+=(--include-translation-heads)
  fi
  CUDA_VISIBLE_DEVICES="" python scripts/diagnostics/merge_translation_refiner_into_hsi.py \
    --hsi-checkpoint "${INIT_HSI_CKPT}" \
    --translation-checkpoint "${STAGE1_CKPT}" \
    --output "${MERGED_CKPT}" \
    --epoch-source translation \
    "${MERGE_ARGS[@]}"
fi

if [[ "${RUN_TRANSLATION_EVAL}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --checkpoint "${STAGE1_CKPT}" \
    --output-dir "${EVAL_ROOT}/translation_metrics" \
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
    --override "model.smpl_translation_refine_max_ray_delta_m=${MAX_RAY_DELTA_M}" \
    --override "model.smpl_translation_refine_max_tangent_delta_m=${MAX_TANGENT_DELTA_M}" \
    --override "model.smpl_translation_refine_max_log_depth_delta=${MAX_LOG_DEPTH_DELTA}" \
    --override "model.smpl_translation_refine_max_box_prior_weight=${MAX_BOX_PRIOR_WEIGHT}"
fi

if [[ "${RUN_HSI_EVAL}" == "1" ]]; then
  [[ -f "${MERGED_CKPT}" ]] || { echo "[ERROR] Missing merged checkpoint for HSI eval: ${MERGED_CKPT}" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_hsi_refine_metrics.py \
    --path-config "${PATH_CONFIG}" \
    --train-config "${HSI_EVAL_CONFIG}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --checkpoint "${MERGED_CKPT}" \
    --output-dir "${EVAL_ROOT}/hsi_metrics" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --use-gt-box-prior \
    --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
    --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
    --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
    --override "data.sequence_length=2" \
    --override "data.max_humans=${MAX_HUMANS}" \
    --override "model.num_smpl_queries=${MAX_HUMANS}" \
    --override "model.enable_camera=true" \
    --override "model.enable_depth=true" \
    --override "model.enable_hsi_refine=true" \
    --override "model.smpl_enable_translation_refine=true" \
    --override "model.smpl_translation_refine_max_ray_delta_m=${MAX_RAY_DELTA_M}" \
    --override "model.smpl_translation_refine_max_tangent_delta_m=${MAX_TANGENT_DELTA_M}" \
    --override "model.smpl_translation_refine_max_log_depth_delta=${MAX_LOG_DEPTH_DELTA}" \
    --override "model.smpl_translation_refine_max_box_prior_weight=${MAX_BOX_PRIOR_WEIGHT}" \
    --override "model.hsi_scene_affine_mode=clip_median"
fi

if [[ "${RUN_BAD_SCAN}" == "1" ]]; then
  [[ -f "${MERGED_CKPT}" ]] || { echo "[ERROR] Missing merged checkpoint for bad scan: ${MERGED_CKPT}" >&2; exit 1; }
  SMPL_CKPT="${MERGED_CKPT}" \
  OUTPUT_DIR="${BAD_SCAN_OUTPUT_DIR}" \
  TRAIN_CONFIG="${BAD_SCAN_CONFIG}" \
  SMPL_TRANSLATION_REFINE_MAX_RAY_DELTA_M="${MAX_RAY_DELTA_M}" \
  SMPL_TRANSLATION_REFINE_MAX_TANGENT_DELTA_M="${MAX_TANGENT_DELTA_M}" \
  SMPL_TRANSLATION_REFINE_MAX_LOG_DEPTH_DELTA="${MAX_LOG_DEPTH_DELTA}" \
  SMPL_TRANSLATION_REFINE_MAX_BOX_PRIOR_WEIGHT="${MAX_BOX_PRIOR_WEIGHT}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
fi

echo "========== SMPL translation ray-refine hard-tail repair finished =========="
echo "Translation-only checkpoint: ${STAGE1_CKPT}"
echo "Merged full-HSI checkpoint : ${MERGED_CKPT}"
echo "Eval root                  : ${EVAL_ROOT}"
