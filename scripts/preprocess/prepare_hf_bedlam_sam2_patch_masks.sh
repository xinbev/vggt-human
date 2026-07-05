#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${DEVICE:-}" =~ ^cuda:([0-9]+)$ ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  fi
  DEVICE=cuda
fi

ARGS=(
  --path-config "${PATH_CONFIG:-configs/path.yaml}"
  --output-split "${OUTPUT_SPLIT:-train}"
  --device "${DEVICE:-cuda}"
  --image-size "${IMAGE_SIZE:-518}"
  --patch-size "${PATCH_SIZE:-16}"
  --max-humans "${MAX_HUMANS:-20}"
  --bbox-expand "${BBOX_EXPAND:-0.15}"
  --mask-patch-threshold "${MASK_PATCH_THRESHOLD:-0.10}"
  --min-mask-patches "${MIN_MASK_PATCHES:-4}"
  --max-npz-files "${MAX_NPZ_FILES:-0}"
  --max-frames "${MAX_FRAMES:-0}"
  --max-output-frames "${MAX_OUTPUT_FRAMES:-0}"
  --log-interval "${LOG_INTERVAL:-100}"
)

if [[ "${TRANSL_ADD_CAM_EXT:-1}" == "0" ]]; then
  ARGS+=(--no-transl-add-cam-ext)
fi
if [[ -n "${IMAGES_ROOT:-}" ]]; then
  ARGS+=(--images-root "${IMAGES_ROOT}")
fi
if [[ -n "${NPZ_ROOT:-}" ]]; then
  ARGS+=(--npz-root "${NPZ_ROOT}")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  ARGS+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${SAM2_ROOT:-}" ]]; then
  ARGS+=(--sam2-root "${SAM2_ROOT}")
fi
if [[ -n "${SAM2_CHECKPOINT:-}" ]]; then
  ARGS+=(--sam2-checkpoint "${SAM2_CHECKPOINT}")
fi
if [[ -n "${SAM2_MODEL_CFG:-}" ]]; then
  ARGS+=(--sam2-model-cfg "${SAM2_MODEL_CFG}")
fi
if [[ "${SAM2_SINGLE_MASK:-0}" == "1" ]]; then
  ARGS+=(--sam2-single-mask)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python scripts/preprocess/prepare_hf_bedlam_sam2_patch_masks.py "${ARGS[@]}"
