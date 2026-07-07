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
  --splits ${SPLITS:-train validation test}
  --device "${DEVICE:-cuda}"
  --image-resolution "${IMAGE_RESOLUTION:-512}"
  --patch-size "${PATCH_SIZE:-16}"
  --mask-patch-threshold "${MASK_PATCH_THRESHOLD:-0.10}"
  --min-mask-patches "${MIN_MASK_PATCHES:-4}"
  --log-interval "${LOG_INTERVAL:-100}"
)

if [[ -n "${IMAGE_SIZE:-}" ]]; then
  ARGS+=(--image-size "${IMAGE_SIZE}")
fi
if [[ -n "${ROOT:-}" ]]; then
  ARGS+=(--root "${ROOT}")
fi
if [[ -n "${ANNOTATION_ROOT:-}" ]]; then
  ARGS+=(--annotation-root "${ANNOTATION_ROOT}")
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
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python scripts/preprocess/prepare_3dpw_sam2_patch_masks.py "${ARGS[@]}"
