#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
OUTPUT_DIR="${REPO_ROOT}/outputs/vis/contact/measure_woman_n3library_20"

# 该阶段只读取 Slurm GPU 阶段生成的缓存，不执行模型推理。
cd "${REPO_ROOT}"
CACHE_DIR="${OUTPUT_DIR}/full_viewer_cache" \
SMPL_EDIT_OUTPUT="${OUTPUT_DIR}/smpl_edit_offsets.json" \
POINT_SIZE=0.006 \
HUMAN_MASK_DILATION_PX=12 \
FILTER_HUMAN_POINTS=false \
PORT="${PORT:-8080}" \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
