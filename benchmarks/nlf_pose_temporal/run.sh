#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"; export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
: "${DATASET:?Set DATASET=3dpw or emdb1}"; : "${TEMPORAL_CHECKPOINT:?Set the V2 checkpoint, or set to an empty value only for base-only debugging}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0}" python benchmarks/nlf_pose_temporal/evaluate.py --dataset "${DATASET}" --path-config "${PATH_CONFIG:-configs/path.yaml}" --config "${CONFIG:-benchmarks/nlf_pose_temporal/config.yaml}" --frames-root "${FRAMES_ROOT:-}" --temporal-checkpoint "${TEMPORAL_CHECKPOINT}" --output-dir "${OUTPUT_DIR:-outputs/eval/nlf_pose_temporal/${DATASET}}" --device "${DEVICE:-cuda:0}" --batch-size "${BATCH_SIZE:-16}" --num-workers "${NUM_WORKERS:-2}"
