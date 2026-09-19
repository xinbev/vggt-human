#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
TUM_CSV="${TUM_CSV:-${PROJECT_ROOT}/outputs/eval/tum_dynamics_ate/vggt_upto500/curve.csv}"
BONN_CSV="${BONN_CSV:-${PROJECT_ROOT}/outputs/eval/bonn_depth_pure_vggt/scale/pure_vggt_scale_curve_points.csv}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/vis/generic_3d_reconstruction/figure1.png}"
SCALE_RATIO="${SCALE_RATIO:-1.0}"
MIDDLE_OFFSET="${MIDDLE_OFFSET:-0.0}"
RIGHT_OFFSET="${RIGHT_OFFSET:-0.0}"
OURS_MIDDLE_OFFSET="${OURS_MIDDLE_OFFSET:-0.0}"
OURS_RIGHT_OFFSET="${OURS_RIGHT_OFFSET:-0.0}"
MIDDLE_Y_MIN="${MIDDLE_Y_MIN:-}"
MIDDLE_Y_MAX="${MIDDLE_Y_MAX:-}"
RIGHT_Y_MIN="${RIGHT_Y_MIN:-}"
RIGHT_Y_MAX="${RIGHT_Y_MAX:-}"

AXIS_ARGS=()
[[ -n "${MIDDLE_Y_MIN}" ]] && AXIS_ARGS+=(--middle-y-min "${MIDDLE_Y_MIN}")
[[ -n "${MIDDLE_Y_MAX}" ]] && AXIS_ARGS+=(--middle-y-max "${MIDDLE_Y_MAX}")
[[ -n "${RIGHT_Y_MIN}" ]] && AXIS_ARGS+=(--right-y-min "${RIGHT_Y_MIN}")
[[ -n "${RIGHT_Y_MAX}" ]] && AXIS_ARGS+=(--right-y-max "${RIGHT_Y_MAX}")

cd "${PROJECT_ROOT}"
python benchmarks/tum_dynamics_ate/plot_figure1.py \
  --tum-csv "${TUM_CSV}" \
  --bonn-csv "${BONN_CSV}" \
  --scale-ratio "${SCALE_RATIO}" \
  --middle-offset "${MIDDLE_OFFSET}" \
  --right-offset "${RIGHT_OFFSET}" \
  --ours-middle-offset "${OURS_MIDDLE_OFFSET}" \
  --ours-right-offset "${OURS_RIGHT_OFFSET}" \
  "${AXIS_ARGS[@]}" \
  --output "${OUTPUT}"

printf 'Figure written to %s\n' "${OUTPUT}"
