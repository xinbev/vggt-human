#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/debug/hsi_contact_teachers_v3_strict_pilot256}"
DEPTH_VISIBILITY_TOLERANCE_M="${DEPTH_VISIBILITY_TOLERANCE_M:-0.20}"
MIN_SOLE_VISIBLE_RATIO="${MIN_SOLE_VISIBLE_RATIO:-0.25}"
MAX_FILES="${MAX_FILES:-0}"

cd "${REPO_ROOT}"
python scripts/smoke/check_hsi_contact_teacher_strict.py \
  --contact-teacher-root "${CONTACT_TEACHER_ROOT}" \
  --depth-tolerance-m "${DEPTH_VISIBILITY_TOLERANCE_M}" \
  --min-sole-visible-ratio "${MIN_SOLE_VISIBLE_RATIO}" \
  --max-files "${MAX_FILES}"
