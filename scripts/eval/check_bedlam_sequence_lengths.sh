#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
SPLIT="${SPLIT:-Training}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bedlam_sequence_lengths}"
TOP_K="${TOP_K:-50}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}/${SPLIT}" ]] || { echo "[ERROR] Missing split directory: ${BEDLAM_ROOT}/${SPLIT}" >&2; exit 1; }

OUT_TXT="${OUTPUT_DIR}/sequence_lengths.txt"
{
  while IFS= read -r -d '' seq_dir; do
    sequence="$(basename "${seq_dir}")"
    rgb_dir="${seq_dir}/rgb"
    depth_dir="${seq_dir}/depth"
    cam_dir="${seq_dir}/cam"
    smpl_dir="${seq_dir}/smpl"
    rgb_count=0
    depth_count=0
    cam_count=0
    smpl_count=0
    if [[ -d "${rgb_dir}" ]]; then
      rgb_count="$(find "${rgb_dir}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
    fi
    if [[ -d "${depth_dir}" ]]; then
      depth_count="$(find "${depth_dir}" -maxdepth 1 -type f -name '*.npy' | wc -l)"
    fi
    if [[ -d "${cam_dir}" ]]; then
      cam_count="$(find "${cam_dir}" -maxdepth 1 -type f -name '*.npz' | wc -l)"
    fi
    if [[ -d "${smpl_dir}" ]]; then
      smpl_count="$(find "${smpl_dir}" -maxdepth 1 -type f -name '*.pkl' | wc -l)"
    fi
    printf "%s,%s,%s,%s,%s,%s\n" "${SPLIT}" "${sequence}" "${rgb_count}" "${depth_count}" "${cam_count}" "${smpl_count}"
  done < <(find "${BEDLAM_ROOT}/${SPLIT}" -mindepth 1 -maxdepth 1 -type d -print0)
} | sort -t, -k3,3n -k2,2 > "${OUT_TXT}.data"
{
  echo "split,sequence,rgb_count,depth_count,cam_count,smpl_count"
  cat "${OUT_TXT}.data"
} > "${OUT_TXT}"
rm -f "${OUT_TXT}.data"

echo "========== BEDLAM sequence length scan =========="
echo "BEDLAM root : ${BEDLAM_ROOT}"
echo "Split       : ${SPLIT}"
echo "Output      : ${OUT_TXT}"
echo "Top-K hint  : ${TOP_K}"
head -n "$((TOP_K + 1))" "${OUT_TXT}"
echo "========== BEDLAM sequence length scan finished =========="
