#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
RAW_ROOT="${RAW_ROOT:-/home/zhw/xyb_space/bedlam/BEDLAM_raw}"
SCENE="${SCENE:-20221014_3_250_batch01hand_orbit_archVizUI3_time15}"
ANNOT_DIR="${ANNOT_DIR:-/home/zhw/xyb_space/bedlam/all_npz_12_training}"
OUTDIR="${OUTDIR:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_raw_scene_links}"

# This wrapper now uses the project-local robust BEDLAM preprocessor. It keeps
# the historical script name because older notes referenced the Human3R path.
PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-${REPO_ROOT}/scripts/preprocess/prepare_bedlam_raw_scene.py}"

SCENE_ROOT="${RAW_ROOT}/${SCENE}"
NESTED_GT="${SCENE_ROOT}/ground_truth/${SCENE}"
NESTED_CAMERA="${NESTED_GT}/ground_truth/camera"
ANNOT_6FPS="${ANNOT_DIR}/${SCENE}_6fps.npz"
ANNOT_30FPS="${ANNOT_DIR}/${SCENE}_30fps.npz"

cd "${REPO_ROOT}"

[[ -d "${SCENE_ROOT}" ]] || { echo "[ERROR] Missing raw scene root: ${SCENE_ROOT}" >&2; exit 1; }
[[ -f "${NESTED_GT}/be_seq.csv" ]] || { echo "[ERROR] Missing nested be_seq.csv: ${NESTED_GT}/be_seq.csv" >&2; exit 1; }
[[ -d "${NESTED_CAMERA}" ]] || { echo "[ERROR] Missing nested camera dir: ${NESTED_CAMERA}" >&2; exit 1; }
[[ -f "${ANNOT_6FPS}" || -f "${ANNOT_30FPS}" ]] || {
  echo "[ERROR] Missing BEDLAM label npz: ${ANNOT_6FPS} or ${ANNOT_30FPS}" >&2
  exit 1
}
[[ -f "${PREPROCESS_SCRIPT}" ]] || {
  echo "[ERROR] Missing preprocess script: ${PREPROCESS_SCRIPT}" >&2
  echo "Set PREPROCESS_SCRIPT=/path/to/scripts/preprocess/prepare_bedlam_raw_scene.py" >&2
  exit 1
}

ln -sfn "${NESTED_GT}/be_seq.csv" "${SCENE_ROOT}/be_seq.csv"
mkdir -p "${SCENE_ROOT}/ground_truth/camera"
for camera_csv in "${NESTED_CAMERA}"/*.csv; do
  ln -sfn "${camera_csv}" "${SCENE_ROOT}/ground_truth/camera/$(basename "${camera_csv}")"
done
mkdir -p "${OUTDIR}"
mkdir -p "${WORK_ROOT}"
ln -sfn "${SCENE_ROOT}" "${WORK_ROOT}/${SCENE}"

echo "========== Prepare raw BEDLAM scene via Human3R preprocessor =========="
echo "Repo        : ${REPO_ROOT}"
echo "Raw root    : ${RAW_ROOT}"
echo "Scene       : ${SCENE}"
echo "Annot dir   : ${ANNOT_DIR}"
echo "Output      : ${OUTDIR}"
echo "Work root   : ${WORK_ROOT}"
echo "Workers     : ${NUM_WORKERS}"
echo "Preprocess  : ${PREPROCESS_SCRIPT}"
echo "be_seq link : ${SCENE_ROOT}/be_seq.csv -> ${NESTED_GT}/be_seq.csv"
echo "camera link : ${SCENE_ROOT}/ground_truth/camera -> ${NESTED_CAMERA}/*.csv"

python "${PREPROCESS_SCRIPT}" \
  --raw-root "${WORK_ROOT}" \
  --scene "${SCENE}" \
  --outdir "${OUTDIR}" \
  --annot-dir "${ANNOT_DIR}" \
  --overwrite

echo "========== Raw BEDLAM scene processed =========="
echo "Output split dirs under: ${OUTDIR}"
echo "Next step: generate box sidecars with scripts/preprocess/prepare_bedlam_boxes.sh"
