#!/usr/bin/env bash
set -euo pipefail

# Official TUM RGB-D Freiburg3 dynamic subset used by Human3R/TTT3R.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/tum_dynamics_raw}"
mkdir -p "${RAW_ROOT}"
cd "${RAW_ROOT}"

BASE_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg3"
ARCHIVES=(
  rgbd_dataset_freiburg3_sitting_static.tgz
  rgbd_dataset_freiburg3_sitting_xyz.tgz
  rgbd_dataset_freiburg3_sitting_halfsphere.tgz
  rgbd_dataset_freiburg3_sitting_rpy.tgz
  rgbd_dataset_freiburg3_walking_static.tgz
  rgbd_dataset_freiburg3_walking_xyz.tgz
  rgbd_dataset_freiburg3_walking_halfsphere.tgz
  rgbd_dataset_freiburg3_walking_rpy.tgz
)

for archive in "${ARCHIVES[@]}"; do
  url="${BASE_URL}/${archive}"
  echo "[download] ${url}"
  wget --continue --show-progress "${url}" -O "${archive}"
  echo "[extract] ${archive}"
  tar -xzf "${archive}"
done

echo "TUM-Dynamics raw data are ready under: ${RAW_ROOT}"
echo "Next: RAW_ROOT=${RAW_ROOT} PREPARED_ROOT=${REPO_ROOT}/data/long_tum_s1 bash benchmarks/tum_dynamics_ate/prepare_tum_dynamics.sh"

