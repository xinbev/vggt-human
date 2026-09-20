#!/usr/bin/env bash
set -euo pipefail

# Download the GVHMR RICH evaluation support bundle. This is separate from the
# official RICH images, scans, and calibration archives.

RICH_ROOT="${RICH_ROOT:-/home/zhw/xyb_space/RICH}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${RICH_ROOT}/official_downloads}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${RICH_ROOT}/hmr4d_support}"
ARCHIVE_PATH="${DOWNLOAD_DIR}/RICH_hmr4d_support.tar.gz"
GDRIVE_URL="https://drive.google.com/file/d/1IyxQ-9VGnROTKqg8EuA5jdQzgtVoORnI/view?usp=sharing"

mkdir -p "${DOWNLOAD_DIR}" "${SUPPORT_ROOT}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed; installing it into the active Python environment..."
  python -m pip install gdown
fi

echo "[download] ${ARCHIVE_PATH}"
gdown --fuzzy --continue --output "${ARCHIVE_PATH}" "${GDRIVE_URL}"

if [[ "$(head -c 2 "${ARCHIVE_PATH}" | od -An -t x1 | tr -d ' \n')" != "1f8b" ]]; then
  echo "error: ${ARCHIVE_PATH} is not a gzip archive." >&2
  echo "Google Drive may have returned an error page or blocked the download quota." >&2
  exit 1
fi

echo "[extract] ${ARCHIVE_PATH}"
tar -xzf "${ARCHIVE_PATH}" -C "${RICH_ROOT}"

labels_path="$(find "${RICH_ROOT}" -type f -name rich_test_labels.pt -print -quit)"
preproc_path="$(find "${RICH_ROOT}" -type f -name rich_test_preproc.pt -print -quit)"

if [[ -z "${labels_path}" ]]; then
  echo "error: rich_test_labels.pt was not found after extraction." >&2
  exit 1
fi
if [[ -z "${preproc_path}" ]]; then
  echo "error: rich_test_preproc.pt was not found after extraction." >&2
  exit 1
fi

link_if_needed() {
  local source_path="$1"
  local target_path="$2"
  if [[ "$(readlink -f "${source_path}")" == "$(readlink -m "${target_path}")" ]]; then
    return 0
  fi
  if [[ -e "${target_path}" || -L "${target_path}" ]]; then
    echo "error: refusing to replace existing ${target_path}" >&2
    exit 1
  fi
  ln -s "${source_path}" "${target_path}"
}

link_if_needed "${labels_path}" "${SUPPORT_ROOT}/rich_test_labels.pt"
link_if_needed "${preproc_path}" "${SUPPORT_ROOT}/rich_test_preproc.pt"

echo
echo "RICH hmr4d support files are ready:"
ls -lh \
  "${SUPPORT_ROOT}/rich_test_labels.pt" \
  "${SUPPORT_ROOT}/rich_test_preproc.pt"
