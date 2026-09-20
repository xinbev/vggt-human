#!/usr/bin/env bash
set -euo pipefail

# Download only the RICH assets needed to prepare the physical-grounding test set.
# The official RICH image archive is large (about 250 GB in JPG form).

RICH_ROOT="${RICH_ROOT:-/home/zhw/xyb_space/RICH}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${RICH_ROOT}/official_downloads}"
CONNECTIONS_PER_FILE="${CONNECTIONS_PER_FILE:-16}"
SPLITS_PER_FILE="${SPLITS_PER_FILE:-16}"
RICH_USERNAME="${RICH_USERNAME:-xinbev@126.com}"

TEST_IMAGES_URL="https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=JPG_images/test.tar.gz"
SCAN_CALIBRATION_URL="https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=scan_calibration.zip"
MULTICAM_TO_WORLD_URL="https://rich.is.tue.mpg.de/media/upload/multicam2world.zip"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "error: aria2c is required for parallel, resumable downloads." >&2
  echo "Install it in the server environment, then run this script again." >&2
  exit 1
fi

if ! [[ "${CONNECTIONS_PER_FILE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${SPLITS_PER_FILE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: CONNECTIONS_PER_FILE and SPLITS_PER_FILE must be positive integers." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"

AUTH_ARGS=()
if [[ -z "${RICH_PASSWORD:-}" ]]; then
  read -r -s -p "RICH password for ${RICH_USERNAME}: " RICH_PASSWORD
  echo
fi
AUTH_ARGS=(--http-user="${RICH_USERNAME}" --http-passwd="${RICH_PASSWORD}")

download_file() {
  local url="$1"
  local output_name="$2"
  local -a auth_args=()

  # The two download.is.tue.mpg.de files may require the RICH portal credentials.
  if [[ "${url}" == https://download.is.tue.mpg.de/* ]]; then
    auth_args=("${AUTH_ARGS[@]}")
  fi

  echo "[start] ${output_name}"
  aria2c \
    --dir="${DOWNLOAD_DIR}" \
    --out="${output_name}" \
    --continue=true \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --max-connection-per-server="${CONNECTIONS_PER_FILE}" \
    --split="${SPLITS_PER_FILE}" \
    --min-split-size=16M \
    --file-allocation=none \
    --retry-wait=5 \
    --max-tries=0 \
    --timeout=60 \
    --connect-timeout=30 \
    --summary-interval=30 \
    --console-log-level=warn \
    "${auth_args[@]}" \
    "${url}"
  echo "[done] ${output_name}"
}

download_file "${TEST_IMAGES_URL}" "rich_test_jpg_images.tar.gz" &
pid_images=$!
download_file "${SCAN_CALIBRATION_URL}" "rich_scan_calibration.zip" &
pid_calibration=$!
download_file "${MULTICAM_TO_WORLD_URL}" "rich_multicam2world.zip" &
pid_world=$!

status=0
for pid in "${pid_images}" "${pid_calibration}" "${pid_world}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "error: one or more downloads failed. Re-run the same command to resume." >&2
  exit "${status}"
fi

require_archive_magic() {
  local path="$1"
  local expected_magic="$2"
  local actual_magic
  local magic_bytes=$(( ${#expected_magic} / 2 ))
  actual_magic="$(head -c "${magic_bytes}" "${path}" | od -An -t x1 | tr -d ' \n')"
  if [[ "${actual_magic}" != "${expected_magic}" ]]; then
    echo "error: ${path} is not the expected archive format." >&2
    echo "The RICH server likely returned an HTML login or permission error page." >&2
    echo "Check RICH_USERNAME (without a backslash before @), then move this file aside and retry." >&2
    exit 1
  fi
}

require_archive_magic "${DOWNLOAD_DIR}/rich_test_jpg_images.tar.gz" "1f8b"
require_archive_magic "${DOWNLOAD_DIR}/rich_scan_calibration.zip" "504b0304"
require_archive_magic "${DOWNLOAD_DIR}/rich_multicam2world.zip" "504b0304"

echo
echo "Downloaded archives:"
ls -lh \
  "${DOWNLOAD_DIR}/rich_test_jpg_images.tar.gz" \
  "${DOWNLOAD_DIR}/rich_scan_calibration.zip" \
  "${DOWNLOAD_DIR}/rich_multicam2world.zip"
echo "Next step: verify and unpack these archives with a separate preparation script."
