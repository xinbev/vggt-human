#!/usr/bin/env bash
set -euo pipefail

# Download only the RICH assets needed to prepare the physical-grounding test set.
# The protected downloads use the form-POST authentication required by the RICH portal.

RICH_ROOT="${RICH_ROOT:-/home/zhw/xyb_space/RICH}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${RICH_ROOT}/official_downloads}"
RICH_USERNAME="${RICH_USERNAME:-xinbev@126.com}"

TEST_IMAGES_URL="https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=JPG_images/test.tar.gz"
SCAN_CALIBRATION_URL="https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=scan_calibration.zip"
MULTICAM_TO_WORLD_URL="https://rich.is.tue.mpg.de/media/upload/multicam2world.zip"

if ! command -v wget >/dev/null 2>&1; then
  echo "error: wget is required by the RICH form-POST download protocol." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"

if [[ -z "${RICH_PASSWORD:-}" ]]; then
  read -r -s -p "RICH password for ${RICH_USERNAME}: " RICH_PASSWORD
  echo
fi

urlencode() {
  local LC_ALL=C
  local input="$1"
  local encoded=""
  local char
  local index
  for ((index = 0; index < ${#input}; index++)); do
    char="${input:index:1}"
    case "${char}" in
      [a-zA-Z0-9.~_-]) encoded+="${char}" ;;
      *) printf -v encoded '%s%%%02X' "${encoded}" "'${char}" ;;
    esac
  done
  printf '%s' "${encoded}"
}

AUTH_PAYLOAD_FILE="$(mktemp "${TMPDIR:-/tmp}/rich-download-auth.XXXXXX")"
cleanup() {
  rm -f "${AUTH_PAYLOAD_FILE}"
}
trap cleanup EXIT INT TERM
chmod 600 "${AUTH_PAYLOAD_FILE}"
printf 'username=%s&password=%s' \
  "$(urlencode "${RICH_USERNAME}")" \
  "$(urlencode "${RICH_PASSWORD}")" >"${AUTH_PAYLOAD_FILE}"
unset RICH_PASSWORD

archive_magic() {
  local path="$1"
  local byte_count="$2"
  head -c "${byte_count}" "${path}" 2>/dev/null | od -An -t x1 | tr -d ' \n'
}

prepare_output() {
  local path="$1"
  local expected_magic="$2"
  local magic_bytes=$(( ${#expected_magic} / 2 ))
  local actual_magic
  local backup_path

  [[ -f "${path}" ]] || return 0
  actual_magic="$(archive_magic "${path}" "${magic_bytes}")"
  if [[ "${actual_magic}" == "${expected_magic}" ]]; then
    echo "[resume] ${path}"
    return 0
  fi

  backup_path="${path}.invalid.$(date +%Y%m%d-%H%M%S)"
  mv "${path}" "${backup_path}"
  if [[ -f "${path}.aria2" ]]; then
    mv "${path}.aria2" "${backup_path}.aria2"
  fi
  echo "[moved] invalid login/error response -> ${backup_path}"
}

download_protected_file() {
  local url="$1"
  local output_name="$2"
  local output_path="${DOWNLOAD_DIR}/${output_name}"

  echo "[start] ${output_name}"
  wget \
    --continue \
    --tries=0 \
    --timeout=60 \
    --read-timeout=60 \
    --retry-connrefused \
    --waitretry=5 \
    --post-file="${AUTH_PAYLOAD_FILE}" \
    --output-document="${output_path}" \
    "${url}"
  echo "[done] ${output_name}"
}

download_public_file() {
  local url="$1"
  local output_name="$2"
  local output_path="${DOWNLOAD_DIR}/${output_name}"

  echo "[start] ${output_name}"
  wget \
    --continue \
    --tries=0 \
    --timeout=60 \
    --read-timeout=60 \
    --retry-connrefused \
    --waitretry=5 \
    --output-document="${output_path}" \
    "${url}"
  echo "[done] ${output_name}"
}

require_archive_magic() {
  local path="$1"
  local expected_magic="$2"
  local magic_bytes=$(( ${#expected_magic} / 2 ))
  local actual_magic
  actual_magic="$(archive_magic "${path}" "${magic_bytes}")"
  if [[ "${actual_magic}" != "${expected_magic}" ]]; then
    echo "error: ${path} is not the expected archive format." >&2
    echo "The RICH server returned an HTML login or permission error page." >&2
    echo "Verify the website password and that this account has dataset access." >&2
    exit 1
  fi
}

prepare_output "${DOWNLOAD_DIR}/rich_test_jpg_images.tar.gz" "1f8b"
prepare_output "${DOWNLOAD_DIR}/rich_scan_calibration.zip" "504b0304"
prepare_output "${DOWNLOAD_DIR}/rich_multicam2world.zip" "504b0304"

# The two large protected archives run concurrently. wget is used because the
# portal requires POST data, which aria2c cannot attach to these download jobs.
download_protected_file "${TEST_IMAGES_URL}" "rich_test_jpg_images.tar.gz" &
pid_images=$!
download_protected_file "${SCAN_CALIBRATION_URL}" "rich_scan_calibration.zip" &
pid_calibration=$!
download_public_file "${MULTICAM_TO_WORLD_URL}" "rich_multicam2world.zip" &
pid_world=$!

status=0
for pid in "${pid_images}" "${pid_calibration}" "${pid_world}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "error: one or more downloads failed. Re-run the script to resume." >&2
  exit "${status}"
fi

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
