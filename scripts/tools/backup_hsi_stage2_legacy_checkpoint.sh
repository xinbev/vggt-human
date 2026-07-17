#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SOURCE_CKPT="${SOURCE_CKPT:?Set SOURCE_CKPT to the legacy Stage2 checkpoint}"
BACKUP_DIR="${BACKUP_DIR:-${REPO_ROOT}/outputs/checkpoint_backups/hsi_stage2_legacy}"
BACKUP_NAME="${BACKUP_NAME:-$(basename "${SOURCE_CKPT}")}"
TARGET_CKPT="${BACKUP_DIR}/${BACKUP_NAME}"

[[ -f "${SOURCE_CKPT}" ]] || { echo "[ERROR] Missing source checkpoint: ${SOURCE_CKPT}" >&2; exit 1; }
mkdir -p "${BACKUP_DIR}"

source_hash="$(sha256sum "${SOURCE_CKPT}" | awk '{print $1}')"
if [[ -e "${TARGET_CKPT}" ]]; then
  [[ -f "${TARGET_CKPT}" ]] || { echo "[ERROR] Backup target is not a file: ${TARGET_CKPT}" >&2; exit 1; }
  target_hash="$(sha256sum "${TARGET_CKPT}" | awk '{print $1}')"
  [[ "${source_hash}" == "${target_hash}" ]] || {
    echo "[ERROR] Existing backup differs from source; refusing to overwrite: ${TARGET_CKPT}" >&2
    exit 1
  }
  echo "[backup] verified existing immutable copy: ${TARGET_CKPT}"
else
  cp --preserve=timestamps --no-clobber "${SOURCE_CKPT}" "${TARGET_CKPT}"
  target_hash="$(sha256sum "${TARGET_CKPT}" | awk '{print $1}')"
  [[ "${source_hash}" == "${target_hash}" ]] || { echo "[ERROR] Backup hash verification failed" >&2; exit 1; }
  chmod a-w "${TARGET_CKPT}"
  echo "[backup] created immutable copy: ${TARGET_CKPT}"
fi

printf '%s  %s\n' "${source_hash}" "$(basename "${TARGET_CKPT}")" > "${TARGET_CKPT}.sha256"
echo "[backup] sha256=${source_hash}"

