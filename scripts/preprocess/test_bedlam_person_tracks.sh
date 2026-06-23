#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SPLIT="${SPLIT:-Training}"
SEQ_INDEX="${SEQ_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-120}"
OUT_ROOT="${OUT_ROOT:-outputs/preprocess/video_tracks}"
VIS_ROOT="${VIS_ROOT:-outputs/vis/video_person_tracks}"
STITCH_MAX_GAP="${STITCH_MAX_GAP:-30}"
STITCH_CENTER_THRESH="${STITCH_CENTER_THRESH:-1.25}"
STITCH_SIZE_LOG_THRESH="${STITCH_SIZE_LOG_THRESH:-0.70}"
STITCH_MIN_SCORE="${STITCH_MIN_SCORE:-0.25}"

python scripts/preprocess/prepare_video_person_tracks.py \
  --bedlam-sequence-index "${SEQ_INDEX}" \
  --bedlam-split "${SPLIT}" \
  --path-config configs/path.yaml \
  --output-root "${OUT_ROOT}" \
  --overwrite \
  --max-frames "${MAX_FRAMES}" \
  --detector-image-size 640 \
  --det-conf 0.25 \
  --det-iou 0.70 \
  --max-age 90 \
  --min-hits 1 \
  --aspect-ratio-thresh 10.0 \
  --stitch-max-gap "${STITCH_MAX_GAP}" \
  --stitch-center-thresh "${STITCH_CENTER_THRESH}" \
  --stitch-size-log-thresh "${STITCH_SIZE_LOG_THRESH}" \
  --stitch-min-score "${STITCH_MIN_SCORE}"

SIDE_ROOT="$(python - <<PY
import json
from pathlib import Path
pointer = Path("${OUT_ROOT}") / "latest_bedlam_tracking.json"
data = json.loads(pointer.read_text())
print(data["output_root"])
PY
)"

SEQ_NAME="$(python - <<PY
import json
from pathlib import Path
pointer = Path("${OUT_ROOT}") / "latest_bedlam_tracking.json"
data = json.loads(pointer.read_text())
print(str(data.get("source_name", "bedlam_sequence")).replace("/", "__"))
PY
)"

python scripts/vis/visualize_video_person_tracks.py \
  --sidecar-root "${SIDE_ROOT}" \
  --output-dir "${VIS_ROOT}/${SPLIT}/${SEQ_NAME}" \
  --max-frames "${MAX_FRAMES}" \
  --write-video
