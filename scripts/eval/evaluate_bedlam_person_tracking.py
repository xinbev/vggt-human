import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam_boxes import extract_best_box, extract_person_id
from vggt_omega.training.config import load_yaml_config, require_path

import numpy
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric

sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.path_config)
    bedlam_root = Path(args.bedlam_root or require_path(config, "datasets.bedlam_root")).expanduser()
    tracking_root = Path(args.tracking_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(bedlam_root, args.split)
    sequences = select_sequences(sequences, args)
    if not sequences:
        raise RuntimeError("No BEDLAM sequences selected")

    rows: list[dict[str, Any]] = []
    for seq_idx, seq_dir in enumerate(sequences):
        rel = str(seq_dir.relative_to(bedlam_root / args.split)).replace("\\", "/")
        sidecar_root = tracking_root / args.split / rel
        if args.run_tracker or not (sidecar_root / "summary.json").is_file():
            run_tracker_for_sequence(rel, args)
        row = evaluate_sequence(seq_dir, sidecar_root, args)
        row["sequence_index"] = seq_idx
        row["sequence_name"] = rel
        rows.append(row)
        print(
            f"[{seq_idx + 1}/{len(sequences)}] {rel} "
            f"recall={row['match_recall']:.3f} precision={row['match_precision']:.3f} "
            f"id_acc={row['id_dominant_accuracy']:.3f} idsw={row['id_switches']}"
        )

    summary = aggregate_rows(rows)
    payload = {
        "bedlam_root": str(bedlam_root),
        "split": args.split,
        "tracking_root": str(tracking_root),
        "num_sequences": len(rows),
        "summary": summary,
        "sequences": rows,
    }
    (output_dir / "bedlam_person_tracking_eval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "bedlam_person_tracking_eval.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted BEDLAM person track IDs against GT person IDs")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--tracking-root", default="outputs/preprocess/video_tracks")
    parser.add_argument("--output-dir", default="outputs/eval/bedlam_person_tracking")
    parser.add_argument("--run-tracker", action="store_true")
    parser.add_argument("--overwrite-tracks", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--sequence-index", action="append", type=int, default=[])
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--tracker-det-conf", type=float, default=0.25)
    parser.add_argument("--tracker-det-iou", type=float, default=0.70)
    parser.add_argument("--stitch-max-gap", type=int, default=30)
    parser.add_argument("--stitch-center-thresh", type=float, default=1.25)
    parser.add_argument("--stitch-size-log-thresh", type=float, default=0.70)
    parser.add_argument("--stitch-min-score", type=float, default=0.25)
    parser.add_argument("--no-reid", action="store_true")
    return parser.parse_args()


def discover_sequences(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")
    return sorted(path for path in split_dir.iterdir() if (path / "rgb").is_dir() and (path / "smpl").is_dir())


def select_sequences(sequences: list[Path], args: argparse.Namespace) -> list[Path]:
    selected = sequences
    if args.sequence:
        wanted = set(args.sequence)
        selected = [path for path in selected if path.name in wanted or str(path.name).replace("\\", "/") in wanted]
    if args.sequence_index:
        selected_by_idx = []
        for idx in args.sequence_index:
            if idx < 0 or idx >= len(sequences):
                raise IndexError(f"--sequence-index out of range: {idx}, valid=[0,{len(sequences) - 1}]")
            selected_by_idx.append(sequences[idx])
        selected = selected_by_idx
    if args.max_sequences and args.max_sequences > 0:
        selected = selected[: args.max_sequences]
    return selected


def run_tracker_for_sequence(sequence_name: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/preprocess/prepare_video_person_tracks.py",
        "--bedlam-sequence",
        sequence_name,
        "--bedlam-split",
        args.split,
        "--path-config",
        args.path_config,
        "--output-root",
        args.tracking_root,
        "--det-conf",
        str(args.tracker_det_conf),
        "--det-iou",
        str(args.tracker_det_iou),
        "--stitch-max-gap",
        str(args.stitch_max_gap),
        "--stitch-center-thresh",
        str(args.stitch_center_thresh),
        "--stitch-size-log-thresh",
        str(args.stitch_size_log_thresh),
        "--stitch-min-score",
        str(args.stitch_min_score),
    ]
    if args.overwrite_tracks:
        cmd.append("--overwrite")
    if args.max_frames_per_sequence and args.max_frames_per_sequence > 0:
        cmd.extend(["--max-frames", str(args.max_frames_per_sequence)])
    if args.no_reid:
        cmd.append("--no-reid")
    subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate_sequence(seq_dir: Path, sidecar_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rgb_paths = sorted((seq_dir / "rgb").glob("*.png"))
    if args.max_frames_per_sequence and args.max_frames_per_sequence > 0:
        rgb_paths = rgb_paths[: args.max_frames_per_sequence]
    per_gt_matches: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    total_gt = 0
    total_pred = 0
    matched_gt = 0
    matched_pred = 0

    for frame_idx, rgb_path in enumerate(rgb_paths):
        gt_people = load_gt_people(seq_dir, rgb_path)
        pred_people = load_pred_people(sidecar_root, rgb_path.stem)
        total_gt += len(gt_people)
        total_pred += len(pred_people)
        matches = greedy_match(gt_people, pred_people, args.iou_threshold)
        matched_gt += len(matches)
        matched_pred += len(matches)
        matched_gt_indices = {gt_idx for gt_idx, _ in matches}
        for gt_idx, pred_idx in matches:
            per_gt_matches[gt_people[gt_idx]["gt_id"]].append((frame_idx, pred_people[pred_idx]["person_id"]))
        for gt_idx, gt in enumerate(gt_people):
            if gt_idx not in matched_gt_indices:
                per_gt_matches[gt["gt_id"]].append((frame_idx, None))

    id_metrics = compute_id_metrics(per_gt_matches)
    return {
        "num_frames": len(rgb_paths),
        "total_gt_person_frames": int(total_gt),
        "total_pred_person_frames": int(total_pred),
        "matched_person_frames": int(matched_gt),
        "false_negative_person_frames": int(total_gt - matched_gt),
        "false_positive_person_frames": int(total_pred - matched_pred),
        "match_recall": ratio(matched_gt, total_gt),
        "match_precision": ratio(matched_pred, total_pred),
        **id_metrics,
    }


def load_gt_people(seq_dir: Path, rgb_path: Path) -> list[dict[str, Any]]:
    with Image.open(rgb_path) as image:
        image_hw = (image.height, image.width)
    smpl_path = seq_dir / "smpl" / f"{rgb_path.stem}.pkl"
    if not smpl_path.is_file():
        return []
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    out = []
    for slot, person in enumerate(persons if isinstance(persons, list) else []):
        if not isinstance(person, dict):
            continue
        box_record = extract_best_box(person, image_hw)
        if not box_record.get("bbox_valid", False):
            continue
        gt_id, valid = extract_person_id(person)
        if not valid:
            gt_id = slot
        out.append({"gt_id": int(gt_id), "bbox": np.asarray(box_record["bbox_xyxy_pixels"], dtype=np.float32)})
    return out


def load_pred_people(sidecar_root: Path, frame_id: str) -> list[dict[str, Any]]:
    path = sidecar_root / "smpl_boxes" / f"{frame_id}.pkl"
    if not path.is_file():
        return []
    with path.open("rb") as file:
        frame = pickle.load(file)
    out = []
    for person in frame.get("persons", []):
        if not person.get("valid", True) or not person.get("bbox_valid", True):
            continue
        out.append({"person_id": int(person["person_id"]), "bbox": np.asarray(person["bbox_xyxy_pixels"], dtype=np.float32)})
    return out


def greedy_match(gt_people: list[dict[str, Any]], pred_people: list[dict[str, Any]], threshold: float) -> list[tuple[int, int]]:
    pairs = []
    for gt_idx, gt in enumerate(gt_people):
        for pred_idx, pred in enumerate(pred_people):
            pairs.append((iou(gt["bbox"], pred["bbox"]), gt_idx, pred_idx))
    pairs.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for score, gt_idx, pred_idx in pairs:
        if score < threshold:
            break
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx))
    return matches


def compute_id_metrics(per_gt_matches: dict[int, list[tuple[int, int | None]]]) -> dict[str, Any]:
    dominant_correct = 0
    matched = 0
    id_switches = 0
    fragmentations = 0
    gt_persons = 0
    for gt_id, rows in per_gt_matches.items():
        rows = sorted(rows)
        assigned = [pred_id for _, pred_id in rows if pred_id is not None]
        if not assigned:
            continue
        gt_persons += 1
        matched += len(assigned)
        dominant_id, dominant_count = Counter(assigned).most_common(1)[0]
        dominant_correct += dominant_count
        previous_pred = None
        previous_matched = False
        segments = 0
        for _, pred_id in rows:
            if pred_id is None:
                previous_matched = False
                continue
            if previous_pred is not None and pred_id != previous_pred:
                id_switches += 1
            if not previous_matched:
                segments += 1
            previous_pred = pred_id
            previous_matched = True
        fragmentations += max(segments - 1, 0)
    return {
        "gt_person_tracks_matched": int(gt_persons),
        "id_dominant_correct_frames": int(dominant_correct),
        "id_dominant_total_frames": int(matched),
        "id_dominant_accuracy": ratio(dominant_correct, matched),
        "id_switches": int(id_switches),
        "id_switches_per_100_matched": 100.0 * ratio(id_switches, matched),
        "fragmentations": int(fragmentations),
        "fragmentations_per_100_matched": 100.0 * ratio(fragmentations, matched),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "num_frames",
        "total_gt_person_frames",
        "total_pred_person_frames",
        "matched_person_frames",
        "false_negative_person_frames",
        "false_positive_person_frames",
        "id_dominant_correct_frames",
        "id_dominant_total_frames",
        "id_switches",
        "fragmentations",
    ]
    summed = {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}
    summed["match_recall"] = ratio(summed["matched_person_frames"], summed["total_gt_person_frames"])
    summed["match_precision"] = ratio(summed["matched_person_frames"], summed["total_pred_person_frames"])
    summed["id_dominant_accuracy"] = ratio(summed["id_dominant_correct_frames"], summed["id_dominant_total_frames"])
    summed["id_switches_per_100_matched"] = 100.0 * ratio(summed["id_switches"], summed["id_dominant_total_frames"])
    summed["fragmentations_per_100_matched"] = 100.0 * ratio(summed["fragmentations"], summed["id_dominant_total_frames"])
    return summed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(float(a[2] - a[0]), 0.0) * max(float(a[3] - a[1]), 0.0)
    area_b = max(float(b[2] - b[0]), 0.0) * max(float(b[3] - b[1]), 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def ratio(num: int | float, denom: int | float) -> float:
    return float(num) / float(denom) if float(denom) > 0 else 0.0


if __name__ == "__main__":
    main()
