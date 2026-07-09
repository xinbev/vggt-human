import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam_boxes import (  # noqa: E402
    BBOX_KEYS,
    J2D_KEYS,
    extract_best_box,
    extract_person_id,
    extract_visibility_stats,
)

import numpy  # noqa: E402
import numpy.core  # noqa: E402
import numpy.core.multiarray  # noqa: E402
import numpy.core.numeric  # noqa: E402

sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "thresholds": {
            "min_visible_joints": int(args.min_visible_joints),
            "min_box_area": float(args.min_box_area),
        },
    }
    all_rows: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []

    for split in args.splits:
        split_summary, rows, samples = inspect_split(dataset_root, split, args)
        summary["splits"][split] = split_summary
        all_rows.extend(rows)
        all_samples.extend(samples)

    write_csv(output_dir / "person_rows.csv", all_rows)
    (output_dir / "person_samples.json").write_text(json.dumps(all_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_human_summary(summary, summary_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BEDLAM processed smpl/*.pkl annotations before box preprocessing")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="outputs/debug/bedlam_annotation_inspection")
    parser.add_argument("--splits", nargs="+", default=["Training"])
    parser.add_argument("--max-sequences", type=int, default=20)
    parser.add_argument("--max-frames-per-sequence", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--min-visible-joints", type=int, default=4)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    return parser.parse_args()


def inspect_split(dataset_root: Path, split: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")

    key_counts: Counter[str] = Counter()
    key_shape_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bbox_field_counts: Counter[str] = Counter()
    j2d_field_counts: Counter[str] = Counter()
    bbox_source_counts: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    persons_per_frame: Counter[int] = Counter()
    visible_joint_hist: Counter[int] = Counter()
    bbox_area_bins: Counter[str] = Counter()
    id_counts: Counter[str] = Counter()

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    total_frames = 0
    total_persons = 0

    seq_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if args.max_sequences > 0:
        seq_dirs = seq_dirs[: args.max_sequences]

    for seq_dir in seq_dirs:
        rgb_dir = seq_dir / "rgb"
        smpl_dir = seq_dir / "smpl"
        if not rgb_dir.is_dir() or not smpl_dir.is_dir():
            continue
        frame_ids = sorted(path.stem for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if args.max_frames_per_sequence > 0:
            frame_ids = frame_ids[: args.max_frames_per_sequence]
        for frame_id in frame_ids:
            smpl_path = smpl_dir / f"{frame_id}.pkl"
            rgb_path = rgb_dir / f"{frame_id}.png"
            if not smpl_path.is_file():
                continue
            persons = load_persons(smpl_path)
            image_hw = load_image_hw(rgb_path)
            total_frames += 1
            total_persons += len(persons)
            persons_per_frame[len(persons)] += 1
            for person_idx, person in enumerate(persons):
                if not isinstance(person, dict):
                    continue
                for key, value in person.items():
                    key_counts[key] += 1
                    key_shape_counts[key][value_signature(value)] += 1
                for key in BBOX_KEYS:
                    if key in person:
                        bbox_field_counts[key] += 1
                for key in J2D_KEYS:
                    if key in person:
                        j2d_field_counts[key] += 1

                person_id, person_id_valid = extract_person_id(person)
                id_counts["valid"] += int(person_id_valid)
                id_counts["missing"] += int(not person_id_valid)
                box = safe_extract_best_box(person, image_hw)
                stats = extract_visibility_stats(person, image_hw, bbox_xyxy_pixels=box.get("bbox_xyxy_pixels"))
                bbox_valid = bool(box.get("bbox_valid", False))
                bbox_source = str(box.get("bbox_source", "error"))
                bbox_source_counts[bbox_source] += 1
                visible_joints = int(stats["visible_joints"])
                visible_joint_hist[visible_joints] += 1
                bbox_area = float(stats["bbox_area_pixels"])
                bbox_area_bins[area_bin(bbox_area)] += 1
                filter_reason = classify_filter(
                    bbox_valid=bbox_valid,
                    has_j2d=bool(stats["has_j2d_visibility"]),
                    visible_joints=visible_joints,
                    bbox_area=bbox_area,
                    min_visible_joints=args.min_visible_joints,
                    min_box_area=args.min_box_area,
                    require_j2d=False,
                )
                strict_filter_reason = classify_filter(
                    bbox_valid=bbox_valid,
                    has_j2d=bool(stats["has_j2d_visibility"]),
                    visible_joints=visible_joints,
                    bbox_area=bbox_area,
                    min_visible_joints=args.min_visible_joints,
                    min_box_area=args.min_box_area,
                    require_j2d=True,
                )
                filter_counts[filter_reason] += 1
                row = {
                    "split": split,
                    "sequence": seq_dir.name,
                    "frame": frame_id,
                    "person_index": person_idx,
                    "person_id": person_id,
                    "person_id_valid": int(person_id_valid),
                    "num_keys": len(person),
                    "keys": ",".join(sorted(person.keys())),
                    "bbox_valid": int(bbox_valid),
                    "bbox_source": bbox_source,
                    "bbox_area_pixels": bbox_area,
                    "has_j2d_visibility": int(bool(stats["has_j2d_visibility"])),
                    "j2d_source": stats["j2d_source"],
                    "visible_joints": visible_joints,
                    "total_joints": int(stats["total_joints"]),
                    "filter_reason_default": filter_reason,
                    "filter_reason_require_j2d": strict_filter_reason,
                }
                rows.append(row)
                if len(samples) < int(args.max_samples):
                    samples.append(sample_record(row, person, box, stats))

    split_summary = {
        "total_frames": total_frames,
        "total_persons": total_persons,
        "persons_per_frame": dict(sorted(persons_per_frame.items())),
        "key_counts": dict(key_counts.most_common()),
        "key_shape_examples": {key: dict(counter.most_common(8)) for key, counter in key_shape_counts.items()},
        "bbox_field_counts": dict(bbox_field_counts.most_common()),
        "j2d_field_counts": dict(j2d_field_counts.most_common()),
        "bbox_source_counts": dict(bbox_source_counts.most_common()),
        "visible_joint_hist": dict(sorted(visible_joint_hist.items())),
        "bbox_area_bins": dict(bbox_area_bins),
        "person_id_counts": dict(id_counts),
        "default_filter_counts": dict(filter_counts.most_common()),
        "strict_require_j2d_filter_counts": count_filter(rows, "filter_reason_require_j2d"),
    }
    return split_summary, rows, samples


def load_persons(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL annotation must be a list of person dicts: {path}")
    return persons


def load_image_hw(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.height, image.width
    except Exception:
        return 512, 512


def safe_extract_best_box(person: dict[str, Any], image_hw: tuple[int, int]) -> dict[str, Any]:
    try:
        return extract_best_box(person, image_hw)
    except Exception as exc:
        return {
            "bbox_valid": False,
            "bbox_source": f"error:{type(exc).__name__}",
            "bbox_xyxy_pixels": [0.0, 0.0, 0.0, 0.0],
            "bbox_cxcywh_norm": [0.0, 0.0, 0.0, 0.0],
        }


def classify_filter(
    *,
    bbox_valid: bool,
    has_j2d: bool,
    visible_joints: int,
    bbox_area: float,
    min_visible_joints: int,
    min_box_area: float,
    require_j2d: bool,
) -> str:
    if not bbox_valid:
        return "missing_box"
    if bbox_area < min_box_area:
        return "box_area_too_small"
    if has_j2d:
        if visible_joints < min_visible_joints:
            return "not_enough_visible_joints"
    elif require_j2d:
        return "missing_j2d_visibility"
    return "ok"


def sample_record(row: dict[str, Any], person: dict[str, Any], box: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "bbox_cxcywh_norm": box.get("bbox_cxcywh_norm"),
        "bbox_xyxy_pixels": box.get("bbox_xyxy_pixels"),
        "visibility": stats,
        "field_summaries": {key: value_signature(value) for key, value in sorted(person.items())},
    }


def value_signature(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return type(value).__name__
    try:
        arr = np.asarray(value)
        return f"{type(value).__name__}{tuple(arr.shape)}:{arr.dtype}"
    except Exception:
        return type(value).__name__


def area_bin(area: float) -> str:
    edges = [0, 10, 50, 100, 500, 1000, 5000, 20000, 100000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= area < hi:
            return f"[{lo},{hi})"
    return f">={edges[-1]}"


def count_filter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(row[key]) for row in rows)
    return dict(counter.most_common())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_human_summary(summary: dict[str, Any], summary_path: Path, output_dir: Path) -> None:
    print("========== BEDLAM annotation inspection ==========")
    print(f"Summary    : {summary_path}")
    print(f"Person CSV : {output_dir / 'person_rows.csv'}")
    print(f"Samples    : {output_dir / 'person_samples.json'}")
    for split, split_summary in summary["splits"].items():
        print(f"[{split}] frames={split_summary['total_frames']} persons={split_summary['total_persons']}")
        print(f"  bbox fields : {split_summary['bbox_field_counts']}")
        print(f"  j2d fields  : {split_summary['j2d_field_counts']}")
        print(f"  bbox sources: {split_summary['bbox_source_counts']}")
        print(f"  default filter      : {split_summary['default_filter_counts']}")
        print(f"  require-j2d filter  : {split_summary['strict_require_j2d_filter_counts']}")
        print(f"  persons/frame       : {split_summary['persons_per_frame']}")
    print("========== inspection finished ==========")


if __name__ == "__main__":
    main()
