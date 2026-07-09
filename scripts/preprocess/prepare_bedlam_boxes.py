import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam_boxes import (
    build_smpl_model_cache,
    extract_best_box,
    extract_visibility_stats,
    optional_smpl_projection_box,
    relative_sequence_name,
)

import numpy
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric

# Some BEDLAM pickle files may reference NumPy 2.x module names. Register the
# compatibility aliases only after project imports have loaded torch, because
# setting numpy._core before torch import can segfault in this environment.
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    smpl_models = None
    if args.use_smpl_projection:
        if not args.smpl_model_dir:
            raise ValueError("--use-smpl-projection requires --smpl-model-dir")
        smpl_models = build_smpl_model_cache(args.smpl_model_dir)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": {},
        "total_frames": 0,
        "total_persons": 0,
        "valid_boxes": 0,
        "trainable_persons": 0,
        "missing_boxes": 0,
        "filtered_persons": 0,
        "filter_reasons": {},
    }
    missing_examples: list[str] = []
    filtered_examples: list[str] = []

    for split in args.splits:
        split_stats = process_split(dataset_root, output_root, split, args, missing_examples, filtered_examples, smpl_models)
        summary["splits"][split] = split_stats
        for key in ("total_frames", "total_persons", "valid_boxes", "trainable_persons", "missing_boxes", "filtered_persons"):
            summary[key] += split_stats[key]
        for reason, count in split_stats["filter_reasons"].items():
            summary["filter_reasons"][reason] = summary["filter_reasons"].get(reason, 0) + count

    summary["missing_examples"] = missing_examples[:50]
    summary["filtered_examples"] = filtered_examples[:50]
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.require_boxes and summary["missing_boxes"]:
        raise RuntimeError(
            f"Preprocessing found {summary['missing_boxes']} people without boxes. "
            f"See {summary_path} for examples; use --use-smpl-projection only after wiring the dataset-specific projection path."
        )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare project-local BEDLAM bbox sidecar annotations")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["Training", "Test"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--require-boxes", action="store_true")
    parser.add_argument("--use-smpl-projection", action="store_true")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--projection-source", choices=("vertices", "joints"), default="vertices")
    parser.add_argument("--visible-only", action="store_true")
    parser.add_argument("--min-visible-joints", type=int, default=4)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--require-j2d-visibility", action="store_true")
    return parser.parse_args()


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    args: argparse.Namespace,
    missing_examples: list[str],
    filtered_examples: list[str],
    smpl_models: dict[str, Any] | None,
) -> dict[str, Any]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")

    stats: dict[str, Any] = {
        "total_frames": 0,
        "total_persons": 0,
        "valid_boxes": 0,
        "trainable_persons": 0,
        "missing_boxes": 0,
        "filtered_persons": 0,
        "filter_reasons": {},
    }
    for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        rgb_dir = seq_dir / "rgb"
        smpl_dir = seq_dir / "smpl"
        if not rgb_dir.is_dir() or not smpl_dir.is_dir():
            continue
        seq_name = relative_sequence_name(seq_dir, split_dir)
        out_dir = output_root / split / seq_name / "smpl_boxes"
        out_dir.mkdir(parents=True, exist_ok=True)

        for rgb_path in sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}):
            smpl_path = smpl_dir / f"{rgb_path.stem}.pkl"
            cam_path = seq_dir / "cam" / f"{rgb_path.stem}.npz"
            if not smpl_path.is_file():
                continue
            frame = process_frame(rgb_path, smpl_path, cam_path, args, missing_examples, filtered_examples, smpl_models)
            with (out_dir / f"{rgb_path.stem}.pkl").open("wb") as file:
                pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
            stats["total_frames"] += 1
            stats["total_persons"] += len(frame["persons"])
            stats["valid_boxes"] += sum(1 for person in frame["persons"] if person["bbox_valid"])
            stats["trainable_persons"] += sum(1 for person in frame["persons"] if person.get("train_valid", person["bbox_valid"]))
            stats["missing_boxes"] += sum(1 for person in frame["persons"] if not person.get("raw_bbox_valid", person["bbox_valid"]))
            for person in frame["persons"]:
                if person.get("train_valid", person["bbox_valid"]):
                    continue
                stats["filtered_persons"] += 1
                reason = str(person.get("filtered_reason", "unknown"))
                stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
    return stats


def process_frame(
    rgb_path: Path,
    smpl_path: Path,
    cam_path: Path,
    args: argparse.Namespace,
    missing_examples: list[str],
    filtered_examples: list[str],
    smpl_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with Image.open(rgb_path) as image:
        image_hw = (image.height, image.width)
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL annotation must be a list of person dicts: {smpl_path}")
    intrinsics = load_intrinsics(cam_path) if args.use_smpl_projection else None

    out_persons = []
    for person_idx, person in enumerate(persons[: args.max_humans]):
        if not isinstance(person, dict):
            continue
        out_person = extract_best_box(person, image_hw)
        if not out_person["bbox_valid"] and args.use_smpl_projection:
            out_person.update(
                optional_smpl_projection_box(
                    person,
                    image_hw,
                    args.smpl_model_dir,
                    intrinsics,
                    smpl_models=smpl_models,
                    projection_source=args.projection_source,
                )
            )
        if not out_person["bbox_valid"] and len(missing_examples) < 50:
            missing_examples.append(f"{smpl_path}:{person_idx}")
        out_person["person_index"] = person_idx
        apply_visibility_filter(out_person, person, image_hw, args)
        if not out_person["train_valid"] and len(filtered_examples) < 50:
            filtered_examples.append(f"{smpl_path}:{person_idx}:{out_person['filtered_reason']}")
        out_persons.append(out_person)

    return {
        "source_rgb": str(rgb_path),
        "source_smpl": str(smpl_path),
        "image_hw": image_hw,
        "target_image_size": int(args.image_size),
        "persons": out_persons,
    }


def load_intrinsics(cam_path: Path) -> np.ndarray:
    if not cam_path.is_file():
        raise FileNotFoundError(f"Camera file not found for SMPL projection: {cam_path}")
    data = np.load(cam_path)
    if "intrinsics" not in data:
        raise ValueError(f"Camera file missing 'intrinsics': {cam_path}")
    return np.asarray(data["intrinsics"], dtype=np.float32).reshape(3, 3)


def apply_visibility_filter(out_person: dict[str, Any], person: dict[str, Any], image_hw: tuple[int, int], args: argparse.Namespace) -> None:
    raw_bbox_valid = bool(out_person.get("bbox_valid", False))
    stats = extract_visibility_stats(person, image_hw, bbox_xyxy_pixels=out_person.get("bbox_xyxy_pixels"))
    out_person.update(stats)
    out_person["raw_bbox_valid"] = raw_bbox_valid
    out_person["raw_bbox_source"] = str(out_person.get("bbox_source", "missing"))
    out_person["raw_bbox_cxcywh_norm"] = list(out_person.get("bbox_cxcywh_norm", [0.0, 0.0, 0.0, 0.0]))
    out_person["raw_bbox_xyxy_pixels"] = list(out_person.get("bbox_xyxy_pixels", [0.0, 0.0, 0.0, 0.0]))

    train_valid = raw_bbox_valid
    reason = "ok" if raw_bbox_valid else "missing_box"
    if train_valid and float(stats["bbox_area_pixels"]) < float(args.min_box_area):
        train_valid = False
        reason = "box_area_too_small"
    if train_valid and bool(args.visible_only):
        if bool(stats["has_j2d_visibility"]):
            if int(stats["visible_joints"]) < int(args.min_visible_joints):
                train_valid = False
                reason = "not_enough_visible_joints"
        elif bool(args.require_j2d_visibility):
            train_valid = False
            reason = "missing_j2d_visibility"

    out_person["train_valid"] = bool(train_valid)
    out_person["valid"] = bool(train_valid)
    out_person["filtered_reason"] = reason
    if not train_valid:
        out_person["bbox_valid"] = False
        out_person["bbox_cxcywh_norm"] = [0.0, 0.0, 0.0, 0.0]
        out_person["bbox_xyxy_pixels"] = [0.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    main()
