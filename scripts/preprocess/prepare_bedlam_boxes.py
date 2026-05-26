import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam_boxes import extract_best_box, optional_smpl_projection_box, relative_sequence_name


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": {},
        "total_frames": 0,
        "total_persons": 0,
        "valid_boxes": 0,
        "missing_boxes": 0,
    }
    missing_examples: list[str] = []

    for split in args.splits:
        split_stats = process_split(dataset_root, output_root, split, args, missing_examples)
        summary["splits"][split] = split_stats
        for key in ("total_frames", "total_persons", "valid_boxes", "missing_boxes"):
            summary[key] += split_stats[key]

    summary["missing_examples"] = missing_examples[:50]
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
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--require-boxes", action="store_true")
    parser.add_argument("--use-smpl-projection", action="store_true")
    parser.add_argument("--smpl-model-dir", default="")
    return parser.parse_args()


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    args: argparse.Namespace,
    missing_examples: list[str],
) -> dict[str, int]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")

    stats = {"total_frames": 0, "total_persons": 0, "valid_boxes": 0, "missing_boxes": 0}
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
            if not smpl_path.is_file():
                continue
            frame = process_frame(rgb_path, smpl_path, args, missing_examples)
            with (out_dir / f"{rgb_path.stem}.pkl").open("wb") as file:
                pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
            stats["total_frames"] += 1
            stats["total_persons"] += len(frame["persons"])
            stats["valid_boxes"] += sum(1 for person in frame["persons"] if person["bbox_valid"])
            stats["missing_boxes"] += sum(1 for person in frame["persons"] if not person["bbox_valid"])
    return stats


def process_frame(rgb_path: Path, smpl_path: Path, args: argparse.Namespace, missing_examples: list[str]) -> dict[str, Any]:
    with Image.open(rgb_path) as image:
        image_hw = (image.height, image.width)
    with smpl_path.open("rb") as file:
        persons = pickle.load(file)
    if not isinstance(persons, list):
        raise TypeError(f"SMPL annotation must be a list of person dicts: {smpl_path}")

    out_persons = []
    for person_idx, person in enumerate(persons[: args.max_humans]):
        if not isinstance(person, dict):
            continue
        out_person = extract_best_box(person, image_hw)
        if not out_person["bbox_valid"] and args.use_smpl_projection:
            out_person.update(optional_smpl_projection_box(person, image_hw, args.smpl_model_dir))
        if not out_person["bbox_valid"] and len(missing_examples) < 50:
            missing_examples.append(f"{smpl_path}:{person_idx}")
        out_person["person_index"] = person_idx
        out_persons.append(out_person)

    return {
        "source_rgb": str(rgb_path),
        "source_smpl": str(smpl_path),
        "image_hw": image_hw,
        "target_image_size": int(args.image_size),
        "persons": out_persons,
    }


if __name__ == "__main__":
    main()
