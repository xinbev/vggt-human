from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.training.hsi_stage2_v4_noise import (
    V4_NOISE_CATEGORIES,
    V4_RAY_LEVELS,
    V4_TANGENT_LEVELS_M,
    deterministic_v4_assignment,
)


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset_indices_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = read_indices(subset_path)
    rows = []
    categories: Counter[str] = Counter()
    ray_signs: Counter[str] = Counter()
    for dataset_index in indices:
        for person_slot in range(args.max_humans):
            assignment = deterministic_v4_assignment(dataset_index, person_slot, args.seed, args.epoch)
            category = V4_NOISE_CATEGORIES[assignment.category_id]
            categories[category] += 1
            if assignment.ray_ratio != 0.0:
                ray_signs["positive" if assignment.ray_ratio > 0.0 else "negative"] += 1
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "person_slot": person_slot,
                    "category": category,
                    "ray_ratio": assignment.ray_ratio,
                    "tangent_x_m": assignment.tangent_x_m,
                    "tangent_y_m": assignment.tangent_y_m,
                }
            )
    assignment_path = output_dir / "overfit_assignments.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "contract": "hsi_stage2_v4_deterministic_v1",
        "seed": args.seed,
        "epoch": args.epoch,
        "max_humans": args.max_humans,
        "num_dataset_indices": len(indices),
        "num_assignments": len(rows),
        "categories": dict(categories),
        "ray_signs": dict(ray_signs),
        "ray_levels": list(V4_RAY_LEVELS),
        "tangent_levels_m": list(V4_TANGENT_LEVELS_M),
        "subset_indices_csv": str(subset_path),
        "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
        "assignments_csv": str(assignment_path),
    }
    (output_dir / "noise_contract.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def read_indices(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "dataset_index" not in reader.fieldnames:
            raise ValueError(f"Expected dataset_index column in {path}")
        return [int(row["dataset_index"]) for row in reader]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-indices-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
