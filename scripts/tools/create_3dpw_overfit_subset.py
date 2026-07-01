from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = int(args.start_index)
    count = int(args.num_samples)
    if count <= 0:
        raise ValueError(f"num_samples must be positive, got {count}")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset_index"])
        writer.writeheader()
        for idx in range(start, start + count):
            writer.writerow({"dataset_index": idx})
    print(f"[3dpw-overfit-subset] wrote {count} indices -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sequential 3DPW dataset_index CSV for overfit diagnostics.")
    parser.add_argument("--output", default="outputs/debug/3dpw_overfit_subset/train_100_indices.csv")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
