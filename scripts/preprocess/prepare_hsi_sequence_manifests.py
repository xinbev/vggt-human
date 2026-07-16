from __future__ import annotations

import argparse
import hashlib
import json
import csv
from pathlib import Path


def main() -> None:
    args = parse_args()
    split_root = Path(args.bedlam_root).expanduser() / args.split
    sequences = sorted(
        path.name
        for path in split_root.iterdir()
        if (path / "rgb").is_dir()
        and (path / "depth").is_dir()
        and (path / "smpl").is_dir()
        and (path / "cam").is_dir()
    )
    if len(sequences) < 2:
        raise RuntimeError(f"Need at least two BEDLAM sequences under {split_root}")
    ordered = sorted(sequences, key=lambda name: hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest())
    val_count = max(1, min(len(ordered) - 1, round(len(ordered) * args.val_ratio)))
    val = sorted(ordered[:val_count])
    train = sorted(ordered[val_count:])
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_sequences.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (output / "val_sequences.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    with (output / "overfit64_indices.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_index"])
        writer.writerows([[index] for index in range(64)])
    summary = {
        "bedlam_root": str(Path(args.bedlam_root).expanduser()),
        "split": args.split,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_train_sequences": len(train),
        "num_val_sequences": len(val),
        "overlap": sorted(set(train) & set(val)),
        "train_manifest": str(output / "train_sequences.txt"),
        "val_manifest": str(output / "val_sequences.txt"),
        "overfit_subset": str(output / "overfit64_indices.csv"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic sequence-level BEDLAM train/val manifests")
    parser.add_argument("--bedlam-root", required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
