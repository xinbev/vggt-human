#!/usr/bin/env python3
"""Validate the native EMDB-2 protocol tree without model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.data import load_emdb2_sequences  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--config", default="benchmarks/emdb2_global/config.yaml")
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--output", default="outputs/debug/emdb2_global_data_check/summary.json")
    args = parser.parse_args()
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    root = Path(
        args.emdb_root
        or require_path(cfg, str(cfg.get("data", {}).get("root_key", "datasets.emdb_root")))
    )
    sequences = load_emdb2_sequences(root)
    rows = [
        {
            "sequence": sequence.name,
            "frames": sequence.frame_count,
            "good_frames": int(sequence.good_frame_indices.size),
            "gender": sequence.gender,
            "annotation": str(sequence.annotation_path),
        }
        for sequence in sequences
    ]
    summary = {
        "emdb_root": str(root),
        "sequence_count": len(sequences),
        "frame_count": sum(row["frames"] for row in rows),
        "good_frame_count": sum(row["good_frames"] for row in rows),
        "sequences": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

