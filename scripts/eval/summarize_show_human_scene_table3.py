#!/usr/bin/env python3
"""Combine 3DPW and EMDB-1 SHOW metrics into one Table-3-style artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")


def main() -> None:
    args = parse_args()
    sources = {"3dpw": Path(args.threedpw), "emdb1": Path(args.emdb1)}
    datasets = {}
    coverage = {}
    for dataset, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset} summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = payload.get("table3")
        if not isinstance(table, dict) or any(table.get(key) is None for key in METRICS):
            raise ValueError(f"Summary does not contain a complete table3 block: {path}")
        datasets[dataset] = {key: float(table[key]) for key in METRICS}
        coverage[dataset] = {
            "num_sequences_evaluated": payload.get("num_sequences_evaluated"),
            "num_person_frames": payload.get("num_person_frames"),
            "num_windows": payload.get("num_windows"),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method,
        "mask_source": "gt_smpl_projection",
        "lower_is_better": True,
        "columns": list(METRICS),
        "datasets": datasets,
        "coverage": coverage,
        "sources": {key: str(value) for key, value in sources.items()},
    }
    json_path = output_dir / "show_table3_summary.json"
    csv_path = output_dir / "show_table3_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "dataset", "mask_source", *METRICS))
        writer.writeheader()
        for dataset, values in datasets.items():
            writer.writerow(
                {
                    "method": args.method,
                    "dataset": dataset,
                    "mask_source": "gt_smpl_projection",
                    **values,
                }
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threedpw", required=True)
    parser.add_argument("--emdb1", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="VGGT-Omega")
    return parser.parse_args()


if __name__ == "__main__":
    main()
