#!/usr/bin/env python3
"""Inspect HMR4D support IDs and raw RGB mappings before expensive evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data import HMR4DSupportEvalDataset
from vggt_omega.training.config import load_yaml_config, require_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="3dpw", choices=["3dpw", "emdb1"])
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--filter", default="")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    config = load_yaml_config(args.path_config)
    support_key = "datasets.threedpw_hmr4d_support_root" if args.dataset == "3dpw" else "datasets.emdb_hmr4d_support_root"
    support_root = Path(args.support_root or require_path(config, support_key))
    frames_root = Path(args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root"))
    dataset = HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sequence_length=9,
        image_resolution=512,
        max_humans=20,
    )
    query = str(args.filter or "").lower().strip()
    rows = []
    for record in dataset.records:
        vname = str(record.label.get("vname", "") or "")
        if query and query not in record.vid.lower() and query not in vname.lower():
            continue
        raw_id = _frame_id(record.label.get("frame_id"), 0)
        candidates = _candidates(frames_root, record, raw_id)
        rows.append(
            {
                "vid": record.vid,
                "vname": vname,
                "length": record.length,
                "frame0_support_index": 0,
                "frame0_source_id": raw_id,
                "rgb_candidates": [str(path) for path in candidates],
                "rgb_existing": [str(path) for path in candidates if path.is_file()],
            }
        )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "support_root": str(support_root),
                "frames_root": str(frames_root),
                "support_record_count": len(dataset.records),
                "filter": args.filter,
                "matched_records": len(rows),
                "records": rows[: int(args.limit)],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _frame_id(value: object, index: int) -> int:
    try:
        import torch

        values = torch.as_tensor(value).reshape(-1)
        if index < values.numel():
            return int(values[index].item())
    except Exception:
        pass
    return int(index)


def _candidates(frames_root: Path, record: object, frame_id: int) -> list[Path]:
    label = record.label
    sequence = str(label.get("vname", "") or record.vid.rsplit("_", 1)[0])
    return [
        frames_root / record.dataset_key / record.safe_vid / "rgb" / "000000.png",
        frames_root / sequence / f"image_{frame_id:05d}.jpg",
        frames_root / sequence / "image_00000.jpg",
    ]


if __name__ == "__main__":
    main()
