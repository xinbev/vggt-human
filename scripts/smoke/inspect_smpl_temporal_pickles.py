#!/usr/bin/env python3
"""Validate native EMDB/3DPW pkl parsing before launching training."""

from __future__ import annotations

import argparse
import json

from vggt_omega.data import SMPLTemporalPickleDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threedpw-root", required=True)
    parser.add_argument("--emdb-root", required=True)
    parser.add_argument("--window-size", type=int, default=9)
    args = parser.parse_args()
    sources = [{"name": "3dpw", "root": args.threedpw_root}, {"name": "emdb", "root": args.emdb_root}]
    dataset = SMPLTemporalPickleDataset(sources, window_size=args.window_size, partition="train", validation_fraction=0.1)
    sample = dataset[0]
    print("========== Temporal-refiner train data summary ==========")
    print(json.dumps(dataset.summary(), indent=2, ensure_ascii=False))
    for key, value in sample.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
