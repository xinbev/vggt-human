from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = parse_args()
    npz_root = Path(args.npz_root).expanduser()
    if not npz_root.is_dir():
        raise FileNotFoundError(f"NPZ root not found: {npz_root}")
    files = sorted(npz_root.glob("*.npz"))
    if args.file:
        files = [npz_root / args.file]
    files = files[: max(int(args.max_files), 1)]
    if not files:
        raise RuntimeError(f"No npz files found under {npz_root}")
    summaries = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            item: dict[str, Any] = {"file": str(path), "keys": []}
            for key in data.files:
                arr = data[key]
                entry: dict[str, Any] = {
                    "key": key,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
                if arr.size > 0 and len(item["keys"]) < int(args.preview_keys):
                    try:
                        sample = arr.reshape(-1)[0]
                        if isinstance(sample, bytes):
                            sample = sample.decode("utf-8", errors="replace")
                        elif not np.isscalar(sample):
                            sample = np.asarray(sample).reshape(-1)[: min(8, np.asarray(sample).size)].tolist()
                        else:
                            sample = sample.item() if hasattr(sample, "item") else sample
                        entry["sample"] = sample
                    except Exception as exc:
                        entry["sample_error"] = str(exc)
                item["keys"].append(entry)
            summaries.append(item)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HuggingFace BEDLAM all_npz annotation keys")
    parser.add_argument("--npz-root", default="/home/zhw/xyb_space/bedlam/all_npz_12_training")
    parser.add_argument("--file", default="")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--preview-keys", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main()
