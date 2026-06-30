import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), {})
    support_root = args.support_root or support_key(config, args.dataset)
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    sidecar_root = args.sidecar_root or str(config.get("datasets", {}).get("hmr4d_eval_tracks_root", "") or "")
    dataset = HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=sidecar_root or None,
        sequence_length=args.sequence_length,
        stride=args.stride,
        image_size=args.image_size,
        max_humans=args.max_humans,
        patch_size=args.patch_size,
        full_sequence=args.full_sequence,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=hmr4d_eval_collate_fn)
    batch = next(iter(loader))
    summary = {
        "dataset": args.dataset,
        "num_windows": len(dataset),
        "batch_size": args.batch_size,
        "meta": batch["meta"],
        "tensor_shapes": tensor_shapes(batch),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check EMDB/RICH/3DPW hmr4d_support data adapter")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "emdb2", "rich", "3dpw"])
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--sidecar-root", default="")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--max-humans", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--full-sequence", action="store_true")
    return parser.parse_args()


def support_key(config: dict[str, Any], dataset: str) -> str:
    key = {
        "emdb1": "datasets.emdb_hmr4d_support_root",
        "emdb2": "datasets.emdb_hmr4d_support_root",
        "rich": "datasets.rich_hmr4d_support_root",
        "3dpw": "datasets.threedpw_hmr4d_support_root",
    }[dataset]
    return require_path(config, key)


def tensor_shapes(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_shapes(value[0])] if value else []
    return type(value).__name__


if __name__ == "__main__":
    main()
