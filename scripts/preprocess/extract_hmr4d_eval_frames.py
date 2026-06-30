import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.hmr4d_eval import EMDB1_NAMES, EMDB2_NAMES, _canonical_dataset_key, _safe_vid, _torch_load
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), {})
    dataset = _canonical_dataset_key(args.dataset)
    support_root = Path(args.support_root or resolve_support_root(config, dataset)).expanduser()
    frames_root = Path(args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")).expanduser()
    records = build_video_records(dataset, support_root)
    if args.max_sequences > 0:
        records = records[: args.max_sequences]
    summary = {"dataset": dataset, "support_root": str(support_root), "frames_root": str(frames_root), "sequences": []}
    for record in records:
        written = extract_video(record, frames_root, overwrite=args.overwrite, png_compression=args.png_compression)
        summary["sequences"].append({**record, "written_frames": written})
        print(f"[frames] {record['dataset_key']} {record['vid']} -> {written} frames", flush=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    summary_path = frames_root / f"{dataset}_extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RGB frames for EMDB/RICH/3DPW hmr4d_support eval adapters")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "emdb2", "rich", "3dpw"])
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--support-root", default="", help="Dataset hmr4d_support directory; defaults to configs/path.yaml")
    parser.add_argument("--frames-root", default="", help="Output root; defaults to datasets.hmr4d_eval_frames_root")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--png-compression", type=int, default=3)
    return parser.parse_args()


def resolve_support_root(config: dict[str, Any], dataset: str) -> str:
    key = {
        "emdb1": "datasets.emdb_hmr4d_support_root",
        "emdb2": "datasets.emdb_hmr4d_support_root",
        "rich": "datasets.rich_hmr4d_support_root",
        "3dpw": "datasets.threedpw_hmr4d_support_root",
    }[dataset]
    return require_path(config, key)


def build_video_records(dataset: str, support_root: Path) -> list[dict[str, Any]]:
    if dataset in {"emdb1", "emdb2"}:
        labels = _torch_load(support_root / "emdb_vit_v4.pt")
        names = EMDB1_NAMES if dataset == "emdb1" else EMDB2_NAMES
        return [
            {
                "dataset_key": dataset,
                "vid": vid,
                "safe_vid": _safe_vid(vid),
                "video_path": str(support_root / "videos" / f"{vid}.mp4"),
                "expected_frames": int(len(labels[vid]["mask"])),
            }
            for vid in names
            if vid in labels
        ]
    if dataset == "rich":
        labels = _torch_load(support_root / "rich_test_labels.pt")
        return [
            {
                "dataset_key": dataset,
                "vid": vid,
                "safe_vid": _safe_vid(vid),
                "video_path": str(support_root / "video" / vid / "video.mp4"),
                "expected_frames": int(len(label["frame_id"])),
            }
            for vid, label in sorted(labels.items())
        ]
    if dataset == "3dpw":
        labels = _torch_load(support_root / "test_3dpw_gt_labels.pt")
        return [
            {
                "dataset_key": dataset,
                "vid": vid,
                "safe_vid": _safe_vid(vid),
                "video_path": str(support_root / "videos" / f"{label['vname']}.mp4"),
                "expected_frames": int(len(label["mask_wham"])),
            }
            for vid, label in sorted(labels.items())
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_video(record: dict[str, Any], frames_root: Path, overwrite: bool, png_compression: int) -> int:
    video_path = Path(record["video_path"]).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found for {record['vid']}: {video_path}")
    out_dir = frames_root / record["dataset_key"] / record["safe_vid"] / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    if existing and not overwrite:
        return len(existing)
    if existing and overwrite:
        for path in existing:
            path.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = out_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)])
            frame_idx += 1
    finally:
        cap.release()
    return frame_idx


if __name__ == "__main__":
    main()
