import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.training.config import load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.path_config)
    bedlam_root = resolve_project_path(args.bedlam_root or require_path(config, "datasets.bedlam_root"))
    output_root = resolve_project_path(args.output_root or require_path(config, "datasets.bedlam_tracks_root"))
    sequences = select_sequences(list_bedlam_sequences(bedlam_root, args.split), args)
    if not sequences:
        raise RuntimeError("No BEDLAM sequences selected")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "bedlam_full_system_tracks",
        "bedlam_root": str(bedlam_root),
        "split": args.split,
        "output_root": str(output_root),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_selected": len(sequences),
        "enable_sam2_masks": not args.no_sam2_masks,
        "sequences": [],
    }

    failures = 0
    for seq_idx, sequence in enumerate(sequences):
        rel = sequence.relative_to(bedlam_root / args.split).as_posix()
        out_dir = output_root / args.split / rel
        summary_path = out_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            row = {"sequence": rel, "status": "skipped_existing", "output_root": str(out_dir)}
            manifest["sequences"].append(row)
            print(f"[skip] {seq_idx + 1}/{len(sequences)} {rel}", flush=True)
            continue

        command = build_sequence_command(args, rel, output_root)
        print(f"[run] {seq_idx + 1}/{len(sequences)} {rel}", flush=True)
        started = time.monotonic()
        proc = subprocess.run(command, cwd=str(ROOT), check=False)
        elapsed = time.monotonic() - started
        if proc.returncode == 0 and summary_path.is_file():
            row = {
                "sequence": rel,
                "status": "ok",
                "output_root": str(out_dir),
                "summary_json": str(summary_path),
                "elapsed_sec": round(elapsed, 3),
            }
        else:
            failures += 1
            row = {
                "sequence": rel,
                "status": "failed",
                "output_root": str(out_dir),
                "returncode": int(proc.returncode),
                "elapsed_sec": round(elapsed, 3),
            }
            print(f"[fail] {rel} returncode={proc.returncode}", flush=True)
            if args.fail_fast:
                manifest["sequences"].append(row)
                write_manifest(output_root, args.split, manifest, failures)
                raise SystemExit(proc.returncode or 1)
        manifest["sequences"].append(row)
        write_manifest(output_root, args.split, manifest, failures)

    write_manifest(output_root, args.split, manifest, failures)
    if failures:
        raise SystemExit(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate full-system BEDLAM detection/SAM2/track sidecars")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--sequence", action="append", default=[], help="Specific sequence name relative to <bedlam-root>/<split>; can repeat")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-log-interval", type=int, default=100)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--max-age", type=int, default=90)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=10.0)
    parser.add_argument("--stitch-max-gap", type=int, default=30)
    parser.add_argument("--stitch-center-thresh", type=float, default=1.25)
    parser.add_argument("--stitch-size-log-thresh", type=float, default=0.70)
    parser.add_argument("--stitch-min-score", type=float, default=0.25)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--use-ecc", action="store_true")
    parser.add_argument("--no-sam2-masks", action="store_true")
    parser.add_argument("--sam2-single-mask", action="store_true")
    return parser.parse_args()


def list_bedlam_sequences(bedlam_root: Path, split: str) -> list[Path]:
    split_dir = bedlam_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")
    sequences = sorted(path for path in split_dir.iterdir() if (path / "rgb").is_dir())
    if not sequences:
        raise RuntimeError(f"No BEDLAM sequences with rgb/ found under {split_dir}")
    return sequences


def select_sequences(sequences: list[Path], args: argparse.Namespace) -> list[Path]:
    if args.sequence:
        by_name = {path.name: path for path in sequences}
        selected = []
        missing = []
        for name in args.sequence:
            key = str(name).strip().replace("\\", "/")
            match = by_name.get(key)
            if match is None:
                missing.append(key)
            else:
                selected.append(match)
        if missing:
            raise FileNotFoundError(f"BEDLAM sequences not found: {missing}")
        return selected
    start = max(int(args.start_index), 0)
    selected = sequences[start:]
    max_sequences = int(args.max_sequences)
    return selected[:max_sequences] if max_sequences > 0 else selected


def build_sequence_command(args: argparse.Namespace, sequence: str, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/preprocess/prepare_video_person_tracks.py",
        "--bedlam-sequence",
        sequence,
        "--bedlam-split",
        str(args.split),
        "--path-config",
        str(args.path_config),
        "--output-root",
        str(output_root),
        "--device",
        str(args.device),
        "--detector-image-size",
        str(args.detector_image_size),
        "--det-conf",
        str(args.det_conf),
        "--det-iou",
        str(args.det_iou),
        "--max-age",
        str(args.max_age),
        "--min-hits",
        str(args.min_hits),
        "--aspect-ratio-thresh",
        str(args.aspect_ratio_thresh),
        "--stitch-max-gap",
        str(args.stitch_max_gap),
        "--stitch-center-thresh",
        str(args.stitch_center_thresh),
        "--stitch-size-log-thresh",
        str(args.stitch_size_log_thresh),
        "--stitch-min-score",
        str(args.stitch_min_score),
        "--frame-log-interval",
        str(args.frame_log_interval),
    ]
    if args.bedlam_root:
        command.extend(["--bedlam-root", str(args.bedlam_root)])
    if args.overwrite:
        command.append("--overwrite")
    if int(args.max_frames) > 0:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.no_reid:
        command.append("--no-reid")
    if args.use_ecc:
        command.append("--use-ecc")
    if not args.no_sam2_masks:
        command.append("--enable-sam2-masks")
    if args.sam2_single_mask:
        command.append("--sam2-single-mask")
    return command


def write_manifest(output_root: Path, split: str, manifest: dict[str, Any], failures: int) -> None:
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["num_failed"] = int(failures)
    manifest["num_ok"] = sum(1 for row in manifest["sequences"] if row.get("status") == "ok")
    manifest["num_skipped"] = sum(1 for row in manifest["sequences"] if row.get("status") == "skipped_existing")
    path = output_root / f"bedlam_{split}_full_system_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
