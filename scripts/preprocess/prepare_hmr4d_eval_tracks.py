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

from scripts.preprocess.extract_hmr4d_eval_frames import build_video_records, resolve_support_root
from vggt_omega.data.hmr4d_eval import _canonical_dataset_key
from vggt_omega.training.config import load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.path_config)
    dataset = _canonical_dataset_key(args.dataset)
    support_root = resolve_project_path(args.support_root or resolve_support_root(config, dataset))
    frames_root = resolve_project_path(args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root"))
    output_root = resolve_project_path(args.output_root or require_path(config, "datasets.hmr4d_eval_tracks_root"))
    records = build_video_records(dataset, support_root)
    if int(args.max_sequences) > 0:
        records = records[: int(args.max_sequences)]
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "kind": "hmr4d_eval_tracks",
        "dataset": dataset,
        "support_root": str(support_root),
        "frames_root": str(frames_root),
        "output_root": str(output_root),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "enable_sam2_masks": not args.no_sam2_masks,
        "sequences": [],
    }
    failures = 0
    for idx, record in enumerate(records):
        safe_vid = record["safe_vid"]
        frames_dir = frames_root / dataset / safe_vid / "rgb"
        out_dir = output_root / dataset / safe_vid
        summary_path = out_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            row = {"vid": record["vid"], "safe_vid": safe_vid, "status": "skipped_existing", "output_root": str(out_dir)}
            manifest["sequences"].append(row)
            print(f"[skip] {idx + 1}/{len(records)} {dataset}/{safe_vid}", flush=True)
            continue
        if not frames_dir.is_dir():
            failures += 1
            row = {
                "vid": record["vid"],
                "safe_vid": safe_vid,
                "status": "missing_frames",
                "frames_dir": str(frames_dir),
                "hint": "Run scripts/preprocess/extract_hmr4d_eval_frames.py first.",
            }
            manifest["sequences"].append(row)
            write_manifest(output_root, dataset, manifest, failures)
            if args.fail_fast:
                raise FileNotFoundError(row["hint"] + f" Missing: {frames_dir}")
            continue

        command = build_track_command(args, dataset, safe_vid, frames_dir, output_root)
        print(f"[run] {idx + 1}/{len(records)} {dataset}/{safe_vid}", flush=True)
        started = time.monotonic()
        proc = subprocess.run(command, cwd=str(ROOT), check=False)
        elapsed = time.monotonic() - started
        if proc.returncode == 0 and summary_path.is_file():
            row = {
                "vid": record["vid"],
                "safe_vid": safe_vid,
                "status": "ok",
                "output_root": str(out_dir),
                "summary_json": str(summary_path),
                "elapsed_sec": round(elapsed, 3),
            }
        else:
            failures += 1
            row = {
                "vid": record["vid"],
                "safe_vid": safe_vid,
                "status": "failed",
                "returncode": int(proc.returncode),
                "output_root": str(out_dir),
                "elapsed_sec": round(elapsed, 3),
            }
            if args.fail_fast:
                manifest["sequences"].append(row)
                write_manifest(output_root, dataset, manifest, failures)
                raise SystemExit(proc.returncode or 1)
        manifest["sequences"].append(row)
        write_manifest(output_root, dataset, manifest, failures)
    write_manifest(output_root, dataset, manifest, failures)
    if failures:
        raise SystemExit(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO/SAM2/BoostTrack sidecars for EMDB/RICH/3DPW eval frames")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "emdb2", "rich", "3dpw"])
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
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


def build_track_command(args: argparse.Namespace, dataset: str, safe_vid: str, frames_dir: Path, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/preprocess/prepare_video_person_tracks.py",
        "--frames-dir",
        str(frames_dir),
        "--source-name",
        safe_vid,
        "--output-subdir",
        f"{dataset}/{safe_vid}",
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


def write_manifest(output_root: Path, dataset: str, manifest: dict[str, Any], failures: int) -> None:
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["num_failed"] = int(failures)
    manifest["num_ok"] = sum(1 for row in manifest["sequences"] if row.get("status") == "ok")
    manifest["num_skipped"] = sum(1 for row in manifest["sequences"] if row.get("status") == "skipped_existing")
    path = output_root / f"{dataset}_tracks_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
