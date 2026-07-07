from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    args = parse_args()
    frames_dir = resolve_project_path(args.frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    source_name = args.source_name or frames_dir.name
    run_root = resolve_project_path(args.output_root) / source_name
    original_dir = run_root / "original_frames"
    erased_dir = run_root / "erased_frames"
    union_mask_dir = run_root / "person_masks"
    tracks_dir = run_root / "tracks"

    if run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {run_root}. Use --overwrite to replace/reuse it.")

    if args.overwrite:
        clear_known_output_dirs(run_root, [original_dir, erased_dir, union_mask_dir, tracks_dir])

    run_root.mkdir(parents=True, exist_ok=True)
    copy_original_frames(frames_dir, original_dir, overwrite=args.overwrite)

    run_tracking(args, original_dir=original_dir, run_root=run_root)
    compose_union_masks(original_dir, tracks_dir, union_mask_dir, dilation=args.mask_dilate)
    run_omnieraser(args, original_dir=original_dir, union_mask_dir=union_mask_dir, erased_dir=erased_dir)

    manifest = {
        "kind": "human_erased_frames",
        "source_frames_dir": str(frames_dir),
        "source_name": source_name,
        "output_root": str(run_root),
        "original_frames_dir": str(original_dir),
        "person_masks_dir": str(union_mask_dir),
        "tracks_dir": str(tracks_dir),
        "erased_frames_dir": str(erased_dir),
        "num_frames": len(list_image_files(original_dir)),
        "mask_dilate": int(args.mask_dilate),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO -> SAM2 -> OmniEraser for a folder of ordered frames."
    )
    parser.add_argument("frames_dir", help="Input directory containing ordered RGB frames.")
    parser.add_argument("--output-root", default="outputs/preprocess/human_erasure")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-log-interval", type=int, default=50)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--max-age", type=int, default=90)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=10.0)
    parser.add_argument("--sam2-single-mask", action="store_true")

    parser.add_argument("--mask-dilate", type=int, default=8)
    parser.add_argument("--erase-resolution", type=int, default=1024)
    parser.add_argument("--erase-steps", type=int, default=28)
    parser.add_argument("--erase-seed", type=int, default=24)
    parser.add_argument("--erase-guidance-scale", type=float, default=3.5)
    parser.add_argument("--erase-true-guidance-scale", type=float, default=1.0)
    parser.add_argument("--erase-controlnet-conditioning-scale", type=float, default=0.9)
    parser.add_argument("--erase-prompt", default="There is nothing here.")
    return parser.parse_args()


def copy_original_frames(source_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in list_image_files(source_dir):
        target = output_dir / frame_path.name
        if target.exists() and not overwrite:
            continue
        shutil.copy2(frame_path, target)


def clear_known_output_dirs(run_root: Path, dirs: list[Path]) -> None:
    resolved_root = run_root.resolve()
    for path in dirs:
        if not path.exists():
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(f"Refusing to remove path outside run root: {resolved_path}")
        shutil.rmtree(resolved_path)


def run_tracking(args: argparse.Namespace, *, original_dir: Path, run_root: Path) -> None:
    command = [
        sys.executable,
        "scripts/preprocess/prepare_video_person_tracks.py",
        "--frames-dir",
        str(original_dir),
        "--path-config",
        str(args.path_config),
        "--output-root",
        str(run_root),
        "--output-subdir",
        "tracks",
        "--source-name",
        str(args.source_name or original_dir.parent.name),
        "--overwrite",
        "--enable-sam2-masks",
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
        "--frame-log-interval",
        str(args.frame_log_interval),
    ]
    if int(args.max_frames) > 0:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.sam2_single_mask:
        command.append("--sam2-single-mask")
    run_command(command)


def compose_union_masks(
    original_dir: Path,
    tracks_dir: Path,
    output_dir: Path,
    *,
    dilation: int,
) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_npz_dir = tracks_dir / "masks"
    kernel = None
    if dilation > 0:
        size = int(dilation) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for frame_path in list_image_files(original_dir):
        image = Image.open(frame_path)
        width, height = image.size
        union = np.zeros((height, width), dtype=np.uint8)
        npz_path = mask_npz_dir / f"{frame_path.stem}.npz"
        if npz_path.is_file():
            with np.load(npz_path) as data:
                for key in data.files:
                    mask = np.asarray(data[key]).astype(bool)
                    if mask.shape != union.shape:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    union[mask] = 255
        if kernel is not None and int(union.max()) > 0:
            union = cv2.dilate(union, kernel, iterations=1)
        Image.fromarray(union, mode="L").save(output_dir / f"{frame_path.stem}.png")


def run_omnieraser(
    args: argparse.Namespace,
    *,
    original_dir: Path,
    union_mask_dir: Path,
    erased_dir: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/preprocess/omnieraser_remove.py",
        "--image-dir",
        str(original_dir),
        "--mask-dir",
        str(union_mask_dir),
        "--output-dir",
        str(erased_dir),
        "--suffix",
        "",
        "--path-config",
        str(args.path_config),
        "--device",
        str(args.device),
        "--resolution",
        str(args.erase_resolution),
        "--steps",
        str(args.erase_steps),
        "--seed",
        str(args.erase_seed),
        "--guidance-scale",
        str(args.erase_guidance_scale),
        "--true-guidance-scale",
        str(args.erase_true_guidance_scale),
        "--controlnet-conditioning-scale",
        str(args.erase_controlnet_conditioning_scale),
        "--prompt",
        str(args.erase_prompt),
        "--skip-empty-masks",
    ]
    run_command(command)


def run_command(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def list_image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
