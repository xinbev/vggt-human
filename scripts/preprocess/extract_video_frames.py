from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = parse_args()
    video_path = resolve_project_path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    source_name = args.source_name or video_path.stem
    output_dir = resolve_project_path(args.output_dir) / source_name
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        image_ext=args.image_ext,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        frame_id_width=args.frame_id_width,
        jpeg_quality=args.jpeg_quality,
        png_compression=args.png_compression,
    )
    manifest = {
        "kind": "video_frames",
        "video_path": str(video_path),
        "source_name": source_name,
        "output_dir": str(output_dir),
        "image_ext": args.image_ext,
        "start_frame": int(args.start_frame),
        "end_frame": int(args.end_frame),
        "stride": int(args.stride),
        "max_frames": int(args.max_frames),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract an ordered frame directory from a video.")
    parser.add_argument("video", help="Input video path.")
    parser.add_argument("--output-dir", default="outputs/preprocess/video_frames")
    parser.add_argument("--source-name", default="", help="Output subdirectory name; defaults to video stem.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="Inclusive end frame; -1 means video end.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--frame-id-width", type=int, default=6)
    parser.add_argument("--image-ext", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0")
    if args.stride <= 0:
        parser.error("--stride must be > 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    return args


def extract_frames(
    *,
    video_path: Path,
    output_dir: Path,
    image_ext: str,
    start_frame: int,
    end_frame: int,
    stride: int,
    max_frames: int,
    frame_id_width: int,
    jpeg_quality: int,
    png_compression: int,
) -> dict[str, int | float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    write_params = []
    if image_ext == "jpg":
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif image_ext == "png":
        write_params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    read_index = int(start_frame)
    written = 0
    try:
        while True:
            if end_frame >= 0 and read_index > end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (read_index - start_frame) % stride == 0:
                frame_name = f"{written:0{frame_id_width}d}.{image_ext}"
                output_path = output_dir / frame_name
                if not cv2.imwrite(str(output_path), frame, write_params):
                    raise RuntimeError(f"Failed to write frame: {output_path}")
                written += 1
                if max_frames > 0 and written >= max_frames:
                    break
            read_index += 1
    finally:
        cap.release()

    return {
        "video_total_frames": total_frames,
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
        "frames_written": written,
    }


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
