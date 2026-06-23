import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def main() -> None:
    args = parse_args()
    sidecar_root = Path(args.sidecar_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_sidecar_frames(sidecar_root)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError(f"No frame sidecars found under {sidecar_root / 'smpl_boxes'}")

    writer = None
    video_path = output_dir / "tracks.mp4"
    summary_rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        image_path = resolve_image_path(frame, args)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        draw_frame(image, frame)
        out_path = output_dir / f"{idx:06d}_{frame['frame_id']}.jpg"
        cv2.imwrite(str(out_path), image)
        if args.write_video:
            if writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (width, height))
            writer.write(image)
        summary_rows.append(
            {
                "frame_id": frame["frame_id"],
                "frame_index": int(frame.get("frame_index", idx)),
                "image_path": str(image_path),
                "output_path": str(out_path),
                "person_ids": [
                    int(person["person_id"])
                    for person in frame.get("persons", [])
                    if person.get("valid", True) and person.get("person_id_valid", True)
                ],
            }
        )
    if writer is not None:
        writer.release()

    summary = {
        "sidecar_root": str(sidecar_root),
        "output_dir": str(output_dir),
        "num_frames": len(frames),
        "video_path": str(video_path) if args.write_video else "",
        "frames": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output_dir", "num_frames", "video_path")}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize video person track sidecars with persistent IDs")
    parser.add_argument("--sidecar-root", required=True, help="Directory containing smpl_boxes/*.pkl")
    parser.add_argument("--output-dir", default="outputs/vis/video_person_tracks")
    parser.add_argument("--dataset-root", default="", help="Optional BEDLAM root used if image_path in sidecar is not available")
    parser.add_argument("--bedlam-split", default="Training")
    parser.add_argument("--bedlam-sequence", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def load_sidecar_frames(sidecar_root: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for path in sorted((sidecar_root / "smpl_boxes").glob("*.pkl")):
        with path.open("rb") as file:
            frame = pickle.load(file)
        if not isinstance(frame, dict):
            raise TypeError(f"Expected dict frame sidecar: {path}")
        frame.setdefault("frame_id", path.stem)
        frames.append(frame)
    return sorted(frames, key=lambda item: int(item.get("frame_index", 0)))


def resolve_image_path(frame: dict[str, Any], args: argparse.Namespace) -> Path:
    image_path = Path(str(frame.get("image_path", ""))).expanduser()
    if image_path.is_file():
        return image_path
    if args.dataset_root and args.bedlam_sequence:
        candidate = (
            Path(args.dataset_root).expanduser()
            / args.bedlam_split
            / args.bedlam_sequence
            / "rgb"
            / f"{frame['frame_id']}.png"
        )
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Frame image not found for frame_id={frame.get('frame_id')}. "
        "Pass --dataset-root/--bedlam-sequence if the sidecar was moved across machines."
    )


def draw_frame(image: np.ndarray, frame: dict[str, Any]) -> None:
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
    title = f"{frame.get('frame_id')}  people={len(frame.get('persons', []))}"
    draw_label(image, title, 8, 8, (255, 255, 255))
    for person in frame.get("persons", []):
        if not person.get("valid", True) or not person.get("bbox_valid", True):
            continue
        person_id = int(person.get("person_id", -1))
        color = color_for_id(person_id)
        x1, y1, x2, y2 = [int(round(float(v))) for v in person["bbox_xyxy_pixels"]]
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"ID {person_id}  {float(person.get('track_confidence', 0.0)):.2f}"
        draw_label(image, label, x1, max(0, y1 - 24), color)


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y), (x + tw + 8, y + th + baseline + 8), (24, 24, 24), -1)
    cv2.rectangle(image, (x, y), (x + tw + 8, y + th + baseline + 8), color, 1)
    cv2.putText(image, text, (x + 4, y + th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_id(person_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(max(person_id, 0) + 1729)
    color = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


if __name__ == "__main__":
    main()
