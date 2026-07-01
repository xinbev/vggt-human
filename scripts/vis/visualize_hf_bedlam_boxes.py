from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.hf_bedlam import _normalize_image_relpath, _resolve_image_path


def main() -> None:
    args = parse_args()
    npz_root = Path(args.npz_root).expanduser()
    images_root = Path(args.images_root).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = resolve_npz(npz_root, args.npz_file)
    with np.load(npz_path, allow_pickle=True) as data:
        names = [str(x) for x in data["imgname"].reshape(-1).tolist()]
        groups = group_by_image(names, npz_path.stem)
        selected_keys = list(groups.keys())[int(args.start_index) : int(args.start_index) + int(args.num_samples)]
        summary = {
            "npz": str(npz_path),
            "images_root": str(images_root),
            "output_dir": str(output_dir),
            "num_groups": len(groups),
            "samples": [],
        }
        for sample_idx, key in enumerate(selected_keys):
            indices = groups[key]
            rel = _normalize_image_relpath(names[indices[0]], npz_path.stem)
            image_path = _resolve_image_path(images_root, rel, npz_path.stem)
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            records = []
            for local_idx, person_idx in enumerate(indices):
                center_box = center_scale_box(data["center"][person_idx], data["scale"][person_idx], image.size)
                gtkps_box = points_box(data["gtkps"][person_idx], image.size) if "gtkps" in data.files else None
                proj_verts_box = points_box(data["proj_verts"][person_idx], image.size) if "proj_verts" in data.files else None
                base_training_box = proj_verts_box if proj_verts_box is not None else (gtkps_box if gtkps_box is not None else center_box)
                training_box = expanded_box(base_training_box, image.size, float(args.bbox_expand))
                draw_box(draw, training_box, (255, 225, 40), width=5)
                draw_box(draw, center_box, (255, 60, 60), width=4)
                if gtkps_box is not None:
                    draw_box(draw, gtkps_box, (40, 220, 90), width=3)
                if proj_verts_box is not None:
                    draw_box(draw, proj_verts_box, (70, 130, 255), width=2)
                cx, cy = data["center"][person_idx].reshape(-1)[:2]
                draw.text((float(cx) + 4, float(cy) + 4), f"p{local_idx}/i{person_idx}", fill=(255, 255, 255))
                records.append(
                    {
                        "person_index": int(person_idx),
                        "center": to_list(data["center"][person_idx]),
                        "scale": float(np.asarray(data["scale"][person_idx]).reshape(-1)[0]),
                        "training_box_xyxy": to_list(training_box),
                        "center_scale_box_xyxy": to_list(center_box),
                        "gtkps_box_xyxy": to_list(gtkps_box) if gtkps_box is not None else None,
                        "proj_verts_box_xyxy": to_list(proj_verts_box) if proj_verts_box is not None else None,
                        "trans_cam": to_list(data["trans_cam"][person_idx]) if "trans_cam" in data.files else None,
                    }
                )
            draw_legend(draw)
            out_path = output_dir / f"hf_bedlam_box_{sample_idx:03d}_{Path(rel).stem}.jpg"
            image.save(out_path, quality=95)
            summary["samples"].append({"image": str(image_path), "output": str(out_path), "persons": records})
            print(f"[vis] wrote {out_path}", flush=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HF BEDLAM center/scale boxes against gtkps/proj_verts boxes.")
    parser.add_argument("--npz-root", default="/home/zhw/xyb_space/bedlam/all_npz_12_training")
    parser.add_argument("--images-root", default="/home/zhw/xyb_space/bedlam/hf_bedlam/training_images")
    parser.add_argument("--npz-file", default="")
    parser.add_argument("--output-dir", default="outputs/vis/hf_bedlam_box_samples")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--bbox-expand", type=float, default=0.15)
    return parser.parse_args()


def resolve_npz(npz_root: Path, name: str) -> Path:
    if name:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = npz_root / path
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        if not path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {path}")
        return path
    files = sorted(npz_root.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found under {npz_root}")
    return files[0]


def group_by_image(names: list[str], scene: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        rel = _normalize_image_relpath(name, scene)
        groups.setdefault(rel, []).append(idx)
    return groups


def center_scale_box(center: np.ndarray, scale: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    cx, cy = np.asarray(center, dtype=np.float32).reshape(-1)[:2]
    side = float(np.asarray(scale, dtype=np.float32).reshape(-1)[0]) * 200.0
    x1 = max(float(cx - 0.5 * side), 0.0)
    y1 = max(float(cy - 0.5 * side), 0.0)
    x2 = min(float(cx + 0.5 * side), float(width - 1))
    y2 = min(float(cy + 0.5 * side), float(height - 1))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def points_box(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray | None:
    width, height = image_size
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        arr = arr.reshape(-1, arr.shape[-1])
    xy = arr[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    if arr.shape[1] >= 3:
        valid &= arr[:, 2] > 0.2
    valid &= xy[:, 0] >= 0
    valid &= xy[:, 1] >= 0
    valid &= xy[:, 0] <= float(width - 1)
    valid &= xy[:, 1] <= float(height - 1)
    if not bool(valid.any()):
        return None
    x1, y1 = xy[valid].min(axis=0)
    x2, y2 = xy[valid].max(axis=0)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def expanded_box(box: np.ndarray, image_size: tuple[int, int], expand: float) -> np.ndarray:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box.reshape(4)]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = 0.5 * float(expand) * bw
    pad_y = 0.5 * float(expand) * bh
    return np.asarray(
        [
            max(x1 - pad_x, 0.0),
            max(y1 - pad_y, 0.0),
            min(x2 + pad_x, float(width - 1)),
            min(y2 + pad_y, float(height - 1)),
        ],
        dtype=np.float32,
    )


def draw_box(draw: ImageDraw.ImageDraw, box: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    x1, y1, x2, y2 = [float(v) for v in box.reshape(4)]
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((10, 10, 430, 104), fill=(0, 0, 0))
    draw.text((20, 18), "yellow: current loader training box", fill=(255, 225, 40))
    draw.text((20, 40), "red: center/scale HMR crop box", fill=(255, 60, 60))
    draw.text((20, 62), "green: gtkps min/max box", fill=(40, 220, 90))
    draw.text((20, 84), "blue: proj_verts min/max box", fill=(70, 130, 255))


def to_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in np.asarray(value).reshape(-1).tolist()]


if __name__ == "__main__":
    main()
