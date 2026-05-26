import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (255, 64, 64),
    (64, 192, 255),
    (64, 255, 128),
    (255, 192, 64),
    (192, 64, 255),
    (255, 64, 192),
]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    boxes_root = Path(args.boxes_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    box_paths = sorted((boxes_root / args.split).glob("*/smpl_boxes/*.pkl"))
    if not box_paths:
        raise RuntimeError(f"No bbox sidecar files found under {boxes_root / args.split}")

    selected = select_samples(box_paths, args.num_samples)
    summary: list[dict[str, Any]] = []
    for index, box_path in enumerate(selected):
        with box_path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict sidecar: {box_path}")
        image_path = Path(data.get("source_rgb", ""))
        if not image_path.is_file():
            image_path = resolve_source_rgb(dataset_root, boxes_root, box_path, args.split)
        out_path = output_dir / f"{index:03d}_{box_path.parent.parent.name}_{box_path.stem}.jpg"
        stats = draw_overlay(image_path, data, out_path, args.draw_joints)
        stats["box_path"] = str(box_path)
        stats["image_path"] = str(image_path)
        stats["output_path"] = str(out_path)
        summary.append(stats)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path), "samples": len(summary)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BEDLAM bbox sidecar annotations")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--boxes-root", required=True)
    parser.add_argument("--output-dir", default="outputs/vis/bedlam_boxes")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--draw-joints", action="store_true")
    return parser.parse_args()


def select_samples(paths: list[Path], num_samples: int) -> list[Path]:
    if num_samples <= 0 or num_samples >= len(paths):
        return paths
    if num_samples == 1:
        return [paths[0]]
    last = len(paths) - 1
    indices = sorted({round(i * last / (num_samples - 1)) for i in range(num_samples)})
    return [paths[i] for i in indices]


def resolve_source_rgb(dataset_root: Path, boxes_root: Path, box_path: Path, split: str) -> Path:
    rel = box_path.relative_to(boxes_root / split)
    sequence = rel.parts[0]
    frame = box_path.stem
    return dataset_root / split / sequence / "rgb" / f"{frame}.png"


def draw_overlay(image_path: Path, data: dict[str, Any], output_path: Path, draw_joints: bool) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    persons = data.get("persons", [])
    valid_count = 0
    invalid_count = 0
    sources: dict[str, int] = {}
    out_of_bounds = 0

    for person_idx, person in enumerate(persons):
        color = COLORS[person_idx % len(COLORS)]
        source = str(person.get("bbox_source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        if not bool(person.get("bbox_valid", False)):
            invalid_count += 1
            continue
        valid_count += 1
        box = [float(v) for v in person["bbox_xyxy_pixels"]]
        if box[0] < 0 or box[1] < 0 or box[2] >= width or box[3] >= height:
            out_of_bounds += 1
        draw.rectangle(box, outline=color, width=4)
        label = f"#{person.get('person_index', person_idx)} {source}"
        draw_label(draw, (box[0], max(0.0, box[1] - 18.0)), label, color)
        if draw_joints:
            draw_projected_joints(draw, person, color)

    draw.rectangle([0, 0, width - 1, height - 1], outline=(255, 255, 255), width=2)
    header = f"valid={valid_count} invalid={invalid_count} {image_path.name}"
    draw_label(draw, (8, 8), header, (255, 255, 255), fill=(0, 0, 0))
    image.save(output_path, quality=95)
    return {
        "valid_boxes": valid_count,
        "invalid_boxes": invalid_count,
        "sources": sources,
        "out_of_bounds_boxes": out_of_bounds,
        "image_size": [height, width],
    }


def draw_projected_joints(draw: ImageDraw.ImageDraw, person: dict[str, Any], color: tuple[int, int, int]) -> None:
    j2ds = person.get("j2ds")
    masks = person.get("j2ds_mask")
    if not isinstance(j2ds, list):
        return
    for joint_idx, xy in enumerate(j2ds):
        if not isinstance(xy, list) or len(xy) < 2:
            continue
        visible = True
        if isinstance(masks, list) and joint_idx < len(masks):
            mask = masks[joint_idx]
            visible = bool(mask[0] if isinstance(mask, list) and mask else mask)
        if not visible:
            continue
        x, y = float(xy[0]), float(xy[1])
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] = (32, 32, 32),
) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=fill, outline=outline, width=1)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


if __name__ == "__main__":
    main()
