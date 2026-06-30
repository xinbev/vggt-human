#!/usr/bin/env python3
"""Create paper-figure visual elements for image patch pooling into person queries.

If no region is provided, this script runs the project YOLO TorchScript person
detector followed by SAM2 box-prompt segmentation. Existing SAM2 masks can also
be passed directly for faster deterministic redraws.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CANVAS_BG = (248, 250, 252)
GRID_LINE = (203, 213, 225)
PATCH_FILL = (255, 255, 255)
PERSON_PATCH = (255, 205, 205)
PERSON_PATCH_STRONG = (244, 124, 124)
PERSON_BORDER = (220, 64, 64)
TOKEN_GREEN = (42, 168, 107)
TOKEN_GREEN_LIGHT = (194, 239, 217)
TEXT_DARK = (31, 41, 51)
MUTED = (102, 112, 133)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_box(text: str) -> Box:
    values = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    if len(values) != 4:
        raise ValueError(f"Expected bbox as x1,y1,x2,y2, got: {text}")
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Box(x1, y1, x2, y2)


def load_boxes(path: Path | None, manual_boxes: list[str]) -> list[Box]:
    boxes = [parse_box(item) for item in manual_boxes]
    if path is None:
        return boxes
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "boxes" in data:
            data = data["boxes"]
        elif "persons" in data:
            data = data["persons"]
        elif "detections" in data:
            data = data["detections"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported boxes JSON schema: {path}")
    for item in data:
        if isinstance(item, dict):
            raw = item.get("bbox") or item.get("box") or item.get("xyxy")
        else:
            raw = item
        if raw is None:
            continue
        if len(raw) != 4:
            continue
        boxes.append(parse_box(",".join(str(x) for x in raw)))
    return boxes


def load_mask(mask_path: Path | None, mask_keys: list[str]) -> np.ndarray | None:
    if mask_path is None:
        return None
    suffix = mask_path.suffix.lower()
    if suffix == ".npz":
        with np.load(mask_path) as data:
            keys = mask_keys or list(data.keys())
            masks = []
            for key in keys:
                if key not in data:
                    raise KeyError(f"Mask key {key!r} not found in {mask_path}. Available keys: {list(data.keys())}")
                masks.append(np.asarray(data[key]).astype(bool))
        if not masks:
            raise ValueError(f"No masks found in {mask_path}")
        return np.logical_or.reduce(masks)
    if suffix == ".npy":
        mask = np.load(mask_path)
        return np.asarray(mask).squeeze().astype(bool)
    image = Image.open(mask_path).convert("L")
    return np.asarray(image) > 0


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        for marker in ("vggt-omega", "vggt-human"):
            parts = path.parts
            if marker in parts:
                marker_idx = parts.index(marker)
                suffix = Path(*parts[marker_idx + 1 :])
                candidates.append(ROOT / suffix)
                break
    else:
        candidates.append(ROOT / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_config_path(config: dict, override: str | None, dotted_key: str) -> str:
    if override:
        return str(resolve_project_path(override))
    from vggt_omega.training.config import require_path

    return str(resolve_project_path(require_path(config, dotted_key)))


def auto_sam2_person_mask(args: argparse.Namespace) -> tuple[np.ndarray, list[Box], dict]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Automatic mask generation requires opencv-python/cv2.") from exc
    from vggt_omega.tracking.detectors import TorchScriptYOLOPersonDetector
    from vggt_omega.tracking.sam2_masks import SAM2BoxMaskPredictor
    from vggt_omega.training.config import load_yaml_config

    config = load_yaml_config(args.path_config)
    yolo_checkpoint = resolve_config_path(config, args.yolo_checkpoint, "checkpoints.yolo8x")
    sam2_root = resolve_config_path(config, args.sam2_root, "third_party.sam2_root")
    sam2_checkpoint = resolve_config_path(config, args.sam2_checkpoint, "third_party.sam2_checkpoint")

    frame_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"Failed to read image with cv2: {args.image}")

    detector = TorchScriptYOLOPersonDetector(
        checkpoint=yolo_checkpoint,
        device=args.device,
        image_size=args.detector_image_size,
        conf_threshold=args.det_conf,
        iou_threshold=args.det_iou,
        person_class_id=0,
        half=args.det_half,
    )
    detections = detector.detect(frame_bgr)
    if not detections:
        raise RuntimeError(
            "No person detected. Provide --bbox/--mask manually or lower --det-conf for this figure asset."
        )
    detections = sorted(detections, key=lambda item: item.score, reverse=True)
    if args.auto_top_k > 0:
        selected_detections = detections[: args.auto_top_k]
    else:
        if args.auto_person_index < 0 or args.auto_person_index >= len(detections):
            raise IndexError(
                f"--auto-person-index={args.auto_person_index} out of range for {len(detections)} detections"
            )
        selected_detections = [detections[args.auto_person_index]]
    for det_idx, det in enumerate(selected_detections):
        det.det_id = det_idx

    predictor = SAM2BoxMaskPredictor(
        sam2_root=sam2_root,
        checkpoint=sam2_checkpoint,
        model_cfg=args.sam2_model_cfg,
        device=args.device,
        multimask_output=not args.sam2_single_mask,
    )
    masks, mask_meta = predictor.predict_for_detections(frame_bgr, selected_detections)
    if not masks:
        raise RuntimeError("SAM2 did not return any person masks.")
    pixel_mask = np.logical_or.reduce([np.asarray(mask).astype(bool) for _, mask in sorted(masks.items())])
    boxes = [Box(*[float(v) for v in det.bbox_xyxy]) for det in selected_detections]
    auto_meta = {
        "enabled": True,
        "detector": {
            "checkpoint": yolo_checkpoint,
            "conf_threshold": float(args.det_conf),
            "iou_threshold": float(args.det_iou),
            "num_detections": len(detections),
            "selected_scores": [float(det.score) for det in selected_detections],
            "selected_boxes_xyxy": [[float(v) for v in det.bbox_xyxy] for det in selected_detections],
        },
        "sam2": {
            "root": sam2_root,
            "checkpoint": sam2_checkpoint,
            "model_cfg": args.sam2_model_cfg,
            "metadata": mask_meta,
        },
    }
    return pixel_mask, boxes, auto_meta


def save_mask_artifacts(output_dir: Path, mask: np.ndarray | None, auto_meta: dict, suffix: str) -> dict:
    if mask is None:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    key = "person_auto" if bool(auto_meta.get("enabled", False)) else "person_mask"
    mask_path = output_dir / f"{suffix}.npz"
    np.savez_compressed(mask_path, **{key: np.asarray(mask).astype(np.uint8)})
    png_path = output_dir / f"{suffix}.png"
    Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L").save(png_path)
    return {
        "npz": str(mask_path),
        "png": str(png_path),
        "array_key": key,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "area": int(np.asarray(mask).sum()),
    }


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_image) > 0


def boxes_from_mask(mask: np.ndarray) -> list[Box]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    return [Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))]


def resize_inputs(
    image: Image.Image,
    boxes: list[Box],
    mask: np.ndarray | None,
    long_side: int,
) -> tuple[Image.Image, list[Box], np.ndarray | None]:
    width, height = image.size
    if mask is not None and mask.shape != (height, width):
        raise ValueError(f"Mask shape {mask.shape} does not match image size {(height, width)}")
    if max(width, height) <= long_side:
        return image.copy(), boxes, None if mask is None else mask.copy()
    scale = float(long_side) / float(max(width, height))
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    scaled_boxes = [Box(b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale) for b in boxes]
    resized_mask = None if mask is None else resize_mask(mask, resized.size)
    return resized, scaled_boxes, resized_mask


def fade_image(image: Image.Image, alpha: float = 0.22) -> Image.Image:
    image = image.convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    return Image.blend(white, image, alpha)


def patch_boxes(width: int, height: int, patch_size: int) -> list[tuple[int, int, Box]]:
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    patches = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = min(width, x1 + patch_size)
            y2 = min(height, y1 + patch_size)
            patches.append((row, col, Box(float(x1), float(y1), float(x2), float(y2))))
    return patches


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def selected_patch_mask_from_boxes(width: int, height: int, patch_size: int, boxes: list[Box], min_overlap: float) -> np.ndarray:
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    selected = np.zeros((rows, cols), dtype=bool)
    clipped_boxes = [box.clipped(width, height) for box in boxes if box.clipped(width, height).area > 0]
    for row, col, patch in patch_boxes(width, height, patch_size):
        for box in clipped_boxes:
            overlap = intersection_area(patch, box) / max(patch.area, 1e-6)
            if overlap >= min_overlap:
                selected[row, col] = True
                break
    return selected


def selected_patch_mask_from_pixel_mask(pixel_mask: np.ndarray, patch_size: int, min_overlap: float) -> np.ndarray:
    if pixel_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pixel_mask.shape}")
    height, width = pixel_mask.shape
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    selected = np.zeros((rows, cols), dtype=bool)
    for row, col, patch in patch_boxes(width, height, patch_size):
        crop = pixel_mask[int(patch.y1) : int(patch.y2), int(patch.x1) : int(patch.x2)]
        if crop.size and float(crop.mean()) >= float(min_overlap):
            selected[row, col] = True
    return selected


def draw_patch_grid(
    image: Image.Image,
    patch_size: int,
    selected: np.ndarray | None = None,
    pixel_mask: np.ndarray | None = None,
    show_image: bool = True,
    selected_alpha: int = 150,
    grid_width: int = 1,
) -> Image.Image:
    base = fade_image(image, 0.25) if show_image else Image.new("RGB", image.size, PATCH_FILL)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    rows = int(np.ceil(height / patch_size))
    cols = int(np.ceil(width / patch_size))
    if pixel_mask is not None:
        mask_layer = Image.fromarray((np.asarray(pixel_mask).astype(np.uint8) * selected_alpha), mode="L")
        red = Image.new("RGBA", image.size, (*PERSON_PATCH, 0))
        red.putalpha(mask_layer)
        overlay = Image.alpha_composite(overlay, red)
        draw = ImageDraw.Draw(overlay)
    elif selected is not None:
        for row in range(rows):
            for col in range(cols):
                if bool(selected[row, col]):
                    x1 = col * patch_size
                    y1 = row * patch_size
                    x2 = min(width, x1 + patch_size)
                    y2 = min(height, y1 + patch_size)
                    draw.rectangle([x1, y1, x2, y2], fill=(*PERSON_PATCH, selected_alpha))
    for x in range(0, width + 1, patch_size):
        draw.line([(x, 0), (x, height)], fill=(*GRID_LINE, 230), width=grid_width)
    for y in range(0, height + 1, patch_size):
        draw.line([(0, y), (width, y)], fill=(*GRID_LINE, 230), width=grid_width)
    if pixel_mask is not None and selected is not None:
        for row in range(rows):
            for col in range(cols):
                if bool(selected[row, col]):
                    x1 = col * patch_size
                    y1 = row * patch_size
                    x2 = min(width, x1 + patch_size)
                    y2 = min(height, y1 + patch_size)
                    draw.rectangle([x1, y1, x2, y2], outline=(*PERSON_PATCH_STRONG, 210), width=max(1, grid_width + 1))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def draw_boxes(image: Image.Image, boxes: list[Box], width: int = 3) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=PERSON_BORDER, width=width)
    return out


def create_person_patch_only(
    image: Image.Image,
    patch_size: int,
    selected: np.ndarray,
    boxes: list[Box],
    pixel_mask: np.ndarray | None = None,
) -> Image.Image:
    width, height = image.size
    out = Image.new("RGB", (width, height), CANVAS_BG)
    source = image.convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    for row, col, patch in patch_boxes(width, height, patch_size):
        rect = (int(patch.x1), int(patch.y1), int(patch.x2), int(patch.y2))
        if bool(selected[row, col]):
            crop = source.crop(rect)
            if pixel_mask is not None:
                crop_mask = pixel_mask[rect[1] : rect[3], rect[0] : rect[2]]
                tinted = Image.blend(Image.new("RGB", crop.size, PERSON_PATCH), crop, 0.38)
                patch_canvas = Image.new("RGB", crop.size, (255, 255, 255))
                patch_canvas.paste(tinted, (0, 0), Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L"))
                crop = patch_canvas
            else:
                crop = Image.blend(Image.new("RGB", crop.size, PERSON_PATCH), crop, 0.38)
            out.paste(crop, rect[:2])
        else:
            draw.rectangle(rect, fill=(255, 255, 255, 120))
    for row, col, patch in patch_boxes(width, height, patch_size):
        rect = [patch.x1, patch.y1, patch.x2, patch.y2]
        line = (*GRID_LINE, 220)
        if bool(selected[row, col]):
            line = (*PERSON_PATCH_STRONG, 240)
        draw.rectangle(rect, outline=line, width=1)
    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(*PERSON_BORDER, 255), width=3)
    return out


def create_token_grid(selected: np.ndarray, cell: int = 32, gap: int = 6, pad: int = 18) -> Image.Image:
    rows, cols = selected.shape
    width = pad * 2 + cols * cell + (cols - 1) * gap
    height = pad * 2 + rows * cell + (rows - 1) * gap
    out = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(out, "RGBA")
    for row in range(rows):
        for col in range(cols):
            if not bool(selected[row, col]):
                continue
            x1 = pad + col * (cell + gap)
            y1 = pad + row * (cell + gap)
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=4,
                fill=(*PERSON_PATCH, 210),
                outline=(*PERSON_PATCH_STRONG, 245),
                width=2,
            )
    return out


def create_query_pooling(selected: np.ndarray, num_queries: int = 6) -> Image.Image:
    token_grid = create_token_grid(selected, cell=26, gap=5, pad=14)
    width = token_grid.width + 360
    height = max(token_grid.height, 240)
    out = Image.new("RGB", (width, height), CANVAS_BG)
    out.paste(token_grid, (0, (height - token_grid.height) // 2), token_grid.getchannel("A"))
    draw = ImageDraw.Draw(out)
    mid_y = height // 2
    start_x = token_grid.width + 28
    draw.line([(token_grid.width + 4, mid_y), (start_x + 70, mid_y)], fill=TOKEN_GREEN, width=4)
    draw.polygon([(start_x + 70, mid_y), (start_x + 54, mid_y - 10), (start_x + 54, mid_y + 10)], fill=TOKEN_GREEN)
    pool_x = start_x + 92
    draw.rounded_rectangle([pool_x, mid_y - 42, pool_x + 88, mid_y + 42], radius=14, fill=TOKEN_GREEN_LIGHT, outline=TOKEN_GREEN, width=3)
    for i in range(5):
        y = mid_y - 24 + i * 12
        draw.line([(pool_x + 18, y), (pool_x + 70, y)], fill=TOKEN_GREEN, width=2)
    qx = pool_x + 130
    for i in range(num_queries):
        y = mid_y - 68 + i * 27
        draw.rounded_rectangle([qx, y, qx + 110, y + 18], radius=8, fill=(255, 255, 255), outline=TOKEN_GREEN, width=2)
        draw.ellipse([qx + 8, y + 5, qx + 16, y + 13], fill=TOKEN_GREEN)
    return out


def svg_color(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def write_token_grid_svg(path: Path, selected: np.ndarray, cell: int = 28, gap: int = 6, pad: int = 16) -> None:
    rows, cols = selected.shape
    width = pad * 2 + cols * cell + (cols - 1) * gap
    height = pad * 2 + rows * cell + (rows - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]
    for row in range(rows):
        for col in range(cols):
            if not bool(selected[row, col]):
                continue
            x = pad + col * (cell + gap)
            y = pad + row * (cell + gap)
            fill = svg_color(PERSON_PATCH)
            stroke = svg_color(PERSON_PATCH_STRONG)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
                f'fill="{fill}" fill-opacity="0.82" stroke="{stroke}" stroke-width="2"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_with_scale(image: Image.Image, path: Path, scale: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image used for paper figure elements.")
    parser.add_argument("--bbox", action="append", default=[], help="Person box as x1,y1,x2,y2. Can be repeated.")
    parser.add_argument("--boxes-json", type=Path, default=None, help="Optional JSON containing boxes/persons/detections.")
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional SAM2/person mask as .npz, .npy, or binary image. This is preferred over bbox selection.",
    )
    parser.add_argument(
        "--mask-key",
        action="append",
        default=[],
        help="Array key inside a SAM2 .npz mask file, e.g. person_3. Can be repeated. Defaults to all arrays.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis/paper_arch_patch_pooling_elements"))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--min-overlap", type=float, default=0.12)
    parser.add_argument("--num-query-chips", type=int, default=6)
    parser.add_argument("--path-config", default="configs/path.yaml", help="Path config for automatic YOLO+SAM2 mode.")
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.7)
    parser.add_argument("--det-half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--auto-person-index",
        type=int,
        default=0,
        help="When auto-generating a mask, select this detection after sorting by confidence.",
    )
    parser.add_argument(
        "--auto-top-k",
        type=int,
        default=0,
        help="When >0, OR-combine SAM2 masks for the top-k detected people instead of using --auto-person-index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    boxes = load_boxes(args.boxes_json, args.bbox)
    pixel_mask = load_mask(args.mask, args.mask_key)
    auto_meta = {"enabled": False}
    if not boxes and pixel_mask is None:
        pixel_mask, boxes, auto_meta = auto_sam2_person_mask(args)
    original_mask_artifact = {}
    if bool(auto_meta.get("enabled", False)) and pixel_mask is not None:
        original_mask_artifact = save_mask_artifacts(out_dir, pixel_mask, auto_meta, "auto_sam2_mask_original")
    image, boxes, pixel_mask = resize_inputs(image, boxes, pixel_mask, args.long_side)
    width, height = image.size
    boxes = [box.clipped(width, height) for box in boxes]
    if pixel_mask is not None:
        selected = selected_patch_mask_from_pixel_mask(pixel_mask, args.patch_size, args.min_overlap)
        if not boxes:
            boxes = boxes_from_mask(pixel_mask)
    else:
        selected = selected_patch_mask_from_boxes(width, height, args.patch_size, boxes, args.min_overlap)

    resized_mask_artifact = {}
    if bool(auto_meta.get("enabled", False)) and pixel_mask is not None:
        resized_mask_artifact = save_mask_artifacts(out_dir, pixel_mask, auto_meta, "auto_sam2_mask_resized")

    grid = draw_patch_grid(image, args.patch_size, selected=None, show_image=True)
    save_with_scale(grid, out_dir / "01_image_patch_grid_faded.png")

    highlighted = draw_patch_grid(image, args.patch_size, selected=selected, pixel_mask=pixel_mask, show_image=True)
    highlighted = draw_boxes(highlighted, boxes)
    save_with_scale(highlighted, out_dir / "02_person_patch_highlight.png")

    person_only = create_person_patch_only(image, args.patch_size, selected, boxes, pixel_mask=pixel_mask)
    save_with_scale(person_only, out_dir / "03_person_patches_extracted.png")

    token_grid = create_token_grid(selected)
    save_with_scale(token_grid, out_dir / "04_patch_token_grid.png")
    write_token_grid_svg(out_dir / "04_patch_token_grid.svg", selected)

    pooling = create_query_pooling(selected, num_queries=args.num_query_chips)
    save_with_scale(pooling, out_dir / "05_pool_to_person_query.png")

    manifest = {
        "image": str(args.image),
        "resized_size": [width, height],
        "patch_size": args.patch_size,
        "grid_shape": list(selected.shape),
        "num_selected_patches": int(selected.sum()),
        "selection_source": "mask" if pixel_mask is not None else "bbox",
        "mask": None if args.mask is None else str(args.mask),
        "mask_keys": args.mask_key,
        "auto_sam2": auto_meta,
        "auto_mask_artifacts": {
            "original": original_mask_artifact,
            "resized": resized_mask_artifact,
        },
        "boxes": [box.__dict__ for box in boxes],
        "files": [
            "01_image_patch_grid_faded.png",
            "02_person_patch_highlight.png",
            "03_person_patches_extracted.png",
            "04_patch_token_grid.png",
            "04_patch_token_grid.svg",
            "05_pool_to_person_query.png",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
