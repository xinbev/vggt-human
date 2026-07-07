from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class ResizeGeometry:
    orig_hw: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    resized_hw: tuple[int, int]
    input_hw: tuple[int, int]
    pad_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    mode: str = "balanced"
    image_resolution: int = 512
    patch_size: int = 16

    @property
    def crop_hw(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.crop_xyxy
        return max(y2 - y1, 1), max(x2 - x1, 1)

    @property
    def scale_xy(self) -> tuple[float, float]:
        crop_h, crop_w = self.crop_hw
        resized_h, resized_w = self.resized_hw
        return float(resized_w) / float(crop_w), float(resized_h) / float(crop_h)

    def to_meta(self) -> dict[str, Any]:
        return {
            "orig_hw": list(self.orig_hw),
            "crop_xyxy": list(self.crop_xyxy),
            "resized_hw": list(self.resized_hw),
            "input_hw": list(self.input_hw),
            "pad_xyxy": list(self.pad_xyxy),
            "mode": self.mode,
            "image_resolution": int(self.image_resolution),
            "patch_size": int(self.patch_size),
        }


def compute_resize_geometry(
    orig_hw: tuple[int, int],
    image_resolution: int = 512,
    patch_size: int = 16,
    mode: str = "balanced",
) -> ResizeGeometry:
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original image size: {orig_hw}")
    if image_resolution <= 0:
        raise ValueError(f"image_resolution must be positive, got {image_resolution}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if image_resolution % patch_size != 0:
        raise ValueError(f"image_resolution={image_resolution} must be divisible by patch_size={patch_size}")

    mode = str(mode or "balanced")
    if mode == "square_legacy":
        return ResizeGeometry(
            orig_hw=(orig_h, orig_w),
            crop_xyxy=(0, 0, orig_w, orig_h),
            resized_hw=(int(image_resolution), int(image_resolution)),
            input_hw=(int(image_resolution), int(image_resolution)),
            mode=mode,
            image_resolution=int(image_resolution),
            patch_size=int(patch_size),
        )
    if mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported resize mode: {mode!r}")

    crop_xyxy = supported_aspect_crop_xyxy(orig_hw)
    x1, y1, x2, y2 = crop_xyxy
    crop_h = y2 - y1
    crop_w = x2 - x1
    aspect_ratio = float(crop_h) / max(float(crop_w), 1.0)
    if mode == "balanced":
        resized_h, resized_w = balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        resized_h, resized_w = max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    return ResizeGeometry(
        orig_hw=(orig_h, orig_w),
        crop_xyxy=crop_xyxy,
        resized_hw=(int(resized_h), int(resized_w)),
        input_hw=(int(resized_h), int(resized_w)),
        mode=mode,
        image_resolution=int(image_resolution),
        patch_size=int(patch_size),
    )


def supported_aspect_crop_xyxy(orig_hw: tuple[int, int], min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0) -> tuple[int, int, int, int]:
    height, width = int(orig_hw[0]), int(orig_hw[1])
    aspect_ratio = float(height) / max(float(width), 1.0)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (int(image_resolution) // int(patch_size)) ** 2
    w_patches = np.sqrt(float(token_number) / max(float(aspect_ratio), 1e-6))
    h_patches = float(token_number) / max(float(w_patches), 1e-6)
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * int(patch_size), w_patches * int(patch_size)


def max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = int(image_resolution)
        width = round_to_patch_multiple(float(image_resolution) / max(float(aspect_ratio), 1e-6), patch_size)
    else:
        width = int(image_resolution)
        height = round_to_patch_multiple(float(image_resolution) * float(aspect_ratio), patch_size)
    return height, width


def round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(int(patch_size), int(np.round(float(value) / float(patch_size))) * int(patch_size))


def resize_image_with_geometry(image: Image.Image, geometry: ResizeGeometry, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    crop = image.crop(geometry.crop_xyxy)
    resized_h, resized_w = geometry.resized_hw
    return crop.resize((int(resized_w), int(resized_h)), resample)


def resize_depth_with_geometry(depth: np.ndarray, geometry: ResizeGeometry) -> np.ndarray:
    x1, y1, x2, y2 = geometry.crop_xyxy
    cropped = np.asarray(depth, dtype=np.float32)[y1:y2, x1:x2]
    resized_h, resized_w = geometry.resized_hw
    image = Image.fromarray(cropped, mode="F").resize((int(resized_w), int(resized_h)), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def resize_mask_with_geometry(mask: np.ndarray, geometry: ResizeGeometry) -> np.ndarray:
    x1, y1, x2, y2 = geometry.crop_xyxy
    cropped = np.asarray(mask).astype(np.float32)[y1:y2, x1:x2]
    resized_h, resized_w = geometry.resized_hw
    image = Image.fromarray(cropped, mode="F").resize((int(resized_w), int(resized_h)), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def transform_intrinsics(intrinsics: np.ndarray | torch.Tensor, geometry: ResizeGeometry) -> torch.Tensor:
    if isinstance(intrinsics, torch.Tensor):
        scaled = intrinsics.clone().float()
    else:
        scaled = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32).copy().reshape(3, 3))
    sx, sy = geometry.scale_xy
    x1, y1, _, _ = geometry.crop_xyxy
    pad_left, pad_top, _, _ = geometry.pad_xyxy
    scaled[0, 0] *= float(sx)
    scaled[0, 2] = (scaled[0, 2] - float(x1)) * float(sx) + float(pad_left)
    scaled[1, 1] *= float(sy)
    scaled[1, 2] = (scaled[1, 2] - float(y1)) * float(sy) + float(pad_top)
    return scaled


def default_intrinsics_for_geometry(geometry: ResizeGeometry) -> torch.Tensor:
    h, w = geometry.input_hw
    focal = float(max(h, w))
    return torch.tensor([[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)


def transform_xyxy_to_normalized_cxcywh(xyxy: np.ndarray | list[float], geometry: ResizeGeometry) -> tuple[np.ndarray, bool]:
    box = np.asarray(xyxy, dtype=np.float32).reshape(4).copy()
    x1, y1, x2, y2 = geometry.crop_xyxy
    sx, sy = geometry.scale_xy
    pad_left, pad_top, _, _ = geometry.pad_xyxy
    box[[0, 2]] = (box[[0, 2]] - float(x1)) * float(sx) + float(pad_left)
    box[[1, 3]] = (box[[1, 3]] - float(y1)) * float(sy) + float(pad_top)
    h, w = geometry.input_hw
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(max(w - 1, 1)))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(max(h - 1, 1)))
    bw = max(float(box[2] - box[0]), 0.0)
    bh = max(float(box[3] - box[1]), 0.0)
    valid = bw > 1.0 and bh > 1.0
    cxcywh = np.asarray(
        [
            (float(box[0]) + 0.5 * bw) / max(float(w), 1.0),
            (float(box[1]) + 0.5 * bh) / max(float(h), 1.0),
            bw / max(float(w), 1.0),
            bh / max(float(h), 1.0),
        ],
        dtype=np.float32,
    )
    return np.clip(cxcywh, 0.0, 1.0), bool(valid)


def normalized_cxcywh_to_xyxy(box: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = int(image_hw[0]), int(image_hw[1])
    cx, cy, bw, bh = box.unbind(dim=-1)
    x1 = (cx - 0.5 * bw) * float(w)
    y1 = (cy - 0.5 * bh) * float(h)
    x2 = (cx + 0.5 * bw) * float(w)
    y2 = (cy + 0.5 * bh) * float(h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_normalized_cxcywh_tensor(xyxy: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = int(image_hw[0]), int(image_hw[1])
    x1, y1, x2, y2 = xyxy.unbind(dim=-1)
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    return torch.stack(
        [
            (x1 + 0.5 * bw) / max(float(w), 1.0),
            (y1 + 0.5 * bh) / max(float(h), 1.0),
            bw / max(float(w), 1.0),
            bh / max(float(h), 1.0),
        ],
        dim=-1,
    )


def pixel_mask_to_patch_mask_hw(pixel_mask: np.ndarray, image_hw: tuple[int, int], patch_size: int, threshold: float = 0.10) -> np.ndarray:
    if pixel_mask.ndim != 2:
        raise ValueError(f"Expected 2D pixel mask, got {pixel_mask.shape}")
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    mask_image = Image.fromarray(np.asarray(pixel_mask).astype(np.float32), mode="F")
    resized = np.asarray(mask_image.resize((image_w, image_h), Image.Resampling.BILINEAR), dtype=np.float32)
    grid_h = image_h // int(patch_size)
    grid_w = image_w // int(patch_size)
    usable_h = grid_h * int(patch_size)
    usable_w = grid_w * int(patch_size)
    resized = resized[:usable_h, :usable_w]
    patch = resized.reshape(grid_h, int(patch_size), grid_w, int(patch_size)).mean(axis=(1, 3))
    return patch >= float(threshold)


def pad_image_batch(
    tensors: list[torch.Tensor],
    patch_size: int,
    value: float = 0.0,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    max_h = max(int(t.shape[-2]) for t in tensors)
    max_w = max(int(t.shape[-1]) for t in tensors)
    max_h = round_to_patch_multiple(max_h, patch_size)
    max_w = round_to_patch_multiple(max_w, patch_size)
    padded = []
    pads = []
    for tensor in tensors:
        h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
        pad_h = max_h - h
        pad_w = max_w - w
        pad_top = (pad_h // (2 * int(patch_size))) * int(patch_size)
        pad_left = (pad_w // (2 * int(patch_size))) * int(patch_size)
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        padded.append(F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=float(value)))
        pads.append((pad_left, pad_top, pad_right, pad_bottom))
    return torch.stack(padded, dim=0), pads


def collate_smpl_geometry_batch(batch: list[dict[str, torch.Tensor]], patch_size: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    image_tensors = [item["images"] for item in batch]
    if any(tensor.ndim != 4 for tensor in image_tensors):
        raise ValueError("Expected each sample images tensor to have shape [S,C,H,W]")
    max_h = round_to_patch_multiple(max(int(t.shape[-2]) for t in image_tensors), patch_size)
    max_w = round_to_patch_multiple(max(int(t.shape[-1]) for t in image_tensors), patch_size)
    padded_hw = (int(max_h), int(max_w))

    out: dict[str, torch.Tensor] = {}
    pads: list[tuple[int, int, int, int]] = []
    sample_hws: list[tuple[int, int]] = []
    padded_images = []
    padded_depths = []
    for item in batch:
        images = item["images"]
        h, w = int(images.shape[-2]), int(images.shape[-1])
        sample_hws.append((h, w))
        pad_left, pad_top, pad_right, pad_bottom = patch_aligned_pad_to_hw((h, w), padded_hw, patch_size)
        pads.append((pad_left, pad_top, pad_right, pad_bottom))
        padded_images.append(F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), value=1.0))
        if "gt_depth" in item:
            padded_depths.append(F.pad(item["gt_depth"], (pad_left, pad_right, pad_top, pad_bottom), value=0.0))

    out["images"] = torch.stack(padded_images, dim=0)
    if padded_depths:
        out["gt_depth"] = torch.stack(padded_depths, dim=0)

    for key in batch[0].keys():
        if key in {"images", "gt_depth", "K_scal3r", "gt_boxes", "smpl_query_boxes", "smpl_query_patch_masks", "image_hw", "valid_hw", "pad_xyxy"}:
            continue
        out[key] = torch.stack([_require_tensor(item[key], key) for item in batch], dim=0)

    if "K_scal3r" in batch[0]:
        out["K_scal3r"] = torch.stack([pad_intrinsics(item["K_scal3r"], pads[idx]) for idx, item in enumerate(batch)], dim=0)

    for box_key in ("gt_boxes", "smpl_query_boxes"):
        if box_key in batch[0]:
            out[box_key] = torch.stack(
                [
                    pad_normalized_boxes(item[box_key], sample_hws[idx], padded_hw, pads[idx])
                    for idx, item in enumerate(batch)
                ],
                dim=0,
            )

    if "smpl_query_patch_masks" in batch[0]:
        out["smpl_query_patch_masks"] = torch.stack(
            [
                pad_patch_masks(item["smpl_query_patch_masks"], sample_hws[idx], padded_hw, pads[idx], patch_size)
                for idx, item in enumerate(batch)
            ],
            dim=0,
        )

    max_s = max(int(item["images"].shape[0]) for item in batch)
    valid_hw = []
    image_hw = []
    pad_xyxy = []
    for idx, item in enumerate(batch):
        num_frames = int(item["images"].shape[0])
        h, w = sample_hws[idx]
        pad_left, pad_top, pad_right, pad_bottom = pads[idx]
        valid = torch.tensor([[h, w]] * num_frames, dtype=torch.long)
        padded = torch.tensor([[max_h, max_w]] * num_frames, dtype=torch.long)
        pads_tensor = torch.tensor([[pad_left, pad_top, pad_right, pad_bottom]] * num_frames, dtype=torch.long)
        if num_frames < max_s:
            repeat = max_s - num_frames
            valid = torch.cat([valid, valid[-1:].repeat(repeat, 1)], dim=0)
            padded = torch.cat([padded, padded[-1:].repeat(repeat, 1)], dim=0)
            pads_tensor = torch.cat([pads_tensor, pads_tensor[-1:].repeat(repeat, 1)], dim=0)
        valid_hw.append(valid)
        image_hw.append(padded)
        pad_xyxy.append(pads_tensor)
    out["valid_hw"] = torch.stack(valid_hw, dim=0)
    out["image_hw"] = torch.stack(image_hw, dim=0)
    out["pad_xyxy"] = torch.stack(pad_xyxy, dim=0)
    return out


def patch_aligned_pad_to_hw(hw: tuple[int, int], target_hw: tuple[int, int], patch_size: int) -> tuple[int, int, int, int]:
    h, w = int(hw[0]), int(hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    patch_size = int(patch_size)
    pad_top = (pad_h // (2 * patch_size)) * patch_size
    pad_left = (pad_w // (2 * patch_size)) * patch_size
    return pad_left, pad_top, pad_w - pad_left, pad_h - pad_top


def pad_intrinsics(intrinsics: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    out = intrinsics.clone().float()
    pad_left, pad_top, _, _ = pad
    out[..., 0, 2] += float(pad_left)
    out[..., 1, 2] += float(pad_top)
    return out


def pad_normalized_boxes(
    boxes: torch.Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    pad: tuple[int, int, int, int],
) -> torch.Tensor:
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    pad_left, pad_top, _, _ = pad
    xyxy = normalized_cxcywh_to_xyxy(boxes.float(), (source_h, source_w))
    xyxy[..., [0, 2]] += float(pad_left)
    xyxy[..., [1, 3]] += float(pad_top)
    return xyxy_to_normalized_cxcywh_tensor(xyxy, (target_h, target_w)).clamp(0.0, 1.0)


def pad_patch_masks(
    patch_masks: torch.Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    pad: tuple[int, int, int, int],
    patch_size: int,
) -> torch.Tensor:
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    grid_h = source_h // int(patch_size)
    grid_w = source_w // int(patch_size)
    target_grid_h = target_h // int(patch_size)
    target_grid_w = target_w // int(patch_size)
    pad_left, pad_top, _, _ = pad
    off_x = int(pad_left) // int(patch_size)
    off_y = int(pad_top) // int(patch_size)
    reshaped = patch_masks.reshape(*patch_masks.shape[:-1], grid_h, grid_w)
    out = torch.zeros(*patch_masks.shape[:-1], target_grid_h, target_grid_w, dtype=patch_masks.dtype, device=patch_masks.device)
    out[..., off_y : off_y + grid_h, off_x : off_x + grid_w] = reshaped
    return out.reshape(*patch_masks.shape[:-1], target_grid_h * target_grid_w)


def _require_tensor(value: torch.Tensor, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
