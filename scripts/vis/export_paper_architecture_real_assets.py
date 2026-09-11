#!/usr/bin/env python3
"""Export real inference assets for the paper architecture figure.

The exporter runs the accepted single-frame chain on one RGB image:

    RGB -> VGGT camera/raw depth -> NLF metric SMPL
        -> analytic coarse scale -> HSI residual affine -> TRSTR spatial refine

It writes figure-ready PNGs plus lossless numeric arrays/JSON.  A single image
cannot provide a real nine-frame temporal observation, so temporal availability
is audited rather than synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.create_hsi_local_probe_real_elements import compute_hsi_probe  # noqa: E402
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    load_image,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    write_ply_vertices_faces,
)
from vggt_omega.models.heads.hsi_refinement_head import _canonical_depth, _project_points  # noqa: E402
from vggt_omega.training.config import deep_update  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


PALETTE = np.asarray(
    [
        (217, 172, 166),
        (175, 202, 178),
        (169, 207, 216),
        (153, 173, 212),
        (203, 186, 216),
        (231, 218, 172),
        (233, 211, 110),
        (47, 52, 56),
    ],
    dtype=np.uint8,
)
INK = (47, 52, 56)
PAPER = (247, 246, 242)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/image/f2/f2.jpg"))
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--model-config", default="configs/infer_smpl_hsi_v3_trstr_spatial.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/smpl_hsi_stage2_trstr_v3_refine/checkpoint_latest.pt",
        help="Checkpoint containing the accepted HSI scale and TRSTR heads.",
    )
    parser.add_argument("--scale-checkpoint", default="", help="Optional HSI-only overlay loaded after --checkpoint.")
    parser.add_argument("--baseline-checkpoint", default="", help="Optional override for checkpoints.vggt_baseline.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis/paper_figure_assets/f2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--coarse-anchor-stride", type=int, default=8)
    parser.add_argument("--coarse-min-anchor-pixels", type=int, default=32)
    parser.add_argument("--coarse-scale-min", type=float, default=0.05)
    parser.add_argument("--coarse-scale-max", type=float, default=25.0)
    parser.add_argument("--local-probe-size", type=int, default=21)
    parser.add_argument("--temporal-frames-dir", type=Path, default=None)
    parser.add_argument("--temporal-window", type=int, default=9)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = resolve_input_path(args.image)
    checkpoint = resolve_input_path(args.checkpoint)
    scale_checkpoint = resolve_input_path(args.scale_checkpoint) if args.scale_checkpoint else None
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.model_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = str(resolve_input_path(args.baseline_checkpoint))
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["enable_hsi_refine"] = True
    model_cfg["enable_hsi_trstr"] = True

    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, checkpoint, device)
    if scale_checkpoint is not None:
        load_training_checkpoint(model, scale_checkpoint, device)
    model.eval()

    image_resolution = int(config.get("data", {}).get("image_resolution", config.get("data", {}).get("image_size", 512)))
    image_tensor, original_rgb = load_image(image_path, image_resolution)
    image_tensor = image_tensor.to(device)
    input_hw = (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))
    processed_rgb = tensor_to_pil(image_tensor[0, 0])
    original_rgb.save(output_dir / "00_input_rgb_original.png")
    processed_rgb.save(output_dir / "00_input_rgb_processed.png")

    trstr_head = getattr(model, "hsi_trstr_head", None)
    model.hsi_trstr_head = None
    try:
        with torch.no_grad():
            first = model(image_tensor)
    finally:
        model.hsi_trstr_head = trstr_head

    required = ("depth", "pose_enc", "pred_pose_6d", "pred_poses", "pred_betas", "pred_transl_cam", "pred_confs")
    missing = [key for key in required if not isinstance(first.get(key), torch.Tensor)]
    if missing:
        raise RuntimeError(f"First-pass inference is missing required outputs: {missing}")

    raw_depth = _canonical_depth(first["depth"]).detach().float()
    first_probe = compute_hsi_probe(first, model, input_hw)
    valid_queries = select_queries(first, args.conf_threshold)
    if not valid_queries:
        raise RuntimeError("NLF produced no person above --conf-threshold; lower the threshold or inspect the checkpoint.")
    export_queries = valid_queries[: max(1, int(args.top_k))]

    hypotheses = compute_coarse_hypotheses(
        smpl_vertices=first_probe["smpl_vertices"][0, 0, valid_queries],
        depth=raw_depth[0, 0],
        pose_enc=first["pose_enc"],
        anchor_stride=args.coarse_anchor_stride,
        scale_min=args.coarse_scale_min,
        scale_max=args.coarse_scale_max,
    )
    valid_ratio = hypotheses["in_range"]
    if int(valid_ratio.sum()) < int(args.coarse_min_anchor_pixels):
        raise RuntimeError(
            "Analytic coarse scale has insufficient valid anchor pixels: "
            f"{int(valid_ratio.sum())} < {args.coarse_min_anchor_pixels}."
        )
    coarse_scale = float(np.median(hypotheses["ratio"][valid_ratio]))
    coarse_depth = raw_depth * coarse_scale

    smpl_override = {
        key: first[key]
        for key in (
            "pred_pose_6d",
            "pred_poses",
            "pred_betas",
            "pred_transl_cam",
            "pred_confs",
            "pred_boxes",
            "pred_cam",
            "base_pred_transl_cam",
        )
        if isinstance(first.get(key), torch.Tensor)
    }
    with torch.no_grad():
        final = model(
            image_tensor,
            smpl_override_outputs=smpl_override,
            hsi_depth_override=coarse_depth,
            hsi_depth_is_metric=True,
            hsi_geometry_mode="smpl_coarse_metric",
        )

    residual_scale = scalar_prediction(final, "hsi_scene_scale", 1.0)
    residual_bias = scalar_prediction(final, "hsi_scene_depth_bias", 0.0)
    metric_depth = final.get("hsi_translation_depth")
    if not isinstance(metric_depth, torch.Tensor):
        metric_depth = coarse_depth * residual_scale + residual_bias
    metric_depth = _canonical_depth(metric_depth).detach().float()

    probe_inputs = dict(final)
    probe_inputs["depth"] = coarse_depth
    hsi_probe = compute_hsi_probe(probe_inputs, model, input_hw)
    intrinsics = hsi_probe["intrinsics"][0].detach().float().cpu().numpy()

    asset_files: list[str] = ["00_input_rgb_original.png", "00_input_rgb_processed.png"]
    asset_files += save_depth_family(output_dir, "01_raw_vggt_depth", raw_depth[0, 0])
    asset_files += save_depth_family(output_dir, "08_coarse_metric_depth", coarse_depth[0, 0])
    asset_files += save_depth_family(output_dir, "12_final_metric_depth", metric_depth[0, 0])
    asset_files.append(save_depth_triptych(output_dir, raw_depth[0, 0], coarse_depth[0, 0], metric_depth[0, 0]))
    asset_files.append(save_intrinsics(output_dir, intrinsics, input_hw))
    asset_files.append(save_detection_overlay(output_dir, processed_rgb, first, export_queries))
    asset_files += save_scale_hypotheses(output_dir, hypotheses, coarse_scale)

    person_records = []
    for rank, query_idx in enumerate(export_queries):
        prefix = f"person{rank:02d}_q{query_idx:02d}"
        record, files = export_person_assets(
            output_dir=output_dir,
            prefix=prefix,
            query_idx=query_idx,
            processed_rgb=processed_rgb,
            predictions=final,
            probe=hsi_probe,
            coarse_depth=coarse_depth[0, 0],
            metric_depth=metric_depth[0, 0],
            trstr_head=trstr_head,
            local_probe_size=args.local_probe_size,
        )
        person_records.append(record)
        asset_files.extend(files)

    hsi_values = {
        "coarse_scale_C_coarse": coarse_scale,
        "residual_scale_R_hsi": residual_scale,
        "residual_bias_b_hsi_m": residual_bias,
        "effective_scale_C_times_R": coarse_scale * residual_scale,
        "formula": "D_metric = (C_coarse * R_hsi) * D_vggt + b_hsi",
    }
    (output_dir / "11_hsi_scale_values.json").write_text(json.dumps(hsi_values, indent=2), encoding="utf-8")
    asset_files.append("11_hsi_scale_values.json")

    temporal = audit_temporal_inputs(args, image_path, output_dir)
    asset_files.extend(temporal["files"])
    manifest = {
        "purpose": "Real-data assets for the RGB/VGGT/NLF/HSI/TRSTR/temporal paper architecture figure.",
        "source_image": str(image_path),
        "model_config": str(args.model_config),
        "checkpoint": str(checkpoint),
        "scale_checkpoint": None if scale_checkpoint is None else str(scale_checkpoint),
        "device": str(device),
        "input_shape": list(image_tensor.shape),
        "selected_queries": export_queries,
        "coarse_scale": hsi_values,
        "people": person_records,
        "temporal": temporal["status"],
        "files": sorted(set(asset_files)),
        "coordinate_contract": {
            "SMPL": "NLF metric camera coordinates, meters",
            "depth": "VGGT processed-image depth plane; raw depth is scale-ambiguous",
            "projection": "processed RGB/depth plane using VGGT intrinsics",
            "TRSTR": "translation-only; pose and betas are unchanged",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_people": len(person_records), "files": len(set(asset_files))}, indent=2))


def resolve_input_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path if path.is_absolute() else candidate


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def select_queries(predictions: dict[str, torch.Tensor], threshold: float) -> list[int]:
    conf = predictions["pred_confs"][0, 0, :, 0].detach().float().cpu()
    order = torch.argsort(conf, descending=True).tolist()
    return [int(index) for index in order if float(conf[index]) >= float(threshold)]


def scalar_prediction(predictions: dict[str, Any], key: str, default: float) -> float:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return float(default)
    return float(value.detach().float().reshape(-1)[0].cpu())


def compute_coarse_hypotheses(
    smpl_vertices: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    anchor_stride: int,
    scale_min: float,
    scale_max: float,
) -> dict[str, np.ndarray]:
    vertices = smpl_vertices[:, :: max(int(anchor_stride), 1)].reshape(-1, 3).to(depth)
    depth_hw = (int(depth.shape[-2]), int(depth.shape[-1]))
    _, k_all = encoding_to_camera(pose_enc, image_size_hw=depth_hw, build_intrinsics=True)
    k = k_all[0, 0].to(depth)
    projected = _project_points(vertices[None], k[None])[0]
    px = projected[:, 0].round().long()
    py = projected[:, 1].round().long()
    valid = (
        torch.isfinite(vertices).all(dim=-1)
        & torch.isfinite(projected).all(dim=-1)
        & (vertices[:, 2] > 1e-6)
        & (px >= 0)
        & (px < depth_hw[1])
        & (py >= 0)
        & (py < depth_hw[0])
    )
    px, py, z_smpl = px[valid], py[valid], vertices[valid, 2]
    z_depth = depth[py, px]
    valid_depth = torch.isfinite(z_depth) & (z_depth > 1e-6)
    px, py, z_smpl, z_depth = px[valid_depth], py[valid_depth], z_smpl[valid_depth], z_depth[valid_depth]
    if px.numel() == 0:
        raise RuntimeError("No valid SMPL/depth correspondences for coarse scale.")

    pixel_id = (py * depth_hw[1] + px).detach().cpu().numpy()
    z_np = z_smpl.detach().float().cpu().numpy()
    order = np.lexsort((z_np, pixel_id))
    sorted_pixels = pixel_id[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
    keep_idx = torch.as_tensor(order[keep], dtype=torch.long, device=depth.device)
    px, py, z_smpl, z_depth = px[keep_idx], py[keep_idx], z_smpl[keep_idx], z_depth[keep_idx]
    ratio = z_smpl / z_depth
    finite = torch.isfinite(ratio) & (ratio > 0)
    px, py, z_smpl, z_depth, ratio = px[finite], py[finite], z_smpl[finite], z_depth[finite], ratio[finite]
    in_range = (ratio >= float(scale_min)) & (ratio <= float(scale_max))
    return {
        "pixel_x": px.detach().cpu().numpy(),
        "pixel_y": py.detach().cpu().numpy(),
        "z_smpl_m": z_smpl.detach().float().cpu().numpy(),
        "z_vggt_raw": z_depth.detach().float().cpu().numpy(),
        "ratio": ratio.detach().float().cpu().numpy(),
        "in_range": in_range.detach().cpu().numpy().astype(bool),
    }


def depth_gray(depth: torch.Tensor) -> Image.Image:
    array = depth.detach().float().cpu().numpy()
    valid = np.isfinite(array) & (array > 1e-6)
    lo, hi = np.percentile(array[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    normalized = np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)
    gray = (242.0 - 190.0 * normalized).astype(np.uint8)
    gray[~valid] = 255
    return Image.fromarray(gray, mode="L").convert("RGB")


def depth_color(depth: torch.Tensor) -> Image.Image:
    array = depth.detach().float().cpu().numpy()
    valid = np.isfinite(array) & (array > 1e-6)
    lo, hi = np.percentile(array[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    t = np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)
    stops = np.asarray([(231, 218, 172), (169, 207, 216), (153, 173, 212), (203, 186, 216), (47, 52, 56)], dtype=np.float32)
    position = t * (len(stops) - 1)
    index = np.floor(position).astype(np.int32).clip(0, len(stops) - 2)
    fraction = (position - index)[..., None]
    rgb = ((1 - fraction) * stops[index] + fraction * stops[index + 1]).clip(0, 255).astype(np.uint8)
    rgb[~valid] = np.asarray(PAPER, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_depth_family(output_dir: Path, stem: str, depth: torch.Tensor) -> list[str]:
    npy = f"{stem}.npy"
    gray = f"{stem}_gray.png"
    color = f"{stem}_color.png"
    np.save(output_dir / npy, depth.detach().float().cpu().numpy())
    depth_gray(depth).save(output_dir / gray)
    depth_color(depth).save(output_dir / color)
    return [npy, gray, color]


def save_depth_triptych(output_dir: Path, raw: torch.Tensor, coarse: torch.Tensor, metric: torch.Tensor) -> str:
    images = [depth_color(raw), depth_color(coarse), depth_color(metric)]
    labels = ["Raw VGGT", "Coarse metric", "HSI metric"]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3 + 48, height + 52), PAPER)
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = index * (width + 24)
        canvas.paste(image, (x, 52))
        draw.text((x + 8, 14), label, fill=INK)
    name = "13_depth_raw_coarse_metric_triptych.png"
    canvas.save(output_dir / name)
    return name


def save_intrinsics(output_dir: Path, intrinsics: np.ndarray, image_hw: tuple[int, int]) -> str:
    name = "02_camera_intrinsics.json"
    payload = {"K": intrinsics.tolist(), "image_hw": list(image_hw), "coordinate_plane": "processed RGB/depth"}
    (output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return name


def normalized_cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = [float(value) for value in box]
    return ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)


def save_detection_overlay(output_dir: Path, rgb: Image.Image, predictions: dict[str, torch.Tensor], queries: list[int]) -> str:
    image = rgb.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = predictions.get("pred_boxes")
    confs = predictions["pred_confs"][0, 0, :, 0].detach().float().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes[0, 0].detach().float().cpu().numpy()
        for rank, query in enumerate(queries):
            color = tuple(int(v) for v in PALETTE[rank % 7])
            xyxy = normalized_cxcywh_to_xyxy(boxes_np[query], image.width, image.height)
            draw.rectangle(xyxy, outline=color, width=4)
            draw.text((xyxy[0] + 4, xyxy[1] + 4), f"q{query}  c={confs[query]:.2f}", fill=color)
    name = "03_nlf_detection_overlay.png"
    image.save(output_dir / name)
    return name


def save_scale_hypotheses(output_dir: Path, hypotheses: dict[str, np.ndarray], coarse_scale: float) -> list[str]:
    csv_name = "06_scale_hypotheses.csv"
    with (output_dir / csv_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pixel_x", "pixel_y", "z_smpl_m", "z_vggt_raw", "scale_ratio", "valid"])
        for row in zip(
            hypotheses["pixel_x"], hypotheses["pixel_y"], hypotheses["z_smpl_m"],
            hypotheses["z_vggt_raw"], hypotheses["ratio"], hypotheses["in_range"], strict=True
        ):
            writer.writerow([float(value) if index < 5 else int(value) for index, value in enumerate(row)])

    width, height = 1200, 280
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    ratios = hypotheses["ratio"]
    valid = hypotheses["in_range"]
    valid_values = ratios[valid]
    lo, hi = np.percentile(valid_values, [1, 99]) if len(valid_values) else (0.0, 1.0)
    pad = max((hi - lo) * 0.12, 1e-3)
    lo, hi = float(lo - pad), float(hi + pad)
    y = 142
    draw.line((70, y, width - 70, y), fill=INK, width=3)
    indices = np.linspace(0, max(len(ratios) - 1, 0), min(96, len(ratios)), dtype=int) if len(ratios) else []
    for item, idx in enumerate(indices):
        x = 70 + (float(ratios[idx]) - lo) / max(hi - lo, 1e-6) * (width - 140)
        x = int(np.clip(x, 70, width - 70))
        color = tuple(int(v) for v in PALETTE[item % 7]) if valid[idx] else (180, 180, 180)
        radius = 6 if valid[idx] else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=INK, width=1)
    median_x = int(70 + (coarse_scale - lo) / max(hi - lo, 1e-6) * (width - 140))
    median_x = int(np.clip(median_x, 70, width - 70))
    draw.line((median_x, 72, median_x, 214), fill=tuple(int(v) for v in PALETTE[6]), width=6)
    draw.text((32, 24), "Real SMPL/VGGT scale hypotheses", fill=INK)
    draw.text((median_x + 10, 78), f"C_coarse={coarse_scale:.4f}", fill=INK)
    png_name = "06_scale_hypothesis_strip.png"
    canvas.save(output_dir / png_name)
    return [csv_name, png_name]


def export_person_assets(
    output_dir: Path,
    prefix: str,
    query_idx: int,
    processed_rgb: Image.Image,
    predictions: dict[str, torch.Tensor],
    probe: dict[str, torch.Tensor],
    coarse_depth: torch.Tensor,
    metric_depth: torch.Tensor,
    trstr_head: torch.nn.Module | None,
    local_probe_size: int,
) -> tuple[dict[str, Any], list[str]]:
    files: list[str] = []
    vertices = probe["smpl_vertices"][0, 0, query_idx].detach().float()
    faces = probe["smpl_faces"].detach().long().cpu().numpy()
    k = probe["intrinsics"][0].detach().float()
    smpl_png = f"04_{prefix}_metric_smpl.png"
    render_projected_mesh(vertices, faces, k, processed_rgb.size, None).save(output_dir / smpl_png)
    files.append(smpl_png)

    smpl_ply = f"04_{prefix}_metric_smpl.ply"
    neutral = np.tile(np.asarray([[169, 207, 216]], dtype=np.uint8), (vertices.shape[0], 1))
    write_ply_vertices_faces(output_dir / smpl_ply, vertices.cpu().numpy(), neutral, faces)
    files.append(smpl_ply)

    anchor_png = f"05_{prefix}_24anchors_overlay.png"
    save_anchor_overlay(output_dir / anchor_png, processed_rgb, probe, query_idx)
    files.append(anchor_png)

    residuals = probe["depth_residual"][0, 0, query_idx].detach().float().cpu().numpy()
    anchor_values = f"05_{prefix}_anchor_values.json"
    anchor_payload = {
        "anchor_xyz_m": probe["anchors"][0, 0, query_idx].detach().float().cpu().numpy().tolist(),
        "projected_uv": probe["projected_depth"][0, 0, query_idx].detach().float().cpu().numpy().tolist(),
        "depth_residual_m": residuals.tolist(),
        "probe_distance_m": probe["distance"][0, 0, query_idx].detach().float().cpu().numpy().tolist(),
    }
    (output_dir / anchor_values).write_text(json.dumps(anchor_payload, indent=2), encoding="utf-8")
    files.append(anchor_values)

    anchor_indices = select_probe_anchors(probe, query_idx, coarse_depth.shape[-2:])
    probe_files, montage = save_local_probe_patches(output_dir, prefix, coarse_depth, probe, query_idx, anchor_indices, local_probe_size)
    files.extend(probe_files)
    files.append(montage)

    token_heatmap = f"07_{prefix}_anchor_residual_tokens.png"
    render_value_grid(residuals, 4, 6, "24 real anchor residuals").save(output_dir / token_heatmap)
    files.append(token_heatmap)

    trstr_record: dict[str, Any] = {"available": False}
    if trstr_head is not None and isinstance(predictions.get("hsi_trstr_region_vote"), torch.Tensor):
        trstr_record, trstr_files = export_trstr_assets(
            output_dir, prefix, query_idx, vertices, faces, k, processed_rgb, predictions, trstr_head
        )
        files.extend(trstr_files)

    record = {
        "query_index": query_idx,
        "confidence": float(predictions["pred_confs"][0, 0, query_idx, 0].detach().cpu()),
        "pose_shape_translation": {
            "pose_6d_shape": list(predictions["pred_pose_6d"][0, 0, query_idx].shape),
            "betas": predictions["pred_betas"][0, 0, query_idx].detach().float().cpu().tolist(),
            "base_translation_m": predictions["pred_transl_cam"][0, 0, query_idx].detach().float().cpu().tolist(),
        },
        "selected_local_probe_anchors": anchor_indices,
        "trstr": trstr_record,
        "files": files,
    }
    return record, files


def project_vertices(vertices: torch.Tensor, k: torch.Tensor) -> np.ndarray:
    return _project_points(vertices[None], k[None])[0].detach().float().cpu().numpy()


def render_projected_mesh(
    vertices: torch.Tensor,
    faces: np.ndarray,
    k: torch.Tensor,
    image_size_wh: tuple[int, int],
    vertex_region_ids: np.ndarray | None,
) -> Image.Image:
    width, height = image_size_wh
    projected = project_vertices(vertices, k)
    z = vertices[:, 2].detach().float().cpu().numpy()
    valid = np.isfinite(projected).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    valid_faces = faces[valid[faces].all(axis=1)]
    if not len(valid_faces):
        return Image.new("RGBA", (640, 900), (255, 255, 255, 0))
    face_depth = z[valid_faces].mean(axis=1)
    valid_faces = valid_faces[np.argsort(face_depth)[::-1]]
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for face in valid_faces:
        polygon = [tuple(projected[index]) for index in face]
        if vertex_region_ids is None:
            color = (169, 207, 216)
        else:
            color = tuple(int(value) for value in PALETTE[int(vertex_region_ids[int(face[0])]) % 7])
        draw.polygon(polygon, fill=(*color, 62), outline=(*INK, 75))
    bbox = canvas.getbbox()
    if bbox is None:
        return canvas
    left, top, right, bottom = bbox
    margin = 16
    crop = canvas.crop((max(0, left - margin), max(0, top - margin), min(width, right + margin), min(height, bottom + margin)))
    target_h = 900
    target_w = max(320, int(crop.width * target_h / max(crop.height, 1)))
    return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)


def save_anchor_overlay(path: Path, rgb: Image.Image, probe: dict[str, torch.Tensor], query_idx: int) -> None:
    image = rgb.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    uv = probe["projected_depth"][0, 0, query_idx].detach().float().cpu().numpy()
    for index, point in enumerate(uv):
        if not np.isfinite(point).all():
            continue
        x, y = float(point[0]), float(point[1])
        color = tuple(int(value) for value in PALETTE[index % 7])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=INK, width=1)
    image.save(path)


def select_probe_anchors(probe: dict[str, torch.Tensor], query_idx: int, depth_hw: tuple[int, int]) -> list[int]:
    uv = probe["projected_depth"][0, 0, query_idx].detach().float().cpu().numpy()
    residual = probe["depth_residual"][0, 0, query_idx].detach().float().cpu().numpy()
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(residual)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < depth_hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth_hw[0])
    indices = np.flatnonzero(valid)
    if len(indices) <= 3:
        return [int(index) for index in indices]
    order = indices[np.argsort(uv[indices, 1])]
    picks = [order[int(round((len(order) - 1) * quantile))] for quantile in (0.15, 0.50, 0.85)]
    return [int(index) for index in dict.fromkeys(picks)]


def save_local_probe_patches(
    output_dir: Path,
    prefix: str,
    depth: torch.Tensor,
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_indices: list[int],
    patch_size: int,
) -> tuple[list[str], str]:
    source = depth_color(depth)
    uv = probe["projected_depth"][0, 0, query_idx].detach().float().cpu().numpy()
    files = []
    patches = []
    radius = max(int(patch_size) // 2, 4)
    for rank, anchor_index in enumerate(anchor_indices):
        x, y = [int(round(value)) for value in uv[anchor_index]]
        padded = Image.new("RGB", (source.width + 2 * radius, source.height + 2 * radius), PAPER)
        padded.paste(source, (radius, radius))
        crop = padded.crop((x, y, x + 2 * radius + 1, y + 2 * radius + 1)).resize((260, 260), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(crop)
        center = 130
        color = tuple(int(value) for value in PALETTE[rank % 7])
        draw.ellipse((center - 9, center - 9, center + 9, center + 9), fill=color, outline=INK, width=2)
        draw.line((center - 28, center, center + 28, center), fill=INK, width=2)
        draw.line((center, center - 28, center, center + 28), fill=INK, width=2)
        name = f"07_{prefix}_local_probe_anchor{anchor_index:02d}.png"
        crop.save(output_dir / name)
        files.append(name)
        patches.append(crop)
    montage_name = f"07_{prefix}_local_scene_probes_triplet.png"
    if patches:
        canvas = Image.new("RGB", (len(patches) * 260 + (len(patches) - 1) * 16, 260), PAPER)
        for index, patch in enumerate(patches):
            canvas.paste(patch, (index * 276, 0))
        canvas.save(output_dir / montage_name)
    else:
        Image.new("RGB", (260, 260), PAPER).save(output_dir / montage_name)
    return files, montage_name


def render_value_grid(values: np.ndarray, rows: int, cols: int, title: str) -> Image.Image:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    width, height = 720, 560
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), title, fill=INK)
    finite = values[np.isfinite(values)]
    bound = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
    cell, gap = 88, 16
    x0, y0 = 48, 90
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        value = values[index] if index < len(values) else np.nan
        if not np.isfinite(value):
            color = (225, 225, 225)
        else:
            t = np.clip(0.5 + 0.5 * float(value) / max(bound, 1e-6), 0, 1)
            low, high = np.asarray(PALETTE[2], dtype=float), np.asarray(PALETTE[3], dtype=float)
            color = tuple(((1 - t) * low + t * high).astype(np.uint8))
        box = (x0 + col * (cell + gap), y0 + row * (cell + gap), x0 + col * (cell + gap) + cell, y0 + row * (cell + gap) + cell)
        draw.rounded_rectangle(box, radius=9, fill=color, outline=INK, width=2)
    return canvas


def export_trstr_assets(
    output_dir: Path,
    prefix: str,
    query_idx: int,
    base_vertices: torch.Tensor,
    faces: np.ndarray,
    k: torch.Tensor,
    rgb: Image.Image,
    predictions: dict[str, torch.Tensor],
    trstr_head: torch.nn.Module,
) -> tuple[dict[str, Any], list[str]]:
    files: list[str] = []
    vote = predictions["hsi_trstr_region_vote"][0, 0, query_idx].detach().float().cpu().numpy()
    gate = predictions["hsi_trstr_region_gate"][0, 0, query_idx, :, 0].detach().float().cpu().numpy()
    logvar = predictions["hsi_trstr_region_logvar"][0, 0, query_idx, :, 0].detach().float().cpu().numpy()
    valid = predictions["hsi_trstr_region_valid"][0, 0, query_idx].detach().bool().cpu().numpy()
    uncertainty = np.exp(logvar)
    magnitude = np.linalg.norm(vote, axis=-1)

    csv_name = f"09_{prefix}_trstr_96_regions.csv"
    with (output_dir / csv_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["region", "valid", "gate", "logvar", "uncertainty", "vote_x_m", "vote_y_m", "vote_z_m", "vote_norm_m"])
        for region in range(len(gate)):
            writer.writerow([region, int(valid[region]), gate[region], logvar[region], uncertainty[region], *vote[region].tolist(), magnitude[region]])
    files.append(csv_name)

    for label, values in (("gate", gate), ("uncertainty", uncertainty), ("vote_norm", magnitude), ("valid", valid.astype(float))):
        name = f"09_{prefix}_trstr_{label}_matrix.png"
        render_value_grid(values, 8, 12, f"TRSTR real {label}: 96 regions").save(output_dir / name)
        files.append(name)

    region_ids = trstr_head.region_bank.vertex_region_ids.detach().cpu().numpy()
    segmented_png = f"09_{prefix}_trstr_segmented_smpl.png"
    render_projected_mesh(base_vertices, faces, k, rgb.size, region_ids).save(output_dir / segmented_png)
    files.append(segmented_png)
    region_colors = PALETTE[region_ids % 7]
    segmented_ply = f"09_{prefix}_trstr_segmented_smpl.ply"
    write_ply_vertices_faces(output_dir / segmented_ply, base_vertices.detach().cpu().numpy(), region_colors, faces)
    files.append(segmented_ply)

    base_t = predictions["pred_transl_cam"][0, 0, query_idx].detach().float()
    refined_t = predictions["hsi_trstr_refined_pred_transl_cam"][0, 0, query_idx].detach().float()
    delta = refined_t - base_t
    refined_vertices = base_vertices + delta[None]
    comparison = rgb.copy().convert("RGBA")
    base_layer = render_mesh_layer(base_vertices, faces, k, rgb.size, (217, 172, 166))
    refined_layer = render_mesh_layer(refined_vertices, faces, k, rgb.size, (153, 173, 212))
    comparison.alpha_composite(base_layer)
    comparison.alpha_composite(refined_layer)
    compare_name = f"10_{prefix}_trstr_base_refined_overlay.png"
    comparison.convert("RGB").save(output_dir / compare_name)
    files.append(compare_name)

    record = {
        "available": True,
        "num_regions": int(len(gate)),
        "valid_region_ratio": float(valid.mean()),
        "gate_mean_valid": float(gate[valid].mean()) if valid.any() else None,
        "uncertainty_mean_valid": float(uncertainty[valid].mean()) if valid.any() else None,
        "vote_norm_mean_valid_m": float(magnitude[valid].mean()) if valid.any() else None,
        "base_translation_m": base_t.cpu().tolist(),
        "refined_translation_m": refined_t.cpu().tolist(),
        "delta_translation_m": delta.cpu().tolist(),
        "delta_norm_m": float(torch.linalg.norm(delta).cpu()),
    }
    json_name = f"10_{prefix}_trstr_summary.json"
    (output_dir / json_name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    files.append(json_name)
    return record, files


def render_mesh_layer(
    vertices: torch.Tensor,
    faces: np.ndarray,
    k: torch.Tensor,
    image_size_wh: tuple[int, int],
    color: tuple[int, int, int],
) -> Image.Image:
    width, height = image_size_wh
    projected = project_vertices(vertices, k)
    z = vertices[:, 2].detach().float().cpu().numpy()
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    step = max(len(faces) // 5000, 1)
    for face in faces[::step]:
        if np.all(np.isfinite(projected[face])) and np.all(z[face] > 0):
            polygon = [tuple(projected[index]) for index in face]
            draw.line(polygon + [polygon[0]], fill=(*color, 110), width=1)
    return canvas


def audit_temporal_inputs(args: argparse.Namespace, image_path: Path, output_dir: Path) -> dict[str, Any]:
    frames_dir = resolve_input_path(args.temporal_frames_dir) if args.temporal_frames_dir else image_path.parent
    frames = (
        sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if frames_dir.is_dir()
        else []
    )
    required = int(args.temporal_window)
    available = len(frames) >= required
    files: list[str] = []
    status = {
        "available": available,
        "frames_dir": str(frames_dir),
        "num_frames_found": len(frames),
        "required_window": required,
        "reason": (
            "rgb_window_available_but_temporal_checkpoint_still_required"
            if available
            else ("insufficient_consecutive_rgb_frames" if frames_dir.is_dir() else "frames_directory_missing")
        ),
        "note": "This exporter does not synthesize temporal motion from a single image.",
    }
    if frames:
        chosen = frames[:required]
        thumbs = [Image.open(path).convert("RGB") for path in chosen]
        thumb_h = 180
        thumbs = [image.resize((max(1, int(image.width * thumb_h / image.height)), thumb_h), Image.Resampling.LANCZOS) for image in thumbs]
        canvas = Image.new("RGB", (sum(image.width for image in thumbs) + 8 * (len(thumbs) - 1), thumb_h), PAPER)
        x = 0
        for image in thumbs:
            canvas.paste(image, (x, 0))
            x += image.width + 8
        name = "14_temporal_rgb_inventory.png"
        canvas.save(output_dir / name)
        files.append(name)
    name = "14_temporal_status.json"
    (output_dir / name).write_text(json.dumps(status, indent=2), encoding="utf-8")
    files.append(name)
    return {"status": status, "files": files}


if __name__ == "__main__":
    main()
