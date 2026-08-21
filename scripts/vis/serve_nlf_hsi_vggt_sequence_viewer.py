#!/usr/bin/env python
"""Serve a Viser viewer for VGGT-Omega + NLF + HSI sequence inference.

The important invariant is that the full selected frame sequence is processed in
one VGGT forward pass.  This preserves the VGGT camera/world frame shared by all
frames in the sequence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
)
from vggt_omega.data.geometry import (  # noqa: E402
    ResizeGeometry,
    compute_resize_geometry,
    pad_image_batch,
    resolve_image_size_config,
    resize_image_with_geometry,
    transform_xyxy_to_normalized_cxcywh,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner  # noqa: E402
from vggt_omega.tracking.io import IMAGE_EXTENSIONS, iter_image_files  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


PALETTE: list[tuple[int, int, int]] = [
    (41, 98, 255),
    (239, 71, 111),
    (6, 180, 162),
    (255, 176, 0),
    (131, 90, 241),
    (46, 204, 113),
    (236, 72, 153),
    (14, 165, 233),
    (217, 119, 6),
    (99, 102, 241),
]

HSI_VISUAL_SCALE_MIN = 0.1
HSI_VISUAL_SCALE_MAX = 10.0
HSI_VISUAL_SCALE_SLIDER_MIN = float(np.log10(HSI_VISUAL_SCALE_MIN))
HSI_VISUAL_SCALE_SLIDER_MAX = float(np.log10(HSI_VISUAL_SCALE_MAX))
HUMAN_MASK_DILATION_MIN_PX = 0
HUMAN_MASK_DILATION_MAX_PX = 32
HUMAN_MASK_DILATION_DEFAULT_PX = 5


def hsi_visual_scale_to_slider(scale: float) -> float:
    clamped = min(HSI_VISUAL_SCALE_MAX, max(HSI_VISUAL_SCALE_MIN, float(scale)))
    return float(np.log10(clamped))


def hsi_visual_slider_to_scale(value: float) -> float:
    scale = float(10.0 ** float(value))
    return min(HSI_VISUAL_SCALE_MAX, max(HSI_VISUAL_SCALE_MIN, scale))


def sync_if_cuda(device: torch.device | None = None) -> None:
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.synchronize(device)


def elapsed_since(start: float, device: torch.device | None = None) -> float:
    sync_if_cuda(device)
    return time.perf_counter() - start


def add_timing_rate(entry: dict[str, float], frames: int) -> dict[str, float]:
    seconds = float(entry.get("seconds", 0.0))
    entry["ms_per_frame"] = 1000.0 * seconds / float(max(int(frames), 1))
    entry["fps"] = float(frames) / seconds if seconds > 0.0 else 0.0
    return entry


def main() -> None:
    total_start = time.perf_counter()
    timings: dict[str, Any] = {}
    args = parse_args()
    if not np.isfinite(args.hsi_visual_scale) or not HSI_VISUAL_SCALE_MIN <= float(args.hsi_visual_scale) <= HSI_VISUAL_SCALE_MAX:
        raise ValueError(
            f"--hsi-visual-scale must be finite and within "
            f"[{HSI_VISUAL_SCALE_MIN}, {HSI_VISUAL_SCALE_MAX}], got {args.hsi_visual_scale}"
        )
    if not HUMAN_MASK_DILATION_MIN_PX <= int(args.human_mask_dilation_px) <= HUMAN_MASK_DILATION_MAX_PX:
        raise ValueError(
            f"--human-mask-dilation-px must be within "
            f"[{HUMAN_MASK_DILATION_MIN_PX}, {HUMAN_MASK_DILATION_MAX_PX}], "
            f"got {args.human_mask_dilation_px}"
        )
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames_dir = resolve_project_path(args.frames_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_start = time.perf_counter()
    frame_paths = select_frames(frames_dir, args)
    if not frame_paths:
        raise RuntimeError(f"No RGB frames found under {frames_dir}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")
    timings["select_frames"] = {"seconds": time.perf_counter() - step_start}

    step_start = time.perf_counter()
    config = load_config(args)
    model_config = config.get("model", {})
    args.hsi_scene_affine_mode = str(model_config.get("hsi_scene_affine_mode", "per_frame"))
    args.hsi_scene_affine_ema_alpha = float(model_config.get("hsi_scene_affine_ema_alpha", 0.25))
    patch_size = int(config.get("model", {}).get("patch_size", 16))
    _, image_resolution = resolve_image_size_config(config.get("data", {}), args.image_size)
    max_humans = int(args.max_humans or config.get("model", {}).get("num_smpl_queries", 20))
    timings["load_config"] = {"seconds": time.perf_counter() - step_start}

    step_start = time.perf_counter()
    images, geometries = load_sequence_images(frame_paths, image_resolution, patch_size, str(config["data"].get("resize_mode", "balanced")))
    priors = build_query_priors(frame_paths, geometries, args, max_humans, device) if args.query_source == "bedlam_sidecar" else None
    sync_if_cuda(device)
    timings["load_images_and_priors"] = {"seconds": time.perf_counter() - step_start}

    step_start = time.perf_counter()
    model = build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model, config, device)
    checkpoint = resolve_stage_checkpoint(args)
    load_training_checkpoint(model, checkpoint, device)
    smpl = SMPLLayer(require_smpl_model_dir(config, args)).to(device).eval()
    sync_if_cuda(device)
    timings["load_models"] = {"seconds": time.perf_counter() - step_start}

    step_start = time.perf_counter()
    image_sequence = images.unsqueeze(0).to(device)
    sync_if_cuda(device)
    timings["transfer_images_to_device"] = {"seconds": time.perf_counter() - step_start}

    with torch.inference_mode():
        step_start = time.perf_counter()
        predictions = run_model(model, image_sequence, priors)
        timings["model_forward"] = {"seconds": elapsed_since(step_start, device)}
    step_start = time.perf_counter()
    geometry_snapshot = snapshot_viewer_geometry(predictions)
    apply_posthoc_tracking_overlay(predictions, args)
    assert_viewer_geometry_unchanged(predictions, geometry_snapshot, args)
    sync_if_cuda(device)
    timings["posthoc_tracking_overlay"] = {"seconds": time.perf_counter() - step_start}

    step_start = time.perf_counter()
    scene = build_scene_data(
        frame_paths=frame_paths,
        images=image_sequence,
        predictions=predictions,
        priors=priors,
        smpl=smpl,
        args=args,
        device=device,
        timings=timings,
    )
    timings["build_scene_data"] = {"seconds": elapsed_since(step_start, device)}
    step_start = time.perf_counter()
    print_human_point_removal_summary(scene, args)
    validate_scene(scene, predictions, image_sequence)
    timings["validate_and_summarize"] = {"seconds": time.perf_counter() - step_start}

    timings["total_before_viewer"] = {"seconds": elapsed_since(total_start, device)}
    for value in timings.values():
        if isinstance(value, dict) and "seconds" in value:
            add_timing_rate(value, len(frame_paths))
    summary = build_summary(args, frame_paths, checkpoint, image_sequence, predictions, scene, output_dir, timings)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"viewer": f"http://127.0.0.1:{int(args.port)}", "summary": str(summary_path)}, indent=2), flush=True)
    if bool(args.smoke_only):
        print("[ok] NLF-HSI VGGT sequence viewer smoke passed", flush=True)
        return

    server = viser.ViserServer(port=int(args.port))
    if hasattr(server, "scene") and hasattr(server.scene, "set_up_direction"):
        server.scene.set_up_direction("-y")
    elif hasattr(server, "set_up_direction"):
        server.set_up_direction("-y")
    viewer = SequenceViewer(server=server, transforms=vtf, scene=scene, args=args)
    viewer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--query-source", choices=["bedlam_sidecar", "nlf_detector"], default="bedlam_sidecar")
    parser.add_argument("--preprocessed-root", default="outputs/preprocess/bedlam_boxes")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--stage2-dir", default="outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl")
    parser.add_argument("--checkpoint", default="", help="Explicit HSI checkpoint. If omitted, rank1 from stage2 checkpoint_topk_index.json is used.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/nlf_hsi_vggt_sequence_viewer")
    parser.add_argument("--device", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--tracking-overlay", choices=["none", "base_smpl"], default="none")
    parser.add_argument("--track-max-age", type=int, default=90)
    parser.add_argument("--track-min-quality", type=float, default=0.25)
    parser.add_argument("--track-max-center-distance", type=float, default=0.25)
    parser.add_argument("--track-max-transl-distance", type=float, default=1.50)
    parser.add_argument("--track-max-beta-l1", type=float, default=0.30)
    parser.add_argument("--depth-point-stride", type=int, default=4, help="Initial Viser point-cloud sampling stride. This can be changed live in the GUI.")
    parser.add_argument("--max-scene-depth", type=float, default=30.0, help="Initial far-depth clipping in meters. Set 0 to disable; this can be changed live in the GUI.")
    parser.add_argument("--viewer-mode", choices=["4D current frame", "3D accumulate", "Hybrid"], default="4D current frame", help="Initial Viser playback mode.")
    parser.add_argument("--environment-display", choices=["points", "mesh", "both"], default="points", help="Initial environment rendering mode. The Viser GUI can still toggle this live.")
    parser.add_argument("--hsi-visual-scale", type=float, default=1.0, help="Initial viewer-only multiplier for HSI environment points and HSI camera positions.")
    parser.add_argument("--human-mask-dilation-px", type=int, default=HUMAN_MASK_DILATION_DEFAULT_PX, help="Pixel dilation around the projected SMPL silhouette when removing human depth points in normal display mode.")
    parser.add_argument("--filter-human-points", action=argparse.BooleanOptionalAction, default=True, help="Initial Viser state for projected-SMPL human point filtering.")
    parser.add_argument("--env-mesh-depth-edge-rtol", type=float, default=0.15, help="Relative depth discontinuity threshold for environment surface mesh faces.")
    parser.add_argument("--env-mesh-color-groups", type=int, default=216, help="Maximum color buckets used to approximate RGB environment mesh face color with Viser simple meshes.")
    parser.add_argument("--env-mesh-color-mode", choices=["point_overlay", "bucketed_mesh"], default="point_overlay", help="How to color depth mesh. point_overlay matches Human3R's Viser RGB point-cloud path over a neutral surface.")
    parser.add_argument("--env-mesh-overlay-point-size-scale", type=float, default=0.75, help="Point-size scale for RGB overlay points in point_overlay mesh color mode.")
    parser.add_argument("--smpl-edit-output", default="", help="Optional JSON path for viewer-only SMPL translation edits. Defaults to <output-dir>/smpl_edit_offsets.json.")
    parser.add_argument("--show-track-ids", action=argparse.BooleanOptionalAction, default=True, help="Initial visibility for SMPL track ID labels. The Viser GUI can still toggle this live.")
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--camera-frustum-scale", type=float, default=0.20)
    parser.add_argument("--alignment-vertex-stride", type=int, default=16)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--smoke-only", action="store_true", help="Run inference, validation, and summary export, then exit without serving Viser.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def ensure_viser_available() -> None:
    try:
        import viser  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "The Viser viewer requires the optional demo dependency 'viser'. "
            "Install it in the server environment with `pip install viser` or install the project demo extra."
        ) from exc


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def select_frames(frames_dir: Path, args: argparse.Namespace) -> list[Path]:
    paths = iter_image_files(frames_dir)
    start = max(0, int(args.start_index))
    stride = max(1, int(args.frame_stride))
    selected = paths[start::stride]
    if int(args.max_frames) > 0:
        selected = selected[: int(args.max_frames)]
    return selected


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(resolve_project_path(args.path_config)), load_yaml_config(resolve_project_path(args.train_config)))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = str(resolve_project_path(args.baseline_checkpoint))
    data_cfg = config.setdefault("data", {})
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    data_cfg["image_size"] = int(image_size)
    data_cfg["image_resolution"] = int(image_resolution)
    data_cfg.setdefault("resize_mode", "balanced")

    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["enable_hsi_refine"] = True
    model_cfg["smpl_provider"] = "nlf"
    model_cfg["num_smpl_queries"] = int(args.max_humans)
    model_cfg["smpl_query_box_prior"] = args.query_source == "bedlam_sidecar"
    model_cfg["smpl_query_patch_pool"] = False
    model_cfg["nlf_use_detector"] = args.query_source == "nlf_detector"
    model_cfg["nlf_require_boxes"] = args.query_source == "bedlam_sidecar"
    model_cfg["smpl_track_assignment_mode"] = "gt" if args.query_source == "bedlam_sidecar" else "none"
    model_cfg["smpl_use_external_track_prior"] = False
    if args.smpl_model_dir:
        config.setdefault("assets", {})["smpl_model_dir"] = str(resolve_project_path(args.smpl_model_dir))
    return config


def resolve_stage_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return resolve_project_path(args.checkpoint)
    stage_dir = resolve_project_path(args.stage2_dir)
    index_path = stage_dir / "checkpoint_topk_index.json"
    if not index_path.is_file():
        latest = stage_dir / "checkpoint_latest.pt"
        if latest.is_file():
            return latest
        raise FileNotFoundError(f"Missing stage2 checkpoint index and latest checkpoint: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"No top-k checkpoint entries in {index_path}")
    return resolve_project_path(entries[0]["path"])


def require_smpl_model_dir(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.smpl_model_dir:
        return str(resolve_project_path(args.smpl_model_dir))
    return require_path(config, "assets.smpl_model_dir", allow_empty=False)


def load_sequence_images(
    frame_paths: list[Path],
    image_resolution: int,
    patch_size: int,
    resize_mode: str,
) -> tuple[torch.Tensor, list[ResizeGeometry]]:
    tensors: list[torch.Tensor] = []
    geometries: list[ResizeGeometry] = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        geometry = compute_resize_geometry((image.height, image.width), image_resolution=image_resolution, patch_size=patch_size, mode=resize_mode)
        resized = resize_image_with_geometry(image, geometry, Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        geometries.append(geometry)
    batch, pads = pad_image_batch(tensors, patch_size=patch_size, value=1.0)
    input_hw = (int(batch.shape[-2]), int(batch.shape[-1]))
    padded_geometries = [
        replace(geometry, input_hw=input_hw, pad_xyxy=tuple(int(v) for v in pads[idx]))
        for idx, geometry in enumerate(geometries)
    ]
    return batch, padded_geometries


def build_query_priors(
    frame_paths: list[Path],
    geometries: list[ResizeGeometry],
    args: argparse.Namespace,
    max_humans: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boxes = torch.zeros(1, len(frame_paths), max_humans, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(1, len(frame_paths), max_humans, dtype=torch.bool, device=device)
    track_ids = torch.full((1, len(frame_paths), max_humans), -1, dtype=torch.long, device=device)
    track_mask = torch.zeros(1, len(frame_paths), max_humans, dtype=torch.bool, device=device)
    preprocessed_root = resolve_project_path(args.preprocessed_root)
    bedlam_root = resolve_project_path(args.bedlam_root) if args.bedlam_root else None

    for frame_idx, image_path in enumerate(frame_paths):
        frame = load_sidecar_frame(preprocessed_root, bedlam_root, image_path)
        persons = frame.get("persons", [])
        if not isinstance(persons, list):
            continue
        image_h, image_w = frame_hw(frame, geometries[frame_idx].orig_hw)
        slot = 0
        for person_idx, person in enumerate(persons):
            if slot >= max_humans:
                break
            if not person_train_valid(person) or not bool(person.get("bbox_valid", False)):
                continue
            xyxy = person_xyxy(person, image_w=image_w, image_h=image_h)
            if xyxy is None:
                continue
            box, valid = transform_xyxy_to_normalized_cxcywh(xyxy, geometries[frame_idx])
            if not valid:
                continue
            boxes[0, frame_idx, slot] = torch.as_tensor(box, dtype=torch.float32, device=device)
            box_mask[0, frame_idx, slot] = True
            track_ids[0, frame_idx, slot] = int(person_track_id(person, person_idx))
            track_mask[0, frame_idx, slot] = True
            slot += 1
    if not bool(box_mask.any()):
        raise RuntimeError("No valid sidecar boxes were loaded. Check FRAMES_DIR, BEDLAM_ROOT, and PREPROCESSED_ROOT.")
    return {"smpl_query_boxes": boxes, "smpl_query_boxes_mask": box_mask, "smpl_track_ids": track_ids, "smpl_track_mask": track_mask}


def load_sidecar_frame(preprocessed_root: Path, bedlam_root: Path | None, image_path: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if bedlam_root is not None:
        try:
            rel = image_path.resolve().relative_to(bedlam_root.resolve())
            parts = rel.parts
            if len(parts) >= 4 and parts[2] == "rgb":
                candidates.append(preprocessed_root / parts[0] / parts[1] / "smpl_boxes" / f"{image_path.stem}.pkl")
        except ValueError:
            pass
    candidates.extend(
        [
            preprocessed_root / "smpl_boxes" / f"{image_path.stem}.pkl",
            preprocessed_root / f"{image_path.stem}.pkl",
        ]
    )
    for path in candidates:
        if path.is_file():
            with path.open("rb") as file:
                data = pickle.load(file)
            if not isinstance(data, dict):
                raise TypeError(f"Sidecar must contain a frame dict: {path}")
            return data
    raise FileNotFoundError(f"Missing sidecar for frame {image_path.name}. Tried: {[str(path) for path in candidates]}")


def frame_hw(frame: dict[str, Any], fallback_hw: tuple[int, int]) -> tuple[int, int]:
    if "image_hw" in frame:
        h, w = frame["image_hw"]
        return int(h), int(w)
    return int(fallback_hw[0]), int(fallback_hw[1])


def person_train_valid(person: dict[str, Any]) -> bool:
    if "train_valid" in person:
        return bool(person["train_valid"])
    if "valid" in person:
        return bool(person["valid"])
    return bool(person.get("bbox_valid", False))


def person_xyxy(person: dict[str, Any], image_w: int, image_h: int) -> np.ndarray | None:
    if "bbox_xyxy_pixels" in person:
        return np.asarray(person["bbox_xyxy_pixels"], dtype=np.float32).reshape(4)
    if "bbox_cxcywh_norm" in person:
        cx, cy, bw, bh = np.asarray(person["bbox_cxcywh_norm"], dtype=np.float32).reshape(4)
        return np.asarray(
            [
                (cx - 0.5 * bw) * float(image_w),
                (cy - 0.5 * bh) * float(image_h),
                (cx + 0.5 * bw) * float(image_w),
                (cy + 0.5 * bh) * float(image_h),
            ],
            dtype=np.float32,
        )
    return None


def person_track_id(person: dict[str, Any], fallback_index: int) -> int:
    for key in ("person_id", "track_id_prior", "track_id", "person_index"):
        if key not in person:
            continue
        try:
            value = int(person[key])
            if value >= 0:
                return value
        except (TypeError, ValueError):
            continue
    return int(fallback_index)


def run_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    priors: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if priors is not None:
        kwargs.update(
            {
                "smpl_query_boxes": priors["smpl_query_boxes"],
                "smpl_query_boxes_mask": priors["smpl_query_boxes_mask"],
                "smpl_track_ids": priors["smpl_track_ids"],
                "smpl_track_mask": priors["smpl_track_mask"],
            }
        )
    predictions = model(images, **kwargs)
    predictions["images"] = images
    return predictions


def apply_posthoc_tracking_overlay(
    predictions: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """Attach display-only IDs after HSI inference so geometry stays unchanged."""
    if str(args.tracking_overlay) == "none":
        return
    required = {
        "pred_boxes": predictions.get("pred_boxes"),
        "pred_betas": predictions.get("hsi_refined_pred_betas", predictions.get("pred_betas")),
        "pred_transl_cam": predictions.get(
            "hsi_refined_pred_transl_cam", predictions.get("pred_transl_cam")
        ),
        "pred_confs": predictions.get("pred_confs"),
    }
    missing = [name for name, value in required.items() if not isinstance(value, torch.Tensor)]
    if missing:
        raise RuntimeError(f"Post-HSI tracking overlay is missing predictions: {missing}")
    boxes = required["pred_boxes"]
    confs = required["pred_confs"]
    confidence = confs.detach().float()
    while confidence.ndim > 3:
        confidence = confidence.mean(dim=-1)
    query_mask = confidence >= float(args.conf_threshold)
    query_mask &= torch.isfinite(boxes).all(dim=-1)
    query_mask &= boxes[..., 2:].gt(0.0).all(dim=-1)
    assigner = BaseSMPLTrackAssigner(
        max_age=int(args.track_max_age),
        min_track_quality=float(args.track_min_quality),
        max_center_distance_norm=float(args.track_max_center_distance),
        max_transl_distance_m=float(args.track_max_transl_distance),
        max_beta_l1=float(args.track_max_beta_l1),
        id_weight=0.0,
    )
    predictions.update(
        assigner.assign(
            boxes=boxes,
            pred_betas=required["pred_betas"],
            pred_transl_cam=required["pred_transl_cam"],
            pred_confs=confs,
            query_mask=query_mask,
        )
    )
    predictions["viewer_tracking_overlay_active"] = boxes.new_ones(())


def snapshot_viewer_geometry(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = (
        "hsi_scene_scale",
        "hsi_scene_depth_bias",
        "pred_poses",
        "pred_betas",
        "pred_transl_cam",
        "hsi_refined_pred_poses",
        "hsi_refined_pred_betas",
        "hsi_refined_pred_transl_cam",
    )
    return {
        key: predictions[key].detach().clone()
        for key in keys
        if isinstance(predictions.get(key), torch.Tensor)
    }


def assert_viewer_geometry_unchanged(
    predictions: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    if str(args.tracking_overlay) == "none":
        return
    changed = [
        key
        for key, before in snapshot.items()
        if not isinstance(predictions.get(key), torch.Tensor)
        or not torch.equal(before, predictions[key])
    ]
    if changed:
        raise RuntimeError(f"Display-only tracking modified Stage2 geometry tensors: {changed}")
    reference = predictions.get("pred_transl_cam")
    if isinstance(reference, torch.Tensor):
        predictions["viewer_tracking_geometry_unchanged"] = reference.new_ones(())


def build_scene_data(
    frame_paths: list[Path],
    images: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    priors: dict[str, torch.Tensor] | None,
    smpl: SMPLLayer,
    args: argparse.Namespace,
    device: torch.device,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_start = time.perf_counter()
    image_hw = tuple(int(v) for v in images.shape[-2:])
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"].detach().float(), image_size_hw=image_hw, build_intrinsics=True)
    raw_depth = canonical_depth(predictions["depth"]).detach().float()
    hsi_depth = raw_depth
    if "hsi_scene_scale" in predictions and "hsi_scene_depth_bias" in predictions:
        scale = predictions["hsi_scene_scale"].detach().float().reshape(raw_depth.shape[:2] + (1, 1)).to(raw_depth.device)
        bias = predictions["hsi_scene_depth_bias"].detach().float().reshape(raw_depth.shape[:2] + (1, 1)).to(raw_depth.device)
        hsi_depth = raw_depth * scale + bias
    if timings is not None:
        timings["scene_camera_and_depth"] = {"seconds": elapsed_since(stage_start, device)}

    stage_start = time.perf_counter()
    people = decode_people(predictions, smpl, args, device)
    if timings is not None:
        timings["decode_people_smpl"] = {"seconds": elapsed_since(stage_start, device)}
    faces = np.asarray(smpl.faces, dtype=np.int64).reshape(-1, 3)
    track_palette: dict[int, int] = {}

    stage_start = time.perf_counter()
    alignment = compute_depth_alignment(predictions, people, raw_depth, hsi_depth, intrinsics, args)
    if timings is not None:
        timings["depth_alignment"] = {"seconds": elapsed_since(stage_start, device)}

    stage_start = time.perf_counter()
    frames = []
    for idx, image_path in enumerate(frame_paths):
        extrinsic = extrinsics[0, idx].detach().float().cpu().numpy()
        intrinsic = intrinsics[0, idx].detach().float().cpu().numpy()
        hsi_scale = prediction_scalar(predictions, "hsi_scene_scale", idx)
        hsi_bias = prediction_scalar(predictions, "hsi_scene_depth_bias", idx)
        hsi_frame_scale = prediction_scalar(predictions, "hsi_frame_scene_scale", idx)
        hsi_frame_bias = prediction_scalar(predictions, "hsi_frame_scene_depth_bias", idx)
        hsi_extrinsic = scale_w2c_extrinsic_translation(extrinsic, float(hsi_scale if hsi_scale is not None else 1.0))
        rgb = images[0, idx].detach().float().cpu()
        frame_people = select_frame_people(predictions, people, priors, idx, hsi_extrinsic, faces, track_palette, args)
        raw_human_mask = projected_human_exclusion_mask(raw_depth[0, idx], frame_people, intrinsic, "base_vertices_cam", args)
        hsi_human_mask = projected_human_exclusion_mask(hsi_depth[0, idx], frame_people, intrinsic, "hsi_vertices_cam", args)
        raw_points_full, raw_colors_full = depth_to_world_points(raw_depth[0, idx], rgb, intrinsic, extrinsic, args)
        hsi_points_full, hsi_colors_full = depth_to_world_points(hsi_depth[0, idx], rgb, intrinsic, hsi_extrinsic, args)
        raw_points, raw_colors = depth_to_world_points(raw_depth[0, idx], rgb, intrinsic, extrinsic, args, exclude_mask=raw_human_mask)
        hsi_points, hsi_colors = depth_to_world_points(hsi_depth[0, idx], rgb, intrinsic, hsi_extrinsic, args, exclude_mask=hsi_human_mask)
        frames.append(
            {
                "frame_index": int(idx),
                "frame_id": image_path.stem,
                "image": str(image_path),
                "raw_points": raw_points,
                "raw_colors": raw_colors,
                "raw_points_full": raw_points_full,
                "raw_colors_full": raw_colors_full,
                "hsi_points": hsi_points,
                "hsi_colors": hsi_colors,
                "hsi_points_full": hsi_points_full,
                "hsi_colors_full": hsi_colors_full,
                "raw_human_exclusion_mask": raw_human_mask,
                "hsi_human_exclusion_mask": hsi_human_mask,
                "raw_depth_map": raw_depth[0, idx].detach().float().cpu().numpy().astype(np.float32, copy=False),
                "hsi_depth_map": hsi_depth[0, idx].detach().float().cpu().numpy().astype(np.float32, copy=False),
                "rgb_chw": rgb.detach().float().cpu().numpy().astype(np.float32, copy=False),
                "intrinsic": intrinsic.astype(np.float32, copy=False),
                "raw_extrinsic": extrinsic.astype(np.float32, copy=False),
                "hsi_extrinsic": hsi_extrinsic.astype(np.float32, copy=False),
                "depth_point_stride": int(args.depth_point_stride),
                "max_scene_depth": float(args.max_scene_depth),
                "people": frame_people,
                "camera": camera_pose_from_extrinsic(hsi_extrinsic, intrinsic),
                "raw_camera": camera_pose_from_extrinsic(extrinsic, intrinsic),
                "hsi_camera": camera_pose_from_extrinsic(hsi_extrinsic, intrinsic),
                "hsi_scene_scale": hsi_scale,
                "hsi_scene_depth_bias": hsi_bias,
                "hsi_frame_scene_scale": hsi_frame_scale,
                "hsi_frame_scene_depth_bias": hsi_frame_bias,
                "depth_alignment": alignment[idx],
            }
        )
    raw_camera_trajectory = np.stack([frame["raw_camera"]["position"] for frame in frames], axis=0).astype(np.float32) if frames else np.zeros((0, 3), dtype=np.float32)
    hsi_camera_trajectory = np.stack([frame["hsi_camera"]["position"] for frame in frames], axis=0).astype(np.float32) if frames else np.zeros((0, 3), dtype=np.float32)
    if timings is not None:
        timings["build_frame_point_clouds"] = {"seconds": elapsed_since(stage_start, device)}
    return {
        "frames": frames,
        "image_hw": list(image_hw),
        "track_palette": track_palette,
        "camera_trajectory": hsi_camera_trajectory,
        "camera_trajectory_raw": raw_camera_trajectory,
        "camera_trajectory_hsi": hsi_camera_trajectory,
        "hsi_scene_affine_mode": str(getattr(args, "hsi_scene_affine_mode", "per_frame")),
        "hsi_scene_affine_ema_alpha": float(getattr(args, "hsi_scene_affine_ema_alpha", 0.25)),
    }


def canonical_depth(tensor: torch.Tensor) -> torch.Tensor:
    depth = tensor
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W] or [B,S,H,W,1], got {tuple(tensor.shape)}")
    return depth


def depth_to_world_points(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    args: argparse.Namespace,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return depth_to_world_points_with_limits(
        depth=depth,
        rgb=rgb,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        depth_point_stride=int(args.depth_point_stride),
        max_scene_depth=float(args.max_scene_depth),
        exclude_mask=exclude_mask,
    )


def depth_to_world_points_with_limits(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    depth_point_stride: int,
    max_scene_depth: float,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = depth.detach().float()
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    step = max(1, int(depth_point_stride))
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=torch.float32),
        torch.arange(0, width, step, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()]
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points = torch.stack([x, y, z], dim=-1)
    rgb_use = rgb.to(device=depth.device, dtype=torch.float32)
    if tuple(rgb_use.shape[-2:]) != (height, width):
        rgb_use = F.interpolate(rgb_use[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    colors = (rgb_use[:, ys.long(), xs.long()].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    mask = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    if float(max_scene_depth) > 0:
        mask = mask & (z <= float(max_scene_depth))
    if exclude_mask is not None:
        exclude = torch.as_tensor(exclude_mask, dtype=torch.bool, device=depth.device)
        if tuple(exclude.shape) != (height, width):
            exclude = F.interpolate(exclude[None, None].float(), size=(height, width), mode="nearest")[0, 0].bool()
        mask = mask & ~exclude[ys.long(), xs.long()]
    points_np = points[mask].detach().cpu().numpy().astype(np.float32, copy=False)
    colors_np = colors[mask].detach().cpu().numpy().astype(np.uint8, copy=False)
    return camera_points_to_world_np(points_np, extrinsic), colors_np


def projected_human_exclusion_mask(
    depth: torch.Tensor,
    people: list[dict[str, Any]],
    intrinsic: np.ndarray,
    vertex_key: str,
    args: argparse.Namespace,
    dilation_px_override: int | None = None,
) -> np.ndarray:
    depth_np = depth.detach().float().cpu().numpy().astype(np.float32, copy=False)
    height, width = depth_np.shape[-2:]
    exclusion = np.zeros((height, width), dtype=bool)
    dilation_px = max(
        0,
        int(args.human_mask_dilation_px if dilation_px_override is None else dilation_px_override),
    )
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    for person in people:
        vertices = person.get(vertex_key)
        if vertices is None and vertex_key == "hsi_vertices_cam":
            vertices = person.get("base_vertices_cam")
        if vertices is None:
            continue
        vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        valid_vertices = np.isfinite(vertices_np).all(axis=1) & (vertices_np[:, 2] > 1e-6)
        if int(valid_vertices.sum()) < 3:
            continue
        projected = np.zeros((vertices_np.shape[0], 2), dtype=np.float32)
        z = vertices_np[valid_vertices, 2]
        projected[valid_vertices, 0] = vertices_np[valid_vertices, 0] / z * fx + cx
        projected[valid_vertices, 1] = vertices_np[valid_vertices, 1] / z * fy + cy
        core_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(core_image)
        faces = np.asarray(person.get("faces", np.empty((0, 3), dtype=np.int64)), dtype=np.int64).reshape(-1, 3)
        face_bounds = (faces >= 0).all(axis=1) & (faces < vertices_np.shape[0]).all(axis=1)
        faces = faces[face_bounds]
        valid_faces = faces[valid_vertices[faces].all(axis=1)] if faces.size > 0 else faces
        triangles = projected[valid_faces] if valid_faces.size > 0 else np.empty((0, 3, 2), dtype=np.float32)
        if triangles.shape[0] > 0:
            intersects_image = (
                (triangles[..., 0].max(axis=1) >= 0.0)
                & (triangles[..., 0].min(axis=1) < float(width))
                & (triangles[..., 1].max(axis=1) >= 0.0)
                & (triangles[..., 1].min(axis=1) < float(height))
            )
            triangles = triangles[intersects_image]
            triangles[..., 0] = np.clip(triangles[..., 0], 0.0, width - 1.0)
            triangles[..., 1] = np.clip(triangles[..., 1], 0.0, height - 1.0)
            for triangle in np.rint(triangles).astype(np.int32):
                draw.polygon([(int(point[0]), int(point[1])) for point in triangle], fill=255)
        else:
            hull = convex_hull_2d(np.rint(projected[valid_vertices]).astype(np.int32))
            if hull.shape[0] >= 3:
                draw.polygon([(int(point[0]), int(point[1])) for point in hull], fill=255)
        silhouette_image = core_image
        if dilation_px > 0:
            kernel_size = 2 * dilation_px + 1
            silhouette_image = core_image.filter(ImageFilter.MaxFilter(kernel_size))
        exclusion |= np.asarray(silhouette_image, dtype=np.uint8) > 0
    return exclusion


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if points_np.shape[0] <= 1:
        return points_np
    points_np = np.unique(points_np, axis=0)
    if points_np.shape[0] <= 2:
        return points_np
    order = np.lexsort((points_np[:, 1], points_np[:, 0]))
    sorted_points = points_np[order]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> int:
        return int(
            (int(first[0]) - int(origin[0])) * (int(second[1]) - int(origin[1]))
            - (int(first[1]) - int(origin[1])) * (int(second[0]) - int(origin[0]))
        )

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in sorted_points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.int32)


def camera_points_to_world_np(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation[None, :]) @ rotation).astype(np.float32)


def scale_w2c_extrinsic_translation(extrinsic: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.asarray(extrinsic, dtype=np.float32).copy()
    scaled[:3, 3] *= float(scale)
    return scaled


def decode_people(
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix, pose_key, beta_key, transl_key in [
        ("base", "pred_poses", "pred_betas", "pred_transl_cam"),
        ("hsi", "hsi_refined_pred_poses", "hsi_refined_pred_betas", "hsi_refined_pred_transl_cam"),
    ]:
        if pose_key not in predictions or beta_key not in predictions or transl_key not in predictions:
            continue
        poses = predictions[pose_key].detach()
        betas = predictions[beta_key].detach()
        transl = predictions[transl_key].detach()
        shape = poses.shape[:3]
        with torch.no_grad():
            vertices, _ = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
        vertices = vertices.reshape(*shape, vertices.shape[-2], 3).to(device=device, dtype=transl.dtype) + transl[..., None, :]
        out[f"{prefix}_vertices_cam"] = vertices.detach()
    return out


def compute_depth_alignment(
    predictions: dict[str, torch.Tensor],
    decoded: dict[str, torch.Tensor],
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    base_vertices = decoded.get("base_vertices_cam")
    hsi_vertices = decoded.get("hsi_vertices_cam", base_vertices)
    confs = predictions["pred_confs"].detach().float()
    frame_count = int(raw_depth.shape[1])
    summaries: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        frame_summary: dict[str, Any] = {"base_raw": [], "base_hsi": [], "hsi_hsi": []}
        if base_vertices is not None:
            frame_summary["base_raw"] = frame_alignment_entries(
                base_vertices[0, frame_idx],
                confs[0, frame_idx, :, 0],
                raw_depth[0, frame_idx],
                intrinsics[0, frame_idx],
                args,
            )
            frame_summary["base_hsi"] = frame_alignment_entries(
                base_vertices[0, frame_idx],
                confs[0, frame_idx, :, 0],
                hsi_depth[0, frame_idx],
                intrinsics[0, frame_idx],
                args,
            )
        if hsi_vertices is not None:
            frame_summary["hsi_hsi"] = frame_alignment_entries(
                hsi_vertices[0, frame_idx],
                confs[0, frame_idx, :, 0],
                hsi_depth[0, frame_idx],
                intrinsics[0, frame_idx],
                args,
            )
        summaries.append(frame_summary)
    return summaries


def frame_alignment_entries(
    vertices_by_query: torch.Tensor,
    confs: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    order = torch.argsort(confs.detach().float(), descending=True).tolist()
    for query_idx in order:
        conf = float(confs[query_idx].detach().cpu().item())
        if conf < float(args.conf_threshold):
            continue
        stats = vertex_depth_alignment(vertices_by_query[query_idx], depth, intrinsic, args)
        if stats["valid_points"] <= 0:
            continue
        stats["query_index"] = int(query_idx)
        stats["confidence"] = conf
        entries.append(stats)
    return entries


def vertex_depth_alignment(
    vertices_cam: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vertices = vertices_cam.detach().float()
    stride = max(1, int(getattr(args, "alignment_vertex_stride", 16)))
    vertices = vertices[::stride]
    z = vertices[:, 2]
    valid = torch.isfinite(vertices).all(dim=-1) & (z > 1e-6)
    if float(args.max_scene_depth) > 0:
        valid = valid & (z <= float(args.max_scene_depth))
    vertices = vertices[valid]
    if vertices.numel() == 0:
        return empty_alignment_stats()
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    fx = intrinsic[0, 0].clamp(min=1e-6)
    fy = intrinsic[1, 1].clamp(min=1e-6)
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    u = vertices[:, 0] / vertices[:, 2] * fx + cx
    v = vertices[:, 1] / vertices[:, 2] * fy + cy
    xi = torch.round(u).long()
    yi = torch.round(v).long()
    in_frame = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    if not bool(in_frame.any()):
        return empty_alignment_stats()
    xi = xi[in_frame]
    yi = yi[in_frame]
    z = vertices[in_frame, 2]
    sampled = depth[yi, xi].detach().float()
    valid_depth = torch.isfinite(sampled) & (sampled > 1e-6)
    if float(args.max_scene_depth) > 0:
        valid_depth = valid_depth & (sampled <= float(args.max_scene_depth))
    if not bool(valid_depth.any()):
        return empty_alignment_stats()
    delta = z[valid_depth] - sampled[valid_depth]
    abs_delta = delta.abs()
    return {
        "valid_points": int(delta.numel()),
        "median_signed_m": float(delta.median().detach().cpu().item()),
        "median_abs_m": float(abs_delta.median().detach().cpu().item()),
        "mean_abs_m": float(abs_delta.mean().detach().cpu().item()),
        "p90_abs_m": float(torch.quantile(abs_delta, 0.90).detach().cpu().item()) if delta.numel() > 1 else float(abs_delta[0].detach().cpu().item()),
    }


def empty_alignment_stats() -> dict[str, Any]:
    return {
        "valid_points": 0,
        "median_signed_m": None,
        "median_abs_m": None,
        "mean_abs_m": None,
        "p90_abs_m": None,
    }


def select_frame_people(
    predictions: dict[str, torch.Tensor],
    decoded: dict[str, torch.Tensor],
    priors: dict[str, torch.Tensor] | None,
    frame_index: int,
    extrinsic: np.ndarray,
    faces: np.ndarray,
    track_palette: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    confs = predictions["pred_confs"][0, frame_index, :, 0].detach().float().cpu()
    assigned_ids = predictions.get("assigned_track_ids")
    assigned_mask = predictions.get("assigned_track_mask")
    assigned_quality = predictions.get("assigned_track_quality")
    assigned_source = predictions.get("assigned_track_source")
    if (
        isinstance(assigned_ids, torch.Tensor)
        and isinstance(assigned_mask, torch.Tensor)
        and bool(assigned_mask[0, frame_index].any())
    ):
        valid = assigned_mask[0, frame_index].detach().cpu().bool()
        track_ids = assigned_ids[0, frame_index].detach().cpu().long()
    elif priors is not None:
        valid = priors["smpl_query_boxes_mask"][0, frame_index].detach().cpu().bool()
        track_ids = priors["smpl_track_ids"][0, frame_index].detach().cpu().long()
    else:
        valid = confs >= float(args.conf_threshold)
        track_ids = torch.arange(confs.numel(), dtype=torch.long)
    order = torch.argsort(confs, descending=True).tolist()
    people: list[dict[str, Any]] = []
    for query_idx in order:
        if not bool(valid[query_idx]) or float(confs[query_idx]) < float(args.conf_threshold):
            continue
        track_id = int(track_ids[query_idx].item()) if int(track_ids[query_idx].item()) >= 0 else int(query_idx)
        color = PALETTE[palette_index_for_track(track_id, track_palette)]
        item: dict[str, Any] = {
            "query_index": int(query_idx),
            "track_id": int(track_id),
            "confidence": float(confs[query_idx].item()),
            "color": color,
            "faces": faces,
        }
        if isinstance(assigned_quality, torch.Tensor):
            item["track_quality"] = float(assigned_quality[0, frame_index, query_idx].detach().float().cpu())
        if isinstance(assigned_source, torch.Tensor):
            item["track_source"] = int(assigned_source[0, frame_index, query_idx].detach().cpu())
        for prefix in ("base", "hsi"):
            key = f"{prefix}_vertices_cam"
            if key in decoded:
                mesh_cam = decoded[key][0, frame_index, query_idx].detach().float().cpu().numpy()
                item[f"{prefix}_vertices_cam"] = mesh_cam.astype(np.float32, copy=False)
                item[f"{prefix}_vertices"] = camera_points_to_world_np(mesh_cam, extrinsic)
        people.append(item)
    return people


def palette_index_for_track(track_id: int, state: dict[int, int]) -> int:
    if track_id not in state:
        state[track_id] = len(state) % len(PALETTE)
    return state[track_id]


def camera_pose_from_extrinsic(extrinsic: np.ndarray, intrinsic: np.ndarray) -> dict[str, Any]:
    rotation_w2c = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    rotation_c2w = rotation_w2c.T
    position = -rotation_c2w @ translation
    fy = max(float(intrinsic[1, 1]), 1e-6)
    height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
    width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
    return {
        "rotation_c2w": rotation_c2w.astype(np.float32),
        "position": position.astype(np.float32),
        "fov": float(2.0 * np.arctan((height * 0.5) / fy)),
        "aspect": float(width / height),
    }


def prediction_scalar(predictions: dict[str, torch.Tensor], key: str, frame_index: int) -> float | None:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return float(value[0, frame_index].detach().float().reshape(-1)[0].cpu())


def validate_scene(scene: dict[str, Any], predictions: dict[str, torch.Tensor], images: torch.Tensor) -> None:
    required = ["pose_enc", "depth", "hsi_scene_scale", "hsi_scene_depth_bias"]
    missing = [key for key in required if key not in predictions]
    if missing:
        raise RuntimeError(f"Missing required prediction fields for HSI viewer: {missing}")
    if "nlf_image_hw" in predictions:
        nlf_hw = [int(v) for v in predictions["nlf_image_hw"].detach().cpu().reshape(-1).tolist()]
        if nlf_hw != [int(images.shape[-2]), int(images.shape[-1])]:
            raise RuntimeError(f"NLF image HW mismatch: nlf={nlf_hw} images={list(images.shape[-2:])}")
    if not scene["frames"]:
        raise RuntimeError("Viewer scene has no frames")
    if max(frame["hsi_points"].shape[0] for frame in scene["frames"]) <= 0:
        raise RuntimeError("HSI/world point cloud has no valid points")
    if not any("hsi_vertices" in person for frame in scene["frames"] for person in frame["people"]):
        raise RuntimeError("No finite HSI SMPL meshes were decoded")


def build_summary(
    args: argparse.Namespace,
    frame_paths: list[Path],
    checkpoint: Path,
    images: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    scene: dict[str, Any],
    output_dir: Path,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "frames_dir": str(resolve_project_path(args.frames_dir)),
        "num_frames": len(frame_paths),
        "checkpoint": str(checkpoint),
        "query_source": str(args.query_source),
        "tracking_overlay": str(args.tracking_overlay),
        "tracking": summarize_tracking(predictions),
        "tracking_geometry_unchanged": bool(
            predictions.get("viewer_tracking_geometry_unchanged", images.new_zeros(())).detach().cpu() > 0.5
        ),
        "image_shape": list(images.shape),
        "nlf_image_hw": [int(v) for v in predictions.get("nlf_image_hw", torch.tensor([], device=images.device)).detach().cpu().reshape(-1).tolist()],
        "point_counts_hsi": [int(frame["hsi_points"].shape[0]) for frame in scene["frames"]],
        "point_counts_hsi_full": [int(frame["hsi_points_full"].shape[0]) for frame in scene["frames"]],
        "human_points_removed_hsi": [
            int(frame["hsi_points_full"].shape[0] - frame["hsi_points"].shape[0]) for frame in scene["frames"]
        ],
        "human_mask_pixels_hsi": [int(np.count_nonzero(frame["hsi_human_exclusion_mask"])) for frame in scene["frames"]],
        "human_point_removal": {
            "method": "projected_smpl_triangle_silhouette_dilated_unconditional",
            "dilation_px": int(args.human_mask_dilation_px),
            "calibration_mode_uses_full_points": True,
        },
        "people_counts": [int(len(frame["people"])) for frame in scene["frames"]],
        "hsi_scene_scale": [frame["hsi_scene_scale"] for frame in scene["frames"]],
        "hsi_scene_depth_bias": [frame["hsi_scene_depth_bias"] for frame in scene["frames"]],
        "hsi_frame_scene_scale": [frame["hsi_frame_scene_scale"] for frame in scene["frames"]],
        "hsi_frame_scene_depth_bias": [frame["hsi_frame_scene_depth_bias"] for frame in scene["frames"]],
        "hsi_scene_affine_mode": str(scene.get("hsi_scene_affine_mode", "per_frame")),
        "hsi_scene_affine_ema_alpha": float(scene.get("hsi_scene_affine_ema_alpha", 0.25)),
        "hsi_visual_scale_initial": float(args.hsi_visual_scale),
        "camera_motion": {
            "raw_vggt": summarize_camera_motion(scene, "camera_trajectory_raw"),
            "hsi_scaled": summarize_camera_motion(scene, "camera_trajectory_hsi"),
        },
        "depth_alignment_note": "Depth alignment is computed by projecting SMPL vertices with VGGT K and sampling VGGT raw/HSI depth in the processed image plane; median_signed_m = z_smpl - z_depth.",
        "depth_alignment_overall": summarize_depth_alignment(scene),
        "depth_alignment_by_frame": [frame["depth_alignment"] for frame in scene["frames"]],
        "timings": timings or {},
        "output_dir": str(output_dir),
    }


def print_human_point_removal_summary(scene: dict[str, Any], args: argparse.Namespace) -> None:
    full_points = sum(int(frame["hsi_points_full"].shape[0]) for frame in scene["frames"])
    kept_points = sum(int(frame["hsi_points"].shape[0]) for frame in scene["frames"])
    removed_points = max(0, full_points - kept_points)
    mask_pixels = sum(int(np.count_nonzero(frame["hsi_human_exclusion_mask"])) for frame in scene["frames"])
    removed_ratio = float(removed_points) / float(max(full_points, 1))
    print(
        f"[human-mask] frames={len(scene['frames'])} mask_pixels={mask_pixels} "
        f"points_removed={removed_points}/{full_points} ({removed_ratio:.2%}) "
        f"dilation={int(args.human_mask_dilation_px)}px",
        flush=True,
    )
    if mask_pixels <= 0 or removed_points <= 0:
        print("[human-mask][warning] no HSI human points were removed; check SMPL detections and projection geometry", flush=True)


def summarize_tracking(predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
    ids = predictions.get("assigned_track_ids")
    mask = predictions.get("assigned_track_mask")
    quality = predictions.get("assigned_track_quality")
    if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return {"active": False, "unique_track_ids": [], "track_ids_by_frame": []}
    ids_cpu = ids.detach().cpu().long()
    mask_cpu = mask.detach().cpu().bool()
    quality_mean = None
    if isinstance(quality, torch.Tensor) and bool(mask.any()):
        quality_mean = float(quality.detach().float()[mask].mean().cpu())
    return {
        "active": bool(mask_cpu.any()),
        "unique_track_ids": sorted({int(value) for value in ids_cpu[mask_cpu].tolist()}),
        "track_ids_by_frame": [ids_cpu[0, frame][mask_cpu[0, frame]].tolist() for frame in range(ids_cpu.shape[1])],
        "mean_track_quality": quality_mean,
    }


def summarize_camera_motion(scene: dict[str, Any], key: str = "camera_trajectory") -> dict[str, Any]:
    trajectory = np.asarray(scene.get(key, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if trajectory.shape[0] <= 0:
        return {
            "num_cameras": 0,
            "positions_world": [],
            "step_distances": [],
            "total_path_m_vggt_units": 0.0,
            "start_end_m_vggt_units": 0.0,
            "axis_range_xyz_vggt_units": [0.0, 0.0, 0.0],
        }
    if trajectory.shape[0] > 1:
        step_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    else:
        step_distances = np.zeros((0,), dtype=np.float32)
    return {
        "num_cameras": int(trajectory.shape[0]),
        "positions_world": trajectory.tolist(),
        "step_distances": step_distances.astype(np.float32).tolist(),
        "total_path_m_vggt_units": float(step_distances.sum()),
        "start_end_m_vggt_units": float(np.linalg.norm(trajectory[-1] - trajectory[0])) if trajectory.shape[0] > 1 else 0.0,
        "axis_range_xyz_vggt_units": (trajectory.max(axis=0) - trajectory.min(axis=0)).astype(np.float32).tolist(),
        "mean_step_m_vggt_units": float(step_distances.mean()) if step_distances.size else 0.0,
        "max_step_m_vggt_units": float(step_distances.max()) if step_distances.size else 0.0,
    }


def summarize_depth_alignment(scene: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("base_raw", "base_hsi", "hsi_hsi"):
        med_abs: list[float] = []
        med_signed: list[float] = []
        valid_points = 0
        people = 0
        for frame in scene["frames"]:
            for entry in frame.get("depth_alignment", {}).get(key, []):
                if entry.get("median_abs_m") is None:
                    continue
                people += 1
                valid_points += int(entry.get("valid_points", 0) or 0)
                med_abs.append(float(entry["median_abs_m"]))
                med_signed.append(float(entry["median_signed_m"]))
        summary[key] = {
            "people": int(people),
            "valid_points": int(valid_points),
            "median_abs_m_mean": float(np.mean(med_abs)) if med_abs else None,
            "median_abs_m_median": float(np.median(med_abs)) if med_abs else None,
            "median_signed_m_mean": float(np.mean(med_signed)) if med_signed else None,
            "median_signed_m_median": float(np.median(med_signed)) if med_signed else None,
        }
    return summary


class SequenceViewer:
    def __init__(self, server: Any, transforms: Any, scene: dict[str, Any], args: argparse.Namespace) -> None:
        self.server = server
        self.transforms = transforms
        self.scene = scene
        self.args = args
        self.handles: list[dict[str, Any]] = []
        self.current_step = 0
        self.clients: dict[int, Any] = {}
        self.point_size_value = float(args.point_size)
        self.camera_scale_value = float(args.camera_frustum_scale)
        self.hsi_visual_scale_value = min(
            HSI_VISUAL_SCALE_MAX,
            max(HSI_VISUAL_SCALE_MIN, float(args.hsi_visual_scale)),
        )
        self.smpl_opacity_value = 1.0
        self.depth_point_stride_value = max(1, int(args.depth_point_stride))
        self.max_scene_depth_value = float(args.max_scene_depth)
        self.filter_human_points_value = bool(args.filter_human_points)
        self.human_mask_dilation_px_value = int(args.human_mask_dilation_px)
        self.env_mesh_depth_edge_rtol = float(getattr(args, "env_mesh_depth_edge_rtol", 0.08))
        self.env_mesh_color_groups = max(1, int(getattr(args, "env_mesh_color_groups", 216)))
        self.env_mesh_color_mode = str(getattr(args, "env_mesh_color_mode", "point_overlay"))
        self.env_mesh_overlay_point_size = max(0.0005, float(args.point_size) * float(getattr(args, "env_mesh_overlay_point_size_scale", 0.75)))
        self._rebuilding_points = False
        self.hsi_calibration_active = False
        self._switching_hsi_calibration = False
        self._hsi_calibration_restore_state: dict[str, Any] = {}
        self.global_handles: dict[str, list[Any]] = {}
        self.measurement_points: list[np.ndarray] = []
        self.measurement_sources: list[str] = []
        self.measurement_handles: list[Any] = []
        self.measurement_pointer_active = False
        self.measurement_pointer_api_mode = "none"
        self.measurement_click_callback = None
        self.measurement_label_handle = None
        self.measurement_label_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.human_entries: list[dict[str, Any]] = []
        self.human_entry_by_key: dict[str, dict[str, Any]] = {}
        self.selected_human_key = "none"
        self._syncing_smpl_controls = False
        self.smpl_edit_output = resolve_smpl_edit_output(args)
        self.transform_controls = None
        self._build_scene()
        self._build_gui()
        self._register_clients()
        self._update_visibility()

    def run(self) -> None:
        try:
            while True:
                if bool(self.play.value):
                    self.current_step = (int(self.timestep.value) + 1) % len(self.scene["frames"])
                    self.timestep.value = self.current_step
                    self._update_visibility()
                    if bool(self.follow_camera.value):
                        self._follow_pred_camera(self.current_step)
                time.sleep(1.0 / max(float(self.fps.value), 1.0))
        except KeyboardInterrupt:
            print("[viewer] stopped", flush=True)

    def _build_scene(self) -> None:
        tracking_only = bool(getattr(self.args, "tracking_only", False))
        for frame in self.scene["frames"]:
            idx = int(frame["frame_index"])
            frame_handles: dict[str, Any] = {
                "raw": [],
                "hsi": [],
                "raw_mesh": [],
                "hsi_mesh": [],
                "base_humans": [],
                "hsi_humans": [],
                "track_labels": [],
                "cameras_raw": [],
                "cameras_hsi": [],
            }
            frame_handles["raw"].append(
                self._add_environment_point_cloud(
                    f"/frames/{idx:04d}/points_raw_depth",
                    self._raw_display_points(frame),
                    self._raw_display_colors(frame),
                    frame_index=idx,
                    source="raw",
                )
            )
            if not tracking_only:
                frame_handles["hsi"].append(
                    self._add_environment_point_cloud(
                        f"/frames/{idx:04d}/points_hsi_depth",
                        self._scaled_hsi_points(frame),
                        self._hsi_point_colors(frame),
                        frame_index=idx,
                        source="hsi",
                    )
                )
            for person in frame["people"]:
                color = tuple(int(v) for v in person["color"])
                track_id = int(person["track_id"])
                query_idx = int(person["query_index"])
                label_handle = None
                if "base_vertices" in person:
                    handle = add_mesh(self.server, f"/frames/{idx:04d}/human_base_t{track_id}_q{query_idx}", person["base_vertices"], person["faces"], color, self.smpl_opacity_value)
                    frame_handles["base_humans"].append(handle)
                    self._register_human_entry(
                        frame_index=idx,
                        frame_id=str(frame["frame_id"]),
                        kind="base",
                        track_id=track_id,
                        query_idx=query_idx,
                        handle=handle,
                        vertices=person["base_vertices"],
                        label_handle=None,
                        label_base_position=None,
                        color=color,
                    )
                if "hsi_vertices" in person:
                    handle = add_mesh(self.server, f"/frames/{idx:04d}/human_hsi_t{track_id}_q{query_idx}", person["hsi_vertices"], person["faces"], color, self.smpl_opacity_value)
                    frame_handles["hsi_humans"].append(handle)
                    self._register_human_entry(
                        frame_index=idx,
                        frame_id=str(frame["frame_id"]),
                        kind="hsi",
                        track_id=track_id,
                        query_idx=query_idx,
                        handle=handle,
                        vertices=person["hsi_vertices"],
                        label_handle=None,
                        label_base_position=None,
                        color=color,
                    )
                label_vertices = person.get("hsi_vertices", person.get("base_vertices"))
                if label_vertices is not None:
                    vertices = np.asarray(label_vertices, dtype=np.float32)
                    label_position = vertices[int(np.argmin(vertices[:, 1]))].copy()
                    label_position[1] -= 0.12
                    quality = person.get("track_quality")
                    label_text = f"ID {track_id}" if quality is None else f"ID {track_id}  {float(quality):.2f}"
                    label_handle = add_label(self.server, f"/frames/{idx:04d}/track_label_t{track_id}_q{query_idx}", label_text, label_position)
                    frame_handles["track_labels"].append(label_handle)
                    for kind in ("base", "hsi"):
                        entry = self.human_entry_by_key.get(human_entry_key(idx, kind, track_id, query_idx))
                        if entry is not None:
                            entry["label_handle"] = label_handle
                            entry["label_base_position"] = label_position.astype(np.float32, copy=False)
            frame_handles["cameras_raw"].append(add_camera(self.server, self.transforms, f"/frames/{idx:04d}/camera_raw_vggt", frame["raw_camera"], self.camera_scale_value, (255, 255, 255)))
            if not tracking_only:
                frame_handles["cameras_hsi"].append(add_camera(self.server, self.transforms, f"/frames/{idx:04d}/camera_hsi_scaled", self._scaled_hsi_camera(frame), self.camera_scale_value, (255, 176, 0)))
            self.handles.append(frame_handles)
        raw_trajectory = np.asarray(self.scene.get("camera_trajectory_raw", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        hsi_trajectory = self._scaled_hsi_trajectory()
        if raw_trajectory.shape[0] > 0:
            self.global_handles["camera_trajectory_raw"] = [
                add_point_cloud(self.server, "/camera_trajectory/raw_vggt_centers", raw_trajectory, camera_trajectory_colors(raw_trajectory.shape[0]), max(self.point_size_value * 2.5, 0.01))
            ]
        if not tracking_only and hsi_trajectory.shape[0] > 0:
            self.global_handles["camera_trajectory_hsi"] = [
                add_point_cloud(self.server, "/camera_trajectory/hsi_scaled_centers", hsi_trajectory, camera_trajectory_colors(hsi_trajectory.shape[0]), max(self.point_size_value * 3.5, 0.012))
            ]

    def _scaled_hsi_points(
        self,
        frame: dict[str, Any],
        scale: float | None = None,
        include_humans: bool = False,
    ) -> np.ndarray:
        use_full = include_humans or not self.filter_human_points_value
        key = "hsi_points_full" if use_full else "hsi_points"
        points = np.asarray(frame.get(key, frame["hsi_points"]), dtype=np.float32)
        visual_scale = self.hsi_visual_scale_value if scale is None else float(scale)
        return points * np.float32(visual_scale)

    def _hsi_point_colors(self, frame: dict[str, Any], include_humans: bool = False) -> np.ndarray:
        use_full = include_humans or not self.filter_human_points_value
        key = "hsi_colors_full" if use_full else "hsi_colors"
        return np.asarray(frame.get(key, frame["hsi_colors"]), dtype=np.uint8)

    def _raw_display_points(self, frame: dict[str, Any]) -> np.ndarray:
        key = "raw_points" if self.filter_human_points_value else "raw_points_full"
        return np.asarray(frame.get(key, frame["raw_points"]), dtype=np.float32)

    def _raw_display_colors(self, frame: dict[str, Any]) -> np.ndarray:
        key = "raw_colors" if self.filter_human_points_value else "raw_colors_full"
        return np.asarray(frame.get(key, frame["raw_colors"]), dtype=np.uint8)

    def _scaled_hsi_camera(self, frame: dict[str, Any], scale: float | None = None) -> dict[str, Any]:
        camera = dict(frame["hsi_camera"])
        visual_scale = self.hsi_visual_scale_value if scale is None else float(scale)
        camera["position"] = np.asarray(camera["position"], dtype=np.float32) * np.float32(visual_scale)
        return camera

    def _scaled_hsi_trajectory(self) -> np.ndarray:
        trajectory = np.asarray(
            self.scene.get("camera_trajectory_hsi", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        return trajectory * np.float32(self.hsi_visual_scale_value)

    def _build_gui(self) -> None:
        self.frame_info = add_text(self.server, "Frame Info", "")
        self.alignment_info = add_text(self.server, "Depth Align", "")
        self.camera_motion_info = add_text(self.server, "Camera Motion", format_camera_motion_short(self.scene))
        self.timestep = add_slider(self.server, "Timestep", 0, len(self.scene["frames"]) - 1, 1, 0)
        self.prev_button = add_button(self.server, "Prev Frame")
        self.next_button = add_button(self.server, "Next Frame")
        self.play = add_checkbox(self.server, "Playing", False)
        self.fps = add_slider(self.server, "FPS", 1, 30, 1, 6)
        self.fps_buttons = add_button_group(self.server, "FPS Preset", ("5", "10", "20", "30"))
        self.mode = add_dropdown(self.server, "Mode", ["4D current frame", "3D accumulate", "Hybrid"], str(getattr(self.args, "viewer_mode", "4D current frame")))
        tracking_only = bool(getattr(self.args, "tracking_only", False))
        self.depth_source = add_dropdown(self.server, "Depth Source", ["hsi_depth", "raw_depth", "both"], "raw_depth" if tracking_only else "hsi_depth")
        self.environment_display = add_dropdown(
            self.server,
            "Environment Display",
            ["points", "mesh", "both"],
            str(getattr(self.args, "environment_display", "points")),
        )
        with add_folder(self.server, "Human Point Filter Controls"):
            self.filter_human_points = add_checkbox(self.server, "Filter Human Points", self.filter_human_points_value)
            self.human_filter_dilation = add_slider(
                self.server,
                "Human Filter Dilation (px)",
                HUMAN_MASK_DILATION_MIN_PX,
                HUMAN_MASK_DILATION_MAX_PX,
                1,
                self.human_mask_dilation_px_value,
            )
            self.apply_human_filter_dilation = add_button(self.server, "Apply Human Filter Size")
            self.reset_human_filter_dilation = add_button(
                self.server,
                f"Reset Human Filter Size to {HUMAN_MASK_DILATION_DEFAULT_PX} px",
            )
            self.human_point_filter_info = add_text(self.server, "Human Point Filter", "")
            set_handle_disabled(self.human_point_filter_info, True)
        with add_folder(self.server, "Point Cloud Measurement"):
            self.measurement_enabled = add_checkbox(self.server, "Enable Point Measurement", False)
            self.measurement_pick_radius = add_slider(
                self.server,
                "Measurement Pick Radius",
                0.005,
                0.5,
                0.005,
                max(0.05, min(0.5, self.point_size_value * 8.0)),
            )
            self.measurement_line_width = add_slider(self.server, "Measurement Line Width", 1.0, 10.0, 0.5, 3.0)
            self.measurement_font_size = add_slider(self.server, "Distance Font Size", 16, 64, 2, 32)
            self.measurement_color = add_rgb(self.server, "Measurement Color", (255, 64, 96))
            self.measurement_info = add_text(self.server, "Measurement Result", "Off")
            set_handle_disabled(self.measurement_info, True)
            self.clear_measurement = add_button(self.server, "Clear Current Measurement")
        with add_folder(self.server, "HSI Scale Controls"):
            self.hsi_calibration_mode = add_checkbox(self.server, "Single-Frame Scale Calibration", False)
            self.hsi_calibration_info = add_text(self.server, "Calibration Status", "Off")
            set_handle_disabled(self.hsi_calibration_info, True)
            self.hsi_scale_strategy_info = add_text(self.server, "Strategy", "")
            self.hsi_model_scale_info = add_text(self.server, "Current Model Scale / Bias", "")
            self.hsi_raw_scale_info = add_text(self.server, "Raw Frame Scale / Bias", "")
            self.hsi_scale_range_info = add_text(self.server, "Sequence Scale Min / Median / Max", "")
            self.hsi_visual_result_info = add_text(self.server, "Applied / Effective", "")
            for handle in (
                self.hsi_scale_strategy_info,
                self.hsi_model_scale_info,
                self.hsi_raw_scale_info,
                self.hsi_scale_range_info,
                self.hsi_visual_result_info,
            ):
                set_handle_disabled(handle, True)
            self.hsi_visual_scale = add_slider(
                self.server,
                "Visual Scale Multiplier (log10)",
                HSI_VISUAL_SCALE_SLIDER_MIN,
                HSI_VISUAL_SCALE_SLIDER_MAX,
                0.01,
                hsi_visual_scale_to_slider(self.hsi_visual_scale_value),
            )
            self.apply_hsi_visual_scale = add_button(self.server, "Apply Scale")
            self.reset_hsi_visual_scale = add_button(self.server, "Reset Scale to 1.0")
            if tracking_only:
                set_handle_disabled(self.hsi_calibration_mode, True)
        self.point_size = add_slider(self.server, "Point Size", 0.0005, 0.08, 0.0005, self.point_size_value)
        self.density_preset = add_dropdown(self.server, "Point Density Preset", ["custom", "dense stride 1", "balanced stride 2", "fast stride 4", "full sequence stride 6"], "custom")
        self.depth_point_stride = add_slider(self.server, "Depth Point Stride", 1, 64, 1, self.depth_point_stride_value)
        self.max_scene_depth = add_slider(self.server, "Max Scene Depth", 0.0, 200.0, 1.0, max(0.0, self.max_scene_depth_value))
        self.camera_size = add_slider(self.server, "Camera Size", 0.01, 1.00, 0.01, self.camera_scale_value)
        self.show_hsi = add_checkbox(self.server, "Show HSI SMPL", not tracking_only)
        self.show_base = add_checkbox(self.server, "Show Base SMPL", tracking_only)
        self.show_track_ids = add_checkbox(self.server, "Show Track IDs", bool(getattr(self.args, "show_track_ids", True)))
        self.smpl_opacity = add_slider(self.server, "SMPL Opacity", 0.05, 1.00, 0.05, self.smpl_opacity_value)
        self.smpl_color = add_rgb(self.server, "SMPL Color", (204, 51, 51))
        self.smpl_downsample = add_slider(self.server, "SMPL Downsample", 1, max(1, len(self.scene["frames"])), 1, 1)
        self.show_cameras = add_checkbox(self.server, "Show Cameras", True)
        self.camera_source = add_dropdown(self.server, "Camera Source", ["auto", "hsi_scaled", "raw_vggt", "both"], "auto")
        self.show_camera_trajectory = add_checkbox(self.server, "Show Camera Trajectory", True)
        self.camera_downsample = add_slider(self.server, "Camera Downsample", 1, max(1, len(self.scene["frames"])), 1, 1)
        self.follow_camera = add_checkbox(self.server, "Follow Pred Camera", False)
        self.smpl_edit_info = add_text(self.server, "SMPL Edit", "Select or click an SMPL mesh to edit viewer-only translation.")
        self.selected_smpl = add_dropdown(self.server, "Selected SMPL", self._human_dropdown_options(), "none")
        self.smpl_edit_scope = add_dropdown(self.server, "SMPL Edit Scope", ["selected frame", "same track all frames"], "selected frame")
        self.smpl_edit_dx = add_slider(self.server, "SMPL dX", -5.0, 5.0, 0.01, 0.0)
        self.smpl_edit_dy = add_slider(self.server, "SMPL dY", -5.0, 5.0, 0.01, 0.0)
        self.smpl_edit_dz = add_slider(self.server, "SMPL dZ", -5.0, 5.0, 0.01, 0.0)
        self.reset_smpl_edit = add_button(self.server, "Reset SMPL Offset")
        self.save_smpl_edits = add_button(self.server, "Save SMPL Offsets")
        self.transform_controls = add_transform_controls(
            self.server,
            "/viewer_controls/selected_smpl_translation",
            position=np.zeros(3, dtype=np.float32),
            scale=0.35,
            visible=False,
        )
        for handle in [
            self.timestep,
            self.mode,
            self.depth_source,
            self.environment_display,
            self.show_hsi,
            self.show_base,
            self.show_track_ids,
            self.smpl_downsample,
            self.show_cameras,
            self.camera_source,
            self.show_camera_trajectory,
            self.camera_downsample,
            self.follow_camera,
        ]:
            bind_update(handle, self._on_gui_update)
        bind_update(self.point_size, self._on_point_size_update)
        bind_update(self.density_preset, self._on_density_preset_update)
        bind_update(self.depth_point_stride, self._on_depth_sampling_update)
        bind_update(self.max_scene_depth, self._on_depth_sampling_update)
        bind_update(self.hsi_visual_scale, self._on_hsi_visual_scale_pending)
        bind_update(self.hsi_calibration_mode, self._on_hsi_calibration_mode_update)
        bind_update(self.filter_human_points, self._on_filter_human_points_update)
        bind_update(self.human_filter_dilation, self._on_human_filter_dilation_pending)
        bind_update(self.measurement_enabled, self._on_measurement_enabled_update)
        bind_update(self.measurement_line_width, self._on_measurement_style_update)
        bind_update(self.measurement_font_size, self._on_measurement_style_update)
        bind_update(self.measurement_color, self._on_measurement_style_update)
        bind_update(self.camera_size, self._on_camera_size_update)
        bind_update(self.smpl_opacity, self._on_smpl_opacity_update)
        bind_update(self.smpl_color, self._on_smpl_color_update)
        bind_update(self.selected_smpl, self._on_selected_smpl_update)
        bind_update(self.smpl_edit_scope, self._on_smpl_offset_slider_update)
        bind_update(self.smpl_edit_dx, self._on_smpl_offset_slider_update)
        bind_update(self.smpl_edit_dy, self._on_smpl_offset_slider_update)
        bind_update(self.smpl_edit_dz, self._on_smpl_offset_slider_update)
        bind_update(self.transform_controls, self._on_transform_controls_update)
        bind_click(self.prev_button, self._prev_frame)
        bind_click(self.next_button, self._next_frame)
        bind_click(self.fps_buttons, self._set_fps_preset)
        bind_click(self.reset_smpl_edit, self._reset_selected_smpl_offset)
        bind_click(self.save_smpl_edits, self._save_smpl_edit_offsets)
        bind_click(self.apply_hsi_visual_scale, self._apply_hsi_visual_scale)
        bind_click(self.reset_hsi_visual_scale, self._reset_hsi_visual_scale)
        bind_click(self.apply_human_filter_dilation, self._apply_human_filter_dilation)
        bind_click(self.reset_human_filter_dilation, self._reset_human_filter_dilation)
        bind_click(self.clear_measurement, self._clear_current_measurement)

    def _register_clients(self) -> None:
        if hasattr(self.server, "on_client_connect"):
            @self.server.on_client_connect
            def _on_connect(client: Any) -> None:
                self.clients[int(getattr(client, "client_id", len(self.clients)))] = client
                self._register_measurement_camera_updates(client)

    def _register_measurement_camera_updates(self, client: Any) -> None:
        camera = getattr(client, "camera", None)
        if camera is None or not hasattr(camera, "on_update"):
            return

        @camera.on_update
        def _(_: Any) -> None:
            if self.measurement_label_handle is None:
                return
            try:
                self.measurement_label_wxyz = np.asarray(camera.wxyz, dtype=np.float32).reshape(4)
                set_handle_wxyz(self.measurement_label_handle, self.measurement_label_wxyz)
            except Exception:
                pass

    def _add_environment_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        frame_index: int,
        source: str,
    ) -> Any:
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return add_point_cloud(self.server, name, points_np, colors, self.point_size_value)

    def _on_measurement_enabled_update(self, _: Any = None) -> None:
        if bool(self.measurement_enabled.value):
            self.play.value = False
            if not self._set_measurement_pointer_active(True):
                set_text_value(
                    self.measurement_info,
                    "Unavailable | this Viser version has no scene click API",
                )
                return
            if len(self.measurement_points) == 0:
                set_text_value(self.measurement_info, "Ready | click the first environment point")
            elif len(self.measurement_points) == 1:
                set_text_value(self.measurement_info, "P1 selected | click the second environment point")
            else:
                self._update_measurement_result_text()
        else:
            self._set_measurement_pointer_active(False)
            if len(self.measurement_points) == 0:
                set_text_value(self.measurement_info, "Off | enable measurement to select points")
            else:
                set_text_value(self.measurement_info, "Paused | current measurement remains visible")

    def _set_measurement_pointer_active(self, active: bool) -> bool:
        api = scene_api(self.server)
        if active:
            if self.measurement_pointer_active:
                return True
            callback = self._on_scene_measurement_click
            if hasattr(api, "on_click"):
                self.measurement_click_callback = api.on_click()(callback)
                self.measurement_pointer_api_mode = "modern"
            elif hasattr(api, "on_pointer_event"):
                self.measurement_click_callback = api.on_pointer_event("click")(callback)
                self.measurement_pointer_api_mode = "legacy"
            else:
                return False
            self.measurement_pointer_active = True
            return True
        if self.measurement_pointer_active:
            try:
                if self.measurement_pointer_api_mode == "modern" and hasattr(api, "remove_click_callback"):
                    api.remove_click_callback(self.measurement_click_callback)
                elif hasattr(api, "remove_pointer_callback"):
                    api.remove_pointer_callback()
            except Exception:
                pass
        self.measurement_pointer_active = False
        self.measurement_pointer_api_mode = "none"
        self.measurement_click_callback = None
        return True

    def _visible_measurement_clouds(self) -> list[tuple[int, str, np.ndarray]]:
        if str(self.environment_display.value) not in {"points", "both"}:
            return []
        current = int(self.timestep.value)
        mode = str(self.mode.value)
        depth_source = str(self.depth_source.value)
        tracking_only = bool(getattr(self.args, "tracking_only", False))
        clouds: list[tuple[int, str, np.ndarray]] = []
        for idx, frame in enumerate(self.scene["frames"]):
            visible = idx == current if mode == "4D current frame" else idx <= current
            if not visible:
                continue
            if depth_source in {"raw_depth", "both"}:
                clouds.append((idx, "raw", self._raw_display_points(frame)))
            if not tracking_only and depth_source in {"hsi_depth", "both"}:
                if self.hsi_calibration_active and idx == current:
                    points = self._scaled_hsi_points(frame, include_humans=True)
                    source = "hsi calibration"
                else:
                    points = self._scaled_hsi_points(frame)
                    source = "hsi"
                clouds.append((idx, source, points))
        return clouds

    def _on_scene_measurement_click(self, event: Any) -> None:
        if not bool(getattr(self.measurement_enabled, "value", False)):
            return
        clouds = self._visible_measurement_clouds()
        if not clouds:
            set_text_value(self.measurement_info, "No visible point cloud | set Environment Display to points or both")
            return
        best: tuple[float, int, np.ndarray, int, str] | None = None
        for frame_index, source, points in clouds:
            selection = nearest_clicked_point_with_distance(event, points)
            if selection is None:
                continue
            point_index, point, ray_distance = selection
            if best is None or ray_distance < best[0]:
                best = (ray_distance, point_index, point, frame_index, source)
        if best is None:
            set_text_value(self.measurement_info, "Selection failed | click in front of the camera")
            return
        ray_distance, point_index, point, frame_index, source = best
        pick_radius = max(0.0, float(self.measurement_pick_radius.value))
        if ray_distance > pick_radius:
            set_text_value(
                self.measurement_info,
                (
                    f"No point within pick radius | nearest={ray_distance:.4f}, "
                    f"radius={pick_radius:.4f}; increase Measurement Pick Radius"
                ),
            )
            return
        client = getattr(event, "client", None)
        if client is not None:
            try:
                self.measurement_label_wxyz = np.asarray(client.camera.wxyz, dtype=np.float32).reshape(4)
            except Exception:
                pass
        self._record_measurement_point(point_index, point, frame_index, source)

    def _record_measurement_point(
        self,
        point_index: int,
        point: np.ndarray,
        frame_index: int,
        source: str,
    ) -> None:
        if len(self.measurement_points) >= 2:
            self._clear_current_measurement(status=None)
        point = np.asarray(point, dtype=np.float32).reshape(3)
        self.measurement_points.append(point)
        self.measurement_sources.append(f"{source} frame {int(frame_index) + 1} point {point_index}")
        self._render_measurement_geometry()
        if len(self.measurement_points) == 1:
            set_text_value(
                self.measurement_info,
                f"P1 {self.measurement_sources[0]} | click the second environment point",
            )
            return
        self._update_measurement_result_text()

    def _on_measurement_style_update(self, _: Any = None) -> None:
        if self.measurement_points:
            self._render_measurement_geometry()

    def _render_measurement_geometry(self) -> None:
        for handle in self.measurement_handles:
            remove_handle(handle)
        self.measurement_handles = []
        self.measurement_label_handle = None
        for index, point in enumerate(self.measurement_points):
            marker_color = (0, 220, 255) if index == 0 else (255, 210, 0)
            self.measurement_handles.append(
                add_point_cloud(
                    self.server,
                    f"/measurements/current/p{index + 1}_marker",
                    np.asarray(point, dtype=np.float32).reshape(1, 3),
                    np.asarray([marker_color], dtype=np.uint8),
                    max(self.point_size_value * 4.0, 0.025),
                )
            )
        if len(self.measurement_points) != 2:
            return
        first, second = self.measurement_points
        distance = float(np.linalg.norm(second - first))
        midpoint = (first + second) * np.float32(0.5)
        color = tuple(int(value) for value in self.measurement_color.value)
        line = add_line_segments(
            self.server,
            "/measurements/current/distance_line",
            np.stack([first, second], axis=0)[None, ...],
            color=color,
            line_width=float(self.measurement_line_width.value),
        )
        label_segments = build_measurement_text_segments(
            f"{distance:.2f}m",
            height=max(0.025, float(self.measurement_font_size.value) * 0.002),
        )
        self.measurement_label_handle = add_line_segments(
            self.server,
            "/measurements/current/distance_label",
            label_segments,
            color=color,
            line_width=max(1.5, float(self.measurement_font_size.value) * 0.09),
            position=midpoint,
            wxyz=self.measurement_label_wxyz,
        )
        self.measurement_handles.extend([line, self.measurement_label_handle])

    def _update_measurement_result_text(self) -> None:
        if len(self.measurement_points) != 2:
            return
        first, second = self.measurement_points
        distance = float(np.linalg.norm(second - first))
        set_text_value(
            self.measurement_info,
            (
                f"Distance={distance:.2f}m | "
                f"P1: {self.measurement_sources[0]} "
                f"({first[0]:.3f}, {first[1]:.3f}, {first[2]:.3f}) | "
                f"P2: {self.measurement_sources[1]} "
                f"({second[0]:.3f}, {second[1]:.3f}, {second[2]:.3f})"
            ),
        )

    def _clear_current_measurement(self, _: Any = None, status: str | None = "button") -> None:
        for handle in self.measurement_handles:
            remove_handle(handle)
        self.measurement_handles = []
        self.measurement_label_handle = None
        self.measurement_points = []
        self.measurement_sources = []
        if status is None:
            return
        if status != "button":
            set_text_value(self.measurement_info, status)
        elif bool(getattr(self.measurement_enabled, "value", False)):
            set_text_value(self.measurement_info, "Cleared | click the first environment point")
        else:
            set_text_value(self.measurement_info, "Cleared | enable measurement to select points")

    def _register_human_entry(
        self,
        frame_index: int,
        frame_id: str,
        kind: str,
        track_id: int,
        query_idx: int,
        handle: Any,
        vertices: np.ndarray,
        label_handle: Any,
        label_base_position: np.ndarray | None,
        color: tuple[int, int, int],
    ) -> None:
        key = human_entry_key(frame_index, kind, track_id, query_idx)
        vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(vertices_np).all(axis=1)
        anchor = vertices_np[finite].mean(axis=0) if bool(finite.any()) else np.zeros(3, dtype=np.float32)
        entry = {
            "key": key,
            "frame_index": int(frame_index),
            "frame_id": str(frame_id),
            "kind": str(kind),
            "track_id": int(track_id),
            "query_index": int(query_idx),
            "handle": handle,
            "label_handle": label_handle,
            "label_base_position": label_base_position,
            "anchor_position": np.asarray(anchor, dtype=np.float32),
            "offset": np.zeros(3, dtype=np.float32),
            "color": tuple(int(v) for v in color),
        }
        self.human_entries.append(entry)
        self.human_entry_by_key[key] = entry
        bind_click(handle, lambda *_, selected_key=key: self._select_human(selected_key, sync_timestep=True))

    def _human_dropdown_options(self) -> list[str]:
        return ["none"] + [str(entry["key"]) for entry in self.human_entries]

    def _select_human(self, key: str, sync_timestep: bool = False) -> None:
        if key not in self.human_entry_by_key:
            key = "none"
        self.selected_human_key = key
        if hasattr(self, "selected_smpl"):
            try:
                if str(self.selected_smpl.value) != key:
                    self.selected_smpl.value = key
            except Exception:
                pass
        if sync_timestep and key != "none":
            entry = self.human_entry_by_key[key]
            try:
                self.timestep.value = int(entry["frame_index"])
            except Exception:
                pass
        self._sync_smpl_edit_controls_from_selection()
        self._update_visibility()

    def _on_selected_smpl_update(self, _: Any = None) -> None:
        self._select_human(str(self.selected_smpl.value), sync_timestep=True)

    def _sync_smpl_edit_controls_from_selection(self) -> None:
        entry = self.human_entry_by_key.get(self.selected_human_key)
        self._syncing_smpl_controls = True
        try:
            if entry is None:
                offset = np.zeros(3, dtype=np.float32)
                set_text_value(self.smpl_edit_info, "No SMPL selected.")
                set_handle_visible(self.transform_controls, False)
            else:
                offset = np.asarray(entry["offset"], dtype=np.float32)
                set_text_value(
                    self.smpl_edit_info,
                    (
                        f"{entry['key']} | frame={entry['frame_id']} "
                        f"offset=({offset[0]:.3f},{offset[1]:.3f},{offset[2]:.3f})"
                    ),
                )
                if self.transform_controls is not None:
                    set_handle_visible(self.transform_controls, True)
                    set_handle_position(self.transform_controls, np.asarray(entry["anchor_position"], dtype=np.float32) + offset)
            self.smpl_edit_dx.value = float(offset[0])
            self.smpl_edit_dy.value = float(offset[1])
            self.smpl_edit_dz.value = float(offset[2])
        finally:
            self._syncing_smpl_controls = False

    def _on_smpl_offset_slider_update(self, _: Any = None) -> None:
        if self._syncing_smpl_controls:
            return
        entry = self.human_entry_by_key.get(self.selected_human_key)
        if entry is None:
            return
        offset = np.asarray(
            [float(self.smpl_edit_dx.value), float(self.smpl_edit_dy.value), float(self.smpl_edit_dz.value)],
            dtype=np.float32,
        )
        self._apply_smpl_offset(entry, offset, scope=str(self.smpl_edit_scope.value))
        self._sync_transform_controls_to_entry(entry)
        self._sync_smpl_edit_controls_from_selection()

    def _on_transform_controls_update(self, _: Any = None) -> None:
        if self._syncing_smpl_controls:
            return
        entry = self.human_entry_by_key.get(self.selected_human_key)
        if entry is None or self.transform_controls is None:
            return
        position = get_handle_position(self.transform_controls)
        if position is None:
            return
        offset = np.asarray(position, dtype=np.float32) - np.asarray(entry["anchor_position"], dtype=np.float32)
        self._syncing_smpl_controls = True
        try:
            self.smpl_edit_dx.value = float(offset[0])
            self.smpl_edit_dy.value = float(offset[1])
            self.smpl_edit_dz.value = float(offset[2])
        finally:
            self._syncing_smpl_controls = False
        self._apply_smpl_offset(entry, offset, scope=str(self.smpl_edit_scope.value))
        self._sync_smpl_edit_controls_from_selection()

    def _apply_smpl_offset(self, selected_entry: dict[str, Any], offset: np.ndarray, scope: str) -> None:
        offset_np = np.asarray(offset, dtype=np.float32).reshape(3)
        entries = [selected_entry]
        if scope == "same track all frames":
            entries = [
                entry
                for entry in self.human_entries
                if int(entry["track_id"]) == int(selected_entry["track_id"]) and str(entry["kind"]) == str(selected_entry["kind"])
            ]
        for entry in entries:
            entry["offset"] = offset_np.copy()
            set_handle_position(entry["handle"], offset_np)
            label_handle = entry.get("label_handle")
            label_base = entry.get("label_base_position")
            if label_handle is not None and label_base is not None:
                set_handle_position(label_handle, np.asarray(label_base, dtype=np.float32) + offset_np)

    def _sync_transform_controls_to_entry(self, entry: dict[str, Any]) -> None:
        if self.transform_controls is None:
            return
        position = np.asarray(entry["anchor_position"], dtype=np.float32) + np.asarray(entry["offset"], dtype=np.float32)
        set_handle_position(self.transform_controls, position)

    def _reset_selected_smpl_offset(self, _: Any = None) -> None:
        entry = self.human_entry_by_key.get(self.selected_human_key)
        if entry is None:
            return
        self._apply_smpl_offset(entry, np.zeros(3, dtype=np.float32), scope=str(self.smpl_edit_scope.value))
        self._sync_smpl_edit_controls_from_selection()

    def _save_smpl_edit_offsets(self, _: Any = None) -> None:
        rows = []
        for entry in self.human_entries:
            offset = np.asarray(entry["offset"], dtype=np.float32)
            if not bool(np.any(np.abs(offset) > 1e-7)):
                continue
            rows.append(
                {
                    "key": str(entry["key"]),
                    "frame_index": int(entry["frame_index"]),
                    "frame_id": str(entry["frame_id"]),
                    "kind": str(entry["kind"]),
                    "track_id": int(entry["track_id"]),
                    "query_index": int(entry["query_index"]),
                    "offset_world_xyz": [float(v) for v in offset.tolist()],
                }
            )
        payload = {
            "note": "Viewer-only SMPL translation offsets. Model predictions and saved run_summary geometry are unchanged.",
            "offsets": rows,
        }
        self.smpl_edit_output.parent.mkdir(parents=True, exist_ok=True)
        self.smpl_edit_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        set_text_value(self.smpl_edit_info, f"Saved {len(rows)} SMPL offset(s) to {self.smpl_edit_output}")

    def _on_gui_update(self, _: Any = None) -> None:
        if self._switching_hsi_calibration:
            return
        self.current_step = int(self.timestep.value)
        if self.hsi_calibration_active:
            self._rebuild_hsi_calibration_frame(self.current_step)
        self._update_visibility()
        if bool(self.follow_camera.value):
            self._follow_pred_camera(self.current_step)

    def _prev_frame(self, _: Any = None) -> None:
        self.timestep.value = (int(self.timestep.value) - 1) % len(self.scene["frames"])
        self._on_gui_update()

    def _next_frame(self, _: Any = None) -> None:
        self.timestep.value = (int(self.timestep.value) + 1) % len(self.scene["frames"])
        self._on_gui_update()

    def _set_fps_preset(self, _: Any = None) -> None:
        try:
            self.fps.value = int(str(self.fps_buttons.value))
        except Exception:
            pass

    def _on_point_size_update(self, _: Any = None) -> None:
        self.point_size_value = float(self.point_size.value)
        self._set_handle_attr(["raw", "hsi"], "point_size", self.point_size_value)

    def _on_density_preset_update(self, _: Any = None) -> None:
        preset_to_stride = {
            "dense stride 1": 1,
            "balanced stride 2": 2,
            "fast stride 4": 4,
            "full sequence stride 6": 6,
        }
        preset = str(self.density_preset.value)
        if preset in preset_to_stride:
            self.depth_point_stride.value = preset_to_stride[preset]
        self._on_depth_sampling_update()

    def _on_depth_sampling_update(self, _: Any = None) -> None:
        if self._rebuilding_points:
            return
        self.depth_point_stride_value = max(1, int(self.depth_point_stride.value))
        self.max_scene_depth_value = float(self.max_scene_depth.value)
        self._rebuild_depth_point_clouds()

    def _on_filter_human_points_update(self, _: Any = None) -> None:
        if self._switching_hsi_calibration or self.hsi_calibration_active:
            return
        requested = bool(self.filter_human_points.value)
        if requested == self.filter_human_points_value:
            self._update_info_text(int(self.timestep.value))
            return
        self.filter_human_points_value = requested
        self._rebuild_environment_point_handles()

    def _requested_human_filter_dilation(self) -> int:
        return min(
            HUMAN_MASK_DILATION_MAX_PX,
            max(HUMAN_MASK_DILATION_MIN_PX, int(round(float(self.human_filter_dilation.value)))),
        )

    def _on_human_filter_dilation_pending(self, _: Any = None) -> None:
        self._update_info_text(int(self.timestep.value))

    def _apply_human_filter_dilation(self, _: Any = None) -> None:
        if self._rebuilding_points or self.hsi_calibration_active:
            return
        requested = self._requested_human_filter_dilation()
        if requested == self.human_mask_dilation_px_value:
            self._update_info_text(int(self.timestep.value))
            return
        self.human_mask_dilation_px_value = requested
        self._rebuild_human_filter_masks_and_points()

    def _reset_human_filter_dilation(self, _: Any = None) -> None:
        if self.hsi_calibration_active:
            return
        self.human_filter_dilation.value = HUMAN_MASK_DILATION_DEFAULT_PX
        self._apply_human_filter_dilation()

    def _rebuild_human_filter_masks_and_points(self) -> None:
        self._rebuilding_points = True
        try:
            for frame in self.scene["frames"]:
                intrinsic = np.asarray(frame["intrinsic"], dtype=np.float32)
                raw_depth = torch.from_numpy(np.asarray(frame["raw_depth_map"], dtype=np.float32))
                hsi_depth = torch.from_numpy(np.asarray(frame["hsi_depth_map"], dtype=np.float32))
                frame["raw_human_exclusion_mask"] = projected_human_exclusion_mask(
                    raw_depth,
                    frame["people"],
                    intrinsic,
                    "base_vertices_cam",
                    self.args,
                    dilation_px_override=self.human_mask_dilation_px_value,
                )
                frame["hsi_human_exclusion_mask"] = projected_human_exclusion_mask(
                    hsi_depth,
                    frame["people"],
                    intrinsic,
                    "hsi_vertices_cam",
                    self.args,
                    dilation_px_override=self.human_mask_dilation_px_value,
                )
                frame["raw_points"], frame["raw_colors"] = rebuild_depth_points_for_frame(
                    frame,
                    depth_key="raw_depth_map",
                    extrinsic_key="raw_extrinsic",
                    depth_point_stride=self.depth_point_stride_value,
                    max_scene_depth=self.max_scene_depth_value,
                    exclude_mask_key="raw_human_exclusion_mask",
                )
                frame["hsi_points"], frame["hsi_colors"] = rebuild_depth_points_for_frame(
                    frame,
                    depth_key="hsi_depth_map",
                    extrinsic_key="hsi_extrinsic",
                    depth_point_stride=self.depth_point_stride_value,
                    max_scene_depth=self.max_scene_depth_value,
                    exclude_mask_key="hsi_human_exclusion_mask",
                )
        finally:
            self._rebuilding_points = False
        self._rebuild_environment_point_handles()

    def _rebuild_environment_point_handles(self) -> None:
        self._clear_current_measurement(status="Cleared because the displayed point clouds changed")
        tracking_only = bool(getattr(self.args, "tracking_only", False))
        for frame, frame_handles in zip(self.scene["frames"], self.handles, strict=True):
            idx = int(frame["frame_index"])
            for handle in frame_handles.get("raw", []) + frame_handles.get("hsi", []):
                remove_handle(handle)
            frame_handles["raw"] = [
                self._add_environment_point_cloud(
                    f"/frames/{idx:04d}/points_raw_depth",
                    self._raw_display_points(frame),
                    self._raw_display_colors(frame),
                    frame_index=idx,
                    source="raw",
                )
            ]
            frame_handles["hsi"] = []
            if not tracking_only:
                frame_handles["hsi"] = [
                    self._add_environment_point_cloud(
                        f"/frames/{idx:04d}/points_hsi_depth",
                        self._scaled_hsi_points(frame),
                        self._hsi_point_colors(frame),
                        frame_index=idx,
                        source="hsi",
                    )
                ]
        self._update_visibility()

    def _on_hsi_visual_scale_pending(self, _: Any = None) -> None:
        if self.hsi_calibration_active:
            self.hsi_visual_scale_value = self._requested_hsi_visual_scale()
            self._rebuild_hsi_calibration_frame(int(self.timestep.value))
            self._update_visibility()
            return
        self._update_hsi_scale_info(int(self.timestep.value))

    def _apply_hsi_visual_scale(self, _: Any = None) -> None:
        if self.hsi_calibration_active:
            return
        requested = self._requested_hsi_visual_scale()
        if abs(requested - self.hsi_visual_scale_value) <= 1e-7:
            self._update_hsi_scale_info(int(self.timestep.value))
            return
        self.hsi_visual_scale_value = requested
        self._rebuild_hsi_visual_geometry()

    def _reset_hsi_visual_scale(self, _: Any = None) -> None:
        self.hsi_visual_scale.value = hsi_visual_scale_to_slider(1.0)
        if self.hsi_calibration_active:
            self.hsi_visual_scale_value = 1.0
            self._rebuild_hsi_calibration_frame(int(self.timestep.value))
            self._update_visibility()
        else:
            self._apply_hsi_visual_scale()

    def _requested_hsi_visual_scale(self) -> float:
        return hsi_visual_slider_to_scale(float(self.hsi_visual_scale.value))

    def _on_hsi_calibration_mode_update(self, _: Any = None) -> None:
        if self._switching_hsi_calibration:
            return
        if bool(self.hsi_calibration_mode.value):
            self._enter_hsi_calibration_mode()
        else:
            self._exit_hsi_calibration_mode()

    def _enter_hsi_calibration_mode(self) -> None:
        if self.hsi_calibration_active or bool(getattr(self.args, "tracking_only", False)):
            return
        controls = {
            "play": self.play,
            "mode": self.mode,
            "depth_source": self.depth_source,
            "environment_display": self.environment_display,
            "filter_human_points": self.filter_human_points,
            "show_hsi": self.show_hsi,
            "show_base": self.show_base,
            "show_cameras": self.show_cameras,
            "camera_source": self.camera_source,
            "show_camera_trajectory": self.show_camera_trajectory,
            "follow_camera": self.follow_camera,
        }
        self._hsi_calibration_restore_state = {key: handle.value for key, handle in controls.items()}
        self.hsi_calibration_active = True
        self._switching_hsi_calibration = True
        try:
            self.play.value = False
            self.mode.value = "4D current frame"
            self.depth_source.value = "hsi_depth"
            self.environment_display.value = "points"
            self.filter_human_points.value = False
            self.filter_human_points_value = False
            self.show_hsi.value = True
            self.show_base.value = False
            self.show_cameras.value = False
            self.camera_source.value = "hsi_scaled"
            self.show_camera_trajectory.value = False
            self.follow_camera.value = False
            for handle in controls.values():
                set_handle_disabled(handle, True)
            set_handle_disabled(self.apply_hsi_visual_scale, True)
            set_handle_disabled(self.human_filter_dilation, True)
            set_handle_disabled(self.apply_human_filter_dilation, True)
            set_handle_disabled(self.reset_human_filter_dilation, True)
        finally:
            self._switching_hsi_calibration = False
        self.current_step = int(self.timestep.value)
        self.hsi_visual_scale_value = self._requested_hsi_visual_scale()
        self._rebuild_hsi_calibration_frame(self.current_step)
        self._update_visibility()

    def _exit_hsi_calibration_mode(self) -> None:
        if not self.hsi_calibration_active:
            return
        restore_state = dict(self._hsi_calibration_restore_state)
        self.hsi_calibration_active = False
        self._switching_hsi_calibration = True
        try:
            controls = {
                "play": self.play,
                "mode": self.mode,
                "depth_source": self.depth_source,
                "environment_display": self.environment_display,
                "filter_human_points": self.filter_human_points,
                "show_hsi": self.show_hsi,
                "show_base": self.show_base,
                "show_cameras": self.show_cameras,
                "camera_source": self.camera_source,
                "show_camera_trajectory": self.show_camera_trajectory,
                "follow_camera": self.follow_camera,
            }
            for key, handle in controls.items():
                set_handle_disabled(handle, False)
                if key != "play" and key in restore_state:
                    handle.value = restore_state[key]
            self.play.value = False
            self.filter_human_points_value = bool(restore_state.get("filter_human_points", True))
            set_handle_disabled(self.apply_hsi_visual_scale, False)
            set_handle_disabled(self.human_filter_dilation, False)
            set_handle_disabled(self.apply_human_filter_dilation, False)
            set_handle_disabled(self.reset_human_filter_dilation, False)
        finally:
            self._switching_hsi_calibration = False
        self._rebuild_hsi_visual_geometry()
        if "play" in restore_state:
            self.play.value = bool(restore_state["play"])
        self._hsi_calibration_restore_state = {}
        self._update_visibility()

    def _rebuild_hsi_calibration_frame(self, frame_index: int) -> None:
        if not self.hsi_calibration_active or bool(getattr(self.args, "tracking_only", False)):
            return
        self._clear_current_measurement(status="Cleared because the calibration point cloud changed")
        idx = int(frame_index)
        frame = self.scene["frames"][idx]
        frame_handles = self.handles[idx]
        for handle in frame_handles.get("hsi", []) + frame_handles.get("hsi_mesh", []):
            remove_handle(handle)
        frame_handles["hsi"] = [
            self._add_environment_point_cloud(
                f"/frames/{idx:04d}/points_hsi_depth",
                self._scaled_hsi_points(frame, include_humans=True),
                self._hsi_point_colors(frame, include_humans=True),
                frame_index=idx,
                source="hsi calibration",
            )
        ]
        frame_handles["hsi_mesh"] = []
        scaled_camera = self._scaled_hsi_camera(frame)
        for handle in frame_handles.get("cameras_hsi", []):
            set_handle_position(handle, scaled_camera["position"])

    def _rebuild_hsi_visual_geometry(self) -> None:
        if bool(getattr(self.args, "tracking_only", False)):
            return
        self._clear_current_measurement(status="Cleared because the HSI visual scale changed")
        for frame, frame_handles in zip(self.scene["frames"], self.handles, strict=True):
            idx = int(frame["frame_index"])
            for handle in frame_handles.get("hsi", []) + frame_handles.get("hsi_mesh", []):
                remove_handle(handle)
            frame_handles["hsi"] = [
                self._add_environment_point_cloud(
                    f"/frames/{idx:04d}/points_hsi_depth",
                    self._scaled_hsi_points(frame),
                    self._hsi_point_colors(frame),
                    frame_index=idx,
                    source="hsi",
                )
            ]
            frame_handles["hsi_mesh"] = []
            scaled_camera = self._scaled_hsi_camera(frame)
            for handle in frame_handles.get("cameras_hsi", []):
                set_handle_position(handle, scaled_camera["position"])

        for handle in self.global_handles.get("camera_trajectory_hsi", []):
            remove_handle(handle)
        hsi_trajectory = self._scaled_hsi_trajectory()
        self.global_handles["camera_trajectory_hsi"] = []
        if hsi_trajectory.shape[0] > 0:
            self.global_handles["camera_trajectory_hsi"] = [
                add_point_cloud(
                    self.server,
                    "/camera_trajectory/hsi_scaled_centers",
                    hsi_trajectory,
                    camera_trajectory_colors(hsi_trajectory.shape[0]),
                    max(self.point_size_value * 3.5, 0.012),
                )
            ]
        self._update_visibility()

    def _rebuild_depth_point_clouds(self) -> None:
        self._clear_current_measurement(status="Cleared because point sampling changed")
        self._rebuilding_points = True
        try:
            for frame, frame_handles in zip(self.scene["frames"], self.handles, strict=True):
                idx = int(frame["frame_index"])
                for handle in (
                    frame_handles.get("raw", [])
                    + frame_handles.get("hsi", [])
                    + frame_handles.get("raw_mesh", [])
                    + frame_handles.get("hsi_mesh", [])
                ):
                    remove_handle(handle)

                raw_points_full, raw_colors_full = rebuild_depth_points_for_frame(
                    frame,
                    depth_key="raw_depth_map",
                    extrinsic_key="raw_extrinsic",
                    depth_point_stride=self.depth_point_stride_value,
                    max_scene_depth=self.max_scene_depth_value,
                )
                raw_points, raw_colors = rebuild_depth_points_for_frame(
                    frame,
                    depth_key="raw_depth_map",
                    extrinsic_key="raw_extrinsic",
                    depth_point_stride=self.depth_point_stride_value,
                    max_scene_depth=self.max_scene_depth_value,
                    exclude_mask_key="raw_human_exclusion_mask",
                )
                frame["raw_points"] = raw_points
                frame["raw_colors"] = raw_colors
                frame["raw_points_full"] = raw_points_full
                frame["raw_colors_full"] = raw_colors_full
                frame["depth_point_stride"] = int(self.depth_point_stride_value)
                frame["max_scene_depth"] = float(self.max_scene_depth_value)
                frame_handles["raw"] = [
                    self._add_environment_point_cloud(
                        f"/frames/{idx:04d}/points_raw_depth",
                        self._raw_display_points(frame),
                        self._raw_display_colors(frame),
                        frame_index=idx,
                        source="raw",
                    )
                ]
                frame_handles["raw_mesh"] = []
                if not bool(getattr(self.args, "tracking_only", False)):
                    hsi_points_full, hsi_colors_full = rebuild_depth_points_for_frame(
                        frame,
                        depth_key="hsi_depth_map",
                        extrinsic_key="hsi_extrinsic",
                        depth_point_stride=self.depth_point_stride_value,
                        max_scene_depth=self.max_scene_depth_value,
                    )
                    hsi_points, hsi_colors = rebuild_depth_points_for_frame(
                        frame,
                        depth_key="hsi_depth_map",
                        extrinsic_key="hsi_extrinsic",
                        depth_point_stride=self.depth_point_stride_value,
                        max_scene_depth=self.max_scene_depth_value,
                        exclude_mask_key="hsi_human_exclusion_mask",
                    )
                    frame["hsi_points"] = hsi_points
                    frame["hsi_colors"] = hsi_colors
                    frame["hsi_points_full"] = hsi_points_full
                    frame["hsi_colors_full"] = hsi_colors_full
                    frame_handles["hsi"] = [
                        self._add_environment_point_cloud(
                            f"/frames/{idx:04d}/points_hsi_depth",
                            self._scaled_hsi_points(frame),
                            self._hsi_point_colors(frame),
                            frame_index=idx,
                            source="hsi",
                        )
                    ]
                    frame_handles["hsi_mesh"] = []
        finally:
            self._rebuilding_points = False
        self._update_visibility()

    def _on_camera_size_update(self, _: Any = None) -> None:
        self.camera_scale_value = float(self.camera_size.value)
        self._set_handle_attr(["cameras_raw", "cameras_hsi"], "scale", self.camera_scale_value)

    def _on_smpl_opacity_update(self, _: Any = None) -> None:
        self.smpl_opacity_value = float(self.smpl_opacity.value)
        self._set_handle_attr(["base_humans", "hsi_humans"], "opacity", self.smpl_opacity_value)

    def _on_smpl_color_update(self, _: Any = None) -> None:
        color = tuple(int(np.clip(value, 0, 255)) for value in self.smpl_color.value)
        self._set_handle_attr(["base_humans", "hsi_humans"], "color", color)

    def _update_visibility(self) -> None:
        current = int(self.timestep.value)
        mode = str(self.mode.value)
        depth_source = str(self.depth_source.value)
        environment_display = str(self.environment_display.value)
        show_env_points = environment_display in {"points", "both"}
        show_env_mesh = environment_display in {"mesh", "both"}
        smpl_stride = max(1, int(self.smpl_downsample.value))
        camera_stride = max(1, int(self.camera_downsample.value))
        for idx, frame_handles in enumerate(self.handles):
            if mode == "3D accumulate":
                show_points = idx <= current
                show_humans = idx <= current
            elif mode == "Hybrid":
                show_points = idx <= current
                show_humans = idx == current
            else:
                show_points = idx == current
                show_humans = idx == current
            show_decimated_smpl = idx == current or (idx % smpl_stride == 0)
            show_decimated_camera = idx == current or (idx % camera_stride == 0)
            show_camera_frame = (idx <= current if mode != "4D current frame" else idx == current)
            show_raw_camera, show_hsi_camera = self._camera_visibility_for_depth(depth_source)
            if show_points and show_env_mesh:
                self._ensure_depth_meshes_for_frame(frame=self.scene["frames"][idx], frame_handles=frame_handles, idx=idx, depth_source=depth_source)
            set_group_visible(frame_handles["raw"], show_points and show_env_points and depth_source in {"raw_depth", "both"})
            set_group_visible(frame_handles["hsi"], show_points and show_env_points and depth_source in {"hsi_depth", "both"})
            set_group_visible(frame_handles["raw_mesh"], show_points and show_env_mesh and depth_source in {"raw_depth", "both"})
            set_group_visible(frame_handles["hsi_mesh"], show_points and show_env_mesh and depth_source in {"hsi_depth", "both"})
            set_group_visible(frame_handles["base_humans"], show_humans and show_decimated_smpl and bool(self.show_base.value))
            set_group_visible(frame_handles["hsi_humans"], show_humans and show_decimated_smpl and bool(self.show_hsi.value))
            set_group_visible(frame_handles["track_labels"], show_humans and show_decimated_smpl and bool(self.show_track_ids.value))
            set_group_visible(frame_handles["cameras_raw"], bool(self.show_cameras.value) and show_raw_camera and show_camera_frame and show_decimated_camera)
            set_group_visible(frame_handles["cameras_hsi"], bool(self.show_cameras.value) and show_hsi_camera and show_camera_frame and show_decimated_camera)
        self._update_info_text(current)

    def _ensure_depth_meshes_for_frame(self, frame: dict[str, Any], frame_handles: dict[str, Any], idx: int, depth_source: str) -> None:
        if depth_source in {"raw_depth", "both"} and not frame_handles.get("raw_mesh"):
            raw_mesh_vertices, raw_mesh_colors, raw_mesh_faces, raw_mesh_face_colors = build_depth_mesh_for_frame(
                frame,
                depth_key="raw_depth_map",
                extrinsic_key="raw_extrinsic",
                depth_point_stride=self.depth_point_stride_value,
                max_scene_depth=self.max_scene_depth_value,
                depth_edge_rtol=self.env_mesh_depth_edge_rtol,
            )
            frame["raw_mesh_vertices"] = raw_mesh_vertices
            frame["raw_mesh_colors"] = raw_mesh_colors
            frame["raw_mesh_faces"] = raw_mesh_faces
            frame["raw_mesh_face_colors"] = raw_mesh_face_colors
            frame_handles["raw_mesh"] = add_vertex_color_mesh(
                self.server,
                f"/frames/{idx:04d}/mesh_raw_depth",
                raw_mesh_vertices,
                raw_mesh_faces,
                raw_mesh_colors,
                face_colors=raw_mesh_face_colors,
                opacity=0.82,
                max_color_groups=self.env_mesh_color_groups,
                color_mode=self.env_mesh_color_mode,
                overlay_point_size=self.env_mesh_overlay_point_size,
            )
        if bool(getattr(self.args, "tracking_only", False)):
            return
        if depth_source in {"hsi_depth", "both"} and not frame_handles.get("hsi_mesh"):
            hsi_mesh_vertices, hsi_mesh_colors, hsi_mesh_faces, hsi_mesh_face_colors = build_depth_mesh_for_frame(
                frame,
                depth_key="hsi_depth_map",
                extrinsic_key="hsi_extrinsic",
                depth_point_stride=self.depth_point_stride_value,
                max_scene_depth=self.max_scene_depth_value,
                depth_edge_rtol=self.env_mesh_depth_edge_rtol,
            )
            frame["hsi_mesh_vertices"] = hsi_mesh_vertices
            frame["hsi_mesh_colors"] = hsi_mesh_colors
            frame["hsi_mesh_faces"] = hsi_mesh_faces
            frame["hsi_mesh_face_colors"] = hsi_mesh_face_colors
            frame_handles["hsi_mesh"] = add_vertex_color_mesh(
                self.server,
                f"/frames/{idx:04d}/mesh_hsi_depth",
                hsi_mesh_vertices,
                hsi_mesh_faces,
                hsi_mesh_colors,
                face_colors=hsi_mesh_face_colors,
                opacity=0.82,
                max_color_groups=self.env_mesh_color_groups,
                color_mode=self.env_mesh_color_mode,
                overlay_point_size=self.env_mesh_overlay_point_size,
            )

    def _camera_visibility_for_depth(self, depth_source: str) -> tuple[bool, bool]:
        if bool(getattr(self.args, "tracking_only", False)):
            return True, False
        camera_source = str(self.camera_source.value)
        if camera_source == "raw_vggt":
            return True, False
        if camera_source == "hsi_scaled":
            return False, True
        if camera_source == "both":
            return True, True
        if depth_source == "raw_depth":
            return True, False
        if depth_source == "both":
            return True, True
        return False, True

    def _update_info_text(self, frame_index: int) -> None:
        frame = self.scene["frames"][int(frame_index)]
        raw_cam_pos = np.asarray(frame["raw_camera"]["position"], dtype=np.float32)
        hsi_cam_pos = np.asarray(self._scaled_hsi_camera(frame)["position"], dtype=np.float32)
        hsi_full_count = int(np.asarray(frame.get("hsi_points_full", frame["hsi_points"])).shape[0])
        hsi_filtered_count = int(frame["hsi_points"].shape[0])
        hsi_mask_pixels = int(np.count_nonzero(frame.get("hsi_human_exclusion_mask", np.zeros((0,), dtype=bool))))
        filter_active = self.filter_human_points_value and not self.hsi_calibration_active
        hsi_display_count = hsi_filtered_count if filter_active else hsi_full_count
        set_text_value(
            self.frame_info,
            (
                f"{int(frame_index) + 1}/{len(self.scene['frames'])} "
                f"{frame['frame_id']} | raw_pts={int(frame['raw_points'].shape[0])} "
                f"hsi_pts={hsi_display_count}/{hsi_full_count} filter={'on' if filter_active else 'off'} "
                f"maskRemoved={hsi_full_count - hsi_filtered_count} "
                f"maskPx={hsi_mask_pixels} "
                f"people={len(frame['people'])} "
                f"stride={int(frame.get('depth_point_stride', self.depth_point_stride_value))} "
                f"maxD={float(frame.get('max_scene_depth', self.max_scene_depth_value)):.1f} "
                f"rawCam=({raw_cam_pos[0]:.3f},{raw_cam_pos[1]:.3f},{raw_cam_pos[2]:.3f}) "
                f"hsiCamVisual=({hsi_cam_pos[0]:.3f},{hsi_cam_pos[1]:.3f},{hsi_cam_pos[2]:.3f}) "
                f"IDs={[int(person['track_id']) for person in frame['people']]}"
            ),
        )
        if self.hsi_calibration_active:
            filter_text = (
                "OFF (forced by single-frame scale calibration) "
                f"| applied dilation={self.human_mask_dilation_px_value} px"
            )
        elif self.filter_human_points_value:
            filter_text = (
                f"ON | applied dilation={self.human_mask_dilation_px_value} px "
                f"| current frame removes {hsi_full_count - hsi_filtered_count} points"
            )
        else:
            filter_text = (
                "OFF | displaying complete raw and HSI point clouds "
                f"| applied dilation={self.human_mask_dilation_px_value} px"
            )
        requested_dilation = self._requested_human_filter_dilation()
        if requested_dilation != self.human_mask_dilation_px_value:
            filter_text += f" | pending={requested_dilation} px (click Apply)"
        set_text_value(self.human_point_filter_info, filter_text)
        self._update_hsi_scale_info(frame_index)
        align = frame.get("depth_alignment", {})
        set_text_value(
            self.alignment_info,
            (
                f"base/raw {format_alignment_short(align.get('base_raw', []))} | "
                f"base/hsi {format_alignment_short(align.get('base_hsi', []))} | "
                f"hsi/hsi {format_alignment_short(align.get('hsi_hsi', []))}"
            ),
        )
        show_raw_camera, show_hsi_camera = self._camera_visibility_for_depth(str(self.depth_source.value))
        set_group_visible(self.global_handles.get("camera_trajectory_raw", []), bool(self.show_camera_trajectory.value) and show_raw_camera)
        set_group_visible(self.global_handles.get("camera_trajectory_hsi", []), bool(self.show_camera_trajectory.value) and show_hsi_camera)

    def _update_hsi_scale_info(self, frame_index: int) -> None:
        frame = self.scene["frames"][int(frame_index)]
        model_scale = float(frame["hsi_scene_scale"])
        model_bias = float(frame["hsi_scene_depth_bias"])
        raw_frame_scale = frame.get("hsi_frame_scene_scale")
        raw_frame_bias = frame.get("hsi_frame_scene_depth_bias")
        model_scales = np.asarray([item["hsi_scene_scale"] for item in self.scene["frames"]], dtype=np.float64)
        raw_scales = np.asarray(
            [item["hsi_frame_scene_scale"] for item in self.scene["frames"] if item.get("hsi_frame_scene_scale") is not None],
            dtype=np.float64,
        )
        mode = str(self.scene.get("hsi_scene_affine_mode", "per_frame"))
        if mode == "clip_median":
            strategy = "clip_median: one robust scale/bias shared by all frames"
        elif mode == "ema":
            alpha = float(self.scene.get("hsi_scene_affine_ema_alpha", 0.25))
            strategy = f"ema(alpha={alpha:.3g}): smoothed scale/bias can vary by frame"
        else:
            strategy = "per_frame: each frame uses its own predicted scale/bias"
        pending = self._requested_hsi_visual_scale()
        applied = float(self.hsi_visual_scale_value)
        raw_text = "unavailable"
        if raw_frame_scale is not None and raw_frame_bias is not None:
            raw_text = f"scale={float(raw_frame_scale):.5g}, bias={float(raw_frame_bias):.5g}"
        raw_range = "unavailable"
        if raw_scales.size > 0:
            raw_range = f"{raw_scales.min():.5g}/{np.median(raw_scales):.5g}/{raw_scales.max():.5g}"
        set_text_value(self.hsi_scale_strategy_info, strategy)
        if self.hsi_calibration_active:
            set_text_value(
                self.hsi_calibration_info,
                f"ON | frame {int(frame_index) + 1}/{len(self.scene['frames'])} | live full-point preview",
            )
        else:
            set_text_value(self.hsi_calibration_info, "OFF | normal sequence display")
        set_text_value(self.hsi_model_scale_info, f"scale={model_scale:.6g}, bias={model_bias:.6g}")
        set_text_value(self.hsi_raw_scale_info, raw_text)
        set_text_value(
            self.hsi_scale_range_info,
            (
                f"model={model_scales.min():.5g} / {np.median(model_scales):.5g} / {model_scales.max():.5g} | "
                f"raw={raw_range}"
            ),
        )
        visual_result = f"x{applied:.3f} | effective scale={model_scale * applied:.6g}, bias={model_bias * applied:.6g}"
        if self.hsi_calibration_active:
            visual_result += " | live on selected frame; turn mode off to apply sequence"
        elif abs(pending - applied) > 1e-7:
            visual_result += f" | pending x{pending:.3f}: click Apply Scale"
        set_text_value(self.hsi_visual_result_info, visual_result)

    def _set_handle_attr(self, groups: list[str], attr: str, value: float) -> None:
        for frame_handles in self.handles:
            for group in groups:
                for handle in frame_handles.get(group, []):
                    try:
                        setattr(handle, attr, value)
                    except Exception:
                        pass

    def _follow_pred_camera(self, step: int) -> None:
        show_raw_camera, show_hsi_camera = self._camera_visibility_for_depth(str(self.depth_source.value))
        camera_key = "raw_camera" if show_raw_camera and not show_hsi_camera else "hsi_camera"
        frame = self.scene["frames"][int(step)]
        camera = frame[camera_key] if camera_key == "raw_camera" else self._scaled_hsi_camera(frame)
        rotation = camera["rotation_c2w"]
        position = camera["position"]
        wxyz = self.transforms.SO3.from_matrix(rotation).wxyz
        clients = list(self.clients.values())
        if hasattr(self.server, "get_clients"):
            try:
                clients = list(self.server.get_clients().values())
            except Exception:
                pass
        for client in clients:
            try:
                client.camera.wxyz = wxyz
                client.camera.position = position
                client.camera.fov = float(camera["fov"])
            except Exception:
                continue


def scene_api(server: Any) -> Any:
    return getattr(server, "scene", server)


def gui_api(server: Any) -> Any:
    return getattr(server, "gui", server)


def add_point_cloud(server: Any, name: str, points: np.ndarray, colors: np.ndarray, point_size: float) -> Any:
    api = scene_api(server)
    try:
        return api.add_point_cloud(name=name, points=points, colors=colors, point_size=point_size)
    except TypeError:
        return api.add_point_cloud(name, points, colors, point_size=point_size)


def build_measurement_text_segments(text: str, height: float) -> np.ndarray:
    strokes = {
        "a": ((0.0, 1.0), (0.6, 1.0)),
        "b": ((0.6, 1.0), (0.6, 0.5)),
        "c": ((0.6, 0.5), (0.6, 0.0)),
        "d": ((0.0, 0.0), (0.6, 0.0)),
        "e": ((0.0, 0.0), (0.0, 0.5)),
        "f": ((0.0, 0.5), (0.0, 1.0)),
        "g": ((0.0, 0.5), (0.6, 0.5)),
    }
    digit_strokes = {
        "0": "abcedf",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcedfg",
        "9": "abfgcd",
    }
    glyphs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    cursor = 0.0
    for character in text:
        if character in digit_strokes:
            glyphs.extend(
                (
                    (cursor + strokes[key][0][0], strokes[key][0][1]),
                    (cursor + strokes[key][1][0], strokes[key][1][1]),
                )
                for key in digit_strokes[character]
            )
            cursor += 0.78
        elif character == ".":
            glyphs.append(((cursor + 0.08, 0.0), (cursor + 0.08, 0.08)))
            cursor += 0.28
        elif character == "m":
            glyphs.extend(
                [
                    ((cursor, 0.0), (cursor, 0.65)),
                    ((cursor, 0.65), (cursor + 0.22, 0.42)),
                    ((cursor + 0.22, 0.42), (cursor + 0.44, 0.65)),
                    ((cursor + 0.44, 0.65), (cursor + 0.44, 0.0)),
                ]
            )
            cursor += 0.62
        else:
            cursor += 0.4
    if not glyphs:
        return np.empty((0, 2, 3), dtype=np.float32)
    segments = np.asarray(
        [[[start[0], start[1], 0.0], [end[0], end[1], 0.0]] for start, end in glyphs],
        dtype=np.float32,
    )
    min_xy = segments[..., :2].min(axis=(0, 1))
    max_xy = segments[..., :2].max(axis=(0, 1))
    segments[..., :2] -= ((min_xy + max_xy) * np.float32(0.5))[None, None, :]
    segments *= np.float32(max(float(height), 1e-4))
    return segments


def nearest_clicked_point_with_distance(event: Any, points: np.ndarray) -> tuple[int, np.ndarray, float] | None:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points_np.shape[0] == 0:
        return None
    ray_origin_value = getattr(event, "ray_origin", None)
    ray_direction_value = getattr(event, "ray_direction", None)
    if ray_origin_value is None or ray_direction_value is None:
        return None
    ray_origin = np.asarray(ray_origin_value, dtype=np.float32).reshape(-1)
    ray_direction = np.asarray(ray_direction_value, dtype=np.float32).reshape(-1)
    if ray_origin.size != 3 or ray_direction.size != 3:
        return None
    direction_norm = float(np.linalg.norm(ray_direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1e-8:
        return None
    ray_direction = ray_direction / np.float32(direction_norm)
    finite = np.isfinite(points_np).all(axis=1)
    if not bool(finite.any()):
        return None
    valid_indices = np.flatnonzero(finite)
    valid_points = points_np[valid_indices]
    offsets = valid_points - ray_origin[None, :]
    along_ray = offsets @ ray_direction
    in_front = along_ray >= 0.0
    if not bool(in_front.any()):
        return None
    candidate_indices = valid_indices[in_front]
    candidate_offsets = offsets[in_front]
    candidate_along = along_ray[in_front]
    distance_sq = np.maximum(
        np.einsum("ij,ij->i", candidate_offsets, candidate_offsets) - candidate_along * candidate_along,
        0.0,
    )
    local_index = int(np.argmin(distance_sq))
    selected = int(candidate_indices[local_index])
    return selected, points_np[selected].copy(), float(np.sqrt(distance_sq[local_index]))


def add_line_segments(
    server: Any,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
    line_width: float,
    position: np.ndarray | None = None,
    wxyz: np.ndarray | None = None,
) -> Any:
    api = scene_api(server)
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 2, 3)
    color_int = tuple(int(np.clip(value, 0, 255)) for value in color)
    transform_kwargs: dict[str, Any] = {}
    if position is not None:
        transform_kwargs["position"] = np.asarray(position, dtype=np.float32).reshape(3)
    if wxyz is not None:
        transform_kwargs["wxyz"] = np.asarray(wxyz, dtype=np.float32).reshape(4)
    if hasattr(api, "add_line_segments"):
        try:
            return api.add_line_segments(
                name=name,
                points=points_np,
                colors=color_int,
                line_width=float(line_width),
                **transform_kwargs,
            )
        except TypeError:
            return api.add_line_segments(
                name,
                points_np,
                color_int,
                line_width=float(line_width),
                **transform_kwargs,
            )
    start, end = points_np[0]
    line_points = np.linspace(start, end, 64, dtype=np.float32)
    line_colors = np.repeat(np.asarray(color_int, dtype=np.uint8)[None, :], line_points.shape[0], axis=0)
    return add_point_cloud(server, name, line_points, line_colors, max(float(line_width) * 0.002, 0.006))


def rebuild_depth_points_for_frame(
    frame: dict[str, Any],
    depth_key: str,
    extrinsic_key: str,
    depth_point_stride: int,
    max_scene_depth: float,
    exclude_mask_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = torch.from_numpy(np.asarray(frame[depth_key], dtype=np.float32))
    rgb = torch.from_numpy(np.asarray(frame["rgb_chw"], dtype=np.float32))
    intrinsic = np.asarray(frame["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(frame[extrinsic_key], dtype=np.float32)
    exclude_mask = None
    if exclude_mask_key is not None and frame.get(exclude_mask_key) is not None:
        exclude_mask = np.asarray(frame[exclude_mask_key], dtype=bool)
    return depth_to_world_points_with_limits(
        depth=depth,
        rgb=rgb,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        depth_point_stride=int(depth_point_stride),
        max_scene_depth=float(max_scene_depth),
        exclude_mask=exclude_mask,
    )


def build_depth_mesh_for_frame(
    frame: dict[str, Any],
    depth_key: str,
    extrinsic_key: str,
    depth_point_stride: int,
    max_scene_depth: float,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = torch.from_numpy(np.asarray(frame[depth_key], dtype=np.float32))
    rgb = torch.from_numpy(np.asarray(frame["rgb_chw"], dtype=np.float32))
    intrinsic = np.asarray(frame["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(frame[extrinsic_key], dtype=np.float32)
    return depth_to_world_surface_mesh_with_limits(
        depth=depth,
        rgb=rgb,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        depth_point_stride=int(depth_point_stride),
        max_scene_depth=float(max_scene_depth),
        depth_edge_rtol=float(depth_edge_rtol),
    )


def depth_to_world_surface_mesh_with_limits(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    depth_point_stride: int,
    max_scene_depth: float,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = depth.detach().float()
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    step = max(1, int(depth_point_stride))
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=torch.float32),
        torch.arange(0, width, step, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()]
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points = torch.stack([x, y, z], dim=-1)
    rgb_use = rgb.to(device=depth.device, dtype=torch.float32)
    if tuple(rgb_use.shape[-2:]) != (height, width):
        rgb_use = F.interpolate(rgb_use[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    colors = (rgb_use[:, ys.long(), xs.long()].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    if float(max_scene_depth) > 0:
        valid = valid & (z <= float(max_scene_depth))
    points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    colors_np = colors.detach().cpu().numpy().astype(np.uint8, copy=False)
    depth_np = z.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid.detach().cpu().numpy().astype(bool, copy=False)
    vertices_cam, vertex_colors, faces, face_colors = depth_sample_grid_to_surface_mesh(
        points_np,
        colors_np,
        depth_np,
        valid_np,
        depth_edge_rtol=float(depth_edge_rtol),
    )
    vertices_world = camera_points_to_world_np(vertices_cam, extrinsic) if vertices_cam.size else vertices_cam
    return vertices_world, vertex_colors, faces, face_colors


def depth_sample_grid_to_surface_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = depth.shape
    index_map = -np.ones((rows, cols), dtype=np.int64)
    index_map[valid] = np.arange(int(valid.sum()), dtype=np.int64)
    vertices = np.asarray(points[valid], dtype=np.float32).reshape(-1, 3)
    vertex_colors = np.asarray(colors[valid], dtype=np.uint8).reshape(-1, 3)
    if rows < 2 or cols < 2 or vertices.shape[0] == 0:
        return vertices, vertex_colors, np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.uint8)

    cell_mask = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    d00 = depth[:-1, :-1]
    d01 = depth[:-1, 1:]
    d10 = depth[1:, :-1]
    d11 = depth[1:, 1:]
    dmax = np.maximum.reduce([d00, d01, d10, d11])
    dmin = np.minimum.reduce([d00, d01, d10, d11])
    dmean = (d00 + d01 + d10 + d11) * 0.25
    cell_mask &= ((dmax - dmin) / np.maximum(np.abs(dmean), 1e-6)) <= max(float(depth_edge_rtol), 0.0)

    ys, xs = np.nonzero(cell_mask)
    if ys.size == 0:
        return vertices, vertex_colors, np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.uint8)
    i00 = index_map[ys, xs]
    i01 = index_map[ys, xs + 1]
    i10 = index_map[ys + 1, xs]
    i11 = index_map[ys + 1, xs + 1]
    c00 = colors[ys, xs]
    c11 = colors[ys + 1, xs + 1]
    faces = np.concatenate(
        [
            np.stack([i00, i01, i10], axis=1),
            np.stack([i10, i01, i00], axis=1),
            np.stack([i01, i10, i11], axis=1),
            np.stack([i11, i10, i01], axis=1),
        ],
        axis=0,
    )
    face_colors = np.concatenate([c00, c00, c11, c11], axis=0)
    return vertices, vertex_colors, faces.astype(np.int64, copy=False), face_colors.astype(np.uint8, copy=False)


def remove_handle(handle: Any) -> None:
    try:
        handle.remove()
    except Exception:
        try:
            handle.visible = False
        except Exception:
            pass


def add_mesh(server: Any, name: str, vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int], opacity: float = 1.0) -> Any:
    api = scene_api(server)
    color_int = tuple(int(np.clip(v, 0, 255)) for v in color)
    color_float = tuple(float(v) / 255.0 for v in color_int)
    try:
        return api.add_mesh_simple(name=name, vertices=vertices, faces=faces, color=color_int, opacity=float(opacity))
    except TypeError:
        try:
            handle = api.add_mesh_simple(name, vertices, faces, color=color_int)
            try:
                handle.opacity = float(opacity)
            except Exception:
                pass
            return handle
        except TypeError:
            try:
                return api.add_mesh_simple(name=name, vertices=vertices, faces=faces, color=color_float, opacity=float(opacity))
            except TypeError:
                handle = api.add_mesh_simple(name, vertices, faces, color=color_float)
                try:
                    handle.opacity = float(opacity)
                except Exception:
                    pass
                return handle


def add_vertex_color_mesh(
    server: Any,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    face_colors: np.ndarray | None = None,
    opacity: float = 1.0,
    max_color_groups: int = 64,
    color_mode: str = "point_overlay",
    overlay_point_size: float = 0.004,
) -> list[Any]:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    face_colors_arr = None if face_colors is None else np.asarray(face_colors, dtype=np.uint8).reshape(-1, 3)
    if colors.shape[0] != vertices.shape[0]:
        colors = np.full((vertices.shape[0], 3), 160, dtype=np.uint8)
    valid_faces = np.all((faces >= 0) & (faces < vertices.shape[0]), axis=1)
    if face_colors_arr is not None and face_colors_arr.shape[0] == faces.shape[0]:
        face_colors_arr = face_colors_arr[valid_faces]
    else:
        face_colors_arr = None
    faces = faces[valid_faces]
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return []
    max_color_groups = max(1, int(max_color_groups))
    if color_mode == "point_overlay":
        surface = add_mesh(server, f"{name}/surface", vertices, faces, (150, 150, 150), opacity=min(float(opacity), 0.28))
        color_points = add_point_cloud(server, f"{name}/rgb_overlay", vertices, colors, float(overlay_point_size))
        return [surface, color_points]

    face_colors_float = face_colors_arr.astype(np.float32) if face_colors_arr is not None else colors[faces].astype(np.float32).mean(axis=1)
    if max_color_groups <= 1 or face_colors_float.shape[0] == 0:
        median_color = np.median(colors, axis=0).astype(np.uint8).tolist() if colors.size else [160, 160, 160]
        return [add_mesh(server, name, vertices, faces, tuple(int(v) for v in median_color), opacity=opacity)]

    levels = int(np.floor(max_color_groups ** (1.0 / 3.0)))
    if levels < 2:
        median_color = np.median(colors, axis=0).astype(np.uint8).tolist() if colors.size else [160, 160, 160]
        return [add_mesh(server, name, vertices, faces, tuple(int(v) for v in median_color), opacity=opacity)]
    bins = np.clip((face_colors_float * float(levels) / 256.0).astype(np.int64), 0, levels - 1)
    bucket_ids = bins[:, 0] * levels * levels + bins[:, 1] * levels + bins[:, 2]
    handles: list[Any] = []
    for bucket_id in np.unique(bucket_ids):
        mask = bucket_ids == int(bucket_id)
        bucket_faces = faces[mask]
        if bucket_faces.size == 0:
            continue
        used_vertices, remapped = np.unique(bucket_faces.reshape(-1), return_inverse=True)
        compact_faces = remapped.reshape(-1, 3).astype(np.int64, copy=False)
        compact_vertices = vertices[used_vertices]
        bucket_color = np.mean(face_colors_float[mask], axis=0).clip(0.0, 255.0).astype(np.uint8)
        color = tuple(int(v) for v in bucket_color.tolist())
        handles.append(
            add_mesh(
                server,
                f"{name}/color_{int(bucket_id):03d}",
                compact_vertices,
                compact_faces,
                color,
                opacity=opacity,
            )
        )
    return handles


def add_label(server: Any, name: str, text: str, position: np.ndarray) -> Any:
    api = scene_api(server)
    try:
        return api.add_label(name=name, text=text, position=position)
    except TypeError:
        return api.add_label(name, text, position)


def add_transform_controls(server: Any, name: str, position: np.ndarray, scale: float, visible: bool) -> Any:
    api = scene_api(server)
    if not hasattr(api, "add_transform_controls"):
        return None
    position = np.asarray(position, dtype=np.float32)
    try:
        handle = api.add_transform_controls(name=name, position=position, scale=float(scale), visible=bool(visible))
    except TypeError:
        try:
            handle = api.add_transform_controls(name=name, position=position, scale=float(scale))
        except TypeError:
            handle = api.add_transform_controls(name, position=position, scale=float(scale))
        set_handle_visible(handle, visible)
    return handle


def add_camera(server: Any, transforms: Any, name: str, camera: dict[str, Any], scale: float, color: tuple[int, int, int] = (255, 255, 255)) -> Any:
    api = scene_api(server)
    wxyz = transforms.SO3.from_matrix(camera["rotation_c2w"]).wxyz
    try:
        return api.add_camera_frustum(
            name=name,
            fov=float(camera["fov"]),
            aspect=float(camera["aspect"]),
            scale=float(scale),
            wxyz=wxyz,
            position=camera["position"],
            color=color,
        )
    except TypeError:
        return api.add_camera_frustum(name, float(camera["fov"]), float(camera["aspect"]), float(scale), wxyz, camera["position"])


def add_slider(server: Any, name: str, min_value: float, max_value: float, step: float, initial: float) -> Any:
    api = gui_api(server)
    try:
        return api.add_slider(name, min=min_value, max=max_value, step=step, initial_value=initial)
    except AttributeError:
        return server.add_gui_slider(name, min=min_value, max=max_value, step=step, initial_value=initial)


def add_rgb(server: Any, name: str, initial: tuple[int, int, int]) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_rgb"):
        return api.add_rgb(name, initial_value=initial)
    return server.add_gui_rgb(name, initial)


def add_folder(server: Any, name: str) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_folder"):
        return api.add_folder(name, expand_by_default=True)
    if hasattr(server, "add_gui_folder"):
        return server.add_gui_folder(name)
    return nullcontext()


def add_checkbox(server: Any, name: str, initial: bool) -> Any:
    api = gui_api(server)
    try:
        return api.add_checkbox(name, initial_value=initial)
    except AttributeError:
        return server.add_gui_checkbox(name, initial)


def add_dropdown(server: Any, name: str, options: list[str], initial: str) -> Any:
    api = gui_api(server)
    try:
        return api.add_dropdown(name, options=options, initial_value=initial)
    except AttributeError:
        return server.add_gui_dropdown(name, options, initial)


def add_button(server: Any, name: str) -> Any:
    api = gui_api(server)
    try:
        return api.add_button(name)
    except AttributeError:
        return server.add_gui_button(name)


def add_button_group(server: Any, name: str, options: tuple[str, ...]) -> Any:
    api = gui_api(server)
    try:
        return api.add_button_group(name, options)
    except AttributeError:
        return server.add_gui_button_group(name, options)


def add_text(server: Any, name: str, initial: str) -> Any:
    api = gui_api(server)
    try:
        return api.add_text(name, initial_value=initial)
    except AttributeError:
        try:
            return server.add_gui_text(name, initial)
        except AttributeError:
            return None


def bind_update(handle: Any, callback: Any) -> None:
    if hasattr(handle, "on_update"):
        handle.on_update(callback)


def bind_click(handle: Any, callback: Any) -> None:
    if hasattr(handle, "on_click"):
        handle.on_click(callback)


def set_text_value(handle: Any, value: str) -> None:
    if handle is None:
        return
    try:
        handle.value = value
    except Exception:
        pass


def set_handle_disabled(handle: Any, disabled: bool) -> None:
    if handle is None:
        return
    try:
        handle.disabled = bool(disabled)
    except Exception:
        pass


def set_handle_position(handle: Any, position: np.ndarray) -> None:
    if handle is None:
        return
    position_np = np.asarray(position, dtype=np.float32).reshape(3)
    try:
        handle.position = position_np
    except Exception:
        pass


def set_handle_wxyz(handle: Any, wxyz: np.ndarray) -> None:
    if handle is None:
        return
    try:
        handle.wxyz = np.asarray(wxyz, dtype=np.float32).reshape(4)
    except Exception:
        pass


def get_handle_position(handle: Any) -> np.ndarray | None:
    if handle is None:
        return None
    try:
        return np.asarray(handle.position, dtype=np.float32).reshape(3)
    except Exception:
        return None


def set_handle_visible(handle: Any, visible: bool) -> None:
    if handle is None:
        return
    try:
        handle.visible = bool(visible)
    except Exception:
        pass


def set_group_visible(handles: list[Any], visible: bool) -> None:
    for handle in handles:
        set_handle_visible(handle, visible)


def human_entry_key(frame_index: int, kind: str, track_id: int, query_idx: int) -> str:
    return f"f{int(frame_index):04d}:{kind}:id{int(track_id)}:q{int(query_idx)}"


def resolve_smpl_edit_output(args: argparse.Namespace) -> Path:
    if getattr(args, "smpl_edit_output", ""):
        return resolve_project_path(str(args.smpl_edit_output))
    return resolve_project_path(args.output_dir) / "smpl_edit_offsets.json"


def camera_trajectory_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    colors = np.zeros((count, 3), dtype=np.uint8)
    colors[:, 0] = np.asarray(255.0 * t, dtype=np.uint8)
    colors[:, 1] = np.asarray(210.0 * (1.0 - np.abs(t - 0.5) * 2.0), dtype=np.uint8)
    colors[:, 2] = np.asarray(255.0 * (1.0 - t), dtype=np.uint8)
    return colors


def format_camera_motion_short(scene: dict[str, Any]) -> str:
    raw = summarize_camera_motion(scene, "camera_trajectory_raw")
    hsi = summarize_camera_motion(scene, "camera_trajectory_hsi")
    return (
        f"raw path={raw['total_path_m_vggt_units']:.4g} end={raw['start_end_m_vggt_units']:.4g} "
        f"range={tuple(round(float(v), 4) for v in raw['axis_range_xyz_vggt_units'])} | "
        f"hsi path={hsi['total_path_m_vggt_units']:.4g} end={hsi['start_end_m_vggt_units']:.4g} "
        f"range={tuple(round(float(v), 4) for v in hsi['axis_range_xyz_vggt_units'])}"
    )


def format_alignment_short(entries: list[dict[str, Any]]) -> str:
    values = [float(item["median_abs_m"]) for item in entries if item.get("median_abs_m") is not None]
    points = sum(int(item.get("valid_points", 0) or 0) for item in entries)
    if not values:
        return "n/a"
    return f"medAbs={float(np.median(values)):.3f}m pts={points}"


if __name__ == "__main__":
    main()
