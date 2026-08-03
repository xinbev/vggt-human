#!/usr/bin/env python
"""Serve NLF SMPL meshes with learned ROI identity tracking in Viser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import (  # noqa: E402
    SequenceViewer,
    build_query_priors,
    build_scene_data,
    ensure_viser_available,
    load_sequence_images,
    require_smpl_model_dir,
    resolve_project_path,
    select_frames,
)
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
)
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update  # noqa: E402


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frames_dir = resolve_project_path(args.frames_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = select_frames(frames_dir, args)
    if not frame_paths:
        raise RuntimeError(f"No frames found under {frames_dir}")

    config = load_config(args)
    patch_size = int(config["model"].get("patch_size", 16))
    _, image_resolution = resolve_image_size_config(config["data"], args.image_size)
    images, geometries = load_sequence_images(
        frame_paths,
        image_resolution,
        patch_size,
        str(config["data"].get("resize_mode", "balanced")),
    )
    priors = build_query_priors(frame_paths, geometries, args, int(args.max_humans), device)

    model = build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model, config, device)
    id_checkpoint = resolve_project_path(args.id_checkpoint)
    load_training_checkpoint(model, id_checkpoint, device)
    smpl = SMPLLayer(require_smpl_model_dir(config, args)).to(device).eval()

    image_sequence = images.unsqueeze(0).to(device)
    with torch.inference_mode():
        predictions = model(
            image_sequence,
            smpl_query_boxes=priors["smpl_query_boxes"],
            smpl_query_boxes_mask=priors["smpl_query_boxes_mask"],
        )
    predictions["images"] = image_sequence
    predictions["hsi_scene_scale"] = torch.ones(
        image_sequence.shape[:2], device=device, dtype=torch.float32
    )
    predictions["hsi_scene_depth_bias"] = torch.zeros_like(predictions["hsi_scene_scale"])
    validate_predictions(predictions)

    args.tracking_only = True
    scene = build_scene_data(
        frame_paths=frame_paths,
        images=image_sequence,
        predictions=predictions,
        priors=priors,
        smpl=smpl,
        args=args,
        device=device,
    )
    summary = build_summary(args, id_checkpoint, predictions, scene)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"viewer": f"http://127.0.0.1:{args.port}", "summary": str(summary_path)}, indent=2), flush=True)
    if args.smoke_only:
        print("[ok] NLF ROI ID tracking V2 viewer smoke passed", flush=True)
        return

    server = viser.ViserServer(port=int(args.port))
    if hasattr(server, "set_up_direction"):
        server.set_up_direction("-y")
    SequenceViewer(server=server, transforms=vtf, scene=scene, args=args).run()


def load_config(args: argparse.Namespace) -> dict:
    config = deep_update(
        load_yaml_config(resolve_project_path(args.path_config)),
        load_yaml_config(resolve_project_path(args.train_config)),
    )
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = str(resolve_project_path(args.baseline_checkpoint))
    data_cfg = config.setdefault("data", {})
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    data_cfg["image_size"] = int(image_size)
    data_cfg["image_resolution"] = int(image_resolution)
    data_cfg.setdefault("resize_mode", "balanced")

    model_cfg = config.setdefault("model", {})
    model_cfg.update(
        {
            "enable_camera": True,
            "enable_depth": True,
            "enable_smpl": True,
            "enable_hsi_refine": False,
            "smpl_provider": "nlf",
            "nlf_use_detector": False,
            "nlf_require_boxes": True,
            "num_smpl_queries": int(args.max_humans),
            "predict_id_embed": True,
            "id_feature_mode": "roi_query",
            "smpl_track_assignment_mode": "base_smpl",
            "smpl_use_external_track_prior": False,
            "smpl_track_assign_id_weight": float(args.id_weight),
            "smpl_track_assign_max_id_distance": float(args.max_id_distance),
        }
    )
    if args.smpl_model_dir:
        config.setdefault("assets", {})["smpl_model_dir"] = str(resolve_project_path(args.smpl_model_dir))
    return config


def validate_predictions(predictions: dict[str, torch.Tensor]) -> None:
    required = (
        "pose_enc",
        "depth",
        "pred_poses",
        "pred_betas",
        "pred_transl_cam",
        "pred_id_embed",
        "assigned_track_ids",
        "assigned_track_mask",
    )
    missing = [key for key in required if not isinstance(predictions.get(key), torch.Tensor)]
    if missing:
        raise RuntimeError(f"Missing ID-tracking viewer predictions: {missing}")


def build_summary(
    args: argparse.Namespace,
    checkpoint: Path,
    predictions: dict[str, torch.Tensor],
    scene: dict,
) -> dict:
    ids = predictions["assigned_track_ids"].detach().cpu()
    mask = predictions["assigned_track_mask"].detach().cpu().bool()
    quality = predictions.get("assigned_track_quality")
    return {
        "mode": "nlf_roi_id_tracking_v2",
        "frames_dir": str(resolve_project_path(args.frames_dir)),
        "id_checkpoint": str(checkpoint),
        "id_weight": float(args.id_weight),
        "max_id_distance": float(args.max_id_distance),
        "num_frames": int(ids.shape[1]),
        "unique_track_ids": sorted({int(v) for v in ids[mask].tolist()}),
        "track_ids_by_frame": [ids[0, s][mask[0, s]].tolist() for s in range(ids.shape[1])],
        "mean_track_quality": (
            float(quality.detach().float().cpu()[mask].mean())
            if isinstance(quality, torch.Tensor) and bool(mask.any())
            else None
        ),
        "people_counts": [len(frame["people"]) for frame in scene["frames"]],
        "output_dir": str(resolve_project_path(args.output_dir)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--id-checkpoint", required=True)
    parser.add_argument("--preprocessed-root", default="outputs/preprocess/bedlam_boxes")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_nlf_roi_id_tracking_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/nlf_roi_id_tracking_v2")
    parser.add_argument("--device", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--id-weight", type=float, default=0.10)
    parser.add_argument("--max-id-distance", type=float, default=2.0)
    parser.add_argument("--depth-point-stride", type=int, default=4)
    parser.add_argument("--max-scene-depth", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--camera-frustum-scale", type=float, default=0.20)
    parser.add_argument("--alignment-vertex-stride", type=int, default=16)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
