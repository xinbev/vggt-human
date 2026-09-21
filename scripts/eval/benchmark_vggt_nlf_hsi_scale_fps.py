#!/usr/bin/env python3
"""Benchmark steady-state FPS for VGGT+NLF and VGGT+NLF+HSI scale.

The timed region excludes configuration/checkpoint loading, image decode and
resize, host-to-device copies, lazy NLF loading, warm-up, and result writing.
It includes every operation needed after a preprocessed CUDA image tensor is
available, including NLF detection and the analytic coarse-scale computation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import build_model, make_state_dict_loadable  # noqa: E402
from scripts.vis.visualize_smpl_inference import estimate_scene_to_smpl_scale  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.tracking.io import iter_image_files  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402


BASE_SMPL_KEYS = (
    "pred_pose_6d",
    "pred_poses",
    "pred_betas",
    "pred_transl_cam",
    "pred_confs",
    "pred_boxes",
    "pred_cam",
    "base_pred_transl_cam",
    "nlf_valid_mask",
    "nlf_intrinsics",
    "nlf_image_hw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--path-config", type=Path, default=Path("configs/path.yaml"))
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path("benchmarks/emdb2_global/inference_config.yaml"),
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--scale-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/vggt_nlf_hsi_scale_fps"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--coarse-scale-min", type=float, default=0.10)
    parser.add_argument("--coarse-scale-max", type=float, default=25.0)
    parser.add_argument("--coarse-anchor-stride", type=int, default=8)
    parser.add_argument("--coarse-min-anchor-pixels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths and frame selection without loading either model.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_output_dir(path: Path) -> Path:
    resolved = resolve(path)
    output_root = (ROOT / "outputs").resolve()
    if not resolved.is_relative_to(output_root):
        raise ValueError(f"Benchmark output must remain under {output_root}, got {resolved}")
    return resolved


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("This benchmark requires a CUDA device for synchronized GPU timing")
    for name in ("num_frames", "frame_stride", "max_humans", "repeats"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.start_index) < 0 or int(args.warmup) < 0:
        raise ValueError("--start-index and --warmup must be non-negative")
    if not 0.0 <= float(args.conf_threshold) <= 1.0:
        raise ValueError("--conf-threshold must be in [0, 1]")
    if not 0 < float(args.coarse_scale_min) <= float(args.coarse_scale_max):
        raise ValueError("coarse-scale bounds must be positive and increasing")


def selected_frames(args: argparse.Namespace) -> list[Path]:
    frame_dir = resolve(args.frames_dir)
    paths = iter_image_files(frame_dir)
    start = int(args.start_index)
    stop = start + int(args.num_frames) * int(args.frame_stride)
    selected = paths[start:stop:int(args.frame_stride)]
    if len(selected) != int(args.num_frames):
        raise ValueError(
            f"Requested {args.num_frames} frames from {frame_dir} with start={start}, "
            f"stride={args.frame_stride}, but selected only {len(selected)} from {len(paths)} files"
        )
    return selected


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(
        load_yaml_config(str(resolve(args.path_config))),
        load_yaml_config(str(resolve(args.inference_config))),
    )
    data_cfg = config.setdefault("data", {})
    data_cfg["image_resolution"] = int(args.image_resolution)
    data_cfg["image_size"] = int(args.image_resolution)
    data_cfg["resize_mode"] = str(args.resize_mode)
    data_cfg["max_humans"] = int(args.max_humans)

    model_cfg = config.setdefault("model", {})
    model_cfg.update(
        {
            "enable_camera": True,
            "enable_depth": True,
            "enable_smpl": True,
            "smpl_provider": "nlf",
            "nlf_use_detector": True,
            "nlf_require_boxes": False,
            "num_smpl_queries": int(args.max_humans),
            "smpl_use_aggregator_queries": False,
            "smpl_query_box_prior": False,
            "smpl_query_patch_pool": False,
            "smpl_track_assignment_mode": "base_smpl",
            "smpl_use_external_track_prior": False,
            "hsi_enable_temporal_momentum": False,
            "hsi_scene_affine_mode": "per_frame",
            "enable_hsi_human_scene_align": False,
            "enable_hsi_translation_refine_v4": False,
            "enable_hsi_contact_refine": False,
            "enable_hsi_foot_contact_intent": False,
            "enable_hsi_grounding": False,
            "enable_hsi_trstr": False,
        }
    )
    config.setdefault("loss", {})["type"] = "hungarian"
    return config


def checkpoint_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.baseline_checkpoint is not None:
        return resolve(args.baseline_checkpoint)
    return resolve(Path(require_path(config, "checkpoints.vggt_baseline", allow_empty=False)))


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    state = {
        str(key).removeprefix("module."): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError("Checkpoint contains no tensor weights")
    return state


def load_baseline(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    state = extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    state, report = make_state_dict_loadable(state, model.state_dict(), adapt_query_tensors=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "path": str(path),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_skipped": len(report["skipped"]),
    }


def load_hsi_scale_overlay(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    prefix = "hsi_refinement_head."
    state = extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    expected = {key: value for key, value in model.state_dict().items() if key.startswith(prefix)}
    if not expected:
        raise RuntimeError("HSI model contains no hsi_refinement_head parameters")
    missing = [key for key in expected if key not in state]
    mismatched = [
        key for key, value in expected.items() if key in state and tuple(state[key].shape) != tuple(value.shape)
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"Incomplete HSI scale checkpoint {path}: missing={missing[:12]} mismatched={mismatched[:12]}"
        )
    selected = {key: state[key] for key in expected}
    _, unexpected = model.load_state_dict(selected, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected HSI scale overlay keys: {unexpected[:12]}")
    return {"path": str(path), "loaded_tensors": len(selected), "prefix": prefix}


def build_runtime_model(
    base_config: dict[str, Any],
    enable_hsi_scale: bool,
    baseline: Path,
    scale_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})["enable_hsi_refine"] = bool(enable_hsi_scale)
    model = build_model(config)
    audit: dict[str, Any] = {"baseline": load_baseline(model, baseline)}
    if enable_hsi_scale:
        audit["hsi_scale"] = load_hsi_scale_overlay(model, scale_checkpoint)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, audit


def canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Expected depth [B,S,H,W] or [B,S,H,W,1], got {tuple(depth.shape)}")


@contextmanager
def hsi_scale_disabled(model: torch.nn.Module):
    head = getattr(model, "hsi_refinement_head", None)
    model.hsi_refinement_head = None
    try:
        yield
    finally:
        model.hsi_refinement_head = head


def decode_base_vertices(
    predictions: dict[str, torch.Tensor], hsi_head: torch.nn.Module
) -> torch.Tensor:
    poses = predictions["pred_poses"].float()
    betas = predictions["pred_betas"].float()
    translations = predictions["pred_transl_cam"].float()
    batch_size, num_frames, num_queries = poses.shape[:3]
    vertices, _ = hsi_head.smpl(
        poses.reshape(batch_size * num_frames * num_queries, 72),
        betas.reshape(batch_size * num_frames * num_queries, -1),
    )
    vertices = vertices.reshape(batch_size, num_frames, num_queries, vertices.shape[-2], 3)
    return vertices + translations[..., None, :]


def analytic_coarse_scale(
    predictions: dict[str, torch.Tensor],
    hsi_head: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    depth = canonical_depth(predictions["depth"]).detach().float()
    if depth.shape[0] != 1:
        raise ValueError(f"Benchmark currently expects batch size 1, got {depth.shape[0]}")
    vertices = decode_base_vertices(predictions, hsi_head)
    confidences = predictions["pred_confs"].detach().float()[..., 0]
    scales: list[float | None] = []
    valid_frames = 0
    for frame_index in range(depth.shape[1]):
        person_valid = confidences[0, frame_index] >= float(args.conf_threshold)
        if not bool(person_valid.any()):
            scales.append(None)
            continue
        result = estimate_scene_to_smpl_scale(
            smpl_vertices=vertices[0, frame_index, person_valid],
            depth=depth[0, frame_index],
            pose_enc=predictions["pose_enc"][:, frame_index : frame_index + 1],
            input_size=max(int(depth.shape[-2]), int(depth.shape[-1])),
            min_anchor_pixels=int(args.coarse_min_anchor_pixels),
            scale_min=float(args.coarse_scale_min),
            scale_max=float(args.coarse_scale_max),
            anchor_stride=int(args.coarse_anchor_stride),
        )
        if bool(result.get("applied", False)) and float(result["scale"]) > 0:
            scales.append(float(result["scale"]))
            valid_frames += 1
        else:
            scales.append(None)

    valid_scales = [value for value in scales if value is not None]
    if valid_scales:
        fallback = float(np.exp(np.median(np.log(np.asarray(valid_scales, dtype=np.float64)))))
    else:
        fallback = 1.0
    filled = [fallback if value is None else value for value in scales]
    scale_tensor = depth.new_tensor(filled).reshape(1, depth.shape[1])
    return scale_tensor, {
        "valid_frames": int(valid_frames),
        "fallback_frames": int(depth.shape[1] - valid_frames),
        "all_frames_unit_fallback": not bool(valid_scales),
    }


def base_forward(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    return model(images)


def hsi_scale_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hsi_head = getattr(model, "hsi_refinement_head", None)
    if hsi_head is None:
        raise RuntimeError("HSI scale benchmark requires hsi_refinement_head")

    # The first pass provides the actual VGGT depth/camera and NLF humans. HSI
    # is disabled here because its residual scale is defined on coarse metric
    # depth, which is constructed immediately afterwards.
    with hsi_scale_disabled(model):
        first = model(images)
    coarse_scale, coarse_stats = analytic_coarse_scale(first, hsi_head, args)
    raw_depth = canonical_depth(first["depth"]).detach().float()
    coarse_depth = raw_depth * coarse_scale[..., None, None]
    smpl_override = {key: first[key] for key in BASE_SMPL_KEYS if key in first}

    # The second VGGT pass is required by the current model interface to expose
    # the intermediate scene tokens to HSI. Reusing the first-pass NLF tensors
    # prevents a redundant second NLF detector/estimator invocation.
    refined = model(
        images,
        smpl_override_outputs=smpl_override,
        hsi_depth_override=coarse_depth,
        hsi_depth_is_metric=True,
        hsi_geometry_mode="smpl_coarse_metric",
    )
    residual_scale = refined["hsi_scene_scale"].float()
    residual_bias = refined["hsi_scene_depth_bias"].float()
    coarse_scale_out = coarse_scale[..., None].to(residual_scale)
    effective_scale = coarse_scale_out * residual_scale
    refined["hsi_residual_scene_scale"] = residual_scale
    refined["hsi_coarse_scene_scale"] = coarse_scale_out
    refined["hsi_scene_scale"] = effective_scale
    refined["hsi_translation_depth"] = raw_depth * effective_scale[..., None] + residual_bias[..., None]
    refined["_benchmark_coarse_stats"] = coarse_stats
    return refined


def validate_output(name: str, output: dict[str, Any], num_frames: int) -> dict[str, Any]:
    required = ["depth", "pose_enc", "pred_poses", "pred_betas", "pred_transl_cam", "pred_confs"]
    if name == "vggt_nlf_hsi_scale":
        required.extend(["hsi_scene_scale", "hsi_scene_depth_bias", "hsi_translation_depth"])
    missing = [key for key in required if not isinstance(output.get(key), torch.Tensor)]
    if missing:
        raise RuntimeError(f"{name} did not produce required tensor outputs: {missing}")
    if int(output["depth"].shape[1]) != int(num_frames):
        raise RuntimeError(f"{name} returned {output['depth'].shape[1]} frames, expected {num_frames}")
    result: dict[str, Any] = {
        "depth_shape": list(output["depth"].shape),
        "pose_shape": list(output["pred_poses"].shape),
        "device": str(output["depth"].device),
        "dtype": str(output["depth"].dtype),
    }
    if "_benchmark_coarse_stats" in output:
        result["coarse_scale"] = output["_benchmark_coarse_stats"]
    return result


def synchronized_benchmark(
    name: str,
    callback: Callable[[], dict[str, Any]],
    num_frames: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output: dict[str, Any] | None = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = callback()
        torch.cuda.synchronize(device)
        if output is None:
            output = callback()
            torch.cuda.synchronize(device)
    validation = validate_output(name, output, num_frames)
    del output

    torch.cuda.reset_peak_memory_stats(device)
    latencies_s: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = callback()
            torch.cuda.synchronize(device)
            latencies_s.append(time.perf_counter() - started)
            del output

    latency_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    mean_s = float(np.mean(latencies_s))
    median_s = float(np.median(latencies_s))
    result = {
        "name": name,
        "timing": "synchronized_wall_clock",
        "warmup_iterations_excluded": int(warmup),
        "timed_iterations": int(repeats),
        "frames_per_iteration": int(num_frames),
        "latency_ms_mean": mean_s * 1000.0,
        "latency_ms_median": median_s * 1000.0,
        "latency_ms_std": float(np.std(latency_ms)),
        "latency_ms_p10": float(np.percentile(latency_ms, 10)),
        "latency_ms_p90": float(np.percentile(latency_ms, 90)),
        "fps_from_mean_latency": float(num_frames / mean_s),
        "fps_from_median_latency": float(num_frames / median_s),
        "peak_memory_allocated_gb": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "peak_memory_reserved_gb": float(torch.cuda.max_memory_reserved(device) / 2**30),
        "latency_ms_each": latency_ms.tolist(),
    }
    return result, validation


def release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def write_results(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fps_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = summary["results"]
    fieldnames = [
        "name",
        "frames_per_iteration",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_std",
        "latency_ms_p90",
        "fps_from_mean_latency",
        "fps_from_median_latency",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
    ]
    with (output_dir / "fps_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def print_results(summary: dict[str, Any]) -> None:
    print("\nSteady-state synchronized wall-clock results", flush=True)
    print("pipeline                         latency mean (ms)    FPS       peak alloc (GB)", flush=True)
    for row in summary["results"]:
        print(
            f"{row['name']:<32} {row['latency_ms_mean']:>12.3f} "
            f"{row['fps_from_mean_latency']:>10.3f} {row['peak_memory_allocated_gb']:>15.3f}",
            flush=True,
        )
    comparison = summary["comparison"]
    print(
        f"HSI scale latency overhead: {comparison['latency_overhead_percent']:.2f}% | "
        f"FPS ratio vs base: {comparison['fps_ratio_hsi_over_base']:.4f}",
        flush=True,
    )
    print(f"JSON: {summary['outputs']['json']}", flush=True)
    print(f"CSV : {summary['outputs']['csv']}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = require_output_dir(args.output_dir)
    frames = selected_frames(args)
    config = resolved_config(args)
    baseline = checkpoint_path(args, config)
    scale_checkpoint = resolve(args.scale_checkpoint)
    required_files = [resolve(args.path_config), resolve(args.inference_config), baseline, scale_checkpoint]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark resources:\n" + "\n".join(missing))

    preflight = {
        "frames_dir": str(resolve(args.frames_dir)),
        "selected_frame_count": len(frames),
        "first_frame": str(frames[0]),
        "last_frame": str(frames[-1]),
        "baseline_checkpoint": str(baseline),
        "scale_checkpoint": str(scale_checkpoint),
        "output_dir": str(output_dir),
        "device": str(args.device),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback is allowed for the FPS benchmark")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    # Decode, resize, stack, and transfer once. None of this is in a timed region.
    images = load_and_preprocess_images(
        [str(path) for path in frames],
        mode=str(args.resize_mode),
        image_resolution=int(args.image_resolution),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
    ).unsqueeze(0).to(device)
    torch.cuda.synchronize(device)

    print(f"[input] tensor={tuple(images.shape)} dtype={images.dtype} device={images.device}", flush=True)
    print("[timing] excluded: model/config/checkpoint load, image I/O/resize/H2D, NLF lazy load, warm-up, writes", flush=True)
    print("[timing] included: synchronized model compute, NLF detector, SMPL coarse scale, HSI residual scale/bias", flush=True)

    base_model, base_audit = build_runtime_model(
        config, False, baseline, scale_checkpoint, device
    )
    base_result, base_validation = synchronized_benchmark(
        name="vggt_nlf",
        callback=lambda: base_forward(base_model, images),
        num_frames=len(frames),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        device=device,
    )
    del base_model
    release_cuda_cache()

    hsi_model, hsi_audit = build_runtime_model(
        config, True, baseline, scale_checkpoint, device
    )
    hsi_result, hsi_validation = synchronized_benchmark(
        name="vggt_nlf_hsi_scale",
        callback=lambda: hsi_scale_forward(hsi_model, images, args),
        num_frames=len(frames),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        device=device,
    )
    del hsi_model
    release_cuda_cache()

    latency_ratio = hsi_result["latency_ms_mean"] / base_result["latency_ms_mean"]
    fps_ratio = hsi_result["fps_from_mean_latency"] / base_result["fps_from_mean_latency"]
    output_json = output_dir / "fps_summary.json"
    output_csv = output_dir / "fps_summary.csv"
    summary = {
        "benchmark": "VGGT+NLF vs VGGT+NLF+HSI scale steady-state inference",
        "protocol": {
            "timing": "wall clock with torch.cuda.synchronize before and after every timed iteration",
            "fps_definition": "num_input_frames / mean synchronized latency",
            "excluded": [
                "model construction and checkpoint loading",
                "image file read, decode, resize, stacking, and host-to-device transfer",
                "lazy NLF TorchScript loading",
                "warm-up iterations",
                "validation and output serialization",
            ],
            "base_included": "one VGGT forward plus NLF detector/SMPL estimation",
            "hsi_included": (
                "base VGGT+NLF pass, SMPL surface decode, analytic coarse scale, "
                "second VGGT+HSI pass with first-pass NLF tensors reused, residual scale/bias composition"
            ),
            "explicitly_disabled": [
                "TRSTR",
                "legacy HSI translation alignment",
                "contact refinement",
                "grounding",
                "temporal HSI momentum",
                "visualization and metric evaluation",
            ],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
        },
        "input": {
            **preflight,
            "tensor_shape": list(images.shape),
            "tensor_dtype": str(images.dtype),
            "image_resolution": int(args.image_resolution),
            "resize_mode": str(args.resize_mode),
            "start_index": int(args.start_index),
            "frame_stride": int(args.frame_stride),
            "max_humans": int(args.max_humans),
        },
        "settings": {
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "conf_threshold": float(args.conf_threshold),
            "coarse_scale_range": [float(args.coarse_scale_min), float(args.coarse_scale_max)],
            "coarse_anchor_stride": int(args.coarse_anchor_stride),
            "coarse_min_anchor_pixels": int(args.coarse_min_anchor_pixels),
        },
        "checkpoint_audit": {"vggt_nlf": base_audit, "vggt_nlf_hsi_scale": hsi_audit},
        "validation": {"vggt_nlf": base_validation, "vggt_nlf_hsi_scale": hsi_validation},
        "results": [base_result, hsi_result],
        "comparison": {
            "latency_ratio_hsi_over_base": float(latency_ratio),
            "latency_overhead_percent": float((latency_ratio - 1.0) * 100.0),
            "fps_ratio_hsi_over_base": float(fps_ratio),
        },
        "outputs": {"json": str(output_json), "csv": str(output_csv)},
    }
    write_results(output_dir, summary)
    print_results(summary)


if __name__ == "__main__":
    main()
