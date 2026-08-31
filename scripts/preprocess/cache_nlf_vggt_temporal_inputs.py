#!/usr/bin/env python3
"""Cache reusable RGB->VGGT(camera)->NLF centre-frame predictions.

The cache intentionally contains no temporal-stabilizer output.  It stores
only raw NLF SMPL observations and model-side track assignment for every
unique 3DPW/EMDB video, allowing arbitrary later V2 checkpoints to be tested
without repeating VGGT or NLF inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint  # noqa: E402
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.integrations.nlf_smpl_provider import NLFSMPLProvider  # noqa: E402
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


WINDOW_SIZE = 9
CENTER_INDEX = WINDOW_SIZE // 2
FORMAT = "nlf_vggt_temporal_observation_cache_v1"


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but PyTorch cannot see a CUDA device")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    config = apply_overrides(config, args.override)
    configure_camera_only_model(config, args.checkpoint)
    camera_model = build_model(config).to(device)
    load_initial_checkpoint(camera_model, config, device)
    camera_model.eval()
    nlf = build_nlf_provider(config).to(device).eval()

    dataset = build_dataset(config, args)
    representative_indices = select_unique_video_records(dataset, args.sequence_filter)
    existing = {
        index
        for index in representative_indices
        if (cache_root / args.dataset / f"{safe_cache_name(video_name(dataset.records[index]))}.pt").is_file() and not args.overwrite
    }
    dataset._index = [item for item in dataset._index if item[0] in set(representative_indices) - existing]
    if not dataset._index and len(existing) != len(representative_indices):
        raise RuntimeError("No cache windows selected")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=hmr4d_eval_collate_fn,
        drop_last=False,
    )
    records_by_vid = {record.vid: record for record in dataset.records}
    remaining = Counter(dataset.records[record_index].vid for record_index, _, _ in dataset._index)
    buffers: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    processed = 0
    print_manifest(config, args, device, dataset, representative_indices, existing)
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            meta = batch["meta"]
            vid = str(meta["vid"][0])
            record = records_by_vid[vid]
            start = int(meta["start"][0])
            centre_index = start + CENTER_INDEX
            if vid not in buffers:
                buffers[vid] = allocate_buffer(record, config, batch)
            prediction = run_centre_nlf(camera_model, nlf, batch, config)
            write_centre_prediction(buffers[vid], centre_index, prediction, batch)
            remaining[vid] -= 1
            processed += 1
            if remaining[vid] == 0:
                path = cache_root / args.dataset / f"{safe_cache_name(video_name(record))}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = finalize_buffer(buffers.pop(vid), record, config, args)
                torch.save(payload, path)
                completed.append({"vid": record.vid, "vname": video_name(record), "path": str(path), "cached_centres": int(payload["center_valid"].sum())})
                print(f"[cache] saved vname={video_name(record)} centres={int(payload['center_valid'].sum())} -> {path}", flush=True)
            if args.log_interval > 0 and processed % int(args.log_interval) == 0:
                print(f"[cache] processed_windows={processed}/{len(dataset)} completed_videos={len(completed)}", flush=True)

    manifest = {
        "format": FORMAT,
        "dataset": args.dataset,
        "cache_root": str(cache_root),
        "vggt_checkpoint": str(args.checkpoint),
        "nlf_checkpoint": str(config.get("checkpoints", {}).get("nlf_smpl", "")),
        "window_size": WINDOW_SIZE,
        "centre_index": CENTER_INDEX,
        "sequence_filter": args.sequence_filter,
        "selected_unique_videos": len(representative_indices),
        "reused_existing_videos": len(existing),
        "newly_completed_videos": completed,
        "cache_contents": ["pred_pose_6d", "pred_betas", "pred_transl_cam", "pred_confs", "pred_boxes", "assigned_track_*", "center_valid"],
        "not_cached": ["V2 temporal refined pose", "GT SMPL", "HSI", "TRSTR"],
    }
    (cache_root / args.dataset / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["3dpw", "emdb1"])
    parser.add_argument("--checkpoint", required=True, help="VGGT checkpoint used only for runtime camera")
    parser.add_argument("--config", default="configs/eval_nlf_pose_stabilizer_v2.yaml")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--cache-root", default="outputs/preprocess/nlf_vggt_temporal_cache")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def configure_camera_only_model(config: dict[str, Any], checkpoint: str) -> None:
    model = config.setdefault("model", {})
    model.update(
        {
            "enable_camera": True,
            "enable_depth": False,
            "enable_smpl": False,
            "enable_hsi_refine": False,
            "enable_hsi_human_scene_align": False,
            "enable_hsi_translation_refine_v4": False,
            "enable_hsi_contact_refine": False,
            "enable_hsi_grounding": False,
            "enable_hsi_foot_contact_intent": False,
            "enable_hsi_trstr": False,
        }
    )
    config.setdefault("checkpoints", {})["vggt_baseline"] = str(checkpoint)
    config.setdefault("checkpoint", {})["load_vggt_baseline"] = True


def build_nlf_provider(config: dict[str, Any]) -> NLFSMPLProvider:
    model = config.get("model", {})
    return NLFSMPLProvider(
        model_path=str(model.get("nlf_model_path", config.get("checkpoints", {}).get("nlf_smpl", ""))),
        third_party_root=str(model.get("nlf_third_party_root", config.get("third_party", {}).get("nlf_root", "third_party/nlf"))),
        model_name=str(model.get("nlf_model_name", "smpl")),
        use_detector=True,
        require_boxes=False,
        internal_batch_size=int(model.get("nlf_internal_batch_size", 128)),
        num_aug=int(model.get("nlf_num_aug", 1)),
        detector_threshold=float(model.get("nlf_detector_threshold", 0.3)),
        detector_nms_iou_threshold=float(model.get("nlf_detector_nms_iou_threshold", 0.7)),
        max_detections=int(model.get("nlf_max_detections", 150)),
    )


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> HMR4DSupportEvalDataset:
    data = config.get("data", {})
    support_key = "datasets.threedpw_hmr4d_support_root" if args.dataset == "3dpw" else "datasets.emdb_hmr4d_support_root"
    support_root = args.support_root or require_path(config, support_key)
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    image_size, image_resolution = resolve_image_size_config(data)
    return HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=None,
        sequence_length=WINDOW_SIZE,
        stride=1,
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data.get("resize_mode", "balanced")),
        max_humans=int(data.get("max_humans", model_query_count(config))),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        full_sequence=False,
    )


def model_query_count(config: dict[str, Any]) -> int:
    return int(config.get("model", {}).get("num_smpl_queries", 20))


def video_name(record: Any) -> str:
    return str(record.label.get("vname", "") or record.vid.rsplit("_", 1)[0])


def safe_cache_name(value: str) -> str:
    return str(value).replace("/", "__").replace("\\", "__").replace(" ", "_")


def select_unique_video_records(dataset: HMR4DSupportEvalDataset, raw_filter: str) -> list[int]:
    query = str(raw_filter or "").lower().strip()
    selected: dict[str, int] = {}
    for index, record in enumerate(dataset.records):
        name = video_name(record)
        if query and query not in record.vid.lower() and query not in name.lower():
            continue
        selected.setdefault(name, index)
    if not selected:
        raise ValueError(f"sequence_filter={raw_filter!r} matched no unique videos")
    return list(selected.values())


def allocate_buffer(record: Any, config: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    length = int(record.length)
    queries = model_query_count(config)
    device = batch["images"].device
    return {
        "pred_pose_6d": torch.zeros(length, queries, 144, dtype=torch.float32),
        "pred_betas": torch.zeros(length, queries, 10, dtype=torch.float32),
        "pred_transl_cam": torch.zeros(length, queries, 3, dtype=torch.float32),
        "pred_confs": torch.zeros(length, queries, 1, dtype=torch.float32),
        "pred_boxes": torch.zeros(length, queries, 4, dtype=torch.float32),
        "center_valid": torch.zeros(length, dtype=torch.bool),
        "image_hw": torch.zeros(length, 2, dtype=torch.long),
        "orig_hw": torch.zeros(length, 2, dtype=torch.long),
        "device": device,
    }


def run_centre_nlf(camera_model: torch.nn.Module, nlf: NLFSMPLProvider, batch: dict[str, Any], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    camera = camera_model(batch["images"])
    pose_enc = camera.get("pose_enc")
    if not isinstance(pose_enc, torch.Tensor):
        raise KeyError("Camera-only VGGT output is missing pose_enc required by NLF")
    return nlf(
        images=batch["images"][:, CENTER_INDEX : CENTER_INDEX + 1],
        pose_enc=pose_enc[:, CENTER_INDEX : CENTER_INDEX + 1],
        smpl_query_boxes=None,
        smpl_query_boxes_mask=None,
        max_humans=model_query_count(config),
    )


def write_centre_prediction(buffer: dict[str, Any], centre_index: int, prediction: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
    if not 0 <= int(centre_index) < int(buffer["center_valid"].numel()):
        raise IndexError(f"centre index {centre_index} is outside cache length")
    for key in ("pred_pose_6d", "pred_betas", "pred_transl_cam", "pred_confs", "pred_boxes"):
        value = prediction.get(key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"NLF centre prediction missing {key}")
        buffer[key][centre_index] = value[0, 0].detach().float().cpu()
    buffer["center_valid"][centre_index] = bool((prediction["pred_confs"][0, 0, :, 0] > 0.0).any())
    buffer["image_hw"][centre_index] = batch["image_hw"][0, CENTER_INDEX].detach().cpu().long()
    buffer["orig_hw"][centre_index] = batch["orig_hw"][0, CENTER_INDEX].detach().cpu().long()


def finalize_buffer(buffer: dict[str, Any], record: Any, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    valid_queries = buffer["pred_confs"][..., 0] > 0.0
    track = BaseSMPLTrackAssigner(
        max_age=int(config.get("model", {}).get("smpl_track_assign_max_age", 90)),
        min_track_quality=float(config.get("model", {}).get("smpl_track_assign_min_quality", 0.25)),
        max_center_distance_norm=float(config.get("model", {}).get("smpl_track_assign_max_center_distance_norm", 0.25)),
        max_transl_distance_m=float(config.get("model", {}).get("smpl_track_assign_max_transl_distance_m", 1.50)),
        max_beta_l1=float(config.get("model", {}).get("smpl_track_assign_max_beta_l1", 0.30)),
        persistent=False,
    ).assign(
        boxes=buffer["pred_boxes"].unsqueeze(0),
        pred_betas=buffer["pred_betas"].unsqueeze(0),
        pred_transl_cam=buffer["pred_transl_cam"].unsqueeze(0),
        pred_confs=buffer["pred_confs"].unsqueeze(0),
        query_mask=valid_queries.unsqueeze(0),
    )
    payload = {key: value for key, value in buffer.items() if key != "device"}
    payload.update({key: value[0].cpu() for key, value in track.items()})
    payload.update(
        {
            "format": FORMAT,
            "dataset": args.dataset,
            "vid": record.vid,
            "vname": video_name(record),
            "length": int(record.length),
            "window_size": WINDOW_SIZE,
            "center_index": CENTER_INDEX,
            "vggt_checkpoint": str(args.checkpoint),
            "nlf_checkpoint": str(config.get("checkpoints", {}).get("nlf_smpl", "")),
            "frame_id": torch.as_tensor(record.label.get("frame_id", torch.arange(record.length)), dtype=torch.long).reshape(-1)[: int(record.length)],
        }
    )
    return payload


def print_manifest(config: dict[str, Any], args: argparse.Namespace, device: torch.device, dataset: HMR4DSupportEvalDataset, representatives: list[int], reused: set[int]) -> None:
    print("========== Reusable VGGT+NLF observation cache ==========")
    print(f"device: {device}")
    print(f"dataset / raw RGB root: {args.dataset} / {dataset.frames_root}")
    print(f"VGGT runtime camera checkpoint: {args.checkpoint}")
    print(f"NLF detector checkpoint: {config.get('checkpoints', {}).get('nlf_smpl', '<missing>')}")
    print("VGGT depth/HSI/TRSTR/SMPL heads: disabled for cache preprocessing")
    print("NLF runs centre frame only; V2 output is not cached")
    print(f"unique videos selected / already reusable: {len(representatives)} / {len(reused)}")


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


if __name__ == "__main__":
    main()
