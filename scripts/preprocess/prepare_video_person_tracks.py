import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.tracking.boosttrack_adapter import BoostTrackPersonTracker
from vggt_omega.tracking.detectors import TorchScriptYOLOPersonDetector
from vggt_omega.tracking.diagnostics import TrackingDiagnostics
from vggt_omega.tracking.io import append_jsonl, iter_image_files, write_frame_sidecar, write_json
from vggt_omega.tracking.postprocess import postprocess_sidecar_tracks
from vggt_omega.tracking.sam2_masks import SAM2BoxMaskPredictor, save_frame_masks
from vggt_omega.tracking.schema import FrameObservations
from vggt_omega.training.config import load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.path_config)
    source = resolve_source(args, config)
    source_name = source["source_name"]
    output_root = Path(args.output_root).expanduser() / source["output_subdir"]
    output_root.mkdir(parents=True, exist_ok=True)
    observations_path = output_root / "observations.jsonl"
    if observations_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {observations_path}. Use --overwrite to replace it.")
    if observations_path.exists():
        observations_path.unlink()

    detector = build_detector(args, config)
    tracker = build_tracker(args, config)
    mask_predictor = build_mask_predictor(args, config) if args.enable_sam2_masks else None

    diagnostics = TrackingDiagnostics()
    summary: dict[str, Any] = {
        "source_name": source_name,
        "source_kind": source["kind"],
        "source_video": str(source["video"]) if source["video"] is not None else None,
        "source_frames_dir": str(source["frames_dir"]) if source["frames_dir"] is not None else None,
        "bedlam": source.get("bedlam"),
        "output_root": str(output_root),
        "detector": {
            "backend": "yolo_torchscript",
            "checkpoint": resolve_config_path(config, args.yolo_checkpoint, "checkpoints.yolo8x"),
            "image_size": int(args.detector_image_size),
            "conf_threshold": float(args.det_conf),
            "iou_threshold": float(args.det_iou),
        },
        "tracker": {
            "backend": "boosttrack",
            "root": resolve_config_path(config, args.boosttrack_root, "third_party.boosttrack_root"),
            "weights_root": resolve_config_path(config, args.boosttrack_weights_root, "third_party.boosttrack_weights_root"),
            "use_reid": not args.no_reid,
            "use_ecc": args.use_ecc,
            "max_age": args.max_age,
            "min_hits": args.min_hits,
        },
        "sam2_masks": {
            "enabled": bool(args.enable_sam2_masks),
            "root": resolve_config_path(config, args.sam2_root, "third_party.sam2_root", allow_missing=True),
            "checkpoint": resolve_config_path(config, args.sam2_checkpoint, "third_party.sam2_checkpoint", allow_missing=True),
        },
    }

    tracker.reset(video_name=source_name)
    frame_iter = iter_frames(args, source)
    for frame_index, frame_id, image_path, frame_bgr in frame_iter:
        detections = detector.detect(frame_bgr)
        for det_idx, det in enumerate(detections):
            det.det_id = det_idx
        if mask_predictor is not None and detections:
            masks, mask_meta = mask_predictor.predict_for_detections(frame_bgr, detections)
            mask_path = output_root / "masks" / f"{frame_id}.npz"
            save_frame_masks(mask_path, masks, prefix="det")
            for det in detections:
                meta = mask_meta.get(int(det.det_id))
                if meta is not None:
                    det.mask = {
                        "format": "npz_bool",
                        "path": str(mask_path),
                        "array_key": f"det_{int(det.det_id):06d}",
                        **meta,
                    }
        tracks = tracker.update(
            frame_bgr=frame_bgr,
            detections=detections,
            frame_id=frame_id,
            frame_index=frame_index,
            video_name=source_name,
        )

        height, width = frame_bgr.shape[:2]
        frame = FrameObservations(
            frame_id=frame_id,
            frame_index=frame_index,
            image_path=str(image_path),
            image_width=width,
            image_height=height,
            detections=detections,
            persons=tracks,
        )
        diagnostics.update(frame)
        write_frame_sidecar(frame, output_root / "smpl_boxes" / f"{frame_id}.pkl")
        append_jsonl(observations_path, [obs.to_dict() for obs in tracks])

        if args.frame_log_interval > 0 and (frame_index + 1) % args.frame_log_interval == 0:
            print(
                f"[{frame_index + 1}] detections={len(detections)} tracks={len(tracks)} "
                f"total_tracks={len(diagnostics.active_track_ids)}"
            )
        if args.max_frames is not None and frame_index + 1 >= args.max_frames:
            break

    tracker.dump_cache()
    if args.no_tracklet_stitch:
        summary.update(diagnostics.to_dict())
        summary["postprocess"] = {"tracklet_stitching": {"enabled": False}}
    else:
        post = postprocess_sidecar_tracks(
            output_root,
            max_gap=args.stitch_max_gap,
            center_thresh=args.stitch_center_thresh,
            size_log_thresh=args.stitch_size_log_thresh,
            min_score=args.stitch_min_score,
            compact_ids=not args.no_compact_track_ids,
        )
        summary.update(post["diagnostics"])
        summary["postprocess"] = {"tracklet_stitching": post["tracklet_stitching"]}
    write_json(output_root / "summary.json", summary)
    write_latest_pointer(args, source, output_root, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare video person bbox/track sidecars for VGGT-Omega SMPL queries")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None, help="Input video path")
    source.add_argument("--frames-dir", default=None, help="Directory of ordered image frames")
    source.add_argument("--bedlam-sequence", default=None, help="BEDLAM sequence name relative to <bedlam-root>/<split>")
    source.add_argument("--bedlam-sequence-index", type=int, default=None, help="BEDLAM sequence index under <bedlam-root>/<split>")
    parser.add_argument("--output-root", default="outputs/preprocess/video_tracks")
    parser.add_argument("--source-name", default="", help="Optional source name override for summaries/tracker tags")
    parser.add_argument("--output-subdir", default="", help="Optional output subdirectory under --output-root")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--bedlam-root", default=None)
    parser.add_argument("--bedlam-split", default="Training")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-id-width", type=int, default=6)
    parser.add_argument("--frame-log-interval", type=int, default=50)

    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.7)
    parser.add_argument("--det-half", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--boosttrack-root", default=None)
    parser.add_argument("--boosttrack-weights-root", default=None)
    parser.add_argument("--boosttrack-dataset", choices=("mot17", "mot20"), default="mot17")
    parser.add_argument("--boosttrack-test-dataset", action="store_true")
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--use-ecc", action="store_true")
    parser.add_argument("--no-auto-link-boosttrack-weights", action="store_true")
    parser.add_argument("--tracker-det-thresh", type=float, default=None)
    parser.add_argument("--tracker-iou-thresh", type=float, default=None)
    parser.add_argument("--max-age", type=int, default=90)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--min-box-area", type=float, default=10.0)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=10.0)
    parser.add_argument("--no-tracklet-stitch", action="store_true")
    parser.add_argument("--no-compact-track-ids", action="store_true")
    parser.add_argument("--stitch-max-gap", type=int, default=30)
    parser.add_argument("--stitch-center-thresh", type=float, default=1.25)
    parser.add_argument("--stitch-size-log-thresh", type=float, default=0.70)
    parser.add_argument("--stitch-min-score", type=float, default=0.25)

    parser.add_argument("--enable-sam2-masks", action="store_true")
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    return parser.parse_args()


def build_detector(args: argparse.Namespace, config: dict[str, Any]) -> TorchScriptYOLOPersonDetector:
    checkpoint = resolve_config_path(config, args.yolo_checkpoint, "checkpoints.yolo8x")
    return TorchScriptYOLOPersonDetector(
        checkpoint=checkpoint,
        device=args.device,
        image_size=args.detector_image_size,
        conf_threshold=args.det_conf,
        iou_threshold=args.det_iou,
        person_class_id=0,
        half=args.det_half,
    )


def build_tracker(args: argparse.Namespace, config: dict[str, Any]) -> BoostTrackPersonTracker:
    root = resolve_config_path(config, args.boosttrack_root, "third_party.boosttrack_root")
    weights_root = resolve_config_path(config, args.boosttrack_weights_root, "third_party.boosttrack_weights_root")
    return BoostTrackPersonTracker(
        boosttrack_root=root,
        weights_root=weights_root,
        dataset=args.boosttrack_dataset,
        test_dataset=args.boosttrack_test_dataset,
        use_reid=not args.no_reid,
        use_ecc=args.use_ecc,
        min_box_area=args.min_box_area,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
        det_thresh=args.tracker_det_thresh,
        iou_threshold=args.tracker_iou_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        auto_link_weights=not args.no_auto_link_boosttrack_weights,
    )


def build_mask_predictor(args: argparse.Namespace, config: dict[str, Any]) -> SAM2BoxMaskPredictor:
    return SAM2BoxMaskPredictor(
        sam2_root=resolve_config_path(config, args.sam2_root, "third_party.sam2_root"),
        checkpoint=resolve_config_path(config, args.sam2_checkpoint, "third_party.sam2_checkpoint"),
        model_cfg=args.sam2_model_cfg,
        device=args.device,
        multimask_output=not args.sam2_single_mask,
    )


def iter_frames(args: argparse.Namespace, source: dict[str, Any]) -> Iterator[tuple[int, str, Path, Any]]:
    if source["frames_dir"] is not None:
        files = iter_image_files(Path(source["frames_dir"]).expanduser())
        for frame_index, path in enumerate(files):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image frame: {path}")
            yield frame_index, path.stem, path, image
        return

    video_path = Path(source["video"]).expanduser()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id = f"{frame_index:0{args.frame_id_width}d}"
            yield frame_index, frame_id, video_path, frame
            frame_index += 1
    finally:
        cap.release()


def resolve_source(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.bedlam_sequence is not None or args.bedlam_sequence_index is not None:
        bedlam_root = Path(resolve_config_path(config, args.bedlam_root, "datasets.bedlam_root")).expanduser()
        split_dir = bedlam_root / args.bedlam_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")
        sequences = sorted(path for path in split_dir.iterdir() if (path / "rgb").is_dir())
        if not sequences:
            raise RuntimeError(f"No BEDLAM sequences with rgb/ found under {split_dir}")
        if args.bedlam_sequence_index is not None:
            if args.bedlam_sequence_index < 0 or args.bedlam_sequence_index >= len(sequences):
                raise IndexError(
                    f"--bedlam-sequence-index out of range: {args.bedlam_sequence_index}, valid=[0,{len(sequences) - 1}]"
                )
            seq_dir = sequences[args.bedlam_sequence_index]
        else:
            candidates = [path for path in sequences if str(path.relative_to(split_dir)).replace("\\", "/") == args.bedlam_sequence]
            if not candidates:
                available = [str(path.relative_to(split_dir)).replace("\\", "/") for path in sequences[:20]]
                raise FileNotFoundError(
                    f"BEDLAM sequence not found: {args.bedlam_sequence}. First available sequences: {available}"
                )
            seq_dir = candidates[0]
        rel = str(seq_dir.relative_to(split_dir)).replace("\\", "/")
        source_name = str(args.source_name or rel.replace("/", "__"))
        output_subdir = Path(args.output_subdir) if args.output_subdir else Path(args.bedlam_split) / rel
        _validate_relative_output_subdir(output_subdir)
        return {
            "kind": "bedlam",
            "video": None,
            "frames_dir": seq_dir / "rgb",
            "source_name": source_name,
            "output_subdir": output_subdir,
            "bedlam": {
                "root": str(bedlam_root),
                "split": args.bedlam_split,
                "sequence": rel,
                "sequence_dir": str(seq_dir),
            },
        }

    if args.frames_dir:
        frames_dir = Path(args.frames_dir).expanduser()
        source_name = str(args.source_name or frames_dir.name)
        output_subdir = Path(args.output_subdir) if args.output_subdir else Path(source_name)
        _validate_relative_output_subdir(output_subdir)
        return {
            "kind": "frames",
            "video": None,
            "frames_dir": frames_dir,
            "source_name": source_name,
            "output_subdir": output_subdir,
        }

    video = Path(args.video).expanduser()
    source_name = str(args.source_name or video.stem)
    output_subdir = Path(args.output_subdir) if args.output_subdir else Path(source_name)
    _validate_relative_output_subdir(output_subdir)
    return {
        "kind": "video",
        "video": video,
        "frames_dir": None,
        "source_name": source_name,
        "output_subdir": output_subdir,
    }


def _validate_relative_output_subdir(path: Path) -> None:
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"--output-subdir must be relative and stay under --output-root, got: {path}")


def resolve_config_path(
    config: dict[str, Any],
    override: str | None,
    dotted_key: str,
    allow_missing: bool = False,
) -> str:
    if override:
        return str(resolve_project_path(override))
    try:
        value = require_path(config, dotted_key)
    except ValueError:
        if allow_missing:
            return ""
        raise
    return str(resolve_project_path(value))


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


def write_latest_pointer(args: argparse.Namespace, source: dict[str, Any], output_root: Path, summary: dict[str, Any]) -> None:
    pointer = {
        "output_root": str(output_root),
        "source_kind": source["kind"],
        "source_name": source["source_name"],
        "summary_json": str(output_root / "summary.json"),
        "num_frames": int(summary.get("total_frames", 0)),
    }
    if source["kind"] == "bedlam":
        pointer["bedlam"] = source.get("bedlam", {})
    write_json(Path(args.output_root).expanduser() / "latest_tracking.json", pointer)
    if source["kind"] == "bedlam":
        write_json(Path(args.output_root).expanduser() / "latest_bedlam_tracking.json", pointer)


if __name__ == "__main__":
    main()
