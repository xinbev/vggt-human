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
        tracks = tracker.update(
            frame_bgr=frame_bgr,
            detections=detections,
            frame_id=frame_id,
            frame_index=frame_index,
            video_name=source_name,
        )
        if mask_predictor is not None and tracks:
            masks, mask_meta = mask_predictor.predict_for_observations(frame_bgr, tracks)
            mask_path = output_root / "masks" / f"{frame_id}.npz"
            save_frame_masks(mask_path, masks)
            for obs in tracks:
                meta = mask_meta.get(int(obs.person_id))
                if meta is not None:
                    obs.mask = {
                        "format": "npz_bool",
                        "path": str(mask_path),
                        "array_key": f"person_{obs.person_id}",
                        **meta,
                    }

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
    summary.update(diagnostics.to_dict())
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
        return {
            "kind": "bedlam",
            "video": None,
            "frames_dir": seq_dir / "rgb",
            "source_name": rel.replace("/", "__"),
            "output_subdir": Path(args.bedlam_split) / rel,
            "bedlam": {
                "root": str(bedlam_root),
                "split": args.bedlam_split,
                "sequence": rel,
                "sequence_dir": str(seq_dir),
            },
        }

    if args.frames_dir:
        frames_dir = Path(args.frames_dir).expanduser()
        return {
            "kind": "frames",
            "video": None,
            "frames_dir": frames_dir,
            "source_name": frames_dir.name,
            "output_subdir": Path(frames_dir.name),
        }

    video = Path(args.video).expanduser()
    return {
        "kind": "video",
        "video": video,
        "frames_dir": None,
        "source_name": video.stem,
        "output_subdir": Path(video.stem),
    }


def resolve_config_path(
    config: dict[str, Any],
    override: str | None,
    dotted_key: str,
    allow_missing: bool = False,
) -> str:
    if override:
        return str(Path(override).expanduser())
    try:
        value = require_path(config, dotted_key)
    except ValueError:
        if allow_missing:
            return ""
        raise
    return str(Path(value).expanduser())


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
