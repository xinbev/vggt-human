#!/usr/bin/env python3
"""Materialize one BEDLAM2 scene in the project's ``BedlamDataset`` layout.

BEDLAM2 keeps RGB PNGs, depth EXRs, and SMPL labels in separate roots.  The
training loader deliberately consumes a self-contained per-frame layout, so
this adapter creates::

    <outdir>/Training/<scene>_<sequence>/{rgb,depth,cam,smpl}/<frame>.*

The source data is never changed.  Depth is saved as float32 metres in NPY
files; ``--depth-scale`` is therefore explicit instead of silently assuming
that every BEDLAM release uses centimetres.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_LABEL_KEYS = ("imgname", "smpl_pose_cam", "smpl_betas", "smpl_trans_cam", "cam_int", "cam_ext")


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels).expanduser()
    rgb_root = Path(args.rgb_root).expanduser()
    depth_root = Path(args.depth_root).expanduser()
    outdir = Path(args.outdir).expanduser()
    _require_dir(rgb_root / "png", "RGB png root")
    _require_dir(depth_root / "exr_depth", "depth EXR root")
    if not labels_path.is_file():
        raise FileNotFoundError(f"SMPL label NPZ not found: {labels_path}")
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError(f"--depth-scale must be finite and positive, got {args.depth_scale}")

    labels = np.load(labels_path, allow_pickle=False)
    missing = [key for key in REQUIRED_LABEL_KEYS if key not in labels]
    if missing:
        raise KeyError(f"BEDLAM2 label is missing required keys {missing}; available={sorted(labels.files)}")
    groups = group_label_rows(labels["imgname"])
    if args.sequence:
        groups = OrderedDict((name, rows) for name, rows in groups.items() if name.split("/", 1)[0] == args.sequence)
    if args.max_frames:
        groups = OrderedDict(list(groups.items())[: args.max_frames])
    if not groups:
        raise RuntimeError("No BEDLAM2 label frames selected. Check --sequence and imgname values.")

    inspection = inspect_samples(
        groups,
        rgb_root / "png",
        depth_root / "exr_depth",
        args.inspect_frames,
        exr_channel=args.exr_channel,
    )
    summary: dict[str, Any] = {
        "scene": args.scene,
        "labels": str(labels_path),
        "rgb_root": str(rgb_root),
        "depth_root": str(depth_root),
        "outdir": str(outdir),
        "split": args.split,
        "depth_scale_to_m": float(args.depth_scale),
        "translation_mode": args.translation_mode,
        "frame_count": len(groups),
        "person_count": sum(len(rows) for rows in groups.values()),
        "sample_inspection": inspection,
        "sequences": {},
    }
    print(json.dumps({key: summary[key] for key in summary if key != "sequences"}, indent=2, ensure_ascii=False))
    if args.inspect_only:
        return

    for rel_image, rows in groups.items():
        seq_name, image_name = split_image_name(rel_image)
        frame_stem = Path(image_name).stem
        rgb_source = rgb_root / "png" / rel_image
        depth_source = depth_root / "exr_depth" / PurePosixPath(rel_image).with_suffix(".exr")
        verify_frame_sources(rgb_source, depth_source)
        image_hw = read_image_hw(rgb_source)
        depth = read_exr_depth(depth_source, expected_hw=image_hw, preferred_channel=args.exr_channel)
        depth = sanitize_depth(depth, args.depth_scale)
        persons = build_persons(labels, rows, args.translation_mode)
        intrinsics, pose = frame_camera(labels, rows)

        seq_out = outdir / args.split / f"{args.scene}_{seq_name}"
        paths = {
            "rgb": seq_out / "rgb" / image_name,
            "depth": seq_out / "depth" / f"{frame_stem}.npy",
            "cam": seq_out / "cam" / f"{frame_stem}.npz",
            "smpl": seq_out / "smpl" / f"{frame_stem}.pkl",
        }
        if not args.dry_run:
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            materialize_rgb(rgb_source, paths["rgb"], args.copy_mode, args.overwrite)
            write_or_check_npy(paths["depth"], depth, args.overwrite)
            write_or_check_npz(paths["cam"], intrinsics, pose, args.overwrite)
            write_or_check_pickle(paths["smpl"], persons, args.overwrite)

        sequence_summary = summary["sequences"].setdefault(seq_name, {"frames": 0, "persons": 0})
        sequence_summary["frames"] += 1
        sequence_summary["persons"] += len(persons)

    if not args.dry_run:
        summary_path = outdir / "_preprocess_summaries" / f"{args.scene}_bedlam2_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"BEDLAM2 conversion complete. Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BEDLAM2 RGB/EXR/SMPL labels to BedlamDataset files")
    parser.add_argument("--rgb-root", required=True, help="Raw BEDLAM2 scene root containing png/")
    parser.add_argument("--depth-root", required=True, help="Raw BEDLAM2-depth scene root containing exr_depth/")
    parser.add_argument("--labels", required=True, help="Converted labels_smpl_6fps/<scene>.npz")
    parser.add_argument("--outdir", required=True, help="Destination BedlamDataset root")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--depth-scale", type=float, required=True, help="Multiplier converting raw EXR values to metres")
    parser.add_argument(
        "--exr-channel",
        default="",
        help="Optional exact EXR channel override. Defaults to auto-detecting WorldDepth, then Depth/Z.",
    )
    parser.add_argument("--translation-mode", choices=("add_cam_ext", "direct"), default="add_cam_ext")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--sequence", default="", help="Optional seq_XXXXXX filter for a small conversion")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap, useful for a smoke conversion")
    parser.add_argument("--inspect-frames", type=int, default=3)
    parser.add_argument("--inspect-only", action="store_true", help="Read and report sample EXRs without writing data")
    parser.add_argument("--dry-run", action="store_true", help="Validate every selected source frame without writing output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def group_label_rows(imgnames: np.ndarray) -> OrderedDict[str, list[int]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, raw_name in enumerate(imgnames):
        name = normalise_imgname(raw_name)
        groups.setdefault(name, []).append(index)
    return groups


def normalise_imgname(raw_name: Any) -> str:
    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    name = name.replace("\\", "/").lstrip("./")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or path.suffix.lower() != ".png":
        raise ValueError(f"Expected label imgname 'seq_xxxxxx/frame.png', got {name!r}")
    return path.as_posix()


def split_image_name(rel_image: str) -> tuple[str, str]:
    path = PurePosixPath(rel_image)
    return path.parts[0], path.name


def inspect_samples(
    groups: OrderedDict[str, list[int]],
    rgb_png_root: Path,
    exr_depth_root: Path,
    count: int,
    exr_channel: str = "",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rel_image in list(groups)[: max(1, count)]:
        rgb_path = rgb_png_root / rel_image
        exr_path = exr_depth_root / PurePosixPath(rel_image).with_suffix(".exr")
        verify_frame_sources(rgb_path, exr_path)
        image_hw = read_image_hw(rgb_path)
        raw_depth, channel = read_exr_depth(
            exr_path,
            expected_hw=image_hw,
            return_channel=True,
            preferred_channel=exr_channel,
        )
        finite_positive = raw_depth[np.isfinite(raw_depth) & (raw_depth > 0)]
        if finite_positive.size == 0:
            raise ValueError(f"No finite positive depth values in {exr_path}")
        results.append(
            {
                "imgname": rel_image,
                "rgb_hw": list(image_hw),
                "depth_hw": list(raw_depth.shape),
                "exr_channel": channel,
                "raw_depth_positive_min": float(finite_positive.min()),
                "raw_depth_positive_median": float(np.median(finite_positive)),
                "raw_depth_positive_max": float(finite_positive.max()),
                "candidate_median_if_cm": float(np.median(finite_positive) * 0.01),
                "candidate_median_if_m": float(np.median(finite_positive)),
            }
        )
    return results


def verify_frame_sources(rgb_path: Path, depth_path: Path) -> None:
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB/label pairing failed; RGB frame not found: {rgb_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"RGB/depth pairing failed; expected same-stem EXR not found: {depth_path}")


def read_image_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def read_exr_depth(
    path: Path,
    expected_hw: tuple[int, int],
    return_channel: bool = False,
    preferred_channel: str = "",
) -> np.ndarray | tuple[np.ndarray, str]:
    try:
        import OpenEXR
    except ImportError as exc:  # pragma: no cover - server-only dependency.
        raise ImportError("OpenEXR is required. Install it in the server environment before conversion.") from exc
    exr = OpenEXR.File(str(path))
    if not exr.parts:
        raise ValueError(f"EXR has no parts: {path}")
    channels = exr.parts[0].channels
    channel_name = select_depth_channel(channels, preferred_channel)
    depth = np.asarray(channels[channel_name].pixels, dtype=np.float32).squeeze()
    if depth.ndim == 1 and depth.size == expected_hw[0] * expected_hw[1]:
        depth = depth.reshape(expected_hw)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D EXR depth channel from {path}, got {depth.shape}")
    if tuple(depth.shape) != tuple(expected_hw):
        raise ValueError(
            f"RGB/depth raster sizes differ for {path}: RGB={expected_hw}, depth={tuple(depth.shape)}. "
            "Do not resize blindly; inspect BEDLAM2 depth-camera metadata first."
        )
    return (depth, channel_name) if return_channel else depth


def select_depth_channel(channels: Any, preferred_channel: str = "") -> str:
    """Choose the scalar scene-depth channel across BEDLAM EXR variants.

    BEDLAM2 Movie Render Queue files encode this as
    ``FinalImageMovieRenderQueue_WorldDepth`` rather than the legacy ``Depth``
    channel.  An exact override stays available for another renderer/export.
    """
    names = list(channels)
    if preferred_channel:
        if preferred_channel not in channels:
            raise KeyError(
                f"Requested EXR channel {preferred_channel!r} is unavailable; available={sorted(names)}"
            )
        return preferred_channel
    for name in names:
        if name.lower().endswith("worlddepth"):
            return name
    for name in names:
        lowered = name.lower()
        if "worlddepth" in lowered or "scene_depth" in lowered or "scenedepth" in lowered:
            return name
    for name in ("Depth", "Z", "depth", "z"):
        if name in channels:
            return name
    raise KeyError(
        "No recognised scalar depth channel. Expected a *WorldDepth*, Depth, or Z channel; "
        f"available={sorted(names)}"
    )


def sanitize_depth(raw_depth: np.ndarray, scale: float) -> np.ndarray:
    depth_m = raw_depth.astype(np.float32, copy=False) * np.float32(scale)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not bool(valid.any()):
        raise ValueError("Depth scale produced no finite positive depth values")
    return np.where(valid, depth_m, np.float32(0.0))


def build_persons(labels: Any, rows: list[int], translation_mode: str) -> list[dict[str, Any]]:
    persons: list[dict[str, Any]] = []
    for row in rows:
        pose = np.asarray(labels["smpl_pose_cam"][row], dtype=np.float32).reshape(-1)
        if pose.size < 66:
            raise ValueError(f"SMPL pose at row {row} has {pose.size} values; expected at least 66")
        transl = np.asarray(labels["smpl_trans_cam"][row], dtype=np.float32).reshape(3)
        if translation_mode == "add_cam_ext":
            cam_ext = np.asarray(labels["cam_ext"][row], dtype=np.float32)
            if cam_ext.shape != (4, 4):
                raise ValueError(f"cam_ext at row {row} has shape {cam_ext.shape}; expected (4, 4)")
            transl = transl + cam_ext[:3, 3]
        gender = "neutral"
        if "gender" in labels:
            raw_gender = labels["gender"][row]
            gender = raw_gender.decode("utf-8") if isinstance(raw_gender, bytes) else str(raw_gender)
            gender = gender.lower()
        persons.append(
            {
                # BedlamDataset keeps the historical smplx_* dictionary contract even
                # when the source has already been converted to SMPL.
                "smplx_root_pose": pose[:3].reshape(1, 3),
                "smplx_body_pose": pose[3:66].reshape(21, 3),
                "smplx_shape": np.asarray(labels["smpl_betas"][row], dtype=np.float32).reshape(-1)[:10],
                "smplx_gender": gender if gender in {"neutral", "male", "female"} else "neutral",
                "smplx_transl": transl.reshape(3),
            }
        )
    return persons


def frame_camera(labels: Any, rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
    first = rows[0]
    intrinsics = np.asarray(labels["cam_int"][first], dtype=np.float32)
    pose = np.asarray(labels["cam_ext"][first], dtype=np.float32)
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError(f"Invalid camera matrices at label row {first}: K={intrinsics.shape}, ext={pose.shape}")
    for row in rows[1:]:
        if not np.allclose(intrinsics, labels["cam_int"][row], rtol=1e-5, atol=1e-5):
            raise ValueError(f"People paired to one image have inconsistent cam_int at row {row}")
        if not np.allclose(pose, labels["cam_ext"][row], rtol=1e-5, atol=1e-5):
            raise ValueError(f"People paired to one image have inconsistent cam_ext at row {row}")
    return intrinsics, pose


def materialize_rgb(source: Path, destination: Path, copy_mode: str, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "symlink":
        destination.symlink_to(source)
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def write_or_check_npy(path: Path, value: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    np.save(path, value.astype(np.float32, copy=False))


def write_or_check_npz(path: Path, intrinsics: np.ndarray, pose: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    np.savez(path, intrinsics=intrinsics.astype(np.float32), pose=pose.astype(np.float32))


def write_or_check_pickle(path: Path, persons: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    with path.open("wb") as handle:
        pickle.dump(persons, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
