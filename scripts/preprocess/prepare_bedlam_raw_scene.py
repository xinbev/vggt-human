#!/usr/bin/env python3
"""Prepare one raw BEDLAM scene into this project's processed layout.

This is a project-local, more tolerant adaptation of the Human3R/CUT3R BEDLAM
preprocessing path. HF BEDLAM folders and label npz files do not always use the
same relative image-name convention, so annotation matching normalizes variants
such as ``seq/file.png``, ``png/seq/file.png`` and ``scene/png/seq/file.png``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import OpenEXR
except ImportError as exc:  # pragma: no cover - depends on server env.
    raise ImportError("Install OpenEXR in the active environment: pip install OpenEXR") from exc


SENSOR_W = 36
SENSOR_H = 20.25
IMG_W = 1280
IMG_H = 720
TEST_LIST = {
    "20221018_1_250_batch01hand_zoom_suburb_b",
    "20221018_3_250_batch01hand_orbit_archVizUI3_time15",
    "20221019_3-8_250_highbmihand_orbit_stadium",
}


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser()
    scene_root = raw_root / args.scene
    outdir = Path(args.outdir).expanduser()
    annot_dir = Path(args.annot_dir).expanduser()
    scene = args.scene

    be_seq_csv = find_be_seq_csv(scene_root, scene)
    camera_dir = find_camera_dir(scene_root, scene)
    annot_path = find_annotation_npz(annot_dir, scene)
    image_root = scene_root / "png"
    depth_root = scene_root / "depth"
    mask_root = scene_root / "masks"
    for path, label in [(image_root, "image root"), (depth_root, "depth root")]:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")

    annot = np.load(annot_path, allow_pickle=True)
    index = build_annotation_index(annot["imgname"])
    csv_data = pd.read_csv(be_seq_csv).to_dict("list")

    summary: dict[str, Any] = {
        "scene": scene,
        "raw_root": str(raw_root),
        "outdir": str(outdir),
        "annotation": str(annot_path),
        "be_seq_csv": str(be_seq_csv),
        "camera_dir": str(camera_dir),
        "sequences": {},
        "total_frames": 0,
        "matched_frames": 0,
        "total_persons": 0,
        "max_humans": 0,
    }

    sequence_names = collect_sequence_names(csv_data)
    image_dirs = discover_image_dirs(image_root)
    depth_dirs = discover_named_dirs(depth_root)
    mask_dirs = discover_named_dirs(mask_root) if mask_root.is_dir() else {}
    camera_csvs = discover_camera_csvs(camera_dir)
    discovered_names = sorted((set(sequence_names) | set(image_dirs) | set(camera_csvs)))
    summary["sequence_name_count_csv"] = len(sequence_names)
    summary["sequence_name_count_image_dirs"] = len(image_dirs)
    summary["sequence_name_count_camera_csvs"] = len(camera_csvs)
    summary["missing_image_examples"] = []
    summary["missing_camera_examples"] = []
    for seq_name in tqdm(discovered_names, desc="Processing sequences"):
        seq_summary = process_sequence(
            scene=scene,
            seq_name=seq_name,
            outdir=outdir,
            image_dir=image_dirs.get(seq_name),
            depth_dir=depth_dirs.get(seq_name),
            mask_dir=mask_dirs.get(seq_name),
            cam_csv=camera_csvs.get(seq_name),
            annot=annot,
            annot_index=index,
            frame_stride=int(args.frame_stride),
            overwrite=bool(args.overwrite),
        )
        if seq_summary["frames"] == 0:
            if seq_summary.get("missing_image") and len(summary["missing_image_examples"]) < 20:
                summary["missing_image_examples"].append(seq_name)
            if seq_summary.get("missing_camera") and len(summary["missing_camera_examples"]) < 20:
                summary["missing_camera_examples"].append(seq_name)
            continue
        summary["sequences"][seq_name] = seq_summary
        summary["total_frames"] += seq_summary["frames"]
        summary["matched_frames"] += seq_summary["matched_frames"]
        summary["total_persons"] += seq_summary["persons"]
        summary["max_humans"] = max(summary["max_humans"], seq_summary["max_humans"])

    summary_path = outdir / "_preprocess_summaries" / f"{scene}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["total_frames"] <= 0:
        raise RuntimeError("No frames were processed. Check png/camera/sequence paths.")
    if summary["matched_frames"] <= 0:
        raise RuntimeError("No frames matched BEDLAM annotations. Check annotation imgname convention.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one raw BEDLAM scene for VGGT-Human")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--annot-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_be_seq_csv(scene_root: Path, scene: str) -> Path:
    candidates = [
        scene_root / "be_seq.csv",
        scene_root / "ground_truth" / scene / "be_seq.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not locate be_seq.csv under {scene_root}")


def find_camera_dir(scene_root: Path, scene: str) -> Path:
    candidates = [
        scene_root / "ground_truth" / "camera",
        scene_root / "ground_truth" / scene / "ground_truth" / "camera",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not locate ground_truth/camera under {scene_root}")


def find_annotation_npz(annot_dir: Path, scene: str) -> Path:
    for suffix in ("_6fps.npz", "_30fps.npz"):
        path = annot_dir / f"{scene}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not locate {scene}_6fps.npz or {scene}_30fps.npz under {annot_dir}")


def collect_sequence_names(csv_data: dict[str, list[Any]]) -> list[str]:
    out: list[str] = []
    for comment in csv_data.get("Comment", []):
        text = str(comment)
        if "sequence_name" not in text:
            continue
        out.append(text.split(";")[0].split("=")[-1])
    return out


def discover_image_dirs(image_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not image_root.is_dir():
        return out
    for png_path in sorted(image_root.rglob("*.png")):
        seq_dir = png_path.parent
        seq_name = seq_dir.name
        out.setdefault(seq_name, seq_dir)
    return out


def discover_named_dirs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        out.setdefault(path.name, path)
    return out


def discover_camera_csvs(camera_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not camera_dir.is_dir():
        return out
    for csv_path in sorted(camera_dir.rglob("*_camera.csv")):
        seq_name = csv_path.name.removesuffix("_camera.csv")
        out.setdefault(seq_name, csv_path)
    return out


def build_annotation_index(imgnames: np.ndarray) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for idx, raw in enumerate(imgnames):
        for key in annotation_name_variants(raw):
            index.setdefault(key, []).append(int(idx))
    return index


def annotation_name_variants(raw: Any) -> set[str]:
    if isinstance(raw, bytes):
        name = raw.decode("utf-8")
    else:
        name = str(raw)
    name = name.replace("\\", "/").lstrip("./")
    parts = [part for part in name.split("/") if part]
    variants = {name}
    if len(parts) >= 2:
        variants.add("/".join(parts[-2:]))
    if "png" in parts:
        png_idx = parts.index("png")
        if png_idx + 2 <= len(parts):
            variants.add("/".join(parts[png_idx + 1 :]))
    if parts:
        variants.add(parts[-1])
    return variants


def process_sequence(
    scene: str,
    seq_name: str,
    outdir: Path,
    image_dir: Path | None,
    depth_dir: Path | None,
    mask_dir: Path | None,
    cam_csv: Path | None,
    annot: Any,
    annot_index: dict[str, list[int]],
    frame_stride: int,
    overwrite: bool,
) -> dict[str, Any]:
    if image_dir is None or not image_dir.is_dir():
        return {"frames": 0, "matched_frames": 0, "persons": 0, "max_humans": 0, "missing_image": True}
    if cam_csv is None or not cam_csv.is_file():
        return {"frames": 0, "matched_frames": 0, "persons": 0, "max_humans": 0, "missing_camera": True}

    split = "Test" if scene in TEST_LIST else "Training"
    seq_out = outdir / split / f"{scene}_{seq_name}"
    rgb_out = seq_out / "rgb"
    depth_out = seq_out / "depth"
    cam_out = seq_out / "cam"
    smpl_out = seq_out / "smpl"
    mask_out = seq_out / "mask"
    for path in (rgb_out, depth_out, cam_out, smpl_out, mask_out):
        path.mkdir(parents=True, exist_ok=True)

    cam_csv_data = pd.read_csv(cam_csv).to_dict("list")
    images = sorted(image_dir.glob("*.png"))
    frames = images[:: max(int(frame_stride), 1)]

    stats = {"frames": 0, "matched_frames": 0, "persons": 0, "max_humans": 0}
    for frame_ord, image_path in enumerate(frames):
        cam_ind = frame_ord * max(int(frame_stride), 1)
        if cam_ind >= len(cam_csv_data.get("x", [])):
            continue
        frame_stem = image_path.stem
        rel_name = f"{seq_name}/{image_path.name}"
        indices = resolve_annotation_indices(rel_name, image_path.name, annot_index)
        persons = build_persons(annot, indices)

        out_rgb = rgb_out / image_path.name
        out_depth = depth_out / f"{frame_stem}.npy"
        out_cam = cam_out / f"{frame_stem}.npz"
        out_smpl = smpl_out / f"{frame_stem}.pkl"
        out_mask = mask_out / image_path.name
        if overwrite or not out_rgb.exists():
            shutil.copyfile(image_path, out_rgb)
        if overwrite or not out_depth.exists():
            np.save(out_depth, load_depth(resolve_depth_path(depth_dir, frame_stem)))
        if overwrite or not out_cam.exists():
            intr = get_cam_int(float(cam_csv_data["focal_length"][cam_ind]), SENSOR_W, SENSOR_H, IMG_W / 2.0, IMG_H / 2.0)
            ext = get_frame_w2c(cam_csv_data, cam_ind)
            np.savez(out_cam, intrinsics=intr.astype(np.float32), pose=ext.astype(np.float32))
        if overwrite or not out_smpl.exists():
            with out_smpl.open("wb") as file:
                pickle.dump(persons, file, protocol=pickle.HIGHEST_PROTOCOL)
        if overwrite or not out_mask.exists():
            write_mask(resolve_mask_path(mask_dir, frame_stem), out_mask)

        stats["frames"] += 1
        if indices:
            stats["matched_frames"] += 1
        stats["persons"] += len(persons)
        stats["max_humans"] = max(stats["max_humans"], len(persons))
    return stats


def resolve_annotation_indices(rel_name: str, basename: str, annot_index: dict[str, list[int]]) -> list[int]:
    if rel_name in annot_index:
        return annot_index[rel_name]
    if basename in annot_index:
        return annot_index[basename]
    return []


def build_persons(annot: Any, indices: list[int]) -> list[dict[str, Any]]:
    persons: list[dict[str, Any]] = []
    pose_cam = annot["pose_cam"]
    cam_ext = annot["cam_ext"]
    shape = annot["shape"]
    trans_cam = annot["trans_cam"]
    for idx in indices:
        pose = pose_cam[idx]
        root_pose = pose[:3]
        body_pose = pose[3:66]
        jaw_pose = pose[66:69]
        leye_pose = pose[69:72]
        reye_pose = pose[72:75]
        left_hand_pose = pose[75:120]
        right_hand_pose = pose[120:165]
        transl = trans_cam[idx] + cam_ext[idx][:, 3][:3]
        persons.append(
            {
                "smplx_root_pose": root_pose.reshape(1, 3),
                "smplx_body_pose": body_pose.reshape(21, 3),
                "smplx_jaw_pose": jaw_pose.reshape(1, 3),
                "smplx_leye_pose": leye_pose.reshape(1, 3),
                "smplx_reye_pose": reye_pose.reshape(1, 3),
                "smplx_left_hand_pose": left_hand_pose.reshape(15, 3),
                "smplx_right_hand_pose": right_hand_pose.reshape(15, 3),
                "smplx_shape": shape[idx].reshape(-1),
                "smplx_gender": "neutral",
                "smplx_transl": transl.reshape(3),
            }
        )
    return persons


def resolve_depth_path(depth_dir: Path | None, frame_stem: str) -> Path:
    if depth_dir is None:
        return Path(f"{frame_stem}_depth.exr")
    candidates = [
        depth_dir / f"{frame_stem}_depth.exr",
        depth_dir / f"{frame_stem}.exr",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(depth_dir.rglob(f"{frame_stem}*_depth.exr"))
    if matches:
        return matches[0]
    return candidates[0]


def resolve_mask_path(mask_dir: Path | None, frame_stem: str) -> Path:
    if mask_dir is None:
        return Path(f"{frame_stem}_env.png")
    candidates = [
        mask_dir / f"{frame_stem}_env.png",
        mask_dir / f"{frame_stem}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(mask_dir.rglob(f"{frame_stem}*.png"))
    if matches:
        return matches[0]
    return candidates[0]


def load_depth(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing depth EXR: {path}")
    depth = OpenEXR.File(str(path)).parts[0].channels["Depth"].pixels
    return depth.astype(np.float32) / 100.0


def write_mask(mask_path: Path, out_path: Path) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    if mask is None:
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    else:
        mask = 255 - mask
    cv2.imwrite(str(out_path), mask)


def focal_length_mm_to_px(focal_length: float, sensor: float, focal_point: float) -> float:
    return (focal_length / sensor) * focal_point * 2.0


def get_cam_int(fl: float, sens_w: float, sens_h: float, cx: float, cy: float) -> np.ndarray:
    flx = focal_length_mm_to_px(fl, sens_w, cx)
    fly = focal_length_mm_to_px(fl, sens_h, cy)
    return np.array([[flx, 0.0, cx], [0.0, fly, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotation_matrix_unreal(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    r_yaw = np.array(
        [[np.cos(-yaw_rad), -np.sin(-yaw_rad), 0], [np.sin(-yaw_rad), np.cos(-yaw_rad), 0], [0, 0, 1]]
    )
    r_pitch = np.array(
        [[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]]
    )
    r_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]]
    )
    return r_roll @ r_pitch @ r_yaw


def convert_rotation_to_opencv(r_unreal: np.ndarray) -> np.ndarray:
    c = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
    return c @ r_unreal @ c.T


def convert_translation_to_opencv(x: float, y: float, z: float) -> np.ndarray:
    return np.array([y, -z, x], dtype=np.float32)


def get_frame_w2c(cam_csv_data: dict[str, list[Any]], idx: int) -> np.ndarray:
    r_unreal = rotation_matrix_unreal(
        float(cam_csv_data["yaw"][idx]),
        float(cam_csv_data["pitch"][idx]),
        float(cam_csv_data["roll"][idx]),
    )
    rot_cv = convert_rotation_to_opencv(r_unreal)
    trans_cv = convert_translation_to_opencv(
        float(cam_csv_data["x"][idx]) / 100.0,
        float(cam_csv_data["y"][idx]) / 100.0,
        float(cam_csv_data["z"][idx]) / 100.0,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot_cv
    c2w[:3, 3] = -rot_cv @ trans_cv
    return np.linalg.inv(c2w)


if __name__ == "__main__":
    main()
