from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import load_yaml_config, require_path
from vggt_omega.utils.rotation import axis_angle_to_rotmat, rotation_matrix_to_axis_angle


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.path_config)
    root = resolve_path(args.root or require_path(config, "datasets.threedpw_root"))
    output_root = resolve_path(args.output_root or require_path(config, "datasets.threedpw_smpl_base_root"))
    smpl_model_dir = resolve_path(args.smpl_model_dir or require_path(config, "assets.smpl_model_dir"))
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    smpl_layers = {
        "male": SMPLLayer(smpl_model_dir, gender="male").to(device).eval(),
        "female": SMPLLayer(smpl_model_dir, gender="female").to(device).eval(),
    }

    summary = {
        "root": str(root),
        "output_root": str(output_root),
        "splits": {},
        "min_keypoint_conf": float(args.min_keypoint_conf),
        "bbox_expand": float(args.bbox_expand),
    }
    for split in args.splits:
        annot, stats = build_split(root, split, smpl_layers, device, args)
        out_path = output_root / f"{split}.pkl"
        with out_path.open("wb") as file:
            pickle.dump(annot, file, protocol=pickle.HIGHEST_PROTOCOL)
        summary["splits"][split] = {**stats, "annotation": str(out_path)}
        print(f"[3dpw] split={split} frames={stats['frames']} persons={stats['persons']} -> {out_path}", flush=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact 3DPW SMPL-base annotations for VGGT-Omega training/eval")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--device", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.50)
    parser.add_argument("--min-valid-keypoints", type=int, default=4)
    parser.add_argument("--bbox-expand", type=float, default=0.15)
    return parser.parse_args()


def build_split(
    root: Path,
    split: str,
    smpl_layers: dict[str, SMPLLayer],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    seq_dir = root / "sequenceFiles" / split
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"3DPW sequence split not found: {seq_dir}")
    files = sorted(seq_dir.glob("*.pkl"))
    if int(args.max_sequences) > 0:
        files = files[: int(args.max_sequences)]
    frames: dict[str, dict[str, Any]] = {}
    stats = {"sequences": len(files), "frames": 0, "persons": 0, "skipped_persons": 0, "missing_images": 0}
    for path in files:
        with path.open("rb") as file:
            meta = pickle.load(file, encoding="latin1")
        seq_name = path.stem
        k = np.asarray(meta["cam_intrinsics"], dtype=np.float32).reshape(3, 3)
        seq_len = len(meta["poses"][0]) if meta.get("poses") else 0
        num_persons = len(meta.get("genders", []))
        for frame_idx in range(seq_len):
            image_rel = find_image_relpath(root, seq_name, frame_idx, meta)
            if image_rel is None:
                stats["missing_images"] += 1
                continue
            image_path = root / "imageFiles" / image_rel
            with Image.open(image_path) as image:
                image_hw = (int(image.height), int(image.width))
            t_w2c = np.asarray(meta["cam_poses"][frame_idx], dtype=np.float32).reshape(4, 4)
            persons = []
            for person_idx in range(num_persons):
                valid = bool(np.asarray(meta["campose_valid"][person_idx])[frame_idx])
                if not valid:
                    continue
                person = build_person(meta, frame_idx, person_idx, t_w2c, image_hw, smpl_layers, device, args)
                if person is None:
                    stats["skipped_persons"] += 1
                    continue
                persons.append(person)
            if not persons:
                continue
            frame_key = f"{seq_name}/{Path(image_rel).name}"
            frames[frame_key] = {
                "sequence": seq_name,
                "frame_index": int(frame_idx),
                "image_relpath": image_rel,
                "image_hw": [int(image_hw[0]), int(image_hw[1])],
                "K": k.astype(np.float32),
                "T_w2c": t_w2c.astype(np.float32),
                "persons": persons,
            }
            stats["frames"] += 1
            stats["persons"] += len(persons)
    return {"split": split, "root": str(root), "frames": frames}, stats


def build_person(
    meta: dict[str, Any],
    frame_idx: int,
    person_idx: int,
    t_w2c: np.ndarray,
    image_hw: tuple[int, int],
    smpl_layers: dict[str, SMPLLayer],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    poses = np.asarray(meta["poses"][person_idx][frame_idx], dtype=np.float32).reshape(24, 3)
    raw_trans = np.asarray(meta["trans"][person_idx][frame_idx], dtype=np.float32).reshape(3)
    betas = np.asarray(meta["betas"][person_idx], dtype=np.float32).reshape(-1)[:10]
    raw_gender = meta["genders"][person_idx]
    gender = "male" if str(raw_gender).lower().startswith("m") else "female"
    r_w2c = t_w2c[:3, :3]
    t = t_w2c[:3, 3]

    root_rot = axis_angle_to_rotmat(torch.from_numpy(poses[0]).float()).numpy()
    root_cam_rot = r_w2c @ root_rot
    root_cam = rotation_matrix_to_axis_angle(torch.from_numpy(root_cam_rot).float()).numpy().reshape(3).astype(np.float32)
    body_pose = poses[1:].astype(np.float32)

    with torch.no_grad():
        vertices, joints = smpl_layers[gender](
            torch.from_numpy(np.concatenate([root_cam.reshape(1, 3), body_pose], axis=0).reshape(1, 72)).to(device=device).float(),
            torch.from_numpy(betas.reshape(1, 10)).to(device=device).float(),
        )
    root_joint = joints[0, 0].detach().cpu().numpy().astype(np.float32)
    transl_cam = (-root_joint + (r_w2c @ (root_joint + raw_trans).reshape(3, 1)).reshape(3) + t).astype(np.float32)
    box = extract_bbox(meta, person_idx, frame_idx, image_hw, args)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    h, w = image_hw
    bbox_cxcywh = np.asarray(
        [
            (x1 + x2) * 0.5 / max(w, 1),
            (y1 + y2) * 0.5 / max(h, 1),
            (x2 - x1) / max(w, 1),
            (y2 - y1) / max(h, 1),
        ],
        dtype=np.float32,
    )
    return {
        "person_id": int(person_idx),
        "smpl_root_pose": root_cam.astype(np.float32),
        "smpl_body_pose": body_pose.astype(np.float32),
        "smpl_shape": betas.astype(np.float32),
        "smpl_transl": transl_cam.astype(np.float32),
        "smpl_gender": gender,
        "smpl_valid": True,
        "bbox_xyxy_pixels": np.asarray(box, dtype=np.float32),
        "bbox_cxcywh_norm": bbox_cxcywh,
        "bbox_valid": True,
    }


def extract_bbox(meta: dict[str, Any], person_idx: int, frame_idx: int, image_hw: tuple[int, int], args: argparse.Namespace) -> list[float] | None:
    poses2d_all = meta.get("poses2d")
    if poses2d_all is None:
        return None
    arr = np.asarray(poses2d_all[person_idx], dtype=np.float32)
    if arr.ndim != 3:
        return None
    if arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 1))
    if frame_idx >= arr.shape[0] or arr.shape[-1] < 3:
        return None
    joints = arr[frame_idx]
    valid = joints[:, 2] > float(args.min_keypoint_conf)
    if int(valid.sum()) < int(args.min_valid_keypoints):
        return None
    xy = joints[valid, :2]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    h, w = image_hw
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    expand = float(args.bbox_expand)
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    return [
        float(np.clip(x1, 0.0, max(w - 1, 1))),
        float(np.clip(y1, 0.0, max(h - 1, 1))),
        float(np.clip(x2, 0.0, max(w - 1, 1))),
        float(np.clip(y2, 0.0, max(h - 1, 1))),
    ]


def find_image_relpath(root: Path, seq_name: str, frame_idx: int, meta: dict[str, Any]) -> str | None:
    candidates = [f"{seq_name}/image_{frame_idx:05d}.jpg"]
    frame_ids = meta.get("img_frame_ids")
    if frame_ids is not None and frame_idx < len(frame_ids):
        try:
            candidates.append(f"{seq_name}/image_{int(frame_ids[frame_idx]):05d}.jpg")
        except (TypeError, ValueError):
            pass
    for rel in candidates:
        if (root / "imageFiles" / rel).is_file():
            return rel
    return None


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
