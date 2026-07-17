from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam import BedlamDataset
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices, estimate_local_support_planes
from vggt_omega.utils.rotation import rot6d_to_axis_angle


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = BedlamDataset(
        root=args.bedlam_root,
        split=args.split,
        sequence_length=3,
        stride=1,
        image_resolution=args.image_resolution,
        max_humans=args.max_humans,
        require_smpl=True,
        require_depth=True,
        boxes_root=args.boxes_root,
        require_boxes=True,
        sequence_manifest=args.sequence_manifest or None,
    )
    smpl = SMPLLayer(args.smpl_model_dir).to(device).eval()
    sole_indices = build_sole_vertex_indices(smpl.layer.v_template.detach(), args.sole_vertices_per_foot).to(device)
    output_root = Path(args.output_root).expanduser()
    written: set[tuple[int, int]] = set()
    valid_people = 0
    contact_feet = 0
    plane_valid_feet = 0
    geometry_valid_feet = 0
    geometry_rejected_feet = 0
    contact_window_indices: set[int] = set()
    contact_frame_keys: set[str] = set()

    def process_window(dataset_idx: int, positions: tuple[int, ...]) -> None:
        nonlocal valid_people, contact_feet, plane_valid_feet, geometry_valid_feet, geometry_rejected_feet
        seq_idx, start_idx = dataset._index[dataset_idx]
        seq_dir, frame_ids = dataset._sequences[seq_idx]
        sample = dataset[dataset_idx]
        pose6d = sample["gt_pose_6d"].to(device)
        betas = sample["gt_betas"].to(device)
        transl = sample["gt_transl_cam"].to(device)
        mask = sample["smpl_mask"].to(device).bool()
        track_ids = sample["gt_track_ids"].to(device)
        depth = sample["gt_depth"].to(device)[:, 0]
        intrinsics = sample["K_scal3r"].to(device)
        aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, _ = smpl(aa.float(), betas.reshape(-1, 10).float())
        vertices = vertices.reshape(3, args.max_humans, -1, 3)
        sole_vertices = vertices[:, :, sole_indices] + transl[:, :, None, None, :]
        sole = sole_vertices.mean(dim=-2)

        for position in positions:
            key = (seq_idx, start_idx + position)
            if key in written:
                continue
            written.add(key)
            frame_id = frame_ids[start_idx + position]
            frame_out = output_root / args.split / seq_dir.name / "contact_teacher" / f"{frame_id}.npz"
            if frame_out.exists() and not args.overwrite:
                existing = np.load(frame_out)
                existing_valid = np.asarray(existing["contact_teacher_valid"], dtype=np.bool_)
                existing_contact = np.asarray(existing["contact_label"], dtype=np.bool_)
                existing_plane = np.asarray(existing["contact_plane_valid"], dtype=np.bool_) if "contact_plane_valid" in existing else existing_valid
                existing_geometry = np.asarray(existing["contact_geometry_valid"], dtype=np.bool_) if "contact_geometry_valid" in existing else existing_valid
                valid_people += int(existing_valid.any(axis=-1).sum())
                contact_feet += int(existing_contact.sum())
                plane_valid_feet += int(existing_plane.sum())
                geometry_valid_feet += int(existing_geometry.sum())
                geometry_rejected_feet += int((existing_plane & ~existing_geometry).sum())
                if position == 1:
                    if bool(existing_contact.any()):
                        contact_window_indices.add(dataset_idx)
                        contact_frame_keys.add(f"{args.split}/{seq_dir.name}/{frame_id}")
                continue
            exclusion = projected_body_mask(
                vertices[position] + transl[position, :, None, :],
                mask[position],
                intrinsics[position],
                depth.shape[-2:],
                sample["images"].shape[-2:],
            )
            frame_idx_tensor = torch.zeros(args.max_humans, dtype=torch.long, device=device)
            planes = estimate_local_support_planes(
                depth[position : position + 1],
                intrinsics[position : position + 1],
                sole[position],
                frame_idx_tensor,
                image_size_hw=tuple(sample["images"].shape[-2:]),
                window_size=args.support_window,
                min_points=args.support_min_points,
                max_rmse_m=args.support_max_rmse_m,
                max_depth_m=args.max_depth_m,
                exclusion_mask=exclusion[None],
            )
            velocity = foot_velocity_for_position(sole, track_ids, mask, position)
            sole_geometry = sole_depth_geometry(
                sole_vertices[position],
                depth[position],
                intrinsics[position],
                sample["gt_boxes"][position].to(device),
                sample["boxes_mask"][position].to(device),
                image_size_hw=tuple(sample["images"].shape[-2:]),
                window_size=args.sole_visibility_window,
                tolerance_m=args.sole_visibility_tolerance_m,
                min_visible_ratio=args.min_sole_visible_ratio,
                max_depth_m=args.max_depth_m,
            )
            plane_valid = planes["valid"] & mask[position, :, None]
            teacher_valid = plane_valid & sole_geometry["valid"]
            contact = teacher_valid & (planes["signed"].abs() <= args.contact_threshold_m) & (velocity <= args.velocity_threshold_m)
            arrays = {
                "contact_plane_center_cam": planes["center"].cpu().numpy().astype(np.float32),
                "contact_plane_normal_cam": planes["normal"].cpu().numpy().astype(np.float32),
                "contact_plane_rmse_m": planes["rmse"].cpu().numpy().astype(np.float32),
                "contact_signed_distance_m": planes["signed"].cpu().numpy().astype(np.float32),
                "contact_foot_velocity_m": velocity.cpu().numpy().astype(np.float32),
                "contact_label": contact.cpu().numpy().astype(np.bool_),
                "contact_teacher_valid": teacher_valid.cpu().numpy().astype(np.bool_),
                "contact_plane_valid": plane_valid.cpu().numpy().astype(np.bool_),
                "contact_geometry_valid": sole_geometry["valid"].cpu().numpy().astype(np.bool_),
                "contact_sole_center_inside_box": sole_geometry["center_inside_box"].cpu().numpy().astype(np.bool_),
                "contact_sole_visible_ratio": sole_geometry["visible_ratio"].cpu().numpy().astype(np.float32),
                "contact_sole_median_depth_delta_m": sole_geometry["median_delta"].cpu().numpy().astype(np.float32),
            }
            frame_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(frame_out, **arrays)
            if position == 1 and bool(contact.any()):
                contact_window_indices.add(dataset_idx)
            if bool(contact.any()):
                contact_frame_keys.add(f"{args.split}/{seq_dir.name}/{frame_id}")
            valid_people += int((teacher_valid.any(dim=-1) & mask[position]).sum().item())
            contact_feet += int(contact.sum().item())
            plane_valid_feet += int(plane_valid.sum().item())
            geometry_valid_feet += int(sole_geometry["valid"].sum().item())
            geometry_rejected_feet += int((plane_valid & ~sole_geometry["valid"]).sum().item())

    total = len(dataset)
    window_limit = total if args.max_windows <= 0 else min(total, args.max_windows)
    partial = window_limit < total
    positions = (0, 1, 2) if partial else (1,)
    for idx in range(window_limit):
        process_window(idx, positions)
        if (idx + 1) % args.log_interval == 0 or idx + 1 == window_limit:
            print(f"[contact-teacher] {idx + 1}/{window_limit} frames={len(written)} valid_people={valid_people} contact_feet={contact_feet}", flush=True)
    if not partial:
        # Add sequence boundary frames, which never appear as the center of a 3-frame window.
        first_by_seq: dict[int, int] = {}
        last_by_seq: dict[int, int] = {}
        for dataset_idx, (seq_idx, _) in enumerate(dataset._index):
            first_by_seq.setdefault(seq_idx, dataset_idx)
            last_by_seq[seq_idx] = dataset_idx
        for dataset_idx in first_by_seq.values():
            process_window(dataset_idx, (0,))
        for dataset_idx in last_by_seq.values():
            process_window(dataset_idx, (2,))

    expected_frames = sum(len(frame_ids) for _, frame_ids in dataset._sequences) if not partial else len(written)
    summary = {
        "frames_written_or_seen": len(written),
        "expected_frames": expected_frames,
        "missing_frames": expected_frames - len(written),
        "valid_people": valid_people,
        "contact_feet": contact_feet,
        "plane_valid_feet": plane_valid_feet,
        "geometry_valid_feet": geometry_valid_feet,
        "geometry_rejected_feet": geometry_rejected_feet,
        "contact_windows": len(contact_window_indices),
        "output_root": str(output_root),
        "split": args.split,
        "partial": partial,
        "processed_windows": window_limit,
        "dataset_windows": total,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "contact_window_indices.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_index"])
        writer.writerows([[index] for index in sorted(contact_window_indices)])
    (output_root / "contact_frames.txt").write_text("\n".join(sorted(contact_frame_keys)) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not partial and len(written) != expected_frames:
        raise RuntimeError(f"Contact teacher preprocessing incomplete: {len(written)}/{expected_frames} frames")


def projected_body_mask(vertices, valid_people, intrinsics, depth_hw, image_hw) -> torch.Tensor:
    height, width = int(depth_hw[0]), int(depth_hw[1])
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    mask = torch.zeros(height, width, dtype=torch.float32, device=vertices.device)
    for person_vertices, person_valid in zip(vertices, valid_people):
        if not bool(person_valid):
            continue
        z = person_vertices[:, 2]
        valid = torch.isfinite(person_vertices).all(dim=-1) & (z > 1e-6)
        x = intrinsics[0, 0] * person_vertices[:, 0] / z.clamp(min=1e-6) + intrinsics[0, 2]
        y = intrinsics[1, 1] * person_vertices[:, 1] / z.clamp(min=1e-6) + intrinsics[1, 2]
        x = (x * float(width) / float(image_w)).round().long()
        y = (y * float(height) / float(image_h)).round().long()
        valid = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        mask[y[valid], x[valid]] = 1.0
    return F.max_pool2d(mask[None, None], kernel_size=5, stride=1, padding=2)[0, 0].bool()


def foot_velocity_for_position(sole, track_ids, mask, position: int) -> torch.Tensor:
    output = torch.full((sole.shape[1], 2), float("inf"), device=sole.device)
    neighbors = [idx for idx in (position - 1, position + 1) if 0 <= idx < sole.shape[0]]
    for slot in range(sole.shape[1]):
        if not bool(mask[position, slot]):
            continue
        track_id = track_ids[position, slot]
        distances = []
        for neighbor in neighbors:
            matches = torch.nonzero(mask[neighbor] & (track_ids[neighbor] == track_id), as_tuple=False).reshape(-1)
            if matches.numel() > 0:
                distances.append(torch.linalg.norm(sole[position, slot] - sole[neighbor, matches[0]], dim=-1))
        if distances:
            output[slot] = torch.stack(distances).mean(dim=0)
    return output


def sole_depth_geometry(
    sole_vertices: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    image_size_hw: tuple[int, int],
    window_size: int,
    tolerance_m: float,
    min_visible_ratio: float,
    max_depth_m: float,
) -> dict[str, torch.Tensor]:
    num_people, num_feet, num_vertices = sole_vertices.shape[:3]
    image_h, image_w = image_size_hw
    depth_h, depth_w = depth.shape[-2:]
    points = sole_vertices.reshape(-1, 3).float()
    z = points[:, 2]
    x_image = intrinsics[0, 0] * points[:, 0] / z.clamp(min=1e-6) + intrinsics[0, 2]
    y_image = intrinsics[1, 1] * points[:, 1] / z.clamp(min=1e-6) + intrinsics[1, 2]
    x_depth = x_image * (float(depth_w) / float(image_w))
    y_depth = y_image * (float(depth_h) / float(image_h))
    in_image = (
        torch.isfinite(points).all(dim=-1)
        & (z > 1e-6)
        & (z <= float(max_depth_m))
        & (x_image >= 0)
        & (x_image < image_w)
        & (y_image >= 0)
        & (y_image < image_h)
    )
    best_delta = torch.full_like(z, float("inf"))
    radius = max(int(window_size), 1) // 2
    cx = x_depth.round().long()
    cy = y_depth.round().long()
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            sx = cx + ox
            sy = cy + oy
            sample_valid = (sx >= 0) & (sx < depth_w) & (sy >= 0) & (sy < depth_h)
            sampled = depth[sy.clamp(0, depth_h - 1), sx.clamp(0, depth_w - 1)]
            sample_valid = sample_valid & torch.isfinite(sampled) & (sampled > 1e-6) & (sampled <= float(max_depth_m))
            delta = (sampled - z).abs()
            best_delta = torch.where(sample_valid & (delta < best_delta), delta, best_delta)

    in_image = in_image.reshape(num_people, num_feet, num_vertices)
    best_delta = best_delta.reshape(num_people, num_feet, num_vertices)
    depth_valid = in_image & torch.isfinite(best_delta)
    visible = depth_valid & (best_delta <= float(tolerance_m))
    visible_ratio = visible.sum(dim=-1).float() / in_image.sum(dim=-1).clamp(min=1).float()
    masked_delta = torch.where(depth_valid, best_delta, torch.full_like(best_delta, float("nan")))
    median_delta = torch.nanmedian(masked_delta, dim=-1).values

    centers = sole_vertices.mean(dim=-2)
    center_z = centers[..., 2].clamp(min=1e-6)
    center_x = intrinsics[0, 0] * centers[..., 0] / center_z + intrinsics[0, 2]
    center_y = intrinsics[1, 1] * centers[..., 1] / center_z + intrinsics[1, 2]
    center_x_norm = center_x / float(image_w)
    center_y_norm = center_y / float(image_h)
    box_cx, box_cy, box_w, box_h = boxes.float().unbind(dim=-1)
    x0 = box_cx - box_w * 0.5
    x1 = box_cx + box_w * 0.5
    y0 = box_cy - box_h * 0.5
    y1 = box_cy + box_h * 0.5
    center_inside_box = (
        boxes_mask.bool()[:, None]
        & (center_x_norm >= x0[:, None])
        & (center_x_norm <= x1[:, None])
        & (center_y_norm >= y0[:, None])
        & (center_y_norm <= y1[:, None])
    )
    valid = (
        center_inside_box
        & (visible_ratio >= float(min_visible_ratio))
        & torch.isfinite(median_delta)
        & (median_delta <= float(tolerance_m))
    )
    return {
        "valid": valid,
        "center_inside_box": center_inside_box,
        "visible_ratio": visible_ratio,
        "median_delta": median_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute robust GT foot contact teachers for BEDLAM")
    parser.add_argument("--bedlam-root", required=True)
    parser.add_argument("--boxes-root", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-manifest", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--sole-vertices-per-foot", type=int, default=48)
    parser.add_argument("--support-window", type=int, default=31)
    parser.add_argument("--support-min-points", type=int, default=32)
    parser.add_argument("--support-max-rmse-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--contact-threshold-m", type=float, default=0.025)
    parser.add_argument("--velocity-threshold-m", type=float, default=0.04)
    parser.add_argument("--sole-visibility-window", type=int, default=3)
    parser.add_argument("--sole-visibility-tolerance-m", type=float, default=0.20)
    parser.add_argument("--min-sole-visible-ratio", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
