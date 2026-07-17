from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam import BedlamDataset
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices
from vggt_omega.utils.rotation import rot6d_to_axis_angle


HAND_JOINTS = (20, 21, 22, 23)
FOOT_JOINTS = (7, 8, 10, 11)
SIDE_NAMES = ("left", "right")
VISIBLE_COLOR = (20, 230, 110)
REJECTED_COLOR = (255, 65, 65)
NO_DEPTH_COLOR = (255, 205, 45)
HAND_COLOR = (50, 220, 255)
FOOT_COLOR = (255, 145, 30)
SOLE_COLOR = (255, 45, 220)
PLANE_COLOR = (255, 255, 255)
NORMAL_COLOR = (70, 140, 255)
SAMPLE_COLOR = (255, 225, 60)


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
        contact_teacher_root=args.contact_teacher_root,
        require_contact_teacher=True,
    )
    smpl = SMPLLayer(args.smpl_model_dir).to(device).eval()
    sole_indices = build_sole_vertex_indices(smpl.layer.v_template.detach(), args.sole_vertices_per_foot).to(device)
    output = Path(args.output_dir).expanduser()
    people_output = output / "people"
    people_output.mkdir(parents=True, exist_ok=True)

    pool_size = len(dataset) if args.sample_pool_size <= 0 else min(len(dataset), args.sample_pool_size)
    count = min(args.num_samples, pool_size)
    indices = np.linspace(0, pool_size - 1, num=count, dtype=np.int64)
    sample_rows: list[dict] = []
    person_rows: list[dict] = []
    foot_rows: list[dict] = []

    for output_idx, dataset_idx in enumerate(indices.tolist()):
        sample = dataset[dataset_idx]
        frame = 1
        image = tensor_to_image(sample["images"][frame])
        image_hw = (image.height, image.width)
        depth = sample["gt_depth"][frame, 0].to(device).float()
        intrinsics = sample["K_scal3r"][frame].to(device).float()
        pose = sample["gt_pose_6d"][frame].to(device)
        betas = sample["gt_betas"][frame].to(device)
        transl = sample["gt_transl_cam"][frame].to(device)
        valid_people = sample["smpl_mask"][frame].to(device).bool()
        boxes = sample["gt_boxes"][frame]
        axis_angle = rot6d_to_axis_angle(pose.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, joints = smpl(axis_angle.float(), betas.float())
        vertices_cam = vertices + transl[:, None, :]
        joints_cam = joints[:, :24] + transl[:, None, :]
        exclusion = projected_body_mask(vertices_cam, valid_people, intrinsics, depth.shape, image_hw)
        depth_cpu = depth.detach().cpu()
        exclusion_cpu = exclusion.detach().cpu()

        overview_all = image.copy()
        overview_visible = image.copy()
        depth_panel_base = depth_to_image(depth, args.max_depth_m).resize((image.width, image.height))
        overview_depth = depth_panel_base.copy()
        draw_all = ImageDraw.Draw(overview_all)
        draw_visible = ImageDraw.Draw(overview_visible)
        draw_depth = ImageDraw.Draw(overview_depth)
        sample_people: list[dict] = []

        for person_idx in torch.nonzero(valid_people, as_tuple=False).reshape(-1).tolist():
            box_xyxy = normalized_box_xyxy(boxes[person_idx], image_hw)
            draw_box(draw_all, box_xyxy, palette(person_idx))
            draw_box(draw_visible, box_xyxy, palette(person_idx))
            mesh_visibility = visibility_to_cpu(
                depth_visibility(
                    vertices_cam[person_idx],
                    depth,
                    intrinsics,
                    image_hw,
                    args.visibility_window,
                    args.depth_tolerance_m,
                    args.max_depth_m,
                )
            )
            joint_visibility = visibility_to_cpu(
                depth_visibility(
                    joints_cam[person_idx],
                    depth,
                    intrinsics,
                    image_hw,
                    args.visibility_window,
                    args.depth_tolerance_m,
                    args.max_depth_m,
                )
            )
            sole_points = vertices_cam[person_idx, sole_indices]
            sole_visibility = [
                visibility_to_cpu(
                    depth_visibility(
                        points,
                        depth,
                        intrinsics,
                        image_hw,
                        args.visibility_window,
                        args.depth_tolerance_m,
                        args.max_depth_m,
                    )
                )
                for points in sole_points
            ]

            draw_mesh_status(draw_all, mesh_visibility, args.draw_vertex_stride, show_rejected=True)
            draw_mesh_status(draw_visible, mesh_visibility, args.draw_vertex_stride, show_rejected=False, visible_color=palette(person_idx))
            person_record = {
                "dataset_index": dataset_idx,
                "person_index": person_idx,
                "mesh": visibility_metrics(mesh_visibility),
                "hands": visibility_metrics(select_visibility(joint_visibility, HAND_JOINTS)),
                "feet_joints": visibility_metrics(select_visibility(joint_visibility, FOOT_JOINTS)),
                "feet": [],
            }

            panels = [image.copy(), image.copy(), depth_panel_base.copy(), image.copy()]
            panel_draws = [ImageDraw.Draw(panel) for panel in panels]
            draw_box(panel_draws[0], box_xyxy, palette(person_idx))
            draw_box(panel_draws[1], box_xyxy, palette(person_idx))
            draw_mesh_status(panel_draws[0], mesh_visibility, args.draw_vertex_stride, show_rejected=True)
            draw_mesh_status(panel_draws[1], mesh_visibility, args.draw_vertex_stride, show_rejected=False)
            draw_endpoint_joints(panel_draws[1], joint_visibility)

            for side_idx, side_name in enumerate(SIDE_NAMES):
                side_visibility = sole_visibility[side_idx]
                side_metrics = visibility_metrics(side_visibility)
                sole_center = sole_points[side_idx].mean(dim=0)
                center_xy = project(sole_center[None], intrinsics)[0].detach().cpu()
                inside_box = point_in_box(center_xy, box_xyxy)
                teacher_valid = bool(sample["contact_teacher_valid"][frame, person_idx, side_idx])
                contact_label = bool(sample["contact_label"][frame, person_idx, side_idx])
                sole_consistent = (
                    side_metrics["visible_ratio"] >= args.min_sole_visible_ratio
                    and side_metrics["median_abs_depth_delta_m"] is not None
                    and side_metrics["median_abs_depth_delta_m"] <= args.depth_tolerance_m
                )
                audit_valid = bool(teacher_valid and inside_box and sole_consistent)
                plane_center = sample["contact_plane_center_cam"][frame, person_idx, side_idx].to(device)
                plane_normal = sample["contact_plane_normal_cam"][frame, person_idx, side_idx].to(device)
                plane_rmse = float(sample["contact_plane_rmse_m"][frame, person_idx, side_idx])
                signed = float(sample["contact_signed_distance_m"][frame, person_idx, side_idx])
                velocity = float(sample["contact_foot_velocity_m"][frame, person_idx, side_idx])
                candidates = support_candidate_pixels(
                    sole_center,
                    depth_cpu,
                    intrinsics,
                    image_hw,
                    exclusion_cpu,
                    args.support_window,
                    args.max_depth_m,
                    args.support_max_depth_delta_m,
                )

                draw_visibility_points(panel_draws[1], side_visibility, radius=2, visible_color=SOLE_COLOR)
                draw_marker(panel_draws[1], center_xy, SOLE_COLOR, radius=5)
                draw_support_candidates(panel_draws[2], candidates)
                draw_marker(panel_draws[2], center_xy, SOLE_COLOR, radius=5)
                draw_plane(panel_draws[2], plane_center, plane_normal, intrinsics, valid=teacher_valid)
                draw_contact_perturbations(panel_draws[3], sole_center, plane_normal, intrinsics, teacher_valid, contact_label)

                foot_record = {
                    "dataset_index": dataset_idx,
                    "person_index": person_idx,
                    "side": side_name,
                    "teacher_valid": teacher_valid,
                    "contact_label": contact_label,
                    "audit_valid": audit_valid,
                    "rejected_by_audit": bool(teacher_valid and not audit_valid),
                    "sole_center_inside_box": inside_box,
                    "sole_visible_ratio": side_metrics["visible_ratio"],
                    "sole_median_abs_depth_delta_m": side_metrics["median_abs_depth_delta_m"],
                    "sole_p90_abs_depth_delta_m": side_metrics["p90_abs_depth_delta_m"],
                    "plane_rmse_m": plane_rmse,
                    "signed_distance_m": signed,
                    "foot_frame_displacement_m": velocity,
                    "support_candidate_points": len(candidates),
                }
                foot_rows.append(foot_record)
                person_record["feet"].append(foot_record)

                status_color = VISIBLE_COLOR if audit_valid else REJECTED_COLOR
                draw_marker(draw_all, center_xy, status_color, radius=5)
                draw_marker(draw_visible, center_xy, status_color, radius=5)
                draw_marker(draw_depth, center_xy, status_color, radius=5)

            metric_line = format_person_metrics(person_record)
            detail_path = people_output / f"audit_{output_idx:03d}_dataset_{dataset_idx:06d}_person_{person_idx:02d}.png"
            save_panel_grid(
                panels,
                detail_path,
                ("mesh status", "visible + endpoints", "depth + support plane", "contact perturbations"),
                metric_line,
            )
            person_record["path"] = str(detail_path)
            person_rows.append(person_record)
            sample_people.append(person_record)

        overview_path = output / f"audit_{output_idx:03d}_dataset_{dataset_idx:06d}_overview.png"
        save_panel_grid(
            [overview_all, overview_visible, overview_depth],
            overview_path,
            ("all projected points", "depth-consistent points", "GT depth + sole validity"),
            f"dataset={dataset_idx} people={len(sample_people)} tol={args.depth_tolerance_m:.3f}m",
        )
        sample_rows.append({"dataset_index": dataset_idx, "people": len(sample_people), "path": str(overview_path)})

    summary = build_summary(args, sample_rows, person_rows, foot_rows)
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worst = sorted(
        foot_rows,
        key=lambda row: (
            not row["rejected_by_audit"],
            not row["teacher_valid"],
            row["sole_visible_ratio"],
            -(row["sole_median_abs_depth_delta_m"] or 0.0),
        ),
    )[: args.num_worst_feet]
    (output / "worst_feet.json").write_text(json.dumps({"feet": worst}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary["aggregate"]}, indent=2))


def depth_visibility(points, depth, intrinsics, image_hw, window, tolerance_m, max_depth_m):
    points = points.float()
    z = points[:, 2]
    xy = project(points, intrinsics)
    image_h, image_w = image_hw
    depth_h, depth_w = depth.shape[-2:]
    dx = xy[:, 0] * (float(depth_w) / float(image_w))
    dy = xy[:, 1] * (float(depth_h) / float(image_h))
    in_image = (
        torch.isfinite(points).all(dim=-1)
        & (z > 1e-6)
        & (z <= float(max_depth_m))
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image_w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image_h)
    )
    radius = max(int(window), 1) // 2
    best_delta = torch.full_like(z, float("inf"))
    best_depth = torch.full_like(z, float("nan"))
    cx = dx.round().long()
    cy = dy.round().long()
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            sx = cx + ox
            sy = cy + oy
            valid = (sx >= 0) & (sx < depth_w) & (sy >= 0) & (sy < depth_h)
            values = depth[sy.clamp(0, depth_h - 1), sx.clamp(0, depth_w - 1)]
            valid = valid & torch.isfinite(values) & (values > 1e-6) & (values <= float(max_depth_m))
            delta = (values - z).abs()
            update = valid & (delta < best_delta)
            best_delta = torch.where(update, delta, best_delta)
            best_depth = torch.where(update, values, best_depth)
    depth_valid = in_image & torch.isfinite(best_depth)
    visible = depth_valid & (best_delta <= float(tolerance_m))
    return {
        "xy": xy,
        "in_image": in_image,
        "depth_valid": depth_valid,
        "visible": visible,
        "abs_delta": best_delta,
        "sampled_depth": best_depth,
    }


def select_visibility(visibility, indices):
    index = torch.as_tensor(indices, device=visibility["xy"].device, dtype=torch.long)
    return {key: value[index] for key, value in visibility.items()}


def visibility_to_cpu(visibility):
    return {key: value.detach().cpu() for key, value in visibility.items()}


def visibility_metrics(visibility):
    in_count = int(visibility["in_image"].sum().item())
    depth_count = int(visibility["depth_valid"].sum().item())
    visible_count = int(visibility["visible"].sum().item())
    values = visibility["abs_delta"][visibility["depth_valid"]]
    return {
        "in_image_points": in_count,
        "depth_valid_points": depth_count,
        "visible_points": visible_count,
        "visible_ratio": float(visible_count / max(in_count, 1)),
        "median_abs_depth_delta_m": quantile_or_none(values, 0.5),
        "p90_abs_depth_delta_m": quantile_or_none(values, 0.9),
    }


def quantile_or_none(values, q):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q).item())


def projected_body_mask(vertices, valid_people, intrinsics, depth_hw, image_hw):
    depth_h, depth_w = int(depth_hw[0]), int(depth_hw[1])
    image_h, image_w = image_hw
    mask = torch.zeros(depth_h, depth_w, dtype=torch.float32, device=vertices.device)
    for person_idx in torch.nonzero(valid_people, as_tuple=False).reshape(-1).tolist():
        points = vertices[person_idx]
        xy = project(points, intrinsics)
        x = (xy[:, 0] * float(depth_w) / float(image_w)).round().long()
        y = (xy[:, 1] * float(depth_h) / float(image_h)).round().long()
        valid = (
            torch.isfinite(points).all(dim=-1)
            & (points[:, 2] > 1e-6)
            & (x >= 0)
            & (x < depth_w)
            & (y >= 0)
            & (y < depth_h)
        )
        mask[y[valid], x[valid]] = 1.0
    return F.max_pool2d(mask[None, None], kernel_size=5, stride=1, padding=2)[0, 0].bool()


def support_candidate_pixels(sole_center, depth, intrinsics, image_hw, exclusion, window, max_depth_m, max_delta_m):
    image_h, image_w = image_hw
    depth_h, depth_w = depth.shape[-2:]
    xy = project(sole_center[None], intrinsics)[0]
    cx = int(round(float(xy[0]) * depth_w / image_w))
    cy = int(round(float(xy[1]) * depth_h / image_h))
    sole_z = float(sole_center[2])
    radius = max(int(window), 3) // 2
    output = []
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= depth_w or y < 0 or y >= depth_h:
                continue
            if abs(x - cx) <= 2 and abs(y - cy) <= 2:
                continue
            value = float(depth[y, x])
            if not np.isfinite(value) or value <= 0 or value > max_depth_m:
                continue
            if abs(value - sole_z) > max_delta_m or bool(exclusion[y, x]):
                continue
            output.append((x * image_w / depth_w, y * image_h / depth_h))
    return output


def build_summary(args, samples, people, feet):
    existing_valid = [row for row in feet if row["teacher_valid"]]
    rejected = [row for row in existing_valid if row["rejected_by_audit"]]
    contacts = [row for row in feet if row["contact_label"]]
    rejected_contacts = [row for row in contacts if row["rejected_by_audit"]]
    ratios = [row["sole_visible_ratio"] for row in feet]
    medians = [row["sole_median_abs_depth_delta_m"] for row in feet if row["sole_median_abs_depth_delta_m"] is not None]
    aggregate = {
        "num_samples": len(samples),
        "num_people": len(people),
        "num_feet": len(feet),
        "existing_teacher_valid_feet": len(existing_valid),
        "audit_valid_feet": sum(int(row["audit_valid"]) for row in feet),
        "existing_valid_rejected_feet": len(rejected),
        "existing_valid_rejection_rate": len(rejected) / max(len(existing_valid), 1),
        "contact_feet": len(contacts),
        "contact_feet_rejected": len(rejected_contacts),
        "contact_rejection_rate": len(rejected_contacts) / max(len(contacts), 1),
        "median_sole_visible_ratio": float(np.median(ratios)) if ratios else None,
        "median_sole_abs_depth_delta_m": float(np.median(medians)) if medians else None,
    }
    return {
        "thresholds": {
            "depth_tolerance_m": args.depth_tolerance_m,
            "visibility_window": args.visibility_window,
            "min_sole_visible_ratio": args.min_sole_visible_ratio,
            "max_depth_m": args.max_depth_m,
        },
        "aggregate": aggregate,
        "samples": samples,
        "people": people,
    }


def tensor_to_image(image):
    array = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def depth_to_image(depth, max_depth):
    array = depth.detach().cpu().numpy()
    valid = np.isfinite(array) & (array > 0) & (array <= max_depth)
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[valid] = np.clip(array[valid] / max_depth, 0, 1)
    rgb = np.stack([normalized, np.sqrt(normalized), 1.0 - normalized], axis=-1)
    rgb[~valid] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def project(points, intrinsics):
    z = points[:, 2].clamp(min=1e-6)
    x = intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]
    return torch.stack([x, y], dim=-1)


def draw_mesh_status(draw, visibility, stride, show_rejected, visible_color=VISIBLE_COLOR):
    for idx in range(0, visibility["xy"].shape[0], max(int(stride), 1)):
        if not bool(visibility["in_image"][idx]):
            continue
        if bool(visibility["visible"][idx]):
            color = visible_color
        elif not bool(visibility["depth_valid"][idx]):
            color = NO_DEPTH_COLOR
        elif show_rejected:
            color = REJECTED_COLOR
        else:
            continue
        draw_marker(draw, visibility["xy"][idx], color, radius=1)


def draw_visibility_points(draw, visibility, radius, visible_color):
    for idx in range(visibility["xy"].shape[0]):
        if not bool(visibility["in_image"][idx]):
            continue
        color = visible_color if bool(visibility["visible"][idx]) else REJECTED_COLOR
        draw_marker(draw, visibility["xy"][idx], color, radius=radius)


def draw_endpoint_joints(draw, visibility):
    for index in HAND_JOINTS:
        color = HAND_COLOR if bool(visibility["visible"][index]) else REJECTED_COLOR
        draw_marker(draw, visibility["xy"][index], color, radius=5)
    for index in FOOT_JOINTS:
        color = FOOT_COLOR if bool(visibility["visible"][index]) else REJECTED_COLOR
        draw_marker(draw, visibility["xy"][index], color, radius=5)


def draw_support_candidates(draw, candidates):
    step = max(len(candidates) // 200, 1)
    for x, y in candidates[::step]:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=SAMPLE_COLOR)


def draw_plane(draw, center, normal, intrinsics, valid):
    if not valid or not bool(torch.isfinite(center).all() & torch.isfinite(normal).all()):
        return
    points = torch.stack([center, center + normal * 0.25], dim=0)
    xy = project(points, intrinsics).detach().cpu().numpy()
    draw.line((tuple(xy[0]), tuple(xy[1])), fill=NORMAL_COLOR, width=3)
    draw.ellipse((xy[0, 0] - 5, xy[0, 1] - 5, xy[0, 0] + 5, xy[0, 1] + 5), outline=PLANE_COLOR, width=2)


def draw_contact_perturbations(draw, center, normal, intrinsics, valid, contact):
    if not valid or not bool(torch.isfinite(normal).all()):
        return
    points = torch.stack([center, center + 0.08 * normal, center - 0.06 * normal], dim=0)
    xy = project(points, intrinsics).detach().cpu().numpy()
    draw.line((tuple(xy[1]), tuple(xy[2])), fill=NORMAL_COLOR, width=2)
    colors = (VISIBLE_COLOR if contact else PLANE_COLOR, HAND_COLOR, REJECTED_COLOR)
    for point, color in zip(xy, colors):
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), outline=color, width=3)


def draw_marker(draw, xy, color, radius):
    x, y = float(xy[0]), float(xy[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def normalized_box_xyxy(box, image_hw):
    image_h, image_w = image_hw
    cx, cy, width, height = [float(value) for value in box]
    return ((cx - width / 2) * image_w, (cy - height / 2) * image_h, (cx + width / 2) * image_w, (cy + height / 2) * image_h)


def draw_box(draw, box, color):
    draw.rectangle(box, outline=color, width=2)


def point_in_box(point, box):
    x, y = float(point[0]), float(point[1])
    return bool(np.isfinite(x) and np.isfinite(y) and box[0] <= x <= box[2] and box[1] <= y <= box[3])


def palette(index):
    colors = ((0, 220, 120), (255, 90, 70), (40, 150, 255), (255, 210, 40), (180, 80, 255))
    return colors[index % len(colors)]


def format_person_metrics(record):
    mesh = record["mesh"]
    hands = record["hands"]
    feet = record["feet_joints"]
    return (
        f"dataset={record['dataset_index']} person={record['person_index']}  "
        f"mesh vis={mesh['visible_ratio']:.3f} med={format_metric(mesh['median_abs_depth_delta_m'])}  "
        f"hands vis={hands['visible_ratio']:.3f} med={format_metric(hands['median_abs_depth_delta_m'])}  "
        f"feet vis={feet['visible_ratio']:.3f} med={format_metric(feet['median_abs_depth_delta_m'])}"
    )


def format_metric(value):
    return "n/a" if value is None else f"{value:.3f}m"


def save_panel_grid(panels, path, titles, header):
    width = panels[0].width
    height = panels[0].height
    header_height = 42
    canvas = Image.new("RGB", (width * len(panels), height + header_height), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), header, fill="white")
    for idx, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (idx * width, header_height))
        draw.rectangle((idx * width, header_height, idx * width + min(190, width), header_height + 18), fill="black")
        draw.text((idx * width + 4, header_height + 3), title, fill="white")
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Depth-visible endpoint and contact-teacher audit for HSI curriculum V2")
    parser.add_argument("--bedlam-root", required=True)
    parser.add_argument("--boxes-root", required=True)
    parser.add_argument("--contact-teacher-root", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--sample-pool-size", type=int, default=0)
    parser.add_argument("--num-worst-feet", type=int, default=50)
    parser.add_argument("--sole-vertices-per-foot", type=int, default=48)
    parser.add_argument("--visibility-window", type=int, default=3)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.20)
    parser.add_argument("--min-sole-visible-ratio", type=float, default=0.25)
    parser.add_argument("--support-window", type=int, default=31)
    parser.add_argument("--support-max-depth-delta-m", type=float, default=0.75)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--draw-vertex-stride", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
