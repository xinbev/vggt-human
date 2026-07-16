from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.bedlam import BedlamDataset
from vggt_omega.models.smpl_layer import SMPLLayer
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
        contact_teacher_root=args.contact_teacher_root or None,
        require_contact_teacher=bool(args.contact_teacher_root),
    )
    smpl = SMPLLayer(args.smpl_model_dir).to(device).eval()
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, num=min(args.num_samples, len(dataset)), dtype=np.int64)
    rows = []
    for output_idx, dataset_idx in enumerate(indices.tolist()):
        sample = dataset[dataset_idx]
        frame = 1
        image = tensor_to_image(sample["images"][frame])
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        pose = sample["gt_pose_6d"][frame].to(device)
        betas = sample["gt_betas"][frame].to(device)
        transl = sample["gt_transl_cam"][frame].to(device)
        valid = sample["smpl_mask"][frame].to(device).bool()
        aa = rot6d_to_axis_angle(pose.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, _ = smpl(aa.float(), betas.float())
        intrinsics = sample["K_scal3r"][frame].to(device)
        boxes = sample["gt_boxes"][frame]
        h, w = image.height, image.width
        people = []
        for person_idx in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
            color = palette(person_idx)
            box = boxes[person_idx]
            cx, cy, bw, bh = [float(value) for value in box]
            draw.rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), outline=color, width=2)
            points = vertices[person_idx, ::20] + transl[person_idx]
            xy = project(points, intrinsics)
            for x, y in xy.cpu().numpy():
                if 0 <= x < w and 0 <= y < h:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            people.append(person_idx)

        noisy = overlay.copy()
        noisy_draw = ImageDraw.Draw(noisy)
        if "contact_plane_normal_cam" in sample:
            normals = sample["contact_plane_normal_cam"][frame].to(device)
            labels = sample["contact_label"][frame].to(device)
            teacher_valid = sample["contact_teacher_valid"][frame].to(device)
            for person_idx in people:
                active = labels[person_idx] & teacher_valid[person_idx]
                if not active.any():
                    continue
                normal = normals[person_idx, active].mean(dim=0)
                shifted = vertices[person_idx, ::20] + transl[person_idx] + 0.08 * normal
                for x, y in project(shifted, intrinsics).cpu().numpy():
                    if 0 <= x < w and 0 <= y < h:
                        noisy_draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 255))

        depth_panel = depth_to_image(sample["gt_depth"][frame, 0], max_depth=args.max_depth_m).resize((w, h))
        canvas = Image.new("RGB", (w * 3, h), "white")
        canvas.paste(overlay, (0, 0))
        canvas.paste(depth_panel, (w, 0))
        canvas.paste(noisy, (w * 2, 0))
        path = output / f"audit_{output_idx:03d}_dataset_{dataset_idx:06d}.png"
        canvas.save(path)
        rows.append({"dataset_index": dataset_idx, "people": len(people), "path": str(path)})
    (output / "audit_summary.json").write_text(json.dumps({"samples": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "num_samples": len(rows)}, indent=2))


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def depth_to_image(depth: torch.Tensor, max_depth: float) -> Image.Image:
    array = depth.detach().cpu().numpy()
    valid = np.isfinite(array) & (array > 0) & (array <= max_depth)
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[valid] = np.clip(array[valid] / max_depth, 0, 1)
    rgb = np.stack([normalized, np.sqrt(normalized), 1.0 - normalized], axis=-1)
    rgb[~valid] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[:, 2].clamp(min=1e-6)
    x = intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]
    return torch.stack([x, y], dim=-1)


def palette(index: int) -> tuple[int, int, int]:
    colors = ((0, 220, 120), (255, 90, 70), (40, 150, 255), (255, 210, 40), (180, 80, 255))
    return colors[index % len(colors)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual audit for HSI V2 GT geometry and contact teachers")
    parser.add_argument("--bedlam-root", required=True)
    parser.add_argument("--boxes-root", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-manifest", default="")
    parser.add_argument("--contact-teacher-root", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
