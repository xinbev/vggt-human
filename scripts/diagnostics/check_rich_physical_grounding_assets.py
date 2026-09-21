import argparse
import json
from pathlib import Path
from typing import Any

import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resolve_scene_asset_name(sequence_name: str) -> str:
    scene_name = sequence_name.split("_", 1)[0]
    if scene_name != "LectureHall":
        return scene_name
    if "wipingchairs" in sequence_name or "reparingprojector" in sequence_name:
        return "LectureHall_chair"
    return "LectureHall_yoga"


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check RICH assets required for physical-grounding evaluation."
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("/home/zhw/xyb_space/RICH/official"),
    )
    parser.add_argument(
        "--support-root",
        type=Path,
        default=Path("/home/zhw/xyb_space/RICH/hmr4d_support"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/debug/rich_physical_grounding/assets_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = args.support_root / "rich_test_labels.pt"
    cam_params_candidates = [
        args.support_root / "cam2params.pt",
        args.support_root / "resource" / "cam2params.pt",
    ]
    cam_params_path = next((path for path in cam_params_candidates if path.is_file()), None)

    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing RICH labels: {labels_path}")
    if cam_params_path is None:
        raise FileNotFoundError(
            "Missing cam2params.pt; checked: "
            + ", ".join(str(path) for path in cam_params_candidates)
        )

    labels = load_torch(labels_path)
    cam_params = load_torch(cam_params_path)
    if not isinstance(labels, dict) or not labels:
        raise RuntimeError(f"Unexpected labels payload: {labels_path}")
    if not isinstance(cam_params, dict) or not cam_params:
        raise RuntimeError(f"Unexpected camera payload: {cam_params_path}")

    records = []
    issue_count = 0
    total_selected_frames = 0
    for vid, label in sorted(labels.items()):
        parts = vid.split("/")
        issues = []
        if len(parts) != 3:
            records.append({"vid": vid, "issues": ["unexpected video key format"]})
            issue_count += 1
            continue

        split, sequence_name, camera_name = parts
        scene_name = sequence_name.split("_", 1)[0]
        scene_asset_name = resolve_scene_asset_name(sequence_name)
        try:
            camera_id = int(camera_name.removeprefix("cam_"))
        except ValueError:
            camera_id = -1
            issues.append(f"invalid camera name: {camera_name}")

        image_dir = args.official_root / split / sequence_name / camera_name
        images = (
            sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if image_dir.is_dir()
            else []
        )
        if not image_dir.is_dir():
            issues.append(f"missing image directory: {image_dir}")
        elif not images:
            issues.append(f"no images found: {image_dir}")

        frame_ids = torch.as_tensor(label.get("frame_id", []), dtype=torch.long).flatten()
        selected_frames = int(frame_ids.numel())
        total_selected_frames += selected_frames
        if selected_frames == 0:
            issues.append("empty frame_id")
        elif images and int(frame_ids.max()) >= len(images):
            issues.append(
                f"frame_id max {int(frame_ids.max())} exceeds {len(images)} available images"
            )

        calibration_root = args.official_root / "scan_calibration" / scene_name
        scan_filename = (
            f"scan_{scene_asset_name.removeprefix('LectureHall_')}_scene_camcoord.ply"
            if scene_name == "LectureHall"
            else "scan_camcoord.ply"
        )
        scan_path = calibration_root / scan_filename
        camera_xml = calibration_root / "calibration" / f"{camera_id:03d}.xml"
        world_path = (
            args.official_root
            / "multicam2world"
            / f"{scene_asset_name}_multicam2world.json"
        )
        cam_key = f"{scene_name}_{camera_id}"

        for path, description in (
            (scan_path, "scene scan"),
            (camera_xml, "camera calibration"),
            (world_path, "multicam-to-world transform"),
        ):
            if not path.is_file():
                issues.append(f"missing {description}: {path}")
        if cam_key not in cam_params:
            issues.append(f"missing camera key in cam2params.pt: {cam_key}")

        issue_count += int(bool(issues))
        records.append(
            {
                "vid": vid,
                "scene": scene_name,
                "scene_asset": scene_asset_name,
                "camera_id": camera_id,
                "available_images": len(images),
                "selected_frames": selected_frames,
                "first_image": images[0].name if images else None,
                "last_image": images[-1].name if images else None,
                "issues": issues,
            }
        )

    recording_count = len({record["vid"].split("/")[1] for record in records})
    report = {
        "official_root": str(args.official_root),
        "support_root": str(args.support_root),
        "labels_path": str(labels_path),
        "cam_params_path": str(cam_params_path),
        "sequence_count": len(records),
        "recording_count": recording_count,
        "sequences_with_issues": issue_count,
        "total_selected_frames": total_selected_frames,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"RICH camera views: {len(records)}")
    print(f"Unique recordings: {recording_count}")
    print(f"Selected frames: {total_selected_frames}")
    print(f"Sequences with issues: {issue_count}")
    print(f"Report: {args.output}")
    if issue_count:
        print("First issues:")
        shown = 0
        for record in records:
            for issue in record["issues"]:
                print(f"  {record['vid']}: {issue}")
                shown += 1
                if shown >= 20:
                    break
            if shown >= 20:
                break
        raise SystemExit(1)


if __name__ == "__main__":
    main()
