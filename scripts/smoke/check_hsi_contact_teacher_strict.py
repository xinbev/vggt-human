from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "contact_label": np.bool_,
    "contact_teacher_valid": np.bool_,
    "contact_plane_valid": np.bool_,
    "contact_geometry_valid": np.bool_,
    "contact_sole_center_inside_box": np.bool_,
    "contact_sole_visible_ratio": np.float32,
    "contact_sole_median_depth_delta_m": np.float32,
}


def main() -> None:
    args = parse_args()
    root = Path(args.contact_teacher_root).expanduser()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing strict teacher summary: {summary_path}")
    paths = sorted(root.rglob("contact_teacher/*.npz"))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise RuntimeError(f"No contact teacher sidecars found under {root}")

    counters = {
        "files": 0,
        "feet": 0,
        "plane_valid": 0,
        "geometry_valid": 0,
        "teacher_valid": 0,
        "contact": 0,
    }
    failures: list[str] = []
    for path in paths:
        with np.load(path) as data:
            missing = sorted(set(REQUIRED) - set(data.files))
            if missing:
                failures.append(f"{path}: missing={missing}")
                continue
            arrays = {key: np.asarray(data[key]) for key in REQUIRED}
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1 or next(iter(shapes))[1:] != (2,):
            failures.append(f"{path}: inconsistent strict-field shapes={sorted(shapes)}")
            continue
        plane = arrays["contact_plane_valid"].astype(bool)
        geometry = arrays["contact_geometry_valid"].astype(bool)
        teacher = arrays["contact_teacher_valid"].astype(bool)
        contact = arrays["contact_label"].astype(bool)
        inside = arrays["contact_sole_center_inside_box"].astype(bool)
        ratio = arrays["contact_sole_visible_ratio"].astype(np.float32)
        delta = arrays["contact_sole_median_depth_delta_m"].astype(np.float32)
        expected_geometry = inside & (ratio >= args.min_sole_visible_ratio) & np.isfinite(delta) & (delta <= args.depth_tolerance_m)
        if np.any(geometry != expected_geometry):
            failures.append(f"{path}: stored geometry_valid disagrees with strict thresholds")
        if np.any(teacher & ~(plane & geometry)):
            failures.append(f"{path}: teacher_valid is not a subset of plane_valid and geometry_valid")
        if np.any(contact & ~teacher):
            failures.append(f"{path}: contact_label is not a subset of teacher_valid")
        counters["files"] += 1
        counters["feet"] += int(teacher.size)
        counters["plane_valid"] += int(plane.sum())
        counters["geometry_valid"] += int(geometry.sum())
        counters["teacher_valid"] += int(teacher.sum())
        counters["contact"] += int(contact.sum())

    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"Strict contact teacher smoke failed with {len(failures)} errors:\n{preview}")
    if counters["teacher_valid"] <= 0 or counters["contact"] <= 0:
        raise RuntimeError(f"Strict pilot has no usable teacher/contact feet: {counters}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({"gate": "pass", "summary": summary, "scanned": counters}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict HSI contact teacher sidecars")
    parser.add_argument("--contact-teacher-root", required=True)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.20)
    parser.add_argument("--min-sole-visible-ratio", type=float, default=0.25)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
