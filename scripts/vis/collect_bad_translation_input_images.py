#!/usr/bin/env python
"""Collect top bad SMPL translation input RGB frames into one folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    bedlam_root = resolve_bedlam_root(args)
    frame_rows = read_csv(Path(args.frame_csv).expanduser())
    person_rows = read_csv(Path(args.person_csv).expanduser()) if str(args.person_csv).strip() else []
    selected = select_top_frames(frame_rows, args)
    person_by_frame = group_people(person_rows)

    records = []
    for rank, row in enumerate(selected, start=1):
        sequence_name = str(row.get("sequence_name", ""))
        frame_name = str(row.get("frame_name", ""))
        image_path = resolve_image_path(bedlam_root, str(args.split), sequence_name, frame_name)
        out_image = output_dir / f"{rank:02d}_{sequence_name}_{frame_name}{image_path.suffix if image_path.suffix else '.png'}"
        copied = False
        if bool(args.copy_images) and image_path.is_file():
            shutil.copy2(image_path, out_image)
            copied = True
        people = sorted(
            person_by_frame.get((sequence_name, frame_name), []),
            key=lambda item: safe_float(item.get("refined_transl_l2_m")),
            reverse=True,
        )
        record = {
            "rank": rank,
            "sequence_name": sequence_name,
            "frame_name": frame_name,
            "source_image": str(image_path),
            "copied_image": str(out_image) if copied else "",
            "image_found": bool(image_path.is_file()),
            "num_people": safe_int(row.get("num_people")),
            "base_mean_transl_l2_m": safe_float(row.get("base_mean_transl_l2_m")),
            "seed_mean_transl_l2_m": safe_float(row.get("seed_mean_transl_l2_m")),
            "refined_mean_transl_l2_m": safe_float(row.get("refined_mean_transl_l2_m")),
            "base_max_transl_l2_m": safe_float(row.get("base_max_transl_l2_m")),
            "seed_max_transl_l2_m": safe_float(row.get("seed_max_transl_l2_m")),
            "refined_max_transl_l2_m": safe_float(row.get("refined_max_transl_l2_m")),
            "refined_bad_people_gt_0p12m": safe_int(row.get("refined_bad_people_gt_0p12m")),
            "refined_bad_people_gt_0p20m": safe_int(row.get("refined_bad_people_gt_0p20m")),
            "refined_bad_people_gt_0p50m": safe_int(row.get("refined_bad_people_gt_0p50m")),
            "people": compact_people(people),
        }
        records.append(record)

    write_outputs(records, output_dir, args)
    if bool(args.contact_sheet):
        make_contact_sheet(records, output_dir, args)

    print(json.dumps({"output_dir": str(output_dir), "num_records": len(records), "copied_images": sum(1 for r in records if r["copied_image"])}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect top bad translation RGB images")
    parser.add_argument("--frame-csv", default="outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_translation_summary.csv")
    parser.add_argument("--person-csv", default="outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_person_translation_metrics.csv")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--bedlam-root", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--output-dir", default="outputs/vis/smpl_translation_v2_bad_input_images_top10")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sort-key", default="refined_max_transl_l2_m")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-sheet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--people-per-frame", type=int, default=5)
    return parser.parse_args()


def resolve_bedlam_root(args: argparse.Namespace) -> Path:
    if str(args.bedlam_root).strip():
        return Path(args.bedlam_root).expanduser()
    config = load_yaml_light(Path(args.path_config).expanduser())
    value = config.get("datasets", {}).get("bedlam_root", "")
    if not value:
        raise ValueError("--bedlam-root is required when path config has no datasets.bedlam_root")
    return Path(str(value)).expanduser()


def load_yaml_light(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except ImportError:
        return parse_simple_yaml(path.read_text(encoding="utf-8"))


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = value.strip().strip('"').strip("'")
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def select_top_frames(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    sort_key = str(args.sort_key)
    if rows and sort_key not in rows[0]:
        raise ValueError(f"sort key {sort_key!r} not found in frame CSV")
    return sorted(rows, key=lambda row: safe_float(row.get(sort_key)), reverse=True)[: max(int(args.top_k), 0)]


def group_people(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("sequence_name", "")), str(row.get("frame_name", "")))
        out.setdefault(key, []).append(row)
    return out


def resolve_image_path(bedlam_root: Path, split: str, sequence_name: str, frame_name: str) -> Path:
    rgb_dir = bedlam_root / split / sequence_name / "rgb"
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = rgb_dir / f"{frame_name}{suffix}"
        if candidate.is_file():
            return candidate
    return rgb_dir / f"{frame_name}.png"


def compact_people(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    people = []
    for row in rows:
        people.append(
            {
                "track_id": row.get("track_id", ""),
                "gt_idx_mode": row.get("gt_idx_mode", ""),
                "query_idx_modes": row.get("query_idx_modes", ""),
                "base_transl_l2_m": safe_float(row.get("base_transl_l2_m")),
                "seed_transl_l2_m": safe_float(row.get("seed_transl_l2_m")),
                "refined_transl_l2_m": safe_float(row.get("refined_transl_l2_m")),
                "transl_l2_delta_m": safe_float(row.get("transl_l2_delta_m")),
                "base_mpjpe_m": safe_float(row.get("base_mpjpe_m")),
                "refined_mpjpe_m": safe_float(row.get("refined_mpjpe_m")),
            }
        )
    return people


def write_outputs(records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    json_path = output_dir / "bad_translation_top10.json"
    json_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    csv_fields = [
        "rank",
        "sequence_name",
        "frame_name",
        "source_image",
        "copied_image",
        "image_found",
        "num_people",
        "base_mean_transl_l2_m",
        "seed_mean_transl_l2_m",
        "refined_mean_transl_l2_m",
        "base_max_transl_l2_m",
        "seed_max_transl_l2_m",
        "refined_max_transl_l2_m",
        "refined_bad_people_gt_0p12m",
        "refined_bad_people_gt_0p20m",
        "refined_bad_people_gt_0p50m",
    ]
    with (output_dir / "bad_translation_top10.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in csv_fields})

    md_lines = [
        "# SMPL Translation V2 Bad Input Images Top 10",
        "",
        f"- Sort key: `{args.sort_key}`",
        f"- Top K: `{args.top_k}`",
        "",
        "Color/metric reminder: lower translation L2 is better. Values are meters.",
        "",
    ]
    for record in records:
        md_lines.extend(record_markdown(record, int(args.people_per_frame)))
    (output_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def record_markdown(record: dict[str, Any], people_per_frame: int) -> list[str]:
    copied = record.get("copied_image") or "<not copied; run on server with BEDLAM images>"
    lines = [
        f"## {record['rank']:02d}. {record['sequence_name']} / {record['frame_name']}",
        "",
        f"- Image: `{copied}`",
        f"- Source: `{record['source_image']}`",
        f"- People: `{record['num_people']}`",
        f"- Frame max L2: base `{fmt_m(record['base_max_transl_l2_m'])}`, seed `{fmt_m(record['seed_max_transl_l2_m'])}`, refined `{fmt_m(record['refined_max_transl_l2_m'])}`",
        f"- Frame mean L2: base `{fmt_m(record['base_mean_transl_l2_m'])}`, seed `{fmt_m(record['seed_mean_transl_l2_m'])}`, refined `{fmt_m(record['refined_mean_transl_l2_m'])}`",
        "",
        "| track | gt | query modes | base | seed | refined | delta |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for person in record.get("people", [])[:people_per_frame]:
        lines.append(
            "| "
            f"{person['track_id']} | {person['gt_idx_mode']} | {person['query_idx_modes']} | "
            f"{fmt_m(person['base_transl_l2_m'])} | {fmt_m(person['seed_transl_l2_m'])} | "
            f"{fmt_m(person['refined_transl_l2_m'])} | {fmt_m(person['transl_l2_delta_m'])} |"
        )
    lines.append("")
    return lines


def make_contact_sheet(records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    loaded = []
    thumb_w = max(int(args.thumb_width), 120)
    label_h = 92
    font = ImageFont.load_default()
    for record in records:
        copied = str(record.get("copied_image") or "")
        if not copied or not Path(copied).is_file():
            continue
        image = Image.open(copied).convert("RGB")
        ratio = thumb_w / max(image.width, 1)
        thumb_h = max(int(round(image.height * ratio)), 1)
        image = image.resize((thumb_w, thumb_h), Image.BILINEAR)
        loaded.append((record, image))
    if not loaded:
        return
    cols = 2 if len(loaded) > 1 else 1
    rows = math.ceil(len(loaded) / cols)
    cell_h = max(image.height for _, image in loaded) + label_h
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for idx, (record, image) in enumerate(loaded):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * cell_h
        sheet.paste(image, (x, y + label_h))
        label = [
            f"{record['rank']:02d} {record['frame_name']}",
            f"max refined {fmt_m(record['refined_max_transl_l2_m'])}",
            f"base {fmt_m(record['base_max_transl_l2_m'])} seed {fmt_m(record['seed_max_transl_l2_m'])}",
            str(record["sequence_name"])[-42:],
        ]
        for line_idx, line in enumerate(label):
            draw.text((x + 6, y + 6 + line_idx * 16), line, fill=(242, 242, 242), font=font)
    sheet.save(output_dir / "bad_translation_top10_contact_sheet.jpg", quality=92)


def safe_float(value: Any) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def fmt_m(value: Any) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}m"


if __name__ == "__main__":
    main()
