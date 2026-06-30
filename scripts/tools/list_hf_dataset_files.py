#!/usr/bin/env python
"""List files in a Hugging Face dataset subdirectory without downloading them."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_REPO_ID = "nguyenquivinhquang/BEDLAM"
DEFAULT_PATH_IN_REPO = "training_images"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT_DIR = Path("outputs/debug/hf_bedlam_training_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List files under a Hugging Face dataset path. This only queries the "
            "repo tree API and does not download the dataset payload."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional fnmatch pattern. Can be repeated, e.g. --include '*seq_000001*'.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Optional fnmatch pattern to remove paths. Can be repeated.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after this many matched files. 0 means no limit.",
    )
    parser.add_argument(
        "--group-depth",
        type=int,
        default=1,
        help="Group summary depth after --path-in-repo. 1 groups by direct child.",
    )
    return parser.parse_args()


def path_matches(path: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(fnmatch.fnmatch(path, pattern) for pattern in include):
        return False
    if exclude and any(fnmatch.fnmatch(path, pattern) for pattern in exclude):
        return False
    return True


def group_key(path: str, prefix: str, depth: int) -> str:
    rel = path
    clean_prefix = prefix.strip("/")
    if clean_prefix and rel.startswith(clean_prefix + "/"):
        rel = rel[len(clean_prefix) + 1 :]
    parts = rel.split("/")
    if depth <= 0:
        return "."
    return "/".join(parts[:depth]) if len(parts) >= depth else rel


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return ""
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def main() -> None:
    args = parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it in the current Python "
            "environment with: python -m pip install huggingface_hub"
        ) from exc

    api = HfApi(endpoint=args.endpoint, token=args.token)
    entries = api.list_repo_tree(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        recursive=True,
    )

    rows: list[dict[str, str | int]] = []
    groups: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {"file_count": 0, "total_size": 0, "first_path": ""}
    )

    for entry in entries:
        if not hasattr(entry, "size") or not hasattr(entry, "path"):
            continue
        if not path_matches(entry.path, args.include, args.exclude):
            continue

        size = int(entry.size or 0)
        key = group_key(entry.path, args.path_in_repo, args.group_depth)
        rows.append(
            {
                "path": entry.path,
                "size_bytes": size,
                "size_human": format_size(size),
                "group": key,
            }
        )
        groups[key]["file_count"] = int(groups[key]["file_count"]) + 1
        groups[key]["total_size"] = int(groups[key]["total_size"]) + size
        if not groups[key]["first_path"]:
            groups[key]["first_path"] = entry.path

        if args.max_files and len(rows) >= args.max_files:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths_path = args.output_dir / "files.txt"
    csv_path = args.output_dir / "files.csv"
    groups_path = args.output_dir / "groups.csv"
    selected_path = args.output_dir / "selected.txt"

    paths_path.write_text("\n".join(str(row["path"]) for row in rows) + ("\n" if rows else ""))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "size_human", "group"])
        writer.writeheader()
        writer.writerows(rows)

    with groups_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "file_count", "total_size_bytes", "total_size_human", "first_path"],
        )
        writer.writeheader()
        for key in sorted(groups):
            total_size = int(groups[key]["total_size"])
            writer.writerow(
                {
                    "group": key,
                    "file_count": groups[key]["file_count"],
                    "total_size_bytes": total_size,
                    "total_size_human": format_size(total_size),
                    "first_path": groups[key]["first_path"],
                }
            )

    if not selected_path.exists():
        selected_path.write_text(
            "# Copy selected file paths from files.txt here, one path per line.\n"
            "# Lines beginning with # are ignored by the download script.\n",
            encoding="utf-8",
        )

    total_size = sum(int(row["size_bytes"]) for row in rows)
    print(f"Repo       : {args.repo_type}:{args.repo_id}")
    print(f"Endpoint   : {args.endpoint}")
    print(f"Path       : {args.path_in_repo}")
    print(f"Files      : {len(rows)}")
    print(f"Total size : {format_size(total_size)}")
    print(f"File list  : {paths_path}")
    print(f"CSV list   : {csv_path}")
    print(f"Groups     : {groups_path}")
    print(f"Selection  : {selected_path}")


if __name__ == "__main__":
    main()
