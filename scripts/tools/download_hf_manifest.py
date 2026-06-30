#!/usr/bin/env python
"""Download selected Hugging Face dataset files from a manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO_ID = "nguyenquivinhquang/BEDLAM"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MANIFEST = Path("outputs/debug/hf_bedlam_training_images/selected.txt")
DEFAULT_LOCAL_DIR = Path("/home/zhw/xyb_space/bedlam/hf_bedlam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download exact HF files listed in a manifest.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    files: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        files.append(line)
    if not files:
        raise ValueError(f"Manifest has no selected files: {path}")
    return files


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it in the current Python "
            "environment with: python -m pip install huggingface_hub"
        ) from exc

    files = read_manifest(args.manifest)
    print(f"Repo      : {args.repo_type}:{args.repo_id}")
    print(f"Endpoint  : {args.endpoint}")
    print(f"Manifest  : {args.manifest}")
    print(f"Local dir : {args.local_dir}")
    print(f"Files     : {len(files)}")

    if args.dry_run:
        for filename in files:
            print(f"[DRY-RUN] {filename}")
        return

    args.local_dir.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {filename}")
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=filename,
            local_dir=args.local_dir,
            token=args.token,
        )

    print("Download complete.")


if __name__ == "__main__":
    main()
