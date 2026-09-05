#!/usr/bin/env python3
"""Prepare TUM-Dynamics RGB/trajectory prefixes used by Human3R.

The released Human3R/TTT3R protocol first associates ``rgb.txt`` with
``groundtruth.txt`` (20 ms tolerance), then evaluates prefixes of the
associated stream.  This script keeps that protocol explicit and writes a
Human3R-compatible tree::

    <output>/<sequence>/rgb_90/*.png
    <output>/<sequence>/groundtruth_90.txt
    <output>/<sequence>/rgb_1000/*.png
    <output>/<sequence>/groundtruth_1000.txt

The source tree is never modified.  Existing destination files are replaced
only when their names are copied again; the script does not remove directories.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import shutil
from typing import Iterable


DEFAULT_LENGTHS = (50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
DEFAULT_SEQUENCE_GLOBS = ("rgbd_dataset_freiburg3_sitting_*", "rgbd_dataset_freiburg3_walking_*")


def read_file_list(path: Path) -> list[tuple[float, list[str]]]:
    """Read a TUM ``timestamp payload`` text file in timestamp order."""

    rows: list[tuple[float, list[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").replace("\t", " ").split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected timestamp and payload")
            try:
                timestamp = float(fields[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid timestamp {fields[0]!r}") from exc
            rows.append((timestamp, fields[1:]))
    rows.sort(key=lambda item: item[0])
    return rows


def associate(
    first: list[tuple[float, list[str]]],
    second: list[tuple[float, list[str]]],
    max_difference: float = 0.02,
) -> list[tuple[tuple[float, list[str]], tuple[float, list[str]]]]:
    """Greedily associate two timestamp streams, matching the TUM tools.

    Candidate generation only inspects neighbouring timestamps with
    ``bisect``.  This is equivalent to the usual all-pairs implementation for
    the TUM streams while avoiding an unnecessary quadratic scan.
    """

    second_times = [item[0] for item in second]
    candidates: list[tuple[float, int, int]] = []
    for first_index, (first_time, _) in enumerate(first):
        pivot = bisect.bisect_left(second_times, first_time)
        for second_index in range(max(0, pivot - 2), min(len(second), pivot + 3)):
            difference = abs(first_time - second[second_index][0])
            if difference < max_difference:
                candidates.append((difference, first_index, second_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[tuple[float, list[str]], tuple[float, list[str]]]] = []
    for _, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first[first_index], second[second_index]))
    matches.sort(key=lambda pair: pair[0][0])
    return matches


def parse_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(token.strip()) for token in raw.split(",") if token.strip()))
    if not values or any(value <= 0 for value in values):
        raise ValueError("--lengths must contain positive comma-separated integers")
    return values


def sequence_dirs(raw_root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in raw_root.glob(pattern) if path.is_dir())
    return sorted(paths, key=lambda path: path.name)


def prepare_sequence(
    source: Path,
    destination: Path,
    lengths: tuple[int, ...],
    sample_interval: int,
    association_tolerance: float,
    include_90: bool,
) -> dict[str, object]:
    rgb_file = source / "rgb.txt"
    gt_file = source / "groundtruth.txt"
    if not rgb_file.is_file() or not gt_file.is_file():
        raise FileNotFoundError(f"{source} must contain rgb.txt and groundtruth.txt")
    rgb_rows = read_file_list(rgb_file)
    gt_rows = read_file_list(gt_file)
    matches = associate(rgb_rows, gt_rows, max_difference=association_tolerance)
    sampled = matches[::sample_interval]
    if not sampled:
        raise RuntimeError(f"No RGB/GT timestamp pairs found in {source}")

    requested_lengths = list(lengths)
    if include_90 and 90 not in requested_lengths:
        requested_lengths.insert(0, 90)
    written: dict[str, int] = {}
    for requested in requested_lengths:
        count = min(requested, len(sampled))
        selected = sampled[:count]
        rgb_destination = destination / f"rgb_{requested}"
        rgb_destination.mkdir(parents=True, exist_ok=True)
        for (rgb_timestamp, rgb_payload), _ in selected:
            del rgb_timestamp
            source_frame = source / Path(rgb_payload[0])
            if not source_frame.is_file():
                raise FileNotFoundError(f"RGB frame listed in {rgb_file} does not exist: {source_frame}")
            shutil.copy2(source_frame, rgb_destination / source_frame.name)
        gt_destination = destination / f"groundtruth_{requested}.txt"
        with gt_destination.open("w", encoding="utf-8") as handle:
            for (rgb_pair, gt_pair) in selected:
                del rgb_pair
                gt_timestamp, gt_payload = gt_pair
                handle.write(" ".join([f"{gt_timestamp:.9f}", *gt_payload]) + "\n")
        written[str(requested)] = count
    return {
        "sequence": source.name,
        "source": str(source),
        "matched_pairs": len(matches),
        "sample_interval": sample_interval,
        "association_tolerance_s": association_tolerance,
        "prefix_lengths": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path, help="Directory containing the extracted TUM sequence folders")
    parser.add_argument("--output-root", required=True, type=Path, help="Human3R-compatible prepared tree")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)), help="Comma-separated prefix lengths")
    parser.add_argument("--sample-interval", type=int, default=1, help="Keep every Nth associated RGB/GT pair")
    parser.add_argument("--association-tolerance", type=float, default=0.02, help="Maximum RGB/GT timestamp difference in seconds")
    parser.add_argument("--no-90-prefix", action="store_true", help="Do not also create the Human3R-compatible rgb_90 prefix")
    parser.add_argument("--sequence", action="append", default=None, help="Prepare only this sequence name; may be repeated")
    args = parser.parse_args()
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    if args.association_tolerance <= 0:
        parser.error("--association-tolerance must be positive")
    lengths = parse_lengths(args.lengths)
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"Raw TUM root does not exist: {args.raw_root}")
    if args.sequence:
        sources = [args.raw_root / name for name in args.sequence]
        missing = [path for path in sources if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Requested sequence folders do not exist: " + ", ".join(map(str, missing)))
    else:
        sources = sequence_dirs(args.raw_root, DEFAULT_SEQUENCE_GLOBS)
    if not sources:
        raise RuntimeError(f"No TUM-Dynamics sequence folders found below {args.raw_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "Human3R/TTT3R TUM-Dynamics prefix preparation",
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "sequences": [],
    }
    for source in sources:
        info = prepare_sequence(
            source,
            args.output_root / source.name,
            lengths,
            args.sample_interval,
            args.association_tolerance,
            include_90=not args.no_90_prefix,
        )
        manifest["sequences"].append(info)
        print(f"[prepared] {source.name}: {info['matched_pairs']} matched pairs; {info['prefix_lengths']}", flush=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
