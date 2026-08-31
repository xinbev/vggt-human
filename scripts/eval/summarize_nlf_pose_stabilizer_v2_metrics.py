#!/usr/bin/env python3
"""Create one compact six-metric table from 3DPW and EMDB-1 evaluation JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("3dpw", "emdb1")
METRICS = ("pa_mpjpe_mm", "mpjpe_mm", "pve_mm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    reports = {}
    for dataset in DATASETS:
        path = root / dataset / f"{dataset}_nlf_pose_stabilizer_v2_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset report: {path}")
        reports[dataset] = json.loads(path.read_text(encoding="utf-8"))
    table = {"metric_protocol": reports["3dpw"].get("metric_protocol"), "rows": {}}
    for output_name in ("nlf_base", "nlf_pose_temporal"):
        row = {}
        for dataset in DATASETS:
            values = reports[dataset].get(output_name, {})
            row[dataset] = {metric: values.get(metric) for metric in METRICS}
            row[dataset]["metric_rows"] = reports[dataset].get("num_metric_rows", 0)
            row[dataset]["temporal_applied_rate"] = reports[dataset].get("coverage", {}).get("temporal_applied_rate")
        table["rows"][output_name] = row
    (root / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RGB -> VGGT -> NLF -> Pose Temporal Stabilizer V2",
        "",
        f"Protocol: `{table['metric_protocol']}`. All primary metrics are mm. Each result scores only unique temporal-window centre frames.",
        "",
        "| Output | 3DPW PA-MPJPE | 3DPW MPJPE | 3DPW PVE | EMDB-1 PA-MPJPE | EMDB-1 MPJPE | EMDB-1 PVE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in table["rows"].items():
        d3, de = row["3dpw"], row["emdb1"]
        lines.append(
            f"| {name} | {d3['pa_mpjpe_mm']:.2f} | {d3['mpjpe_mm']:.2f} | {d3['pve_mm']:.2f} | "
            f"{de['pa_mpjpe_mm']:.2f} | {de['mpjpe_mm']:.2f} | {de['pve_mm']:.2f} |"
        )
    lines.extend(["", "Temporal coverage is recorded in each dataset JSON and must be reported with the table."])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
