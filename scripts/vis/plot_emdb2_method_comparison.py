#!/usr/bin/env python3
"""Draw a publication-style EMDB-2 method comparison.

With only ``--x-metric`` the script produces a one-dimensional dot plot. Once
the CSV contains a second metric, add ``--y-metric`` to produce a two-dimensional
trade-off plot like the reference figure supplied for this task.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRIC_PRESETS = {
    "wa_mpjpe_mm": (r"WA-MPJPE$_{100}$ (mm)", "min"),
    "w_mpjpe_mm": ("W-MPJPE (mm)", "min"),
    "rte_percent": ("RTE (%)", "min"),
    "fps": ("Inference Speed (FPS)", "max"),
    "latency_s": ("Latency (s)", "min"),
    "gpu_memory_gb": ("GPU Memory (GB)", "min"),
    "params_m": ("Parameters (M)", "min"),
}

COLORS = {
    "baseline": "#159A8C",
    "ours": "#D9573F",
}

STYLE_PRESETS = {
    "baseline_ff": {
        "c": "#159A8C",
        "marker": "o",
        "s": 105,
        "edgecolors": "#202020",
        "linewidths": 0.7,
        "zorder": 3,
    },
    "baseline_tto": {
        "c": "#D9573F",
        "marker": "o",
        "s": 105,
        "edgecolors": "#202020",
        "linewidths": 0.7,
        "zorder": 3,
    },
    "unicon_ff": {
        "c": "#159A8C",
        "marker": "o",
        "s": 105,
        "edgecolors": "#202020",
        "linewidths": 0.7,
        "zorder": 4,
    },
    "unicon_tto": {
        "c": "#D9573F",
        "marker": "*",
        "s": 230,
        "edgecolors": "#202020",
        "linewidths": 0.7,
        "zorder": 4,
    },
    "ours_base": {
        "c": "#4472C4",
        "marker": "*",
        "s": 320,
        "edgecolors": "#202020",
        "linewidths": 0.9,
        "zorder": 5,
    },
    "ours_hsi": {
        "c": "#E68632",
        "marker": "*",
        "s": 320,
        "edgecolors": "#202020",
        "linewidths": 0.9,
        "zorder": 5,
    },
}


@dataclass(frozen=True)
class Record:
    method: str
    group: str
    is_ours: bool
    plot_style: str
    link_group: str
    values: dict[str, float | None]
    label_dx: float
    label_dy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Method-metric CSV.")
    parser.add_argument("--x-metric", default="wa_mpjpe_mm")
    parser.add_argument(
        "--y-metric",
        default=None,
        help="Second metric. Omit it to draw the available one-dimensional comparison.",
    )
    parser.add_argument("--x-label", default=None)
    parser.add_argument("--y-label", default=None)
    parser.add_argument("--x-direction", choices=("min", "max"), default=None)
    parser.add_argument("--y-direction", choices=("min", "max"), default=None)
    parser.add_argument("--x-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--title", default=None)
    parser.add_argument("--show-pareto", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated: png,pdf,svg.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def optional_float(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric value: {raw!r}")
    return value


def read_records(path: Path) -> tuple[list[Record], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"method", "group", "is_ours"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        reserved = required | {"plot_style", "link_group", "label_dx", "label_dy"}
        metric_fields = [field for field in fields if field not in reserved]
        records = []
        for row_number, row in enumerate(reader, start=2):
            method = (row.get("method") or "").strip()
            if not method:
                raise ValueError(f"{path}:{row_number}: method cannot be empty")
            raw_is_ours = (row.get("is_ours") or "0").strip().lower()
            if raw_is_ours not in {"0", "1", "false", "true", "no", "yes"}:
                raise ValueError(f"{path}:{row_number}: invalid is_ours={raw_is_ours!r}")
            parsed_label_dx = optional_float(row.get("label_dx"))
            parsed_label_dy = optional_float(row.get("label_dy"))
            is_ours = raw_is_ours in {"1", "true", "yes"}
            records.append(
                Record(
                    method=method,
                    group=(row.get("group") or "Baseline").strip(),
                    is_ours=is_ours,
                    plot_style=(row.get("plot_style") or ("ours_base" if is_ours else "baseline_ff")).strip(),
                    link_group=(row.get("link_group") or "").strip(),
                    values={field: optional_float(row.get(field)) for field in metric_fields},
                    label_dx=6.0 if parsed_label_dx is None else parsed_label_dx,
                    label_dy=6.0 if parsed_label_dy is None else parsed_label_dy,
                )
            )
    if not records:
        raise ValueError(f"{path} contains no method rows")
    return records, metric_fields


def metric_spec(metric: str, label: str | None, direction: str | None) -> tuple[str, str]:
    preset_label, preset_direction = METRIC_PRESETS.get(
        metric, (metric.replace("_", " ").title(), "max")
    )
    resolved_direction = direction or preset_direction
    arrow = "↓" if resolved_direction == "min" else "↑"
    resolved_label = label or preset_label
    if not resolved_label.rstrip().endswith(("↓", "↑")):
        resolved_label = f"{resolved_label} {arrow}"
    return resolved_label, resolved_direction


def require_metric(records: Iterable[Record], fields: list[str], metric: str) -> None:
    if metric not in fields:
        raise ValueError(f"unknown metric {metric!r}; available columns: {', '.join(fields)}")
    missing = [record.method for record in records if record.values[metric] is None]
    if missing:
        methods = ", ".join(missing)
        raise ValueError(
            f"metric {metric!r} is blank for: {methods}. Fill the CSV values before drawing "
            "a quantitative comparison; placeholder coordinates are intentionally unsupported."
        )


def style_for(record: Record) -> dict[str, object]:
    try:
        return STYLE_PRESETS[record.plot_style]
    except KeyError as error:
        choices = ", ".join(sorted(STYLE_PRESETS))
        raise ValueError(
            f"unknown plot_style {record.plot_style!r} for {record.method}; choose from: {choices}"
        ) from error


def add_legend(axis, records: list[Record]) -> None:
    from matplotlib.lines import Line2D

    legend_labels = {
        "baseline_ff": "Unified Feed-Forward",
        "baseline_tto": "Test-Time Optimization",
        "unicon_ff": "UniCon3R",
        "unicon_tto": "UniCon3R*",
        "ours_base": "Ours",
        "ours_hsi": "Ours + HSI",
    }
    present_styles = {record.plot_style for record in records}
    handles = []
    for style_name, label in legend_labels.items():
        if style_name not in present_styles:
            continue
        style = STYLE_PRESETS[style_name]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=str(style["marker"]),
                color="none",
                markerfacecolor=str(style["c"]),
                markeredgecolor="#202020",
                markersize=11 if style["marker"] == "*" else 7,
                label=label,
            )
        )
    axis.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.96,
        edgecolor="#B8B8B8",
    )


def draw_variant_links(axis, records: list[Record], x_metric: str, y_metric: str) -> None:
    groups: dict[str, list[Record]] = {}
    for record in records:
        if record.link_group:
            groups.setdefault(record.link_group, []).append(record)

    for linked_records in groups.values():
        if len(linked_records) < 2:
            continue
        for first, second in zip(linked_records, linked_records[1:]):
            x0 = float(first.values[x_metric])
            y0 = float(first.values[y_metric])
            x1 = float(second.values[x_metric])
            y1 = float(second.values[y_metric])
            curve_x = []
            curve_y = []
            log_y0 = math.log(y0)
            log_y1 = math.log(y1)
            for index in range(41):
                t = index / 40.0
                curve_x.append(
                    (1.0 - t) * x0
                    + t * x1
                    - 0.10 * abs(x1 - x0) * math.sin(math.pi * t)
                )
                curve_y.append(math.exp((1.0 - t) * log_y0 + t * log_y1))
            axis.plot(
                curve_x,
                curve_y,
                color="#9A9A9A",
                linestyle="--",
                linewidth=1.05,
                zorder=1,
            )


def pareto_frontier(
    records: list[Record], x_metric: str, y_metric: str, x_direction: str, y_direction: str
) -> list[Record]:
    frontier = []
    for candidate in records:
        cx = candidate.values[x_metric]
        cy = candidate.values[y_metric]
        assert cx is not None and cy is not None
        dominated = False
        for other in records:
            if other is candidate:
                continue
            ox = other.values[x_metric]
            oy = other.values[y_metric]
            assert ox is not None and oy is not None
            x_at_least = ox <= cx if x_direction == "min" else ox >= cx
            y_at_least = oy <= cy if y_direction == "min" else oy >= cy
            x_strict = ox < cx if x_direction == "min" else ox > cx
            y_strict = oy < cy if y_direction == "min" else oy > cy
            if x_at_least and y_at_least and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: float(item.values[x_metric]))


def draw_single_metric(axis, records: list[Record], metric: str, label: str) -> None:
    ordered = sorted(records, key=lambda item: float(item.values[metric]))
    for index, record in enumerate(ordered):
        value = record.values[metric]
        assert value is not None
        axis.scatter(value, index, **style_for(record))
        axis.annotate(
            f"{value:g}",
            (value, index),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold" if record.is_ours else "normal",
        )
    axis.set_yticks(range(len(ordered)), [record.method for record in ordered])
    axis.invert_yaxis()
    axis.set_xlabel(label)
    axis.set_ylabel("")
    axis.margins(x=0.18, y=0.16)


def draw_tradeoff(
    axis,
    records: list[Record],
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    x_direction: str,
    y_direction: str,
    show_pareto: bool,
) -> None:
    draw_variant_links(axis, records, x_metric, y_metric)
    if show_pareto:
        frontier = pareto_frontier(records, x_metric, y_metric, x_direction, y_direction)
        axis.plot(
            [record.values[x_metric] for record in frontier],
            [record.values[y_metric] for record in frontier],
            color="#8A8A8A",
            linestyle="--",
            linewidth=1.1,
            zorder=1,
            label="Pareto frontier",
        )

    for record in records:
        x_value = record.values[x_metric]
        y_value = record.values[y_metric]
        assert x_value is not None and y_value is not None
        axis.scatter(x_value, y_value, **style_for(record))
        axis.annotate(
            record.method,
            (x_value, y_value),
            xytext=(record.label_dx, record.label_dy),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold"
            if record.is_ours or record.group.lower() == "unicon3r"
            else "normal",
            ha="left" if record.label_dx >= 0 else "right",
            va="bottom" if record.label_dy >= 0 else "top",
            zorder=5,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.margins(x=0.14, y=0.18)


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    records, fields = read_records(args.csv)
    require_metric(records, fields, args.x_metric)
    if args.y_metric:
        require_metric(records, fields, args.y_metric)

    x_label, x_direction = metric_spec(args.x_metric, args.x_label, args.x_direction)
    y_label = None
    y_direction = None
    if args.y_metric:
        y_label, y_direction = metric_spec(args.y_metric, args.y_label, args.y_direction)

    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.35, 4.5), constrained_layout=True)
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="#D7D7D7", linestyle=":", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.8)
    axis.set_xscale(args.x_scale)

    if args.y_metric:
        assert y_label is not None and y_direction is not None
        axis.set_yscale(args.y_scale)
        draw_tradeoff(
            axis,
            records,
            args.x_metric,
            args.y_metric,
            x_label,
            y_label,
            x_direction,
            y_direction,
            args.show_pareto,
        )
        if args.y_metric == "fps" and args.y_scale == "log":
            fps_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            axis.set_yticks(fps_ticks)
            axis.set_yticklabels(["0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0"])
        output_name = args.output_name or f"{args.x_metric}_vs_{args.y_metric}"
    else:
        draw_single_metric(axis, records, args.x_metric, x_label)
        output_name = args.output_name or f"{args.x_metric}_preview"

    if args.title:
        axis.set_title(args.title, pad=10)
    add_legend(axis, records)

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must contain at least one of: png, pdf, svg")
    invalid_formats = sorted(set(formats).difference({"png", "pdf", "svg"}))
    if invalid_formats:
        raise ValueError(f"unsupported output formats: {invalid_formats}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output_path = args.output_dir / f"{output_name}.{file_format}"
        figure.savefig(
            output_path,
            dpi=args.dpi if file_format == "png" else None,
            facecolor="white",
        )
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
