#!/usr/bin/env python
"""Interactively tune and save one scale multiplier per cached RICH window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.serve_stage2_viewer_cache import (  # noqa: E402
    add_button,
    add_checkbox,
    add_dropdown,
    add_mesh,
    add_point_cloud,
    add_slider,
    add_text,
    bind_click,
    bind_update,
    set_text_value,
)
from vggt_omega.evaluation.rich_physical_grounding import GroundEstimatorConfig  # noqa: E402
from vggt_omega.evaluation.rich_scale_oracle import (  # noqa: E402
    CACHE_FORMAT,
    evaluate_scale_window_cache,
    load_scale_window_cache,
    reconstruct_scale_frame,
)


SCALE_FILE_FORMAT = "vggt_omega_rich_manual_scale_selections_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scale-file", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.008)
    parser.add_argument("--initial-window", type=int, default=0)
    parser.add_argument("--scale-min", type=float, default=0.10)
    parser.add_argument("--scale-max", type=float, default=10.0)
    return parser.parse_args()


class RichManualScaleViewer:
    def __init__(self, server: Any, args: argparse.Namespace) -> None:
        self.server = server
        self.args = args
        self.cache_dir = args.cache_dir.expanduser().resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"Unsupported cache manifest: {self.manifest_path}")
        self.windows = list(self.manifest.get("windows", []))
        if not self.windows:
            raise RuntimeError(f"No cached windows in {self.manifest_path}")
        self.faces = np.load(
            self.cache_dir / str(self.manifest["smpl_faces_file"]), allow_pickle=False
        ).astype(np.int32, copy=False)
        self.ground_config = GroundEstimatorConfig(**self.manifest["ground_config"])
        self.scale_file = (
            args.scale_file.expanduser().resolve()
            if args.scale_file is not None
            else self.cache_dir / "manual_scales.json"
        )
        self.selections = self._load_selections()
        self.window_index = min(max(int(args.initial_window), 0), len(self.windows) - 1)
        self.frame_index = 0
        self.cache: dict[str, np.ndarray] = {}
        self.geometry_handles: list[Any] = []
        self._switching = False
        self._load_window(self.window_index)
        self._build_gui()
        self._rebuild_geometry()

    def _load_selections(self) -> dict[str, Any]:
        if self.scale_file.is_file():
            payload = json.loads(self.scale_file.read_text(encoding="utf-8"))
            if payload.get("format") != SCALE_FILE_FORMAT:
                raise ValueError(f"Unsupported manual scale file: {self.scale_file}")
            return payload
        return {
            "format": SCALE_FILE_FORMAT,
            "cache_manifest": str(self.manifest_path),
            "windows": {},
        }

    def _load_window(self, index: int) -> None:
        self.window_index = int(index)
        record = self.windows[self.window_index]
        self.cache = load_scale_window_cache(self.cache_dir / str(record["cache_file"]))
        self.frame_index = min(self.frame_index, int(record["frame_count"]) - 1)

    def _build_gui(self) -> None:
        options = [str(record["window_id"]) for record in self.windows]
        current_id = options[self.window_index]
        self.window_choice = add_dropdown(self.server, "Window", options, current_id)
        self.previous_window = add_button(self.server, "Previous Window")
        self.next_window = add_button(self.server, "Next Window")
        self.frame = add_slider(
            self.server,
            "Frame in Window",
            0,
            max(int(self.windows[self.window_index]["frame_count"]) - 1, 0),
            1,
            self.frame_index,
        )
        initial_scale = self._saved_scale(current_id)
        self.scale_log10 = add_slider(
            self.server,
            "Scale Multiplier (log10)",
            float(np.log10(self.args.scale_min)),
            float(np.log10(self.args.scale_max)),
            0.005,
            float(np.log10(initial_scale)),
        )
        self.save_scale = add_button(self.server, "Save Window Scale")
        self.preview_metrics = add_button(self.server, "Preview Window Metrics")
        self.window_info = add_text(self.server, "Window Status", "")
        self.frame_info = add_text(self.server, "Frame Status", "")
        self.metric_info = add_text(self.server, "Metric Preview", "Not computed")
        self.include_human_points = add_checkbox(
            self.server, "Keep Human Point Cloud", True
        )
        self.accumulate_frames = add_checkbox(
            self.server, "Accumulate Window Frames", False
        )
        bind_update(self.window_choice, self._on_window_choice)
        bind_click(self.previous_window, self._on_previous_window)
        bind_click(self.next_window, self._on_next_window)
        bind_update(self.frame, self._on_frame)
        bind_update(self.scale_log10, self._on_scale)
        bind_update(self.include_human_points, self._on_display_option)
        bind_update(self.accumulate_frames, self._on_display_option)
        bind_click(self.save_scale, self._on_save)
        bind_click(self.preview_metrics, self._on_preview_metrics)

    def _saved_scale(self, window_id: str) -> float:
        record = self.selections.get("windows", {}).get(window_id, {})
        return float(record.get("scale_multiplier", 1.0))

    def _current_scale(self) -> float:
        return float(10.0 ** float(self.scale_log10.value))

    def _switch_window(self, index: int) -> None:
        index = min(max(int(index), 0), len(self.windows) - 1)
        self._switching = True
        try:
            self._load_window(index)
            record = self.windows[index]
            self.window_choice.value = str(record["window_id"])
            try:
                self.frame.max = max(int(record["frame_count"]) - 1, 0)
            except Exception:
                pass
            self.frame_index = 0
            self.frame.value = 0
            self.scale_log10.value = float(np.log10(self._saved_scale(str(record["window_id"]))))
            set_text_value(self.metric_info, "Not computed for this scale")
        finally:
            self._switching = False
        self._rebuild_geometry()

    def _on_window_choice(self, _: Any = None) -> None:
        if self._switching:
            return
        selected = str(self.window_choice.value)
        self._switch_window(next(i for i, row in enumerate(self.windows) if row["window_id"] == selected))

    def _on_previous_window(self, _: Any = None) -> None:
        self._switch_window(self.window_index - 1)

    def _on_next_window(self, _: Any = None) -> None:
        self._switch_window(self.window_index + 1)

    def _on_frame(self, _: Any = None) -> None:
        if self._switching:
            return
        maximum = int(self.windows[self.window_index]["frame_count"]) - 1
        self.frame_index = min(max(int(self.frame.value), 0), maximum)
        if int(self.frame.value) != self.frame_index:
            self._switching = True
            try:
                self.frame.value = self.frame_index
            finally:
                self._switching = False
        self._rebuild_geometry()

    def _on_scale(self, _: Any = None) -> None:
        if not self._switching:
            set_text_value(self.metric_info, "Scale changed; preview metrics again before saving")
            self._rebuild_geometry()

    def _on_display_option(self, _: Any = None) -> None:
        if not self._switching:
            self._rebuild_geometry()

    def _on_save(self, _: Any = None) -> None:
        record = self.windows[self.window_index]
        window_id = str(record["window_id"])
        scale = self._current_scale()
        self.selections.setdefault("windows", {})[window_id] = {
            "scale_multiplier": scale,
            "vid": record["vid"],
            "window_index": int(record["window_index"]),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.selections["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.scale_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.scale_file.with_suffix(self.scale_file.suffix + ".tmp")
        temporary.write_text(json.dumps(self.selections, indent=2), encoding="utf-8")
        temporary.replace(self.scale_file)
        self._update_info(status=f"SAVED x{scale:.5f}")
        print(f"[saved] {window_id} scale={scale:.8g} -> {self.scale_file}", flush=True)

    def _on_preview_metrics(self, _: Any = None) -> None:
        rows, report = evaluate_scale_window_cache(self.cache, self._current_scale(), self.ground_config)
        metrics = report.get("metrics")
        if not metrics:
            set_text_value(self.metric_info, "No valid ground estimate")
            return
        set_text_value(
            self.metric_info,
            (
                f"valid={metrics['valid_frames']}/{len(rows)} | collision={metrics['collision_ratio_pct']:.3f}% | "
                f"penetrate={metrics['penetrate_cm']:.3f}cm | float={metrics['float_cm']:.3f}cm | "
                f"max={metrics['penetration_max_cm']:.3f}cm | abs={metrics['mean_abs_clearance_cm']:.3f}cm"
            ),
        )

    def _rebuild_geometry(self) -> None:
        for handle in self.geometry_handles:
            try:
                handle.remove()
            except Exception:
                try:
                    handle.visible = False
                except Exception:
                    pass
        self.geometry_handles = []
        valid = bool(self.cache["selected_valid"][self.frame_index])
        frame_indices = (
            list(range(self.frame_index + 1))
            if bool(self.accumulate_frames.value)
            else [self.frame_index]
        )
        for frame_index in frame_indices:
            frame_valid = bool(self.cache["selected_valid"][frame_index])
            points, colors, vertices = reconstruct_scale_frame(
                self.cache,
                frame_index,
                self._current_scale(),
                self.ground_config,
                # Viewer mode intentionally keeps human pixels visible. The
                # metric evaluator continues to exclude them independently.
                exclude_person=not bool(self.include_human_points.value),
            )
            points_np = points.numpy()
            if points_np.size:
                self.geometry_handles.append(
                    add_point_cloud(
                        self.server,
                        f"/rich_scale/scene/frame_{frame_index:04d}",
                        points_np,
                        colors,
                        float(self.args.point_size),
                    )
                )
            if frame_valid:
                self.geometry_handles.append(
                    add_mesh(
                        self.server,
                        f"/rich_scale/person/frame_{frame_index:04d}",
                        vertices.numpy(),
                        self.faces,
                        (232, 142, 82),
                        0.95,
                    )
                )
        self._update_info(status="saved" if self._is_current_saved() else "unsaved")

    def _is_current_saved(self) -> bool:
        window_id = str(self.windows[self.window_index]["window_id"])
        saved = self.selections.get("windows", {}).get(window_id)
        return bool(saved) and abs(float(saved["scale_multiplier"]) - self._current_scale()) < 1e-7

    def _update_info(self, status: str) -> None:
        record = self.windows[self.window_index]
        scale = self._current_scale()
        model_scale = float(self.cache["model_scale"][self.frame_index])
        set_text_value(
            self.window_info,
            f"{self.window_index + 1}/{len(self.windows)} | {record['vid']} | "
            f"window {record['window_index']} ({record['frame_count']} frames, one scale) | {status}",
        )
        set_text_value(
            self.frame_info,
            (
                f"frame {self.frame_index + 1}/{record['frame_count']} | source={record['source_frame_ids'][self.frame_index]} | "
                f"detected={bool(self.cache['selected_valid'][self.frame_index])} | model scale={model_scale:.5g} | "
                f"manual x{scale:.5f} | effective={model_scale * scale:.5g}"
            ),
        )

    def run(self) -> None:
        print(
            f"[rich-manual-scale] http://127.0.0.1:{self.args.port} | "
            f"windows={len(self.windows)} | save={self.scale_file}",
            flush=True,
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[rich-manual-scale] stopped", flush=True)


def main() -> None:
    args = parse_args()
    if args.scale_min <= 0 or args.scale_max <= args.scale_min:
        raise ValueError("Expected 0 < --scale-min < --scale-max")
    try:
        import viser
    except ImportError as exc:
        raise ImportError("This viewer requires viser in the server environment") from exc
    server = viser.ViserServer(port=int(args.port))
    scene = getattr(server, "scene", server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    RichManualScaleViewer(server, args).run()


if __name__ == "__main__":
    main()
