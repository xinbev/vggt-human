#!/usr/bin/env python
"""Serve several full SequenceViewer caches as one transformable timeline.

Each cache is a layer.  The layer controls apply one rigid transform and one
uniform scale to every point, SMPL mesh, label, and camera belonging to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import load_full_sequence_viewer_cache  # noqa: E402
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import (  # noqa: E402
    add_camera,
    add_checkbox,
    add_dropdown,
    add_folder,
    add_label,
    add_mesh,
    add_point_cloud,
    add_slider,
    bind_click,
    bind_update,
    camera_trajectory_colors,
    ensure_viser_available,
    remove_handle,
    set_group_visible,
    set_handle_disabled,
    set_text_value,
)


@dataclass
class LayerTransform:
    scale: float = 1.0
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def matrix(self) -> np.ndarray:
        rx, ry, rz = np.deg2rad(self.rotation_deg.astype(np.float64))
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return (rzm @ rym @ rxm).astype(np.float32)

    def points(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        return (points @ self.matrix().T) * np.float32(self.scale) + self.translation

    def camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        result = dict(camera)
        rotation = self.matrix()
        result["position"] = self.points(np.asarray(camera["position"], dtype=np.float32))[0]
        result["rotation_c2w"] = rotation @ np.asarray(camera["rotation_c2w"], dtype=np.float32)
        return result


@dataclass
class Layer:
    name: str
    cache_dir: Path
    scene: dict[str, Any]
    transform: LayerTransform = field(default_factory=LayerTransform)
    frame_handles: list[dict[str, list[Any]]] = field(default_factory=list)
    boundary_handles: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    trajectory_handles: list[Any] = field(default_factory=list)
    gui: dict[str, Any] = field(default_factory=dict)


class MultiCacheViewer:
    def __init__(self, server: Any, transforms: Any, layers: list[Layer], args: argparse.Namespace) -> None:
        self.server = server
        self.transforms = transforms
        self.layers = layers
        self.args = args
        self.timeline: list[tuple[int, int]] = []
        for layer_index, layer in enumerate(layers):
            for frame_index in range(len(layer.scene["frames"])):
                self.timeline.append((layer_index, frame_index))
        if not self.timeline:
            raise ValueError("No frames found in the supplied caches")
        self.current_step = min(max(int(args.initial_timestep), 0), len(self.timeline) - 1)
        self._rebuilding = False
        self.layout_path = Path(args.layout_json).expanduser().resolve() if args.layout_json else None
        self.alignment_active = bool(args.alignment_preview)
        self.selected_pair = 0
        self._build_scene()
        self._build_gui()
        self._update_visibility()

    def run(self) -> None:
        try:
            while True:
                if bool(self.play.value):
                    self.current_step = (int(self.timestep.value) + 1) % len(self.timeline)
                    self.timestep.value = self.current_step
                    self._update_visibility()
                time.sleep(1.0 / max(float(self.fps.value), 1.0))
        except KeyboardInterrupt:
            print("[multi-cache-viewer] stopped", flush=True)

    def _build_scene(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            layer.frame_handles = []
            layer.boundary_handles = {}
            layer.trajectory_handles = []
            self._build_one_layer(layer_index, layer)

    def _build_gui(self) -> None:
        self.frame_info = self._text("Timeline", "")
        self.timestep = add_slider(self.server, "Global Timestep", 0, len(self.timeline) - 1, 1, self.current_step)
        self.prev_button = self._button("Prev Frame")
        self.next_button = self._button("Next Frame")
        self.play = add_checkbox(self.server, "Playing", False)
        self.fps = add_slider(self.server, "FPS", 1, 30, 1, 6)
        self.mode = add_dropdown(self.server, "Mode", ["4D current frame", "3D accumulate", "Hybrid"], str(self.args.viewer_mode))
        self.smpl_display_frames = add_slider(self.server, "SMPL Display Frames", 1, max(1, len(self.timeline)), 1, min(max(1, int(self.args.smpl_display_frames)), len(self.timeline)))
        self.show_cameras = add_checkbox(self.server, "Show Cameras", bool(self.args.show_cameras))
        self.show_trajectories = add_checkbox(self.server, "Show Camera Trajectories", bool(self.args.show_trajectories))
        pair_options = self._pair_options()
        self.alignment_preview = add_checkbox(self.server, "Boundary Alignment Preview", self.alignment_active)
        self.alignment_pair = add_dropdown(self.server, "Alignment Pair", pair_options, pair_options[0])
        self.alignment_info = self._text("Alignment", "Off")
        self.save_layout = self._button("Save Multi-cache Layout")
        self.load_layout = self._button("Load Multi-cache Layout")
        set_handle_disabled(self.alignment_preview, len(pair_options) <= 1)
        set_handle_disabled(self.alignment_pair, len(pair_options) <= 1)
        bind_update(self.timestep, self._on_timeline_update)
        bind_click(self.prev_button, lambda *_: self._step(-1))
        bind_click(self.next_button, lambda *_: self._step(1))
        for control in (self.mode, self.smpl_display_frames, self.show_cameras, self.show_trajectories):
            bind_update(control, lambda *_: self._update_visibility())
        bind_update(self.alignment_preview, self._on_alignment_preview_update)
        bind_update(self.alignment_pair, self._on_alignment_pair_update)
        bind_click(self.save_layout, self._save_layout)
        bind_click(self.load_layout, self._load_layout_from_disk)
        for layer_index, layer in enumerate(self.layers):
            with add_folder(self.server, f"Cache {layer_index}: {layer.name}"):
                layer.gui["scale"] = add_slider(self.server, "Overall Scale (log10)", -2.0, 2.0, 0.01, float(np.clip(np.log10(layer.transform.scale), -2.0, 2.0)))
                layer.gui["tx"] = add_slider(self.server, "Translate X", -100.0, 100.0, 0.01, 0.0)
                layer.gui["ty"] = add_slider(self.server, "Translate Y", -100.0, 100.0, 0.01, 0.0)
                layer.gui["tz"] = add_slider(self.server, "Translate Z", -100.0, 100.0, 0.01, 0.0)
                layer.gui["rx"] = add_slider(self.server, "Rotate X (deg)", -180.0, 180.0, 1.0, 0.0)
                layer.gui["ry"] = add_slider(self.server, "Rotate Y (deg)", -180.0, 180.0, 1.0, 0.0)
                layer.gui["rz"] = add_slider(self.server, "Rotate Z (deg)", -180.0, 180.0, 1.0, 0.0)
                layer.gui["reset"] = self._button("Reset Cache Transform")
                layer.gui["info"] = self._text("Transform", "scale=1.000 | t=(0,0,0) | r=(0,0,0)")
                for control in layer.gui.values():
                    if control is not layer.gui["reset"] and control is not layer.gui["info"]:
                        bind_update(control, lambda *_args, index=layer_index: self._on_layer_transform(index))
                bind_click(layer.gui["reset"], lambda *_args, index=layer_index: self._reset_layer_transform(index))

    def _text(self, name: str, value: str) -> Any:
        from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import add_text
        return add_text(self.server, name, value)

    def _button(self, name: str) -> Any:
        from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import add_button
        return add_button(self.server, name)

    def _on_timeline_update(self, *_: Any) -> None:
        self.current_step = int(self.timestep.value)
        self._update_visibility()

    def _step(self, delta: int) -> None:
        self.current_step = (int(self.timestep.value) + int(delta)) % len(self.timeline)
        self.timestep.value = self.current_step
        self._update_visibility()

    def _on_layer_transform(self, layer_index: int) -> None:
        if self._rebuilding:
            return
        layer = self.layers[layer_index]
        layer.transform.scale = float(10.0 ** float(layer.gui["scale"].value))
        layer.transform.translation = np.asarray([float(layer.gui[key].value) for key in ("tx", "ty", "tz")], dtype=np.float32)
        layer.transform.rotation_deg = np.asarray([float(layer.gui[key].value) for key in ("rx", "ry", "rz")], dtype=np.float32)
        set_text_value(layer.gui["info"], f"scale={layer.transform.scale:.3f} | t={tuple(np.round(layer.transform.translation, 3))} | r={tuple(np.round(layer.transform.rotation_deg, 1))}")
        self._rebuild_layer(layer_index, boundary_only=self.alignment_active)

    def _reset_layer_transform(self, layer_index: int) -> None:
        layer = self.layers[layer_index]
        self._rebuilding = True
        try:
            for key in ("scale", "tx", "ty", "tz", "rx", "ry", "rz"):
                layer.gui[key].value = 0.0 if key == "scale" else 0.0
        finally:
            self._rebuilding = False
        self._on_layer_transform(layer_index)

    def _remove_layer(self, layer: Layer) -> None:
        for frame_handles in layer.frame_handles:
            for handles in frame_handles.values():
                for handle in handles:
                    remove_handle(handle)
        for handle in layer.trajectory_handles:
            remove_handle(handle)
        for endpoint_handles in layer.boundary_handles.values():
            for handles in endpoint_handles.values():
                for handle in handles:
                    remove_handle(handle)

    def _rebuild_layer(self, layer_index: int, boundary_only: bool = False) -> None:
        layer = self.layers[layer_index]
        if boundary_only:
            for endpoint_handles in layer.boundary_handles.values():
                for handles in endpoint_handles.values():
                    for handle in handles:
                        remove_handle(handle)
            layer.boundary_handles = {}
            self._build_boundary_handles(layer_index, layer)
        else:
            self._remove_layer(layer)
            layer.frame_handles = []
            layer.boundary_handles = {}
            layer.trajectory_handles = []
            self._build_one_layer(layer_index, layer)
        self._update_visibility()

    def _build_one_layer(self, layer_index: int, layer: Layer) -> None:
        for local_index, frame in enumerate(layer.scene["frames"]):
            handles: dict[str, list[Any]] = {"points": [], "humans": [], "labels": [], "cameras": []}
            prefix = f"/caches/{layer_index:02d}_{layer.name}/frames/{local_index:04d}"
            points = layer.transform.points(frame.get("hsi_points", frame.get("raw_points", np.zeros((0, 3)))))
            colors = frame.get("hsi_colors", frame.get("raw_colors", np.zeros((0, 3), dtype=np.uint8)))
            if len(points):
                handles["points"].append(add_point_cloud(self.server, f"{prefix}/points", points, colors, float(self.args.point_size)))
            for person in frame.get("people", []):
                vertices = person.get("hsi_vertices", person.get("base_vertices"))
                if vertices is None:
                    continue
                vertices_world = layer.transform.points(vertices)
                track_id = int(person.get("track_id", -1))
                query_index = int(person.get("query_index", -1))
                handles["humans"].append(add_mesh(self.server, f"{prefix}/smpl_t{track_id}_q{query_index}", vertices_world, person["faces"], tuple(int(v) for v in person.get("color", (232, 142, 82)))))
                label_local = np.asarray(vertices, dtype=np.float32)[int(np.argmin(np.asarray(vertices)[:, 1]))].copy()
                label_local[1] -= 0.12
                label_position = layer.transform.points(label_local)[0]
                handles["labels"].append(add_label(self.server, f"{prefix}/label_t{track_id}_q{query_index}", f"ID {track_id}", label_position))
            camera = frame.get("hsi_camera") or frame.get("raw_camera")
            if camera is not None:
                handles["cameras"].append(add_camera(self.server, self.transforms, f"{prefix}/camera", layer.transform.camera(camera), float(self.args.camera_scale) * layer.transform.scale, (255, 176, 0)))
            layer.frame_handles.append(handles)
        self._build_boundary_handles(layer_index, layer)
        trajectory = layer.scene.get("camera_trajectory_hsi", layer.scene.get("camera_trajectory_raw"))
        if trajectory is not None and len(trajectory):
            points = layer.transform.points(trajectory)
            layer.trajectory_handles = [add_point_cloud(self.server, f"/caches/{layer_index:02d}_{layer.name}/camera_trajectory", points, camera_trajectory_colors(len(points)), max(float(self.args.point_size) * 2.5, 0.01))]

    def _build_boundary_handles(self, layer_index: int, layer: Layer) -> None:
        """Build full HSI point clouds for the endpoints used during alignment."""
        frame_count = len(layer.scene["frames"])
        endpoints = {"first": 0, "last": frame_count - 1}
        for endpoint, local_index in endpoints.items():
            frame = layer.scene["frames"][local_index]
            handles: dict[str, list[Any]] = {"points": [], "humans": [], "labels": [], "cameras": []}
            prefix = f"/alignment/{layer_index:02d}_{layer.name}/{endpoint}"
            points = layer.transform.points(frame.get("hsi_points_full", frame.get("hsi_points", np.zeros((0, 3)))))
            colors = frame.get("hsi_colors_full", frame.get("hsi_colors", np.zeros((0, 3), dtype=np.uint8)))
            if len(points):
                handles["points"].append(add_point_cloud(self.server, f"{prefix}/full_hsi_points", points, colors, float(self.args.point_size) * 1.15))
            for person in frame.get("people", []):
                vertices = person.get("hsi_vertices", person.get("base_vertices"))
                if vertices is None:
                    continue
                vertices_world = layer.transform.points(vertices)
                track_id = int(person.get("track_id", -1))
                query_index = int(person.get("query_index", -1))
                handles["humans"].append(add_mesh(self.server, f"{prefix}/smpl_t{track_id}_q{query_index}", vertices_world, person["faces"], tuple(int(v) for v in person.get("color", (232, 142, 82)))))
                label_local = np.asarray(vertices, dtype=np.float32)[int(np.argmin(np.asarray(vertices)[:, 1]))].copy()
                label_local[1] -= 0.12
                handles["labels"].append(add_label(self.server, f"{prefix}/label_t{track_id}_q{query_index}", f"ID {track_id}", layer.transform.points(label_local)[0]))
            camera = frame.get("hsi_camera") or frame.get("raw_camera")
            if camera is not None:
                handles["cameras"].append(add_camera(self.server, self.transforms, f"{prefix}/camera", layer.transform.camera(camera), float(self.args.camera_scale) * layer.transform.scale, (255, 176, 0)))
            layer.boundary_handles[endpoint] = handles

    def _pair_options(self) -> list[str]:
        if len(self.layers) < 2:
            return ["No adjacent cache pair"]
        return [f"{index}: {self.layers[index].name} (last)  ->  {self.layers[index + 1].name} (first)" for index in range(len(self.layers) - 1)]

    def _on_alignment_preview_update(self, *_: Any) -> None:
        if self._rebuilding:
            return
        self.alignment_active = bool(self.alignment_preview.value)
        if not self.alignment_active:
            for layer_index in range(len(self.layers)):
                self._rebuild_layer(layer_index, boundary_only=False)
        self._update_visibility()

    def _on_alignment_pair_update(self, *_: Any) -> None:
        if self._rebuilding:
            return
        try:
            self.selected_pair = max(0, int(str(self.alignment_pair.value).split(":", 1)[0]))
        except (ValueError, IndexError):
            self.selected_pair = 0
        self._update_visibility()

    def _layout_payload(self) -> dict[str, Any]:
        return {
            "format": "vggt_omega_multi_cache_layout_v1",
            "cache_dirs": [str(layer.cache_dir) for layer in self.layers],
            "cache_names": [layer.name for layer in self.layers],
            "transforms": [
                {
                    "scale": float(layer.transform.scale),
                    "translation": [float(value) for value in layer.transform.translation],
                    "rotation_deg": [float(value) for value in layer.transform.rotation_deg],
                }
                for layer in self.layers
            ],
            "alignment_preview": bool(self.alignment_active),
            "alignment_pair": int(self.selected_pair),
            "timeline_step": int(self.timestep.value),
        }

    def _save_layout(self, *_: Any) -> None:
        if self.layout_path is None:
            set_text_value(self.alignment_info, "No layout path configured.")
            return
        self.layout_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout_path.write_text(json.dumps(self._layout_payload(), indent=2, ensure_ascii=False), encoding="utf-8")
        set_text_value(self.alignment_info, f"Saved layout: {self.layout_path}")

    def _load_layout_from_disk(self, *_: Any) -> None:
        if self.layout_path is None or not self.layout_path.is_file():
            set_text_value(self.alignment_info, f"Layout not found: {self.layout_path}")
            return
        payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        self._apply_layout_payload(payload)
        set_text_value(self.alignment_info, f"Loaded layout: {self.layout_path}")

    def _apply_layout_payload(self, payload: dict[str, Any]) -> None:
        transforms = payload.get("transforms", [])
        self._rebuilding = True
        try:
            for index, layer in enumerate(self.layers):
                record = transforms[index] if index < len(transforms) else {}
                layer.transform.scale = max(0.01, float(record.get("scale", 1.0)))
                layer.transform.translation = np.asarray(record.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                layer.transform.rotation_deg = np.asarray(record.get("rotation_deg", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                if layer.gui:
                    layer.gui["scale"].value = float(np.log10(layer.transform.scale))
                    for key, value in zip(("tx", "ty", "tz"), layer.transform.translation):
                        layer.gui[key].value = float(value)
                    for key, value in zip(("rx", "ry", "rz"), layer.transform.rotation_deg):
                        layer.gui[key].value = float(value)
            self.selected_pair = max(0, min(len(self.layers) - 2, int(payload.get("alignment_pair", 0)))) if len(self.layers) > 1 else 0
            self.alignment_active = bool(payload.get("alignment_preview", self.alignment_active))
            if hasattr(self, "alignment_pair") and len(self._pair_options()) > 1:
                self.alignment_pair.value = self._pair_options()[self.selected_pair]
            if hasattr(self, "alignment_preview"):
                self.alignment_preview.value = self.alignment_active
            if hasattr(self, "timestep"):
                self.timestep.value = min(max(int(payload.get("timeline_step", 0)), 0), len(self.timeline) - 1)
        finally:
            self._rebuilding = False
        for index in range(len(self.layers)):
            self._rebuild_layer(index, boundary_only=False)

    def _boundary_selection(self) -> set[tuple[int, str]]:
        if not self.alignment_active or len(self.layers) < 2:
            return set()
        pair = max(0, min(len(self.layers) - 2, self.selected_pair))
        return {(pair, "last"), (pair + 1, "first")}

    def _update_visibility(self) -> None:
        current = int(self.timestep.value)
        if self.alignment_active:
            selected_boundaries = self._boundary_selection()
            for global_index, (layer_index, local_index) in enumerate(self.timeline):
                handles = self.layers[layer_index].frame_handles[local_index]
                set_group_visible(handles["points"], False)
                set_group_visible(handles["humans"], False)
                set_group_visible(handles["labels"], False)
                set_group_visible(handles["cameras"], False)
            for layer_index, layer in enumerate(self.layers):
                for endpoint, handles in layer.boundary_handles.items():
                    visible = (layer_index, endpoint) in selected_boundaries
                    set_group_visible(handles["points"], visible)
                    set_group_visible(handles["humans"], visible)
                    set_group_visible(handles["labels"], visible)
                    set_group_visible(handles["cameras"], visible and bool(self.show_cameras.value))
                set_group_visible(layer.trajectory_handles, False)
            if selected_boundaries:
                left_index, left_endpoint = sorted(selected_boundaries)[0]
                right_index, right_endpoint = sorted(selected_boundaries)[1]
                set_text_value(
                    self.alignment_info,
                    f"Boundary preview: {self.layers[left_index].name} last frame + {self.layers[right_index].name} first frame | full HSI points + SMPL",
                )
            return
        for layer in self.layers:
            for endpoint_handles in layer.boundary_handles.values():
                for handles in endpoint_handles.values():
                    set_group_visible(handles, False)
        mode = str(self.mode.value)
        target = max(1, min(len(self.timeline), int(self.smpl_display_frames.value)))
        sample_indices = set(np.asarray(np.linspace(0, len(self.timeline) - 1, target), dtype=np.int64).tolist())
        for global_index, (layer_index, local_index) in enumerate(self.timeline):
            handles = self.layers[layer_index].frame_handles[local_index]
            if mode == "4D current frame":
                show_points = global_index == current
                show_humans = global_index == current
            elif mode == "Hybrid":
                show_points = global_index <= current
                show_humans = global_index == current
            else:
                show_points = global_index <= current
                show_humans = global_index in sample_indices
            set_group_visible(handles["points"], show_points)
            set_group_visible(handles["humans"], show_humans)
            set_group_visible(handles["labels"], show_humans)
            set_group_visible(handles["cameras"], bool(self.show_cameras.value) and show_points)
        for layer in self.layers:
            set_group_visible(layer.trajectory_handles, bool(self.show_trajectories.value))
        layer_index, local_index = self.timeline[current]
        frame = self.layers[layer_index].scene["frames"][local_index]
        set_text_value(self.frame_info, f"{current + 1}/{len(self.timeline)} | cache={self.layers[layer_index].name} | cache frame={local_index + 1}/{len(self.layers[layer_index].scene['frames'])} | source={frame.get('source_frame_index', frame.get('frame_index', local_index))}")
        set_text_value(self.alignment_info, "Off | enable Boundary Alignment Preview to align adjacent caches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", default=[], help="Full cache directory; repeat for each cache.")
    parser.add_argument("--cache-name", action="append", default=[], help="Optional display name; repeat in cache order.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--camera-scale", type=float, default=0.05)
    parser.add_argument("--smpl-display-frames", type=int, default=50)
    parser.add_argument("--initial-timestep", type=int, default=0)
    parser.add_argument("--viewer-mode", choices=["4D current frame", "3D accumulate", "Hybrid"], default="Hybrid")
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layout-json", default="", help="JSON file used to save/load cache transforms and alignment state.")
    parser.add_argument("--alignment-preview", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    layout_path = resolve_path(args.layout_json) if args.layout_json else None
    layout_payload: dict[str, Any] = {}
    if layout_path is not None and layout_path.is_file():
        layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
    cli_cache_dirs = list(args.cache_dir)
    if not args.cache_dir:
        args.cache_dir = [str(value) for value in layout_payload.get("cache_dirs", [])]
    if not args.cache_dir:
        raise ValueError("At least one --cache-dir is required, or --layout-json must contain cache_dirs")
    if cli_cache_dirs and layout_payload.get("cache_dirs"):
        requested_paths = [str(resolve_path(value)) for value in cli_cache_dirs]
        saved_paths = [str(resolve_path(value)) for value in layout_payload.get("cache_dirs", [])]
        if requested_paths != saved_paths:
            print("[multi-cache-viewer] layout cache list does not match CLI caches; ignoring saved transforms", flush=True)
            layout_payload = {}
    if not args.cache_name and layout_payload.get("cache_names"):
        args.cache_name = [str(value) for value in layout_payload["cache_names"]]
    if "alignment_preview" in layout_payload:
        args.alignment_preview = bool(layout_payload["alignment_preview"])
    if "timeline_step" in layout_payload:
        args.initial_timestep = int(layout_payload["timeline_step"])
    if args.cache_name and len(args.cache_name) not in {1, len(args.cache_dir)}:
        raise ValueError("--cache-name must be supplied once or once per --cache-dir")
    names = (
        args.cache_name * len(args.cache_dir)
        if len(args.cache_name) == 1
        else (args.cache_name if len(args.cache_name) == len(args.cache_dir) else [f"cache_{i}" for i in range(len(args.cache_dir))])
    )
    layers: list[Layer] = []
    for index, cache_value in enumerate(args.cache_dir):
        cache_path = resolve_path(cache_value)
        scene, _, manifest_path = load_full_sequence_viewer_cache(cache_path)
        layer = Layer(name=str(names[index]).replace("/", "_"), cache_dir=cache_path, scene=scene)
        if index < len(layout_payload.get("transforms", [])):
            transform = layout_payload["transforms"][index]
            layer.transform.scale = max(0.01, float(transform.get("scale", 1.0)))
            layer.transform.translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            layer.transform.rotation_deg = np.asarray(transform.get("rotation_deg", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        layers.append(layer)
        print(f"[multi-cache-viewer] loaded {len(scene['frames'])} frames from {manifest_path}", flush=True)

    server = viser.ViserServer(port=int(args.port))
    scene_api = getattr(server, "scene", server)
    if hasattr(scene_api, "set_up_direction"):
        scene_api.set_up_direction("-y")
    print(f"[multi-cache-viewer] serving {len(layers)} caches / {sum(len(layer.scene['frames']) for layer in layers)} timeline frames at http://127.0.0.1:{int(args.port)}", flush=True)
    viewer = MultiCacheViewer(server, vtf, layers, args)
    if layout_payload:
        viewer._apply_layout_payload(layout_payload)
    viewer.run()


if __name__ == "__main__":
    main()
