#!/usr/bin/env python
"""Serve several full SequenceViewer caches as one transformable timeline.

Each cache is a layer.  The layer controls apply one rigid transform and one
uniform scale to every point, SMPL mesh, label, and camera belonging to it.
"""

from __future__ import annotations

import argparse
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
        bind_update(self.timestep, self._on_timeline_update)
        bind_click(self.prev_button, lambda *_: self._step(-1))
        bind_click(self.next_button, lambda *_: self._step(1))
        for control in (self.mode, self.smpl_display_frames, self.show_cameras, self.show_trajectories):
            bind_update(control, lambda *_: self._update_visibility())
        for layer_index, layer in enumerate(self.layers):
            with add_folder(self.server, f"Cache {layer_index}: {layer.name}"):
                layer.gui["scale"] = add_slider(self.server, "Overall Scale", 0.01, 20.0, 0.01, 1.0)
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
        layer.transform.scale = float(layer.gui["scale"].value)
        layer.transform.translation = np.asarray([float(layer.gui[key].value) for key in ("tx", "ty", "tz")], dtype=np.float32)
        layer.transform.rotation_deg = np.asarray([float(layer.gui[key].value) for key in ("rx", "ry", "rz")], dtype=np.float32)
        set_text_value(layer.gui["info"], f"scale={layer.transform.scale:.3f} | t={tuple(np.round(layer.transform.translation, 3))} | r={tuple(np.round(layer.transform.rotation_deg, 1))}")
        self._rebuild_layer(layer_index)

    def _reset_layer_transform(self, layer_index: int) -> None:
        layer = self.layers[layer_index]
        self._rebuilding = True
        try:
            for key in ("scale", "tx", "ty", "tz", "rx", "ry", "rz"):
                layer.gui[key].value = 1.0 if key == "scale" else 0.0
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

    def _rebuild_layer(self, layer_index: int) -> None:
        layer = self.layers[layer_index]
        self._remove_layer(layer)
        layer.frame_handles = []
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
        trajectory = layer.scene.get("camera_trajectory_hsi", layer.scene.get("camera_trajectory_raw"))
        if trajectory is not None and len(trajectory):
            points = layer.transform.points(trajectory)
            layer.trajectory_handles = [add_point_cloud(self.server, f"/caches/{layer_index:02d}_{layer.name}/camera_trajectory", points, camera_trajectory_colors(len(points)), max(float(self.args.point_size) * 2.5, 0.01))]

    def _update_visibility(self) -> None:
        current = int(self.timestep.value)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, help="Full cache directory; repeat for each cache.")
    parser.add_argument("--cache-name", action="append", default=[], help="Optional display name; repeat in cache order.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--camera-scale", type=float, default=0.05)
    parser.add_argument("--smpl-display-frames", type=int, default=50)
    parser.add_argument("--initial-timestep", type=int, default=0)
    parser.add_argument("--viewer-mode", choices=["4D current frame", "3D accumulate", "Hybrid"], default="Hybrid")
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-trajectories", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    if args.cache_name and len(args.cache_name) not in {1, len(args.cache_dir)}:
        raise ValueError("--cache-name must be supplied once or once per --cache-dir")
    names = args.cache_name if len(args.cache_name) == len(args.cache_dir) else [f"cache_{i}" for i in range(len(args.cache_dir))]
    layers: list[Layer] = []
    for index, cache_value in enumerate(args.cache_dir):
        cache_path = resolve_path(cache_value)
        scene, _, manifest_path = load_full_sequence_viewer_cache(cache_path)
        layer = Layer(name=str(names[index]).replace("/", "_"), cache_dir=cache_path, scene=scene)
        layers.append(layer)
        print(f"[multi-cache-viewer] loaded {len(scene['frames'])} frames from {manifest_path}", flush=True)

    server = viser.ViserServer(port=int(args.port))
    scene_api = getattr(server, "scene", server)
    if hasattr(scene_api, "set_up_direction"):
        scene_api.set_up_direction("-y")
    print(f"[multi-cache-viewer] serving {len(layers)} caches / {sum(len(layer.scene['frames']) for layer in layers)} timeline frames at http://127.0.0.1:{int(args.port)}", flush=True)
    MultiCacheViewer(server, vtf, layers, args).run()


if __name__ == "__main__":
    main()
