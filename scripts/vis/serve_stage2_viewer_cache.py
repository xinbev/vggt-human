#!/usr/bin/env python
"""Serve a saved Stage2 point-cloud/SMPL cache without loading models or checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.sequence_sampling import uniform_sample_indices  # noqa: E402
from scripts.vis.viewer_cache_io import load_sequence_viewer_manifest  # noqa: E402


def main() -> None:
    args = parse_args()
    try:
        import viser  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("The cached viewer requires viser in the server environment") from exc

    cache_root, manifest = load_sequence_viewer_manifest(resolve_path(args.cache_dir))
    server = viser.ViserServer(port=int(args.port))
    scene = getattr(server, "scene", server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    viewer = CachedSequenceViewer(server, cache_root, manifest, args)
    print(
        f"[cached-viewer] ready: http://127.0.0.1:{int(args.port)} | "
        f"frames={len(viewer.frame_handles)} cache={cache_root}",
        flush=True,
    )
    viewer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--smpl-display-frames", type=int, default=50)
    parser.add_argument("--display-people", type=int, default=0, help="Maximum people displayed per frame; 0 shows all.")
    parser.add_argument("--initial-timestep", type=int, default=-1, help="Negative starts at the final cached frame.")
    parser.add_argument(
        "--viewer-mode",
        choices=["4D current frame", "3D accumulate", "Hybrid"],
        default="3D accumulate",
    )
    parser.add_argument("--show-track-ids", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class CachedSequenceViewer:
    def __init__(self, server: Any, cache_root: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
        self.server = server
        self.cache_root = cache_root
        self.manifest = manifest
        self.args = args
        self.faces = np.load(cache_root / str(manifest["smpl_faces_file"]), allow_pickle=False).astype(np.int32, copy=False)
        self.frame_handles: list[dict[str, list[Any]]] = []
        self.current_step = self._initial_timestep()
        self._build_scene()
        self._build_gui()
        self._update_visibility()

    def _initial_timestep(self) -> int:
        frame_count = len(self.manifest["frames"])
        requested = int(self.args.initial_timestep)
        return frame_count - 1 if requested < 0 else min(max(requested, 0), frame_count - 1)

    def _build_scene(self) -> None:
        frame_count = len(self.manifest["frames"])
        for position, record in enumerate(self.manifest["frames"]):
            frame_path = self.cache_root / str(record["file"])
            if not frame_path.is_file():
                raise FileNotFoundError(f"Missing cached frame: {frame_path}")
            handles: dict[str, list[Any]] = {"points": [], "humans": [], "labels": [], "people": []}
            with np.load(frame_path, allow_pickle=False) as data:
                points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
                colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
                if points.shape[0] > 0:
                    handles["points"].append(
                        add_point_cloud(
                            self.server,
                            f"/frames/{position:04d}/hsi_points",
                            points,
                            colors,
                            float(self.args.point_size),
                        )
                    )
                vertices = np.asarray(data["smpl_vertices"], dtype=np.float32)
                track_ids = np.asarray(data["smpl_track_ids"], dtype=np.int32)
                query_indices = np.asarray(data["smpl_query_indices"], dtype=np.int32)
                track_qualities = np.asarray(data["smpl_track_qualities"], dtype=np.float32)
                mesh_colors = np.asarray(data["smpl_colors"], dtype=np.uint8).reshape(-1, 3)
                for person_index in range(vertices.shape[0]):
                    track_id = int(track_ids[person_index])
                    query_index = int(query_indices[person_index])
                    mesh = vertices[person_index]
                    color = tuple(int(value) for value in mesh_colors[person_index])
                    person_handles: dict[str, list[Any]] = {"humans": [], "labels": []}
                    mesh_handle = add_mesh(
                            self.server,
                            f"/frames/{position:04d}/human_t{track_id}_q{query_index}",
                            mesh,
                            self.faces,
                            color,
                            1.0,
                        )
                    handles["humans"].append(mesh_handle)
                    person_handles["humans"].append(mesh_handle)
                    label_position = mesh[int(np.argmin(mesh[:, 1]))].copy()
                    label_position[1] -= 0.12
                    quality = float(track_qualities[person_index])
                    label_text = f"ID {track_id}" if not np.isfinite(quality) else f"ID {track_id}  {quality:.2f}"
                    label_handle = add_label(
                            self.server,
                            f"/frames/{position:04d}/track_t{track_id}_q{query_index}",
                            label_text,
                            label_position,
                        )
                    handles["labels"].append(label_handle)
                    person_handles["labels"].append(label_handle)
                    handles["people"].append(person_handles)
            self.frame_handles.append(handles)
            if (position + 1) % 25 == 0 or position + 1 == frame_count:
                print(f"[cached-viewer] loaded {position + 1}/{frame_count} frames", flush=True)

    def _build_gui(self) -> None:
        frame_count = len(self.frame_handles)
        target = max(1, min(frame_count, int(self.args.smpl_display_frames)))
        max_people = max(1, max((len(handles["people"]) for handles in self.frame_handles), default=0))
        requested_people = int(self.args.display_people)
        people_initial = max_people if requested_people <= 0 else min(max_people, requested_people)
        self.frame_info = add_text(self.server, "Frame Info", "")
        self.sampling_info = add_text(self.server, "SMPL Sampling", "")
        set_disabled(self.frame_info, True)
        set_disabled(self.sampling_info, True)
        self.timestep = add_slider(self.server, "Timestep", 0, frame_count - 1, 1, self.current_step)
        self.prev_button = add_button(self.server, "Prev Frame")
        self.next_button = add_button(self.server, "Next Frame")
        self.play = add_checkbox(self.server, "Playing", False)
        self.fps = add_slider(self.server, "FPS", 1, 30, 1, 6)
        self.mode = add_dropdown(
            self.server,
            "Mode",
            ["4D current frame", "3D accumulate", "Hybrid"],
            str(self.args.viewer_mode),
        )
        self.show_points = add_checkbox(self.server, "Show Point Clouds", True)
        self.show_smpl = add_checkbox(self.server, "Show SMPL", True)
        self.show_track_ids = add_checkbox(self.server, "Show Track IDs", bool(self.args.show_track_ids))
        self.display_people = add_slider(self.server, "Max People Per Frame", 1, max_people, 1, people_initial)
        self.smpl_display_frames = add_slider(self.server, "Accumulated SMPL Frames", 1, frame_count, 1, target)
        self.point_size = add_slider(self.server, "Point Size", 0.0005, 0.08, 0.0005, float(self.args.point_size))
        self.smpl_opacity = add_slider(self.server, "SMPL Opacity", 0.05, 1.0, 0.05, 1.0)
        for handle in (
            self.timestep,
            self.mode,
            self.show_points,
            self.show_smpl,
            self.show_track_ids,
            self.display_people,
            self.smpl_display_frames,
        ):
            bind_update(handle, self._on_gui_update)
        bind_update(self.point_size, self._on_point_size_update)
        bind_update(self.smpl_opacity, self._on_smpl_opacity_update)
        bind_click(self.prev_button, self._previous_frame)
        bind_click(self.next_button, self._next_frame)

    def run(self) -> None:
        try:
            while True:
                if bool(self.play.value):
                    self.current_step = (int(self.timestep.value) + 1) % len(self.frame_handles)
                    self.timestep.value = self.current_step
                    self._update_visibility()
                time.sleep(1.0 / max(float(self.fps.value), 1.0))
        except KeyboardInterrupt:
            print("[cached-viewer] stopped", flush=True)

    def _on_gui_update(self, _: Any = None) -> None:
        self.current_step = int(self.timestep.value)
        self._update_visibility()

    def _previous_frame(self, _: Any = None) -> None:
        self.current_step = (int(self.timestep.value) - 1) % len(self.frame_handles)
        self.timestep.value = self.current_step
        self._update_visibility()

    def _next_frame(self, _: Any = None) -> None:
        self.current_step = (int(self.timestep.value) + 1) % len(self.frame_handles)
        self.timestep.value = self.current_step
        self._update_visibility()

    def _on_point_size_update(self, _: Any = None) -> None:
        value = float(self.point_size.value)
        for handles in self.frame_handles:
            set_group_attr(handles["points"], "point_size", value)

    def _on_smpl_opacity_update(self, _: Any = None) -> None:
        value = float(self.smpl_opacity.value)
        for handles in self.frame_handles:
            set_group_attr(handles["humans"], "opacity", value)

    def _update_visibility(self) -> None:
        current = int(self.timestep.value)
        mode = str(self.mode.value)
        target = max(1, min(len(self.frame_handles), int(self.smpl_display_frames.value)))
        smpl_indices = set(uniform_sample_indices(len(self.frame_handles), target))
        display_people = max(1, int(self.display_people.value))
        for index, handles in enumerate(self.frame_handles):
            if mode == "3D accumulate":
                show_points_at_frame = index <= current
                show_human_at_frame = index <= current and index in smpl_indices
            elif mode == "Hybrid":
                show_points_at_frame = index <= current
                show_human_at_frame = index == current
            else:
                show_points_at_frame = index == current
                show_human_at_frame = index == current
            set_group_visible(handles["points"], bool(self.show_points.value) and show_points_at_frame)
            for person_rank, person_handles in enumerate(handles["people"]):
                show_person = bool(self.show_smpl.value) and show_human_at_frame and person_rank < display_people
                set_group_visible(person_handles["humans"], show_person)
                set_group_visible(person_handles["labels"], show_person and bool(self.show_track_ids.value))
        record = self.manifest["frames"][current]
        set_text_value(
            self.frame_info,
            f"{current + 1}/{len(self.frame_handles)} | {record['frame_id']} | "
            f"points={record['point_count']} people={record['people_count']}",
        )
        if mode == "3D accumulate":
            visible_count = sum(index <= current for index in smpl_indices)
            set_text_value(
                self.sampling_info,
                f"{visible_count} visible now / {target} across {len(self.frame_handles)} cached frames",
            )
        else:
            set_text_value(self.sampling_info, f"Target {target} applies to 3D accumulate mode")


def scene_api(server: Any) -> Any:
    return getattr(server, "scene", server)


def gui_api(server: Any) -> Any:
    return getattr(server, "gui", server)


def add_point_cloud(server: Any, name: str, points: np.ndarray, colors: np.ndarray, point_size: float) -> Any:
    api = scene_api(server)
    try:
        return api.add_point_cloud(name=name, points=points, colors=colors, point_size=point_size)
    except TypeError:
        return api.add_point_cloud(name, points, colors, point_size=point_size)


def add_mesh(
    server: Any,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> Any:
    api = scene_api(server)
    try:
        return api.add_mesh_simple(
            name=name,
            vertices=vertices,
            faces=faces,
            color=color,
            opacity=float(opacity),
        )
    except TypeError:
        handle = api.add_mesh_simple(name, vertices, faces, color=color)
        try:
            handle.opacity = float(opacity)
        except Exception:
            pass
        return handle


def add_label(server: Any, name: str, text: str, position: np.ndarray) -> Any:
    api = scene_api(server)
    try:
        return api.add_label(name=name, text=text, position=position)
    except TypeError:
        return api.add_label(name, text, position)


def add_slider(server: Any, name: str, minimum: float, maximum: float, step: float, initial: float) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_slider"):
        return api.add_slider(name, min=minimum, max=maximum, step=step, initial_value=initial)
    return server.add_gui_slider(name, min=minimum, max=maximum, step=step, initial_value=initial)


def add_checkbox(server: Any, name: str, initial: bool) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_checkbox"):
        return api.add_checkbox(name, initial_value=initial)
    return server.add_gui_checkbox(name, initial)


def add_dropdown(server: Any, name: str, options: list[str], initial: str) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_dropdown"):
        return api.add_dropdown(name, options=options, initial_value=initial)
    return server.add_gui_dropdown(name, options, initial)


def add_button(server: Any, name: str) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_button"):
        return api.add_button(name)
    return server.add_gui_button(name)


def add_text(server: Any, name: str, initial: str) -> Any:
    api = gui_api(server)
    if hasattr(api, "add_text"):
        return api.add_text(name, initial_value=initial)
    if hasattr(server, "add_gui_text"):
        return server.add_gui_text(name, initial)
    return None


def bind_update(handle: Any, callback: Any) -> None:
    if handle is None or not hasattr(handle, "on_update"):
        return
    try:
        handle.on_update(callback)
    except TypeError:
        handle.on_update(lambda event: callback(event))


def bind_click(handle: Any, callback: Any) -> None:
    if handle is None or not hasattr(handle, "on_click"):
        return
    try:
        handle.on_click(callback)
    except TypeError:
        handle.on_click(lambda event: callback(event))


def set_group_visible(handles: list[Any], visible: bool) -> None:
    set_group_attr(handles, "visible", bool(visible))


def set_group_attr(handles: list[Any], attribute: str, value: Any) -> None:
    for handle in handles:
        try:
            setattr(handle, attribute, value)
        except Exception:
            pass


def set_text_value(handle: Any, value: str) -> None:
    if handle is None:
        return
    try:
        handle.value = value
    except Exception:
        pass


def set_disabled(handle: Any, disabled: bool) -> None:
    if handle is None:
        return
    try:
        handle.disabled = bool(disabled)
    except Exception:
        pass


if __name__ == "__main__":
    main()
