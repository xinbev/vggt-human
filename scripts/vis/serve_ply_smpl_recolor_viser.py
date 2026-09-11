#!/usr/bin/env python
"""Small standalone Viser UI for opening and recoloring an SMPL PLY mesh."""

from __future__ import annotations

import argparse
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ply",
        action="append",
        default=[],
        help="Optional PLY loaded when the server starts. Repeat or separate entries with ';'.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/vis/ply_smpl_recolor"))
    return parser.parse_args()


def gui_api(server: Any) -> Any:
    return getattr(server, "gui", server)


def scene_api(server: Any) -> Any:
    return getattr(server, "scene", server)


def bind_update(handle: Any, callback: Any) -> None:
    handle.on_update(callback)


def bind_click(handle: Any, callback: Any) -> None:
    handle.on_click(callback)


def set_status(handle: Any, message: str) -> None:
    handle.value = message


def add_dropdown(server: Any, name: str, options: list[str], initial: str) -> Any:
    api = gui_api(server)
    try:
        return api.add_dropdown(name, options=options, initial_value=initial)
    except AttributeError:
        return server.add_gui_dropdown(name, options, initial)


def add_checkbox(server: Any, name: str, initial: bool) -> Any:
    api = gui_api(server)
    try:
        return api.add_checkbox(name, initial_value=initial)
    except AttributeError:
        return server.add_gui_checkbox(name, initial)


@dataclass
class MeshEntry:
    key: str
    label: str
    source_name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]
    opacity: float = 1.0
    handle: Any = None


class PlyRecolorViewer:
    def __init__(self, server: Any, output_dir: Path) -> None:
        self.server = server
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.meshes: list[MeshEntry] = []
        self.download_handle: Any = None
        self._next_id = 1
        self._syncing_gui = False
        self._build_gui()

    def _build_gui(self) -> None:
        gui = gui_api(self.server)
        self.status = gui.add_text("Status", initial_value="Upload mesh PLY files or load server-side paths.")
        try:
            self.status.disabled = True
        except Exception:
            pass

        self.upload = gui.add_upload_button("Open local PLY", mime_type="application/octet-stream")
        self.path = gui.add_text("Server-side PLY path(s)", initial_value="")
        self.load_path = gui.add_button("Load path(s)")
        self.selected = add_dropdown(self.server, "Selected mesh", ["none"], "none")
        self.apply_all = add_checkbox(self.server, "Apply color/opacity to all", False)
        self.color = gui.add_rgb("SMPL color", initial_value=(169, 207, 216))
        self.opacity = gui.add_slider("Opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0)
        self.fit_camera = gui.add_button("Fit camera")
        self.save = gui.add_button("Export selected PLY")
        self.save_all = gui.add_button("Export all PLYs")

        if hasattr(self.upload, "on_upload"):
            self.upload.on_upload(self._on_upload)
        else:
            bind_update(self.upload, self._on_upload)
        bind_click(self.load_path, self._on_load_path)
        bind_update(self.selected, self._on_selected_change)
        bind_update(self.color, self._on_appearance_change)
        bind_update(self.opacity, self._on_appearance_change)
        bind_click(self.fit_camera, self._on_fit_camera)
        bind_click(self.save, self._on_save)
        bind_click(self.save_all, self._on_save_all)

    def _on_upload(self, _: Any = None) -> None:
        try:
            upload = self.upload.value
            if upload is None:
                return
            self.load_bytes(bytes(upload.content), str(upload.name))
        except Exception as exc:
            set_status(self.status, f"Load failed: {exc}")

    def _on_load_path(self, _: Any = None) -> None:
        try:
            raw = str(self.path.value).strip().strip('"')
            if not raw:
                raise ValueError("enter one or more PLY paths first")
            paths = parse_path_list(raw)
            if not paths:
                raise ValueError("enter one or more PLY paths first")
            loaded = []
            for raw_path in paths:
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = ROOT / path
                self.load_path_file(path.resolve())
                loaded.append(str(path.resolve()))
            self.path.value = "; ".join(loaded)
        except Exception as exc:
            set_status(self.status, f"Load failed: {exc}")

    def load_path_file(self, path: Path) -> None:
        if path.suffix.lower() != ".ply":
            raise ValueError("only .ply files are supported")
        if not path.is_file():
            raise FileNotFoundError(path)
        self.load_bytes(path.read_bytes(), path.name)

    def load_bytes(self, payload: bytes, filename: str) -> None:
        if Path(filename).suffix.lower() != ".ply":
            raise ValueError("only .ply files are supported")
        vertices, faces, source_color = read_mesh_ply(payload)
        with self.lock:
            key = f"mesh_{self._next_id:03d}"
            self._next_id += 1
            label = self._make_label(Path(filename).name, key)
            mesh = MeshEntry(
                key=key,
                label=label,
                source_name=Path(filename).name,
                vertices=vertices,
                faces=faces,
                color=source_color,
                opacity=float(self.opacity.value),
            )
            self.meshes.append(mesh)
            self._render_mesh(mesh)
            self._refresh_selected_options(mesh.label)
            self._set_controls_from_mesh(mesh)
        set_status(
            self.status,
            f"Loaded {label}: {len(vertices):,} vertices, {len(faces):,} triangles; total meshes: {len(self.meshes)}",
        )
        self._fit_all_clients()

    def _on_appearance_change(self, _: Any = None) -> None:
        if self._syncing_gui:
            return
        targets = self._target_meshes()
        if not targets:
            return
        with self.lock:
            color = tuple(int(value) for value in self.color.value)
            opacity = float(self.opacity.value)
            for mesh in targets:
                mesh.color = color
                mesh.opacity = opacity
                self._render_mesh(mesh)

    def _render_mesh(self, mesh: MeshEntry) -> None:
        if mesh.handle is not None:
            try:
                mesh.handle.remove()
            except Exception:
                pass
        mesh.handle = scene_api(self.server).add_mesh_simple(
            name=f"/smpl/{mesh.key}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=mesh.color,
            opacity=mesh.opacity,
        )

    def _on_fit_camera(self, _: Any = None) -> None:
        self._fit_all_clients()

    def _fit_all_clients(self) -> None:
        if not self.meshes:
            return
        vertices = np.concatenate([mesh.vertices for mesh in self.meshes], axis=0)
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.1)
        for client in getattr(self.server, "get_clients", lambda: {})().values():
            try:
                client.camera.up = (0.0, -1.0, 0.0)
                client.camera.look_at = center
                client.camera.position = center + np.asarray((0.0, 0.0, -2.5 * radius))
            except Exception:
                pass

    def _on_save(self, _: Any = None) -> None:
        try:
            mesh = self._selected_mesh()
            if mesh is None:
                raise RuntimeError("load a mesh first")
            payload, output = self._export_mesh(mesh)
            output.write_bytes(payload)
            self._replace_download(payload, output.name)
            set_status(self.status, f"Saved: {output}")
        except Exception as exc:
            set_status(self.status, f"Export failed: {exc}")

    def _on_save_all(self, _: Any = None) -> None:
        try:
            if not self.meshes:
                raise RuntimeError("load a mesh first")
            outputs = []
            for mesh in self.meshes:
                payload, output = self._export_mesh(mesh)
                output.write_bytes(payload)
                outputs.append(output)
            set_status(self.status, f"Saved {len(outputs)} files under: {self.output_dir}")
        except Exception as exc:
            set_status(self.status, f"Export failed: {exc}")

    def _replace_download(self, payload: bytes, filename: str) -> None:
        if self.download_handle is not None:
            try:
                self.download_handle.remove()
            except Exception:
                pass
        gui = gui_api(self.server)
        if hasattr(gui, "add_download_button"):
            self.download_handle = gui.add_download_button(
                "Download last export",
                filename=filename,
                content=payload,
            )

    def _make_label(self, filename: str, key: str) -> str:
        stem = Path(filename).name or "smpl.ply"
        existing = {mesh.label for mesh in self.meshes}
        label = stem
        if label in existing:
            label = f"{stem} ({key})"
        return label

    def _refresh_selected_options(self, selected_label: str | None = None) -> None:
        options = [mesh.label for mesh in self.meshes] or ["none"]
        try:
            self.selected.options = options
        except Exception:
            pass
        if selected_label in options:
            self.selected.value = selected_label
        elif self.selected.value not in options:
            self.selected.value = options[0]

    def _on_selected_change(self, _: Any = None) -> None:
        mesh = self._selected_mesh()
        if mesh is None:
            return
        self._set_controls_from_mesh(mesh)
        set_status(self.status, f"Selected {mesh.label}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")

    def _selected_mesh(self) -> MeshEntry | None:
        value = str(getattr(self.selected, "value", ""))
        return next((mesh for mesh in self.meshes if mesh.label == value), self.meshes[-1] if self.meshes else None)

    def _target_meshes(self) -> list[MeshEntry]:
        if bool(getattr(self.apply_all, "value", False)):
            return list(self.meshes)
        mesh = self._selected_mesh()
        return [mesh] if mesh is not None else []

    def _export_mesh(self, mesh: MeshEntry) -> tuple[bytes, Path]:
        payload = write_mesh_ply(mesh.vertices, mesh.faces, mesh.color)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(mesh.source_name).stem).strip("._") or mesh.key
        color = mesh.color
        output = self.output_dir / f"{stem}_{mesh.key}_rgb_{color[0]}_{color[1]}_{color[2]}.ply"
        return payload, output

    def _set_controls_from_mesh(self, mesh: MeshEntry) -> None:
        self._syncing_gui = True
        try:
            self.color.value = mesh.color
            self.opacity.value = float(mesh.opacity)
        finally:
            self._syncing_gui = False


def parse_path_list(raw: str) -> list[str]:
    return [item.strip().strip('"').strip("'") for item in raw.split(";") if item.strip().strip('"').strip("'")]


PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def read_mesh_ply(payload: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    data_offset, ply_format, elements = parse_ply_header(payload)
    rows = read_ply_rows(payload, data_offset, ply_format, elements)
    vertex_rows = rows.get("vertex", [])
    face_rows = rows.get("face", [])
    if not vertex_rows:
        raise ValueError("PLY contains no vertices")
    vertices = np.asarray(
        [[row["x"], row["y"], row["z"]] for row in vertex_rows],
        dtype=np.float32,
    )
    triangles: list[tuple[int, int, int]] = []
    for row in face_rows:
        indices = row.get("vertex_indices", row.get("vertex_index"))
        if indices is None:
            indices = next((value for value in row.values() if isinstance(value, list)), None)
        if indices is None or len(indices) < 3:
            continue
        root = int(indices[0])
        triangles.extend((root, int(indices[i]), int(indices[i + 1])) for i in range(1, len(indices) - 1))
    faces = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if faces.size == 0:
        raise ValueError("PLY must contain mesh faces, not only points")
    if vertices.size == 0 or not np.isfinite(vertices).all():
        raise ValueError("PLY vertices are empty or non-finite")
    valid_faces = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid_faces]
    if faces.size == 0:
        raise ValueError("PLY has no valid triangle faces")

    color = (169, 207, 216)
    if all(all(channel in row for channel in ("red", "green", "blue")) for row in vertex_rows):
        vertex_colors = np.asarray(
            [[row["red"], row["green"], row["blue"]] for row in vertex_rows],
            dtype=np.float64,
        )
        if vertex_colors.size and float(vertex_colors.max()) <= 1.0:
            vertex_colors *= 255.0
        median = np.clip(np.median(vertex_colors, axis=0), 0, 255).astype(np.uint8)
        color = tuple(int(value) for value in median)
    return vertices, faces, color


def parse_ply_header(payload: bytes) -> tuple[int, str, list[dict[str, Any]]]:
    match = re.search(br"end_header[ \t]*\r?\n", payload)
    if match is None:
        raise ValueError("invalid PLY header")
    lines = payload[:match.end()].decode("ascii", errors="strict").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY file")
    ply_format = ""
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines[1:]:
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info", "end_header"}:
            continue
        if parts[0] == "format":
            ply_format = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["properties"].append(("scalar", parts[1], parts[2]))
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {ply_format or '<missing>'}")
    return match.end(), ply_format, elements


def read_ply_rows(
    payload: bytes,
    data_offset: int,
    ply_format: str,
    elements: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if ply_format == "ascii":
        return read_ascii_ply_rows(payload[data_offset:], elements)
    endian = "<" if ply_format == "binary_little_endian" else ">"
    return read_binary_ply_rows(payload, data_offset, elements, endian)


def read_ascii_ply_rows(data: bytes, elements: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    line_iter = iter(data.decode("ascii", errors="strict").splitlines())
    result: dict[str, list[dict[str, Any]]] = {}
    for element in elements:
        output_rows: list[dict[str, Any]] = []
        for _ in range(int(element["count"])):
            tokens = next(line_iter).split()
            cursor = 0
            row: dict[str, Any] = {}
            for prop in element["properties"]:
                if prop[0] == "scalar":
                    row[prop[2]] = float(tokens[cursor])
                    cursor += 1
                else:
                    count = int(tokens[cursor])
                    cursor += 1
                    row[prop[3]] = [int(tokens[cursor + i]) for i in range(count)]
                    cursor += count
            output_rows.append(row)
        result[str(element["name"])] = output_rows
    return result


def read_binary_ply_rows(
    payload: bytes,
    data_offset: int,
    elements: list[dict[str, Any]],
    endian: str,
) -> dict[str, list[dict[str, Any]]]:
    cursor = data_offset
    result: dict[str, list[dict[str, Any]]] = {}

    def unpack(type_name: str) -> int | float:
        nonlocal cursor
        code = PLY_TYPES.get(type_name)
        if code is None:
            raise ValueError(f"unsupported PLY property type: {type_name}")
        value = struct.unpack_from(endian + code, payload, cursor)[0]
        cursor += struct.calcsize(code)
        return value

    for element in elements:
        output_rows: list[dict[str, Any]] = []
        for _ in range(int(element["count"])):
            row: dict[str, Any] = {}
            for prop in element["properties"]:
                if prop[0] == "scalar":
                    row[prop[2]] = unpack(prop[1])
                else:
                    count = int(unpack(prop[1]))
                    row[prop[3]] = [int(unpack(prop[2])) for _ in range(count)]
            output_rows.append(row)
        result[str(element["name"])] = output_rows
    return result


def write_mesh_ply(vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]) -> bytes:
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    lines = [header]
    r, g, b = color
    lines.extend(f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}\n" for x, y, z in vertices)
    lines.extend(f"3 {a} {b_} {c}\n" for a, b_, c in faces)
    return "".join(lines).encode("ascii")


def main() -> None:
    args = parse_args()
    try:
        import viser
    except ImportError as exc:
        raise ImportError("viser is required; install the project's demo dependencies") from exc

    server = viser.ViserServer(host=str(args.host), port=int(args.port))
    scene = scene_api(server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    viewer = PlyRecolorViewer(server, Path(args.output_dir).expanduser().resolve())
    for raw in args.ply:
        for item in parse_path_list(str(raw)):
            viewer.load_path_file(Path(item).expanduser().resolve())
    print(f"[ply-recolor] open http://{args.host}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[ply-recolor] stopped", flush=True)


if __name__ == "__main__":
    main()
