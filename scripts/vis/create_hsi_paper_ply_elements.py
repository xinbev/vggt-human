#!/usr/bin/env python3
"""Create deterministic PLY elements for the HSI paper architecture figure.

The exported PLY files are geometry-only screenshot assets. Add labels,
formulas, and arrows in the final SVG/PDF layout so text stays editable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


COLOR = {
    "human": (233, 138, 58),
    "human_light": (246, 191, 142),
    "scene": (60, 125, 217),
    "scene_light": (164, 199, 242),
    "hsi": (42, 168, 107),
    "hsi_light": (173, 226, 198),
    "calib": (26, 166, 166),
    "calib_light": (151, 222, 222),
    "neutral": (215, 220, 227),
    "dark": (31, 41, 51),
    "muted": (102, 112, 133),
    "white": (250, 252, 255),
}


class MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[list[float]] = []
        self.colors: list[tuple[int, int, int]] = []
        self.faces: list[list[int]] = []

    def add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int] | np.ndarray,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        offset = len(self.vertices)
        self.vertices.extend(vertices.tolist())
        if isinstance(color, tuple):
            self.colors.extend([color] * len(vertices))
        else:
            color_arr = np.asarray(color, dtype=np.uint8).reshape(-1, 3)
            if color_arr.shape[0] != vertices.shape[0]:
                raise ValueError("Per-vertex colors must match vertex count")
            self.colors.extend([tuple(int(c) for c in row) for row in color_arr])
        self.faces.extend((faces + offset).tolist())

    def merge(self, other: "MeshBuilder") -> None:
        self.add_mesh(other.as_arrays()[0], other.as_arrays()[2], other.as_arrays()[1])

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = np.asarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        return vertices, colors, faces

    def write(self, path: Path) -> None:
        vertices, colors, faces = self.as_arrays()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {vertices.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write(f"element face {faces.shape[0]}\n")
            file.write("property list uchar int vertex_indices\n")
            file.write("end_header\n")
            for vertex, color in zip(vertices, colors, strict=True):
                file.write(
                    f"{float(vertex[0]):.6f} {float(vertex[1]):.6f} {float(vertex[2]):.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
            for face in faces:
                file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def basis_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = unit(direction)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(w, helper))) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = unit(np.cross(w, helper))
    v = unit(np.cross(w, u))
    return u, v, w


def add_cylinder(mesh: MeshBuilder, p0, p1, radius: float, color, sections: int = 16) -> None:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    u, v, _ = basis_from_direction(p1 - p0)
    verts = []
    for base in (p0, p1):
        for i in range(sections):
            angle = 2.0 * math.pi * i / sections
            verts.append(base + radius * (math.cos(angle) * u + math.sin(angle) * v))
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + i])
        faces.append([j, sections + j, sections + i])
    mesh.add_mesh(np.asarray(verts), np.asarray(faces), color)


def add_cone(mesh: MeshBuilder, base, tip, radius: float, color, sections: int = 16) -> None:
    base = np.asarray(base, dtype=np.float32)
    tip = np.asarray(tip, dtype=np.float32)
    u, v, _ = basis_from_direction(tip - base)
    verts = []
    for i in range(sections):
        angle = 2.0 * math.pi * i / sections
        verts.append(base + radius * (math.cos(angle) * u + math.sin(angle) * v))
    verts.append(tip)
    faces = [[i, (i + 1) % sections, sections] for i in range(sections)]
    mesh.add_mesh(np.asarray(verts), np.asarray(faces), color)


def add_arrow(
    mesh: MeshBuilder,
    start,
    end,
    radius: float,
    color,
    head_radius: float | None = None,
    head_length: float | None = None,
) -> None:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return
    head_length = min(head_length or radius * 7.0, length * 0.35)
    head_radius = head_radius or radius * 3.0
    shaft_end = end - direction / length * head_length
    add_cylinder(mesh, start, shaft_end, radius, color)
    add_cone(mesh, shaft_end, end, head_radius, color)


def add_uv_sphere(
    mesh: MeshBuilder,
    center,
    radius: float,
    color,
    rings: int = 10,
    segments: int = 18,
) -> None:
    center = np.asarray(center, dtype=np.float32)
    verts = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            verts.append(
                center
                + radius
                * np.array(
                    [
                        math.sin(phi) * math.cos(theta),
                        math.sin(phi) * math.sin(theta),
                        math.cos(phi),
                    ],
                    dtype=np.float32,
                )
            )
    faces = []
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + s
            d = (r + 1) * segments + (s + 1) % segments
            faces.append([a, c, b])
            faces.append([b, c, d])
    mesh.add_mesh(np.asarray(verts), np.asarray(faces), color)


def add_box(mesh: MeshBuilder, center, size, color) -> None:
    cx, cy, cz = np.asarray(center, dtype=np.float32)
    sx, sy, sz = np.asarray(size, dtype=np.float32) * 0.5
    verts = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    mesh.add_mesh(verts, faces, color)


def add_depth_surface(
    mesh: MeshBuilder,
    origin=(0.0, 0.0, 0.0),
    width: float = 2.2,
    height: float = 1.6,
    nx: int = 26,
    ny: int = 18,
    color_low=(172, 211, 247),
    color_high=(37, 99, 180),
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(origin, dtype=np.float32)
    xs = np.linspace(-width * 0.5, width * 0.5, nx)
    ys = np.linspace(-height * 0.5, height * 0.5, ny)
    verts = []
    colors = []
    z_vals = []
    for y in ys:
        for x in xs:
            z = 0.12 * math.sin(2.6 * x) + 0.08 * math.cos(3.0 * y) + 0.06 * x
            point = origin + np.array([x, y, z], dtype=np.float32)
            verts.append(point)
            z_vals.append(z)
    z_vals_arr = np.asarray(z_vals)
    t_vals = (z_vals_arr - z_vals_arr.min()) / max(float(np.ptp(z_vals_arr)), 1e-6)
    low = np.asarray(color_low, dtype=np.float32)
    high = np.asarray(color_high, dtype=np.float32)
    for t in t_vals:
        colors.append(((1.0 - t) * low + t * high).clip(0, 255))
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = j * nx + i + 1
            c = (j + 1) * nx + i
            d = (j + 1) * nx + i + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    mesh.add_mesh(np.asarray(verts), np.asarray(faces), np.asarray(colors, dtype=np.uint8))
    return np.asarray(verts).reshape(ny, nx, 3), np.asarray(xs)


def add_patch_grid(
    mesh: MeshBuilder,
    origin=(0.0, 0.0, -0.08),
    cols: int = 8,
    rows: int = 6,
    tile: float = 0.17,
    gap: float = 0.025,
    color=COLOR["scene_light"],
    highlight: set[tuple[int, int]] | None = None,
) -> None:
    highlight = highlight or set()
    x0 = -(cols - 1) * (tile + gap) * 0.5
    y0 = -(rows - 1) * (tile + gap) * 0.5
    origin = np.asarray(origin, dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tile_color = COLOR["hsi_light"] if (r, c) in highlight else color
            z_lift = 0.035 if (r, c) in highlight else 0.0
            add_box(
                mesh,
                origin + np.array([x0 + c * (tile + gap), y0 + r * (tile + gap), z_lift], dtype=np.float32),
                (tile, tile, 0.025),
                tile_color,
            )


def body_anchor_points() -> tuple[dict[str, np.ndarray], list[tuple[str, str]], list[np.ndarray]]:
    joints = {
        "pelvis": np.array([0.0, 0.0, 0.95]),
        "spine": np.array([0.0, 0.0, 1.25]),
        "chest": np.array([0.0, 0.0, 1.55]),
        "neck": np.array([0.0, 0.0, 1.78]),
        "head": np.array([0.0, 0.0, 1.98]),
        "l_shoulder": np.array([-0.32, 0.0, 1.62]),
        "l_elbow": np.array([-0.55, 0.02, 1.32]),
        "l_wrist": np.array([-0.72, 0.04, 1.08]),
        "r_shoulder": np.array([0.32, 0.0, 1.62]),
        "r_elbow": np.array([0.55, -0.02, 1.32]),
        "r_wrist": np.array([0.72, -0.04, 1.08]),
        "l_hip": np.array([-0.22, 0.0, 0.9]),
        "l_knee": np.array([-0.25, 0.05, 0.48]),
        "l_ankle": np.array([-0.22, 0.08, 0.08]),
        "l_foot": np.array([-0.20, 0.22, 0.02]),
        "r_hip": np.array([0.22, 0.0, 0.9]),
        "r_knee": np.array([0.26, -0.04, 0.48]),
        "r_ankle": np.array([0.25, -0.08, 0.08]),
        "r_foot": np.array([0.22, -0.22, 0.02]),
    }
    bones = [
        ("pelvis", "spine"),
        ("spine", "chest"),
        ("chest", "neck"),
        ("neck", "head"),
        ("chest", "l_shoulder"),
        ("l_shoulder", "l_elbow"),
        ("l_elbow", "l_wrist"),
        ("chest", "r_shoulder"),
        ("r_shoulder", "r_elbow"),
        ("r_elbow", "r_wrist"),
        ("pelvis", "l_hip"),
        ("l_hip", "l_knee"),
        ("l_knee", "l_ankle"),
        ("l_ankle", "l_foot"),
        ("pelvis", "r_hip"),
        ("r_hip", "r_knee"),
        ("r_knee", "r_ankle"),
        ("r_ankle", "r_foot"),
    ]
    anchors = [
        joints["spine"],
        joints["chest"],
        joints["neck"],
        joints["head"],
        joints["l_shoulder"],
        joints["l_elbow"],
        joints["l_wrist"],
        joints["r_shoulder"],
        joints["r_elbow"],
        joints["r_wrist"],
        joints["l_hip"],
        joints["l_knee"],
        joints["l_ankle"],
        joints["l_foot"],
        joints["r_hip"],
        joints["r_knee"],
        joints["r_ankle"],
        joints["r_foot"],
        (joints["pelvis"] + joints["spine"]) * 0.5,
        (joints["chest"] + joints["head"]) * 0.5,
        (joints["l_hip"] + joints["r_hip"]) * 0.5,
        (joints["l_elbow"] + joints["l_wrist"]) * 0.5,
        (joints["r_elbow"] + joints["r_wrist"]) * 0.5,
        np.mean(np.stack(list(joints.values()), axis=0), axis=0),
    ]
    return joints, bones, [np.asarray(p, dtype=np.float32) for p in anchors]


def add_skeleton(
    mesh: MeshBuilder,
    offset=(0.0, 0.0, 0.0),
    scale: float = 1.0,
    show_token_chips: bool = True,
    selected_anchor: int | None = None,
) -> list[np.ndarray]:
    offset = np.asarray(offset, dtype=np.float32)
    joints, bones, anchors = body_anchor_points()
    joints = {name: offset + scale * point for name, point in joints.items()}
    anchors = [offset + scale * point for point in anchors]
    for a, b in bones:
        add_cylinder(mesh, joints[a], joints[b], 0.015 * scale, COLOR["human"])
    add_uv_sphere(mesh, joints["head"], 0.095 * scale, COLOR["human_light"])
    for idx, point in enumerate(anchors):
        radius = 0.046 * scale if idx == selected_anchor else 0.031 * scale
        add_uv_sphere(mesh, point, radius, COLOR["hsi"] if idx != selected_anchor else COLOR["calib"])
    if show_token_chips:
        chip_origin = offset + scale * np.array([1.0, -0.45, 1.65], dtype=np.float32)
        for i in range(24):
            row = i // 8
            col = i % 8
            center = chip_origin + scale * np.array([0.11 * col, 0.0, -0.11 * row], dtype=np.float32)
            add_uv_sphere(mesh, center, 0.03 * scale, COLOR["hsi"])
        for idx in (5, 11, 17, 23):
            add_arrow(mesh, anchors[idx], chip_origin + scale * np.array([0.11 * (idx % 8), 0, -0.11 * (idx // 8)], dtype=np.float32), 0.006 * scale, COLOR["hsi"])
    return anchors


def add_camera_frustum(mesh: MeshBuilder, origin=(-0.7, -1.2, 1.3), scale: float = 0.55) -> None:
    origin = np.asarray(origin, dtype=np.float32)
    corners = [
        origin + scale * np.array([-0.45, 0.9, -0.32]),
        origin + scale * np.array([0.45, 0.9, -0.32]),
        origin + scale * np.array([0.45, 0.9, 0.32]),
        origin + scale * np.array([-0.45, 0.9, 0.32]),
    ]
    add_box(mesh, origin, (0.18 * scale, 0.1 * scale, 0.12 * scale), COLOR["scene"])
    for corner in corners:
        add_cylinder(mesh, origin, corner, 0.008 * scale, COLOR["scene"])
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        add_cylinder(mesh, a, b, 0.008 * scale, COLOR["scene"])


def build_body_anchors_scene() -> MeshBuilder:
    mesh = MeshBuilder()
    add_skeleton(mesh, offset=(-0.9, 0.0, 0.0), scale=1.0, show_token_chips=True, selected_anchor=13)
    add_box(mesh, (-0.9, 0.0, -0.035), (1.4, 0.7, 0.03), COLOR["neutral"])
    return mesh


def build_local_probe_scene() -> MeshBuilder:
    mesh = MeshBuilder()
    anchors = add_skeleton(mesh, offset=(-1.35, 0.0, 0.0), scale=0.95, show_token_chips=False, selected_anchor=13)
    selected = anchors[13]
    depth_origin = np.array([1.0, 0.0, 0.82], dtype=np.float32)
    depth_points, _ = add_depth_surface(mesh, origin=depth_origin, width=1.8, height=1.25, nx=16, ny=12)
    add_patch_grid(mesh, origin=depth_origin + np.array([0.0, 0.0, -0.22], dtype=np.float32), cols=7, rows=5, tile=0.13, gap=0.02, highlight={(2, 3), (2, 2), (2, 4), (1, 3), (3, 3), (1, 2), (1, 4), (3, 2), (3, 4)})
    local_point = depth_points[6, 8] + np.array([0.0, 0.0, 0.08], dtype=np.float32)
    add_arrow(mesh, selected, local_point, 0.012, COLOR["hsi"], head_length=0.13)
    add_uv_sphere(mesh, local_point, 0.06, COLOR["scene"])
    scene_point = local_point + np.array([0.18, 0.18, -0.23], dtype=np.float32)
    add_uv_sphere(mesh, scene_point, 0.055, COLOR["calib"])
    add_arrow(mesh, local_point, scene_point, 0.012, COLOR["scene"], head_length=0.12)
    add_arrow(mesh, selected, scene_point, 0.01, COLOR["human"], head_length=0.12)
    add_arrow(mesh, scene_point, scene_point + np.array([0.0, -0.28, 0.32], dtype=np.float32), 0.01, COLOR["calib"], head_length=0.1)
    add_cylinder(mesh, selected, np.array([selected[0], selected[1], scene_point[2]], dtype=np.float32), 0.006, COLOR["muted"])
    add_box(mesh, depth_origin + np.array([0.0, 0.0, -0.32], dtype=np.float32), (1.55, 1.1, 0.02), COLOR["white"])
    add_camera_frustum(mesh, origin=(-0.1, -1.25, 1.15), scale=0.45)
    return mesh


def build_transformer_tokens_scene() -> MeshBuilder:
    mesh = MeshBuilder()
    body_tokens = []
    for i in range(24):
        row = i // 8
        col = i % 8
        center = np.array([-0.75 + 0.22 * col, -0.25, 1.1 - 0.22 * row], dtype=np.float32)
        body_tokens.append(center)
        add_uv_sphere(mesh, center, 0.055, COLOR["hsi"])
    for i in range(24):
        for j in range(i + 1, 24):
            if (i + j) % 9 == 0:
                add_cylinder(mesh, body_tokens[i], body_tokens[j], 0.004, COLOR["hsi_light"], sections=8)
    scene_tokens = []
    for r in range(3):
        for c in range(3):
            center = np.array([-0.2 + 0.2 * c, 0.45, 0.85 + 0.2 * r], dtype=np.float32)
            scene_tokens.append(center)
            add_box(mesh, center, (0.13, 0.035, 0.13), COLOR["scene"])
    for idx in (3, 10, 17):
        for scene_point in scene_tokens:
            add_arrow(mesh, scene_point, body_tokens[idx], 0.004, COLOR["scene"], head_length=0.05)
    add_box(mesh, (0.0, 0.04, 0.85), (2.1, 0.04, 1.4), COLOR["hsi_light"])
    return mesh


def build_scene_affine_scene() -> MeshBuilder:
    mesh = MeshBuilder()
    query_points = []
    for i, x in enumerate([-0.65, -0.25, 0.15, 0.55]):
        center = np.array([x, -0.65, 0.75 + 0.15 * (i % 2)], dtype=np.float32)
        query_points.append(center)
        add_uv_sphere(mesh, center, 0.075, COLOR["hsi"])
        add_box(mesh, center + np.array([0.0, 0.0, -0.14], dtype=np.float32), (0.16, 0.05, 0.06), COLOR["human"])
    scale_bias = np.array([0.0, 0.0, 0.95], dtype=np.float32)
    add_uv_sphere(mesh, scale_bias, 0.13, COLOR["calib"])
    for point in query_points:
        add_arrow(mesh, point, scale_bias, 0.01, COLOR["hsi"], head_length=0.09)
    add_depth_surface(mesh, origin=(0.0, 0.75, 0.75), width=1.35, height=0.9, nx=14, ny=10, color_low=COLOR["calib_light"], color_high=COLOR["calib"])
    add_arrow(mesh, scale_bias, np.array([0.0, 0.48, 0.82], dtype=np.float32), 0.014, COLOR["calib"], head_length=0.12)
    add_box(mesh, (0.0, 0.0, 0.36), (1.55, 1.65, 0.035), COLOR["neutral"])
    return mesh


def build_full_scene() -> MeshBuilder:
    mesh = MeshBuilder()
    add_skeleton(mesh, offset=(-2.2, -0.15, 0.0), scale=0.8, show_token_chips=True, selected_anchor=13)
    add_camera_frustum(mesh, origin=(-2.75, -0.95, 0.95), scale=0.38)
    add_depth_surface(mesh, origin=(-0.25, -0.1, 0.75), width=1.35, height=0.95, nx=16, ny=12)
    add_patch_grid(mesh, origin=(-0.25, -0.1, 0.43), cols=6, rows=4, tile=0.12, gap=0.018, highlight={(1, 2), (1, 3), (2, 2), (2, 3)})
    add_arrow(mesh, (-1.65, -0.05, 0.06), (-0.55, -0.08, 0.75), 0.011, COLOR["hsi"], head_length=0.11)
    transformer = build_transformer_tokens_scene()
    tv, tc, tf = transformer.as_arrays()
    tv = tv * np.array([0.55, 0.55, 0.55], dtype=np.float32) + np.array([1.15, -0.1, 0.45], dtype=np.float32)
    mesh.add_mesh(tv, tf, tc)
    add_arrow(mesh, (0.42, -0.08, 0.85), (0.86, -0.08, 0.9), 0.012, COLOR["hsi"], head_length=0.1)
    add_arrow(mesh, (1.72, -0.08, 0.9), (2.15, -0.08, 1.0), 0.012, COLOR["calib"], head_length=0.1)
    add_skeleton(mesh, offset=(2.35, -0.12, 0.0), scale=0.58, show_token_chips=False, selected_anchor=None)
    add_depth_surface(mesh, origin=(2.45, 0.75, 0.7), width=0.9, height=0.6, nx=10, ny=8, color_low=COLOR["calib_light"], color_high=COLOR["calib"])
    add_box(mesh, (0.1, -0.35, -0.04), (5.7, 1.85, 0.035), COLOR["neutral"])
    return mesh


def write_manifest(output_dir: Path, files: dict[str, str]) -> None:
    manifest = {
        "purpose": "Geometry-only PLY screenshot assets for the HSI paper architecture figure.",
        "source_component": "vggt_omega.models.heads.hsi_refinement_head.HSIRefinementHead",
        "baseline_preserved": True,
        "elements": {
            "hsi_body_anchors.ply": "Orange simplified SMPL skeleton with 24 green body-anchored HSI points and token chips.",
            "hsi_local_scene_probe.ply": "Anchor projection, 3x3 local scene window, scene xyz, offset/distance, normal, and z-residual geometry.",
            "hsi_transformer_tokens.ply": "Body-token self-attention plus local scene cross-attention as geometry-only token nodes.",
            "hsi_scene_affine.ply": "Human-query scale/bias aggregation feeding calibrated depth.",
            "hsi_full_paper_elements.ply": "Combined overview scene for one screenshot pass.",
        },
        "colors": COLOR,
        "files": files,
        "notes": [
            "PLY files intentionally contain no text. Add labels and formulas in SVG/PDF.",
            "Use D_hsi = s_hsi * D_vggt + b_hsi in the final vector figure.",
            "Optional temporal memory is omitted from these PLY assets because it should stay secondary in the paper figure.",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vis/paper_hsi_ply_elements"),
        help="Directory for generated PLY files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = {
        "hsi_body_anchors.ply": build_body_anchors_scene(),
        "hsi_local_scene_probe.ply": build_local_probe_scene(),
        "hsi_transformer_tokens.ply": build_transformer_tokens_scene(),
        "hsi_scene_affine.ply": build_scene_affine_scene(),
        "hsi_full_paper_elements.ply": build_full_scene(),
    }
    files = {}
    for name, mesh in scenes.items():
        path = output_dir / name
        mesh.write(path)
        vertices, _, faces = mesh.as_arrays()
        files[name] = str(path)
        print(f"[OK] {path} vertices={vertices.shape[0]} faces={faces.shape[0]}")
    write_manifest(output_dir, files)
    print(f"[OK] {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
