#!/usr/bin/env python
"""Render the base neutral SMPL mesh once for every RGB color in a palette."""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

# chumpy, which is used while loading legacy SMPL pickle files, imports these
# NumPy aliases directly. NumPy 1.24+ removed them, so restore only the missing
# names before smplx/chumpy is imported.
_LEGACY_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}
for _name, _value in _LEGACY_NUMPY_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# Older SMPL pickles and smplx versions still expect this Python API.
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple

    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=root / "checkpoints/body_models/smpl",
        help="Directory containing SMPL_NEUTRAL.pkl or gendered SMPL files.",
    )
    parser.add_argument(
        "--colors",
        type=Path,
        default=root / "assets/smpl_colors.txt",
        help="Text file with one R,G,B color per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/vis/smpl_color_palette",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--grid-columns", type=int, default=9)
    parser.add_argument("--background", type=int, nargs=3, default=(245, 245, 245))
    parser.add_argument(
        "--backend",
        choices=("egl", "osmesa", "default"),
        default="egl",
        help="Pyrender OpenGL backend. Use osmesa if EGL is unavailable.",
    )
    return parser.parse_args()


def read_colors(path: Path) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected R,G,B")
        rgb = tuple(int(field) for field in fields)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"{path}:{line_number}: RGB values must be in [0, 255]")
        colors.append(rgb)
    if not colors:
        raise ValueError(f"No colors found in {path}")
    return colors


def load_smpl_vertices(model_dir: Path):
    import torch
    import smplx

    model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=False,
    ).eval()
    with torch.no_grad():
        output = model(
            global_orient=torch.zeros((1, 3)),
            body_pose=torch.zeros((1, 69)),
            betas=torch.zeros((1, 10)),
        )
    return output.vertices[0].cpu().numpy(), model.faces.copy()


def make_mesh(vertices, faces, rgb: tuple[int, int, int]):
    import numpy as np
    import trimesh

    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.repeat(rgba[None, :], len(vertices), axis=0),
        process=False,
    )


def render_color(mesh, rgb: tuple[int, int, int], width: int, height: int, background):
    import numpy as np
    import pyrender

    scene = pyrender.Scene(
        bg_color=[*([channel / 255.0 for channel in background]), 1.0],
        ambient_light=(0.35, 0.35, 0.35),
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*(channel / 255.0 for channel in rgb), 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.72,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material), name="smpl")

    # SMPL uses +Y up; this camera looks toward the origin along -Z.
    camera = pyrender.OrthographicCamera(xmag=1.25, ymag=1.25, znear=0.01, zfar=20.0)
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[2, 3] = 3.5
    scene.add(camera, pose=camera_pose)
    for position in ((2.5, 3.0, 4.0), (-3.0, 2.0, 2.0), (0.0, 4.0, -2.0)):
        light_pose = np.eye(4, dtype=np.float32)
        light_pose[:3, 3] = position
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.2),
            pose=light_pose,
        )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        image, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return image


def add_label(image, text: str):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((8, 8, 205, 38), fill=(255, 255, 255))
    draw.text((14, 13), text, fill=(20, 20, 20), font=font)
    return canvas


def save_grid(images: Iterable, path: Path, columns: int, background: tuple[int, int, int]) -> None:
    from PIL import Image

    images = list(images)
    tile_width, tile_height = images[0].size
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (tile_width * columns, tile_height * rows), background)
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
    grid.save(path)


def main() -> None:
    args = parse_args()
    if args.backend != "default":
        os.environ["PYOPENGL_PLATFORM"] = args.backend

    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = read_colors(args.colors)
    vertices, faces = load_smpl_vertices(args.smpl_model_dir)

    # Center and scale the canonical mesh for a stable, comparable camera view.
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices *= 1.75 / (vertices.max(axis=1) - vertices.min(axis=1)).max()
    background = tuple(args.background)
    grid_images = []
    for index, rgb in enumerate(colors, start=1):
        mesh = make_mesh(vertices, faces, rgb)
        image = render_color(mesh, rgb, args.width, args.height, background)
        labeled = add_label(image, f"{index:02d}  RGB {rgb[0]},{rgb[1]},{rgb[2]}")
        labeled.save(args.output_dir / f"color_{index:03d}_{rgb[0]}_{rgb[1]}_{rgb[2]}.png")
        grid_images.append(labeled)

    save_grid(grid_images, args.output_dir / "palette_grid.png", args.grid_columns, background)
    (args.output_dir / "palette.txt").write_text(
        "\n".join(f"{index:03d}: {r},{g},{b}" for index, (r, g, b) in enumerate(colors, start=1))
        + "\n"
    )
    print(f"Rendered {len(colors)} colors to {args.output_dir}")
    print(f"Grid: {args.output_dir / 'palette_grid.png'}")


if __name__ == "__main__":
    main()
