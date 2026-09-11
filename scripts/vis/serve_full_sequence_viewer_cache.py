#!/usr/bin/env python
"""Load a complete cached SequenceViewer scene and serve it without inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import (  # noqa: E402
    apply_full_viewer_startup_overrides,
    load_full_sequence_viewer_cache,
)
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import (  # noqa: E402
    SequenceViewer,
    ensure_viser_available,
)


def main() -> None:
    cli_args = parse_args()
    ensure_viser_available()
    import viser  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415

    scene, viewer_args, manifest_path = load_full_sequence_viewer_cache(resolve_path(cli_args.cache_dir))
    viewer_args.port = int(cli_args.port)
    apply_full_viewer_startup_overrides(
        viewer_args,
        point_size=cli_args.point_size,
        human_mask_dilation_px=cli_args.human_mask_dilation_px,
        filter_human_points=cli_args.filter_human_points,
    )
    if cli_args.smpl_edit_output:
        viewer_args.smpl_edit_output = str(resolve_path(cli_args.smpl_edit_output))

    server = viser.ViserServer(port=int(cli_args.port))
    if hasattr(server, "scene") and hasattr(server.scene, "set_up_direction"):
        server.scene.set_up_direction("-y")
    elif hasattr(server, "set_up_direction"):
        server.set_up_direction("-y")
    print(
        f"[full-viewer-cache] serving {len(scene['frames'])} frames from {manifest_path} "
        f"at http://127.0.0.1:{int(cli_args.port)}",
        flush=True,
    )
    SequenceViewer(server=server, transforms=vtf, scene=scene, args=viewer_args).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--smpl-edit-output", default="")
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--human-mask-dilation-px", type=int, default=None)
    parser.add_argument("--filter-human-points", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
