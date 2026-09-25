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
from scripts.vis.rich_gt_smpl import attach_rich_gt_smpl  # noqa: E402
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
    if cli_args.show_gt_smpl:
        metadata = attach_rich_gt_smpl(
            scene,
            support_root=cli_args.rich_support_root,
            smplx_model_dir=cli_args.smplx_model_dir,
            smplx_to_smpl_path=cli_args.smplx_to_smpl,
            sequence=cli_args.rich_sequence,
            coordinate_source=cli_args.gt_coordinate_source,
            device=cli_args.gt_device,
        )
        viewer_args.show_gt_smpl = True
        viewer_args.gt_smpl_color = metadata["color"]
        print(
            f"[full-viewer-cache] RICH GT SMPL attached: sequence={metadata['sequence']} "
            f"frames={metadata['gt_frame_count']} source={metadata['coordinate_source']}",
            flush=True,
        )
    viewer_args.port = int(cli_args.port)
    apply_full_viewer_startup_overrides(
        viewer_args,
        point_size=cli_args.point_size,
        human_mask_dilation_px=cli_args.human_mask_dilation_px,
        filter_human_points=cli_args.filter_human_points,
    )
    if cli_args.viewer_mode:
        viewer_args.viewer_mode = str(cli_args.viewer_mode)
    if cli_args.smpl_display_frames is not None:
        if int(cli_args.smpl_display_frames) < 0:
            raise ValueError("--smpl-display-frames must be >= 0")
        viewer_args.smpl_display_frames = int(cli_args.smpl_display_frames)
    if cli_args.display_people is not None:
        if int(cli_args.display_people) < 0:
            raise ValueError("--display-people must be >= 0")
        viewer_args.display_people = int(cli_args.display_people)
    if cli_args.smpl_edit_output:
        viewer_args.smpl_edit_output = str(resolve_path(cli_args.smpl_edit_output))
    if cli_args.smpl_visual_scale is not None:
        if not 0.1 <= float(cli_args.smpl_visual_scale) <= 10.0:
            raise ValueError("--smpl-visual-scale must be within [0.1, 10.0]")
        viewer_args.smpl_visual_scale = float(cli_args.smpl_visual_scale)
    if cli_args.show_pinned_root_trajectory is not None:
        viewer_args.show_pinned_root_trajectory_initial = bool(cli_args.show_pinned_root_trajectory)

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
    parser.add_argument("--smpl-visual-scale", type=float, default=None)
    parser.add_argument(
        "--show-pinned-root-trajectory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Initial state of the pin-panel SMPL root trajectory checkbox.",
    )
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--human-mask-dilation-px", type=int, default=None)
    parser.add_argument("--filter-human-points", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--viewer-mode",
        choices=["4D current frame", "3D accumulate", "Hybrid"],
        default="",
    )
    parser.add_argument("--smpl-display-frames", type=int, default=None)
    parser.add_argument("--display-people", type=int, default=None)
    parser.add_argument("--show-gt-smpl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rich-sequence", default="")
    parser.add_argument("--rich-support-root", default="/home/zhw/xyb_space/RICH/hmr4d_support")
    parser.add_argument("--smplx-model-dir", default="/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smplx")
    parser.add_argument("--smplx-to-smpl", default="/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/utils/smplx2smpl.pkl")
    parser.add_argument("--gt-coordinate-source", choices=["hsi_scaled", "raw_vggt"], default="hsi_scaled")
    parser.add_argument("--gt-device", default="cuda")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
