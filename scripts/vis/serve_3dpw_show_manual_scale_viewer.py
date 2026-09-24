#!/usr/bin/env python3
"""Viser replay viewer for cached HMR4D SHOW per-sequence samples."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.serve_stage2_viewer_cache import (  # noqa: E402
    add_button,
    add_mesh,
    add_point_cloud,
    add_slider,
    add_text,
    bind_click,
    bind_update,
    set_text_value,
)
from vggt_omega.evaluation import HumanSceneConsistencyConfig, compute_human_scene_consistency, render_mesh_silhouette  # noqa: E402


CACHE_FORMAT = "vggt_omega_show_hmr4d_manual_scale_cache_v1"
SCALE_FILE_FORMAT = "vggt_omega_show_hmr4d_manual_scale_selections_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scale-file", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--initial-sequence", type=int, default=0)
    parser.add_argument("--scale-min", type=float, default=0.10)
    parser.add_argument("--scale-max", type=float, default=10.0)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--point-size", type=float, default=0.006)
    return parser.parse_args()


class ShowManualScaleViewer:
    def __init__(self, server: Any, args: argparse.Namespace) -> None:
        self.server = server
        self.args = args
        self.cache_dir = Path(args.cache_dir).expanduser().resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"Unsupported cache manifest: {self.manifest_path}")
        self.manifest = manifest
        self.windows = list(manifest.get("sequences", []))
        if not self.windows:
            raise RuntimeError("The cache contains no sequences")
        self.faces = np.load(self.cache_dir / str(manifest["faces_file"]), allow_pickle=False).astype(np.int32)
        self.metric_config = HumanSceneConsistencyConfig(**manifest["metric_config"])
        self.scale_file = Path(args.scale_file).expanduser().resolve() if args.scale_file else self.cache_dir / "manual_scales.json"
        self.selections = self._load_selections()
        self.window_index = min(max(int(args.initial_sequence), 0), len(self.windows) - 1)
        self.frame_index = 0
        self.frames: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.switching = False
        self._load_window(self.window_index)
        self._build_gui()
        self._rebuild_geometry()

    def _load_selections(self) -> dict[str, Any]:
        if self.scale_file.is_file():
            payload = json.loads(self.scale_file.read_text(encoding="utf-8"))
            if payload.get("format") != SCALE_FILE_FORMAT:
                raise ValueError(f"Unsupported scale file: {self.scale_file}")
            return payload
        return {"format": SCALE_FILE_FORMAT, "cache_manifest": str(self.manifest_path), "sequences": {}}

    def _load_window(self, index: int) -> None:
        self.window_index = int(index)
        record = self.windows[self.window_index]
        with (self.cache_dir / str(record["cache_file"])).open("rb") as file:
            self.frames = pickle.load(file)  # trusted local cache
        if not self.frames:
            raise ValueError(f"Empty cached sequence: {record['sequence_id']}")
        self.frame_index = min(self.frame_index, len(self.frames) - 1)

    def _build_gui(self) -> None:
        record = self.windows[self.window_index]
        self.window_choice = add_slider(self.server, "Sequence", 0, len(self.windows) - 1, 1, self.window_index)
        self.previous_window = add_button(self.server, "Previous Sequence")
        self.next_window = add_button(self.server, "Next Sequence")
        self.frame = add_slider(self.server, "Frame in Sample", 0, max(len(self.frames) - 1, 0), 1, 0)
        scale = self._saved_scale(str(record["sequence_id"]))
        self.scale_log10 = add_slider(self.server, "Manual Scale (log10)", float(np.log10(self.args.scale_min)), float(np.log10(self.args.scale_max)), 0.005, float(np.log10(scale)))
        self.save_scale = add_button(self.server, "Save Sequence Scale")
        self.preview = add_button(self.server, "Preview SHOW Metrics")
        self.window_info = add_text(self.server, "Sequence Status", "")
        self.frame_info = add_text(self.server, "Frame Status", "")
        self.metric_info = add_text(self.server, "Metric Preview", "Not computed")
        bind_update(self.window_choice, self._on_window)
        bind_click(self.previous_window, lambda _: self._switch_window(self.window_index - 1))
        bind_click(self.next_window, lambda _: self._switch_window(self.window_index + 1))
        bind_update(self.frame, self._on_frame)
        bind_update(self.scale_log10, lambda _: self._changed())
        bind_click(self.save_scale, self._on_save)
        bind_click(self.preview, self._on_preview)

    def _saved_scale(self, window_id: str) -> float:
        return float(self.selections.get("sequences", {}).get(window_id, {}).get("scale_multiplier", 1.0))

    def _current_scale(self) -> float:
        return float(10.0 ** float(self.scale_log10.value))

    def _on_window(self, _: Any = None) -> None:
        if not self.switching:
            self._switch_window(int(self.window_choice.value))

    def _switch_window(self, index: int) -> None:
        index = min(max(int(index), 0), len(self.windows) - 1)
        self.switching = True
        try:
            self._load_window(index)
            self.window_choice.value = index
            try:
                self.frame.max = max(len(self.frames) - 1, 0)
            except Exception:
                pass
            self.frame.value = 0
            self.scale_log10.value = float(np.log10(self._saved_scale(str(self.windows[index]["sequence_id"]))))
            set_text_value(self.metric_info, "Not computed")
        finally:
            self.switching = False
        self._rebuild_geometry()

    def _on_frame(self, _: Any = None) -> None:
        if self.switching:
            return
        self.frame_index = min(max(int(self.frame.value), 0), len(self.frames) - 1)
        self._rebuild_geometry()

    def _changed(self) -> None:
        if not self.switching:
            set_text_value(self.metric_info, "Scale changed; preview or save again")
            self._rebuild_geometry()

    def _on_save(self, _: Any = None) -> None:
        record = self.windows[self.window_index]
        window_id = str(record["sequence_id"])
        scale = self._current_scale()
        self.selections.setdefault("sequences", {})[window_id] = {
            "scale_multiplier": scale, "vid": record["vid"], "sequence_index": int(record["sequence_index"]),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.selections["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.scale_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.scale_file.with_suffix(self.scale_file.suffix + ".tmp")
        temporary.write_text(json.dumps(self.selections, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.scale_file)
        set_text_value(self.window_info, f"saved x{scale:.6g} | {window_id}")
        print(f"[show-manual-scale] saved {window_id} x{scale:.8g} -> {self.scale_file}", flush=True)

    def _on_preview(self, _: Any = None) -> None:
        scale = self._current_scale()
        values = []
        for frame in self.frames:
            if not bool(frame.get("eval_valid", True)):
                continue
            pred = torch.from_numpy(np.asarray(frame["pred_vertices_cam"], dtype=np.float32))
            gt = torch.from_numpy(np.asarray(frame["gt_vertices_cam"], dtype=np.float32))
            mask = render_mesh_silhouette(gt, torch.from_numpy(frame["gt_intrinsics"]), tuple(frame["scene_depth"].shape), faces=torch.from_numpy(self.faces).long(), backend=self.metric_config.visibility_backend)
            values.append(compute_human_scene_consistency(pred, torch.from_numpy(frame["scene_depth"]).float() * scale, torch.from_numpy(frame["pred_intrinsics"]), mask, faces=torch.from_numpy(self.faces).long(), scene_valid_mask=torch.from_numpy(frame["scene_valid_mask"]).bool(), config=self.metric_config))
        summary = summarize_metrics(values)
        record = self.windows[self.window_index]
        set_text_value(self.metric_info, f"sample={len(values)}/{record.get('original_frame_count', len(values))} | HS-V5={summary.get('hs_v5', 0):.5g} HS-V10={summary.get('hs_v10', 0):.5g} HS-CF5={summary.get('hs_cf5', 0):.5g} HS-CF10={summary.get('hs_cf10', 0):.5g}")

    def _rebuild_geometry(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        frame = self.frames[self.frame_index]
        depth = np.asarray(frame["scene_depth"], dtype=np.float32) * np.float32(self._current_scale())
        valid = np.asarray(frame["scene_valid_mask"], dtype=bool)
        points = depth_to_points(depth, np.asarray(frame["pred_intrinsics"], dtype=np.float32), valid, int(self.args.point_stride))
        if points.shape[0]:
            colors = np.tile(np.asarray([[150, 170, 190]], dtype=np.uint8), (points.shape[0], 1))
            self.handles.append(add_point_cloud(self.server, "/show_manual_scale/scene", points, colors, float(self.args.point_size)))
        vertices = np.asarray(frame["pred_vertices_cam"], dtype=np.float32)
        self.handles.append(add_mesh(self.server, "/show_manual_scale/person", vertices, self.faces, (232, 142, 82), 0.95))
        record = self.windows[self.window_index]
        set_text_value(self.window_info, f"{self.window_index + 1}/{len(self.windows)} | {record['vid']} | sampled={record.get('sampled_frame_count', len(self.frames))}/{record.get('original_frame_count', len(self.frames))} | scale=x{self._current_scale():.6g}")
        target_iou = frame.get("target_iou", float("nan"))
        set_text_value(
            self.frame_info,
            f"frame {self.frame_index + 1}/{len(self.frames)} | source={frame['source_frame_id']} | "
            f"query={frame['query_idx']} | GT-box IoU={float(target_iou):.3f} | "
            f"match_valid={bool(frame.get('match_valid', True))}",
        )

    def run(self) -> None:
        print(f"[show-manual-scale] http://127.0.0.1:{self.args.port} | sequences={len(self.windows)} | scale_file={self.scale_file}", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


def depth_to_points(depth: np.ndarray, K: np.ndarray, valid: np.ndarray, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    yy, xx = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    good = valid[::stride, ::stride] & np.isfinite(z) & (z > 1e-6)
    x = (xx.astype(np.float32) - float(K[0, 2])) * z / float(K[0, 0])
    y = (yy.astype(np.float32) - float(K[1, 2])) * z / float(K[1, 1])
    return np.stack((x[good], y[good], z[good]), axis=-1).astype(np.float32, copy=False)


def summarize_metrics(values: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for trim in (5, 10):
        for name in ("hs_v", "hs_cf"):
            key = f"{name}{trim}"
            numbers = [float(row[key]) for row in values if row.get(key) is not None and np.isfinite(float(row[key]))]
            out[f"{name}_{trim}"] = float(np.mean(numbers)) if numbers else 0.0
    return out


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
    ShowManualScaleViewer(server, args).run()


if __name__ == "__main__":
    main()
