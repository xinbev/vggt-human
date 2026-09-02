#!/usr/bin/env python3
"""Evaluate official NLF fit_tdpw outputs against raw gender-specific 3DPW GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.human3r_style_3dpw.data import decode_gt_camera_space, load_test_sequences
from benchmarks.human3r_style_3dpw.metrics import human3r_camera_metrics
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import load_yaml_config, require_path


class Totals:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}; self.counts: dict[str, int] = {}

    def add(self, values: dict[str, torch.Tensor]) -> None:
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().sum().cpu())
            self.counts[key] = self.counts.get(key, 0) + int(value.numel())

    def summary(self) -> dict[str, Any]:
        return {**{key: value / max(self.counts[key], 1) for key, value in self.sums.items()}, "count": self.counts}


@torch.no_grad()
def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    cfg = load_yaml_config(args.path_config)
    root = Path(args.threedpw_root or require_path(cfg, "datasets.threedpw_root"))
    fitted_root = Path(args.fitted_root)
    layers = {
        "neutral": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="neutral").to(device).eval(),
        "male": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="male").to(device).eval(),
        "female": SMPLLayer(require_path(cfg, "assets.smpl_model_dir"), gender="female").to(device).eval(),
    }
    total = Totals(); coverage = {"sequences": 0, "valid_gt_people": 0, "valid_fitted_people": 0, "missing_fitted_files": 0}
    for sequence in load_test_sequences(root, args.sequence_filter):
        path = fitted_root / "test" / f"{sequence.name}.pkl"
        if not path.is_file():
            coverage["missing_fitted_files"] += 1; print(f"[missing] {path}", flush=True); continue
        with path.open("rb") as file: fitted = pickle.load(file)
        pose, betas, trans = normalize_fitted(fitted, device)
        frames = min(sequence.length, pose.shape[1])
        for frame in range(frames):
            gt_v, gt_j, person_ids = decode_gt_camera_space(sequence, frame, layers, device)
            coverage["valid_gt_people"] += int(person_ids.numel())
            if not person_ids.numel(): continue
            ids = person_ids[person_ids < pose.shape[0]]
            if not ids.numel(): continue
            pred_valid = torch.isfinite(pose[ids, frame]).all(dim=-1) & torch.isfinite(betas[ids, frame]).all(dim=-1) & torch.isfinite(trans[ids, frame]).all(dim=-1)
            if not bool(pred_valid.any()): continue
            ids = ids[pred_valid]
            # GT output order equals ascending raw original person IDs.
            gt_select = torch.nonzero(torch.isin(person_ids, ids), as_tuple=False).reshape(-1)
            ordered_ids = person_ids[gt_select]
            order = torch.argsort(ids)
            ids = ids[order]
            # person_ids in 3DPW are ascending, so this keeps prediction/GT aligned.
            if not torch.equal(ids, ordered_ids):
                raise RuntimeError(f"Official gtassoc ordering mismatch for {sequence.name} frame={frame}")
            pred_v, pred_j = layers["neutral"](pose[ids, frame], betas[ids, frame])
            pred_v, pred_j = pred_v + trans[ids, frame, None], pred_j[:, :24] + trans[ids, frame, None]
            total.add(human3r_camera_metrics(pred_j, gt_j[gt_select], pred_v, gt_v[gt_select]))
            coverage["valid_fitted_people"] += int(ids.numel())
        coverage["sequences"] += 1
        print(f"[eval] {sequence.name}", flush=True)
    result = total.summary(); targets = {"pa_mpjpe_mm": 37.3, "mpjpe_mm": 60.3, "pve_mm": 71.4}
    delta = {key: result.get(key, float("nan")) - value for key, value in targets.items()}
    summary = {"benchmark": "nlf_official_predict_fit_3dpw_calibration", "fitted_root": str(fitted_root), "coverage": coverage, "metrics": result, "published_target": targets, "delta_to_target_mm": delta, "calibrated_within_1mm": all(abs(value) <= 1.0 for value in delta.values())}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--fitted-root", required=True); p.add_argument("--output", default="outputs/benchmarks/nlf_official_3dpw/calibration.json"); p.add_argument("--threedpw-root", default=""); p.add_argument("--path-config", default="configs/path.yaml"); p.add_argument("--sequence-filter", default=""); p.add_argument("--device", default="cuda:0"); return p.parse_args()


def normalize_fitted(payload: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = torch.as_tensor(payload["pose"], dtype=torch.float32, device=device)
    betas = torch.as_tensor(payload["betas"], dtype=torch.float32, device=device)
    trans = torch.as_tensor(payload["trans"], dtype=torch.float32, device=device)
    if pose.ndim != 3 or pose.shape[-1] != 72: raise ValueError(f"Expected fitted pose [P,F,72], got {tuple(pose.shape)}")
    if betas.shape[:2] != pose.shape[:2] or trans.shape[:2] != pose.shape[:2]: raise ValueError("fitted pose/betas/trans have inconsistent P/F axes")
    return pose, betas[..., :10], trans


if __name__ == "__main__": main()
