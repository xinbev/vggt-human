from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = json.loads((output_dir / "metrics_latest.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = payload.get("val") or payload.get("train") or {}
    required = [
        "loss_hsi_v4_candidate_transl",
        "loss_hsi_v4_candidate_ray",
        "loss_hsi_v4_candidate_tangent",
        "metric_hsi_v4_geometry_eligibility_coverage",
        "metric_hsi_v4_base_active_l2_median",
        "metric_hsi_v4_candidate_active_l2_median",
        "metric_hsi_v4_candidate_improvement_rate",
        "metric_hsi_v4_candidate_ray_sign_acc",
        "metric_hsi_v4_candidate_tangent_l1_delta",
    ]
    missing = [key for key in required if key not in metrics or not math.isfinite(float(metrics[key]))]
    if missing:
        raise SystemExit(f"V4-A1 missing/non-finite metrics: {missing}")
    assert_config(config)
    if args.mode == "overfit":
        base = float(metrics["metric_hsi_v4_base_active_l2_median"])
        candidate = float(metrics["metric_hsi_v4_candidate_active_l2_median"])
        checks = {
            "eligibility_coverage>=0.95": float(metrics["metric_hsi_v4_geometry_eligibility_coverage"]) >= 0.95,
            "improvement_rate>=0.95": float(metrics["metric_hsi_v4_candidate_improvement_rate"]) >= 0.95,
            "candidate_median<=0.25*base": base > 0.0 and candidate <= 0.25 * base,
            "ray_sign_acc>=0.95": float(metrics["metric_hsi_v4_candidate_ray_sign_acc"]) >= 0.95,
            "tangent_degradation<=0.0005": float(metrics["metric_hsi_v4_candidate_tangent_l1_delta"]) <= 0.0005,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(f"V4-A1 overfit gate failed: {failed}")
    print(json.dumps({"gate": "pass", "mode": args.mode, "metrics": {key: metrics[key] for key in required}}, indent=2))


def assert_config(config: dict) -> None:
    expected = {
        "model.enable_hsi_translation_refine_v4": True,
        "model.enable_hsi_human_scene_align": False,
        "model.train_hsi_v4_correction_only": True,
        "model.hsi_v4_phase": "correction",
        "model.hsi_v4_ray_parameterization": "residual_gain",
        "model.hsi_geometry_mode": "gt_metric",
        "training_prior.smpl_translation_noise_contract": "v4_deterministic",
        "loss.hsi_v4_candidate_transl_weight": 8.0,
        "loss.hsi_align_point_weight": 0.0,
        "loss.hsi_no_worse_weight": 0.0,
    }
    for path, value in expected.items():
        cursor = config
        for key in path.split("."):
            cursor = cursor[key]
        if cursor != value:
            raise SystemExit(f"Unexpected resolved config {path}: {cursor!r} != {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "overfit"), default="smoke")
    return parser.parse_args()


if __name__ == "__main__":
    main()
