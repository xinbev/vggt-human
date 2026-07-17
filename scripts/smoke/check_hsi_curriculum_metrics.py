from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data = json.loads((output_dir / "metrics_latest.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    check_resolved_config(config, args.stage)
    metric_source = "val" if data.get("val") else "train"
    metrics = data.get(metric_source) or {}
    required = {
        "stage2": [
            "metric_stage2_selection",
            "metric_hsi_base_transl_l2_median",
            "metric_hsi_transl_l2_median",
            "metric_hsi_transl_improvement_rate",
            "metric_hsi_base_transl_noisy_l2_median",
            "metric_hsi_transl_noisy_l2_median",
            "metric_hsi_transl_noisy_improvement_rate",
            "metric_hsi_transl_clean_displacement_mean_m",
            "metric_hsi_transl_clean_gate_mean",
            "metric_hsi_transl_noisy_gate_mean",
            "metric_hsi_tangent_delta_base_l1",
            "metric_hsi_tangent_delta_refined_l1",
        ],
        "stage3": [
            "metric_stage3_selection",
            "metric_hsi_contact_float_p95_m",
            "metric_hsi_contact_penetration_p95_m",
            "metric_hsi_contact_base_abs_p95_m",
            "metric_hsi_contact_refined_abs_p95_m",
            "metric_hsi_contact_false_pull_rate",
            "metric_hsi_contact_contact_gate_mean",
            "metric_hsi_contact_swing_gate_mean",
            "metric_hsi_contact_swing_displacement_mean_m",
        ],
    }[args.stage]
    missing = [key for key in required if key not in metrics or not math.isfinite(float(metrics[key]))]
    if missing:
        raise SystemExit(f"Missing or non-finite gate metrics: {missing}")
    if args.mode == "overfit":
        if args.stage == "stage2" and float(metrics["metric_hsi_transl_noisy_improvement_rate"]) < 0.90:
            raise SystemExit("Stage2 overfit gate failed: noisy improvement_rate < 0.90")
        if args.stage == "stage2":
            base = float(metrics["metric_hsi_base_transl_noisy_l2_median"])
            refined = float(metrics["metric_hsi_transl_noisy_l2_median"])
            if base <= 0.0 or refined > 0.30 * base:
                raise SystemExit("Stage2 overfit gate failed: noisy refined median is above 30% of base")
            if float(metrics["metric_hsi_transl_clean_displacement_mean_m"]) > 0.005:
                raise SystemExit("Stage2 overfit gate failed: clean displacement exceeds 5 mm")
            if float(metrics["metric_hsi_transl_clean_gate_mean"]) >= float(metrics["metric_hsi_transl_noisy_gate_mean"]):
                raise SystemExit("Stage2 overfit gate failed: align gate did not separate clean and noisy people")
            tangent_base = float(metrics["metric_hsi_tangent_delta_base_l1"])
            tangent_refined = float(metrics["metric_hsi_tangent_delta_refined_l1"])
            if tangent_refined > tangent_base + 0.0005:
                raise SystemExit("Stage2 overfit gate failed: tangent translation degraded by more than 0.5 mm")
        if args.stage == "stage3":
            if float(metrics["metric_hsi_contact_false_pull_rate"]) > 0.05:
                raise SystemExit("Stage3 overfit gate failed: false_pull_rate > 0.05")
            if float(metrics["metric_hsi_contact_contact_gate_mean"]) <= float(metrics["metric_hsi_contact_swing_gate_mean"]):
                raise SystemExit("Stage3 overfit gate failed: contact gate did not separate from swing gate")
            base = float(metrics["metric_hsi_contact_base_abs_p95_m"])
            refined = float(metrics["metric_hsi_contact_refined_abs_p95_m"])
            if base <= 0.0 or refined > 0.30 * base:
                raise SystemExit("Stage3 overfit gate failed: contact p95 reduction is below 70%")
    if args.mode == "distribution" and args.stage == "stage2":
        if float(metrics["metric_hsi_transl_noisy_improvement_rate"]) <= 0.50:
            raise SystemExit("Stage2 distribution gate failed: noisy improvement_rate <= 0.50")
    if args.mode == "distribution" and args.stage == "stage3":
        base = float(metrics["metric_hsi_contact_base_abs_p95_m"])
        refined = float(metrics["metric_hsi_contact_refined_abs_p95_m"])
        if base <= 0.0 or refined >= base:
            raise SystemExit("Stage3 distribution gate failed: contact p95 did not improve")
        if float(metrics["metric_hsi_contact_false_pull_rate"]) > 0.10:
            raise SystemExit("Stage3 distribution gate failed: false_pull_rate > 0.10")
    print(
        json.dumps(
            {
                "gate": "pass",
                "stage": args.stage,
                "mode": args.mode,
                "metric_source": metric_source,
                "metrics": {key: metrics[key] for key in required},
            },
            indent=2,
        )
    )


def check_resolved_config(config: dict, stage: str) -> None:
    expected = {
        "loss.hsi_depth_teacher_weight": 0.0,
        "loss.hsi_anchor_depth_weight": 0.0,
        "loss.hsi_anchor_scene_xyz_weight": 0.0,
        "loss.hsi_smpl_scale_teacher_weight": 0.0,
        "loss.hsi_betas_weight": 0.0,
        "loss.hsi_gate_reg_weight": 0.0,
        "model.hsi_geometry_mode": "gt_metric",
    }
    if stage == "stage2":
        expected.update(
            {
                "model.enable_hsi_contact_refine": False,
                "loss.hsi_projected_joints2d_weight": 0.0,
                "loss.hsi_delta_reg_weight": 0.0,
                "loss.hsi_no_worse_weight": 2.0,
                "loss.hsi_transl_clean_identity_weight": 20.0,
                "loss.hsi_transl_noise_gate_weight": 2.0,
            }
        )
    else:
        expected.update(
            {
                "model.enable_hsi_contact_refine": True,
                "loss.hsi_projected_joints2d_weight": 0.0,
                "loss.hsi_delta_reg_weight": 0.05,
                "loss.hsi_no_worse_weight": 8.0,
                "loss.hsi_contact_refine_plane_weight": 6.0,
                "loss.hsi_contact_refine_class_weight": 0.2,
                "loss.hsi_contact_refine_swing_no_pull_weight": 5.0,
            }
        )

    mismatches = []
    for path, expected_value in expected.items():
        actual = nested_value(config, path)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{path}: expected {expected_value!r}, got {actual!r}")
    if mismatches:
        raise SystemExit("Resolved curriculum config gate failed:\n  " + "\n  ".join(mismatches))


def nested_value(data: dict, path: str):
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("stage2", "stage3"), required=True)
    parser.add_argument("--mode", choices=("smoke", "overfit", "distribution"), default="smoke")
    return parser.parse_args()


if __name__ == "__main__":
    main()
