from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


ALLOWED_NONZERO_WEIGHTS = {
    "hsi_grounding_gate_weight",
    "hsi_grounding_gate_positive_weight",
    "hsi_grounding_gate_negative_weight",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate severe-float Gate-only grounding config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    loss = config.get("loss", {})

    problems: list[str] = []
    if loss.get("hsi_grounding_gate_target_mode") != "severe_float":
        problems.append("loss.hsi_grounding_gate_target_mode must be severe_float")
    if model.get("hsi_grounding_hard_gate_train") is not True:
        problems.append("model.hsi_grounding_hard_gate_train must be true")
    if model.get("hsi_grounding_hard_gate_eval") is not True:
        problems.append("model.hsi_grounding_hard_gate_eval must be true")
    model_threshold = float(model.get("hsi_grounding_gate_threshold", float("nan")))
    loss_threshold = float(loss.get("hsi_grounding_gate_decision_threshold", float("nan")))
    if not math.isfinite(model_threshold) or abs(model_threshold - loss_threshold) > 1e-8:
        problems.append("model/loss Gate decision thresholds must match")
    if float(loss.get("hsi_grounding_gate_weight", 0.0)) <= 0.0:
        problems.append("loss.hsi_grounding_gate_weight must be positive")

    conflicting = {
        key: float(value)
        for key, value in loss.items()
        if key.endswith("_weight")
        and key not in ALLOWED_NONZERO_WEIGHTS
        and isinstance(value, (int, float))
        and abs(float(value)) > 0.0
    }
    if conflicting:
        problems.append(f"non-Gate loss weights must be zero: {conflicting}")
    if problems:
        raise RuntimeError("Invalid severe-float grounding config:\n- " + "\n- ".join(problems))
    print(
        "[grounding-config] pass "
        f"target=severe_float threshold={model_threshold:.2f} "
        f"float_min={float(loss['hsi_grounding_severe_float_threshold_m']):.3f}m "
        "loss=gate_only"
    )


if __name__ == "__main__":
    main()
