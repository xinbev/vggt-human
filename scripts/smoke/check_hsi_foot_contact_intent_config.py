from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = config.get("model", {})
    data = config.get("data", {})
    loss = config.get("loss", {})
    checkpoint = config.get("checkpoint", {})
    optim = config.get("optim", {})
    prior = config.get("training_prior", {})
    checks = {
        "intent_enabled": model.get("enable_hsi_foot_contact_intent") is True,
        "intent_only": model.get("train_hsi_foot_contact_intent_only") is True,
        "grounding_disabled": model.get("enable_hsi_grounding") is False,
        "legacy_contact_disabled": model.get("enable_hsi_contact_refine") is False,
        "gt_provider": model.get("smpl_provider") == "gt_perturbed",
        "five_frame_clip": int(data.get("sequence_length", 0)) >= 5,
        "joint_temporal_support": (
            model.get("hsi_foot_contact_intent_feature_version") == "camera_motion_v3_joint5"
        ),
        "fast_gt_path": model.get("hsi_foot_contact_intent_fast_gt") is True,
        "center_frame_supervision": loss.get("hsi_foot_contact_intent_center_frame_only") is True,
        "full_window_distribution": not data.get("train_contact_only") and not data.get("val_contact_only"),
        "clean_smpl": (
            prior.get("smpl_perturb_mode") == "translation"
            and float(prior.get("smpl_transl_ray_noise_schedule", 1.0)) == 0.0
            and float(prior.get("smpl_transl_tangent_noise_schedule_m", 1.0)) == 0.0
            and float(prior.get("smpl_transl_ray_noise_clean_prob", 0.0)) == 1.0
        ),
        "isolated_save": checkpoint.get("save_prefixes") == ["hsi_foot_contact_intent_head."],
        "isolated_trainable": optim.get("allowed_trainable_prefixes") == ["hsi_foot_contact_intent_head."],
        "intent_loss_enabled": float(loss.get("hsi_foot_contact_intent_weight", 0.0)) > 0.0,
        "grounding_loss_disabled": float(loss.get("hsi_grounding_gate_weight", 0.0)) == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Invalid contact-intent config: " + ", ".join(failed))
    print("[config] HSI foot contact intent contract passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
