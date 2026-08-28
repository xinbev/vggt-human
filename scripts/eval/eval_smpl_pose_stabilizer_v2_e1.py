#!/usr/bin/env python3
"""E1 held-out evaluation for the conservative SMPL pose stabilizer.

The E0 checkpoint is evaluated without any gradient update on fixed validation
windows under four conditions: clean, centre-only jitter, full small flicker,
and full medium flicker.  Metrics are broken down by root, torso, limbs and
fast-moving joints so an apparent average gain cannot hide pose oversmoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from vggt_omega.data import SMPLTemporalPickleDataset
from vggt_omega.models import PoseStabilizerConfig, PoseTemporalStabilizer
from vggt_omega.training import PoseNoiseConfig, corrupt_pose_sequence
from vggt_omega.training.config import deep_update, load_yaml_config
from vggt_omega.utils.rotation import rotation_matrix_to_axis_angle, rot6d_to_rotmat


ROOT = (0,)
TORSO = (3, 6, 9, 12, 15)
LIMBS = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23)
GROUPS = {"all": tuple(range(24)), "root": ROOT, "torso": TORSO, "limbs": LIMBS}


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_dotted(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _balanced_fixed_batch(dataset: SMPLTemporalPickleDataset, batch_size: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    by_source: dict[str, list[int]] = {}
    for sample_index, (record_index, _) in enumerate(dataset.index):
        by_source.setdefault(dataset.records[record_index].dataset_name, []).append(sample_index)
    if len(by_source) < 2:
        raise RuntimeError("E1 expects both EMDB and 3DPW validation windows")
    rng = random.Random(seed)
    names = sorted(by_source)
    base_count, remainder = divmod(batch_size, len(names))
    selected: list[int] = []
    counts: dict[str, int] = {}
    for offset, name in enumerate(names):
        count = base_count + int(offset < remainder)
        if len(by_source[name]) < count:
            raise RuntimeError(f"Not enough held-out {name} windows: need {count}, found {len(by_source[name])}")
        selected.extend(rng.sample(by_source[name], count))
        counts[name] = count
    rng.shuffle(selected)
    samples = [dataset[index] for index in selected]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}, counts


def _geodesic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rotation_matrix_to_axis_angle(pred @ target.transpose(-1, -2)).norm(dim=-1)


def _mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return float(((value * weight).sum() / weight.sum().clamp_min(1.0)).detach().cpu())


@torch.no_grad()
def _evaluate_case(
    model: PoseTemporalStabilizer,
    target_pose: torch.Tensor,
    observed_pose: torch.Tensor,
    valid_mask: torch.Tensor,
    fast_speed_threshold_rad: float,
    blend_threshold: float,
    focus_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    outputs = model(observed_pose, valid_mask)
    batch, steps, _ = target_pose.shape
    target = rot6d_to_rotmat(target_pose.reshape(batch, steps, 24, 6))
    observed = rot6d_to_rotmat(observed_pose.reshape(batch, steps, 24, 6))
    final = rot6d_to_rotmat(outputs["refined_pose_6d"].reshape(batch, steps, 24, 6))
    context = outputs["context_valid"] & valid_mask
    base_error = _geodesic(observed, target)
    final_error = _geodesic(final, target)
    displacement = _geodesic(final, observed)
    per_group: dict[str, dict[str, float]] = {}
    for name, joint_ids in GROUPS.items():
        index = torch.as_tensor(joint_ids, device=target.device)
        base_group = base_error.index_select(-1, index).mean(dim=-1)
        final_group = final_error.index_select(-1, index).mean(dim=-1)
        disp_group = displacement.index_select(-1, index).mean(dim=-1)
        per_group[name] = {
            "base_geodesic_rad": _mean(base_group, context),
            "final_geodesic_rad": _mean(final_group, context),
            "improvement_rad": _mean(base_group - final_group, context),
            "improvement_rate": _mean((final_group < base_group).to(target.dtype), context),
            "final_displacement_rad": _mean(disp_group, context),
        }
    # The target angular speed identifies real fast motion independent of any
    # injected error.  It is only evaluated at context-valid centre frames.
    speed = torch.zeros_like(base_error)
    speed[:, 1:] = _geodesic(target[:, 1:], target[:, :-1])
    fast = (speed >= fast_speed_threshold_rad) & context.unsqueeze(-1)
    fast_count = int(fast.sum().detach().cpu())
    if fast_count:
        fast_metrics = {
            "joint_frame_count": fast_count,
            "base_geodesic_rad": _mean(base_error, fast),
            "final_geodesic_rad": _mean(final_error, fast),
            "improvement_rad": _mean(base_error - final_error, fast),
            "final_displacement_rad": _mean(displacement, fast),
        }
    else:
        fast_metrics = {"joint_frame_count": 0, "base_geodesic_rad": None, "final_geodesic_rad": None, "improvement_rad": None, "final_displacement_rad": None}
    focus_metrics = None
    if focus_mask is not None:
        focus = focus_mask.to(device=target.device, dtype=torch.bool) & context
        focus_groups: dict[str, dict[str, float]] = {}
        for name, joint_ids in GROUPS.items():
            index = torch.as_tensor(joint_ids, device=target.device)
            base_group = base_error.index_select(-1, index).mean(dim=-1)
            final_group = final_error.index_select(-1, index).mean(dim=-1)
            focus_groups[name] = {
                "base_geodesic_rad": _mean(base_group, focus),
                "final_geodesic_rad": _mean(final_group, focus),
                "improvement_rad": _mean(base_group - final_group, focus),
                "improvement_rate": _mean((final_group < base_group).to(target.dtype), focus),
                "final_displacement_rad": _mean(displacement.index_select(-1, index).mean(dim=-1), focus),
            }
        focus_metrics = {"frame_count": int(focus.sum().detach().cpu()), "groups": focus_groups}
    return {
        "context_frames": int(context.sum().detach().cpu()),
        "context_fraction": float(context.to(dtype=target.dtype).mean().detach().cpu()),
        "blend_mean": _mean(outputs["blend"].mean(dim=-2).squeeze(-1), context),
        "blend_active_rate": _mean((outputs["blend"].mean(dim=-2).squeeze(-1) > blend_threshold).to(target.dtype), context),
        "groups": per_group,
        "fast_motion": fast_metrics,
        "corrupted_centre": focus_metrics,
    }


def _resolve_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config["data"].get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must be non-empty")
    return [dict(source) for source in sources]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    for override in args.override:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override!r}")
        key, value = override.split("=", 1)
        _set_dotted(config, key, _parse_value(value))
    seed = int(config["experiment"].get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    data = config["data"]
    dataset = SMPLTemporalPickleDataset(
        sources=_resolve_sources(config),
        window_size=int(data["window_size"]),
        stride=int(data.get("stride", 1)),
        partition="val",
        validation_fraction=float(data.get("validation_fraction", 0.1)),
        min_valid_frames=int(data.get("min_valid_frames", data["window_size"])),
    )
    batch, source_counts = _balanced_fixed_batch(dataset, int(data["eval_batch_size"]), seed)
    target = batch["target_pose_6d"].to(device)
    valid = batch["valid_mask"].to(device)
    checkpoint = torch.load(config["checkpoint"]["path"], map_location=device, weights_only=False)
    if checkpoint.get("format") not in {"smpl_temporal_stabilizer_v2_pose_e0", "smpl_temporal_stabilizer_v2_pose_mixture"}:
        raise ValueError("E1 requires a V2 pose E0 or mixed-training checkpoint")
    model = PoseTemporalStabilizer(PoseStabilizerConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    cases = config["cases"]
    generator = torch.Generator(device=device)
    generated: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {"clean": (target, None)}
    for name in ("centre_jitter", "small", "medium"):
        item = cases[name]
        generator.manual_seed(seed + int(item.get("noise_seed_offset", 0)))
        full = corrupt_pose_sequence(target, valid, PoseNoiseConfig(**item["noise"]), generator=generator)
        if name == "centre_jitter":
            observed = target.clone()
            centre = target.shape[1] // 2
            observed[:, centre] = full[:, centre]
            focus_mask = torch.zeros(target.shape[:2], device=device, dtype=torch.bool)
            focus_mask[:, centre] = True
            generated[name] = (observed, focus_mask)
        else:
            generated[name] = (full, None)
    results: dict[str, Any] = {}
    for name, (observed, focus_mask) in generated.items():
        results[name] = _evaluate_case(
            model,
            target,
            observed,
            valid,
            fast_speed_threshold_rad=float(config["metrics"]["fast_speed_threshold_rad"]),
            blend_threshold=float(config["metrics"]["blend_active_threshold"]),
            focus_mask=focus_mask,
        )
    clean = results["clean"]
    pass_rules = {
        "clean_root_displacement_max_rad": float(config["acceptance"]["clean_root_displacement_max_rad"]),
        "clean_all_displacement_max_rad": float(config["acceptance"]["clean_all_displacement_max_rad"]),
        "centre_jitter_improvement_min_rad": float(config["acceptance"]["centre_jitter_improvement_min_rad"]),
        "small_improvement_min_rad": float(config["acceptance"]["small_improvement_min_rad"]),
        "medium_improvement_min_rad": float(config["acceptance"]["medium_improvement_min_rad"]),
    }
    passed = (
        clean["groups"]["root"]["final_displacement_rad"] <= pass_rules["clean_root_displacement_max_rad"]
        and clean["groups"]["all"]["final_displacement_rad"] <= pass_rules["clean_all_displacement_max_rad"]
        and results["centre_jitter"]["groups"]["all"]["improvement_rad"] >= pass_rules["centre_jitter_improvement_min_rad"]
        and results["small"]["groups"]["all"]["improvement_rad"] >= pass_rules["small_improvement_min_rad"]
        and results["medium"]["groups"]["all"]["improvement_rad"] >= pass_rules["medium_improvement_min_rad"]
    )
    summary = {
        "checkpoint": str(config["checkpoint"]["path"]),
        "device": str(device),
        "held_out_dataset_summary": dataset.summary(),
        "fixed_batch_source_counts": source_counts,
        "metrics": results,
        "acceptance": pass_rules,
        "passed": passed,
        "note": "Fast-motion metrics are diagnostic in E1; a dedicated real-NLF evaluation is required before deployment.",
    }
    (output_dir / "e1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("========== Pose Stabilizer V2 E1 held-out evaluation ==========")
    for name, result in results.items():
        all_group = result["groups"]["all"]
        print(f"[{name}] base={all_group['base_geodesic_rad']:.5f} final={all_group['final_geodesic_rad']:.5f} improve={all_group['improvement_rad']:.5f} root_disp={result['groups']['root']['final_displacement_rad']:.5f} blend={result['blend_mean']:.3f}")
    print(f"[E1 {'PASS' if passed else 'FAIL'}] {output_dir / 'e1_summary.json'}")
    wandb_config = config.get("logging", {}).get("wandb", {})
    if bool(wandb_config.get("enabled", False)):
        try:
            import wandb

            run = wandb.init(
                project=str(wandb_config.get("project", "vggt-human")),
                entity=str(wandb_config.get("entity", "")) or None,
                name=str(wandb_config.get("name", config["experiment"]["name"])),
                group=str(wandb_config.get("group", "smpl_temporal_stabilizer_v2")),
                mode=str(wandb_config.get("mode", "online")),
                config=config,
                tags=list(wandb_config.get("tags", ["smpl", "pose-stabilizer", "e1-held-out"])),
            )
            log = {"e1/passed": int(passed)}
            for case_name, result in results.items():
                for group_name, metrics in result["groups"].items():
                    for metric_name, value in metrics.items():
                        log[f"{case_name}/{group_name}/{metric_name}"] = value
                for metric_name, value in result["fast_motion"].items():
                    if value is not None:
                        log[f"{case_name}/fast_motion/{metric_name}"] = value
                log[f"{case_name}/blend_mean"] = result["blend_mean"]
                log[f"{case_name}/blend_active_rate"] = result["blend_active_rate"]
                if result["corrupted_centre"] is not None:
                    for group_name, metrics in result["corrupted_centre"]["groups"].items():
                        for metric_name, value in metrics.items():
                            log[f"{case_name}/corrupted_centre/{group_name}/{metric_name}"] = value
            run.log(log)
            run.finish()
        except ImportError as error:
            raise RuntimeError("E1 W&B logging was enabled but wandb is unavailable") from error


if __name__ == "__main__":
    main()
