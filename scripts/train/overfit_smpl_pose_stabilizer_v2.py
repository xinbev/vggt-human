#!/usr/bin/env python3
"""E0: prove conservative per-joint SO(3) pose stabilization on fixed data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from vggt_omega.data import SMPLTemporalPickleDataset
from vggt_omega.models import PoseStabilizerConfig, PoseStabilizerLoss, PoseTemporalStabilizer
from vggt_omega.training import PoseNoiseConfig, corrupt_pose_sequence
from vggt_omega.training.config import deep_update, load_yaml_config


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


def _resolve_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config["data"].get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must be a non-empty list")
    return [dict(source) for source in sources]


def _fixed_batch(dataset: SMPLTemporalPickleDataset, batch_size: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    by_source: dict[str, list[int]] = {}
    for sample_index, (record_index, _) in enumerate(dataset.index):
        by_source.setdefault(dataset.records[record_index].dataset_name, []).append(sample_index)
    rng = random.Random(seed)
    selected: list[int] = []
    names = sorted(by_source)
    per_source, remainder = divmod(batch_size, len(names))
    counts: dict[str, int] = {}
    for offset, name in enumerate(names):
        count = per_source + int(offset < remainder)
        selected.extend(rng.sample(by_source[name], count))
        counts[name] = count
    rng.shuffle(selected)
    samples = [dataset[index] for index in selected]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}, counts


def _floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


@torch.no_grad()
def _evaluate(model: PoseTemporalStabilizer, criterion: PoseStabilizerLoss, target: torch.Tensor, observed: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    model.eval()
    return _floats(criterion(model(observed, valid), target, observed, valid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    for override in args.override:
        key, value = override.split("=", 1)
        _set_dotted(config, key, _parse_value(value))
    seed = int(config["experiment"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    data_config = config["data"]
    dataset = SMPLTemporalPickleDataset(
        sources=_resolve_sources(config),
        window_size=int(data_config["window_size"]),
        stride=int(data_config.get("stride", 1)),
        partition="train",
        validation_fraction=float(data_config.get("validation_fraction", 0.1)),
        min_valid_frames=int(data_config.get("min_valid_frames", data_config["window_size"])),
    )
    batch, source_counts = _fixed_batch(dataset, int(data_config["overfit_batch_size"]), seed)
    target = batch["target_pose_6d"].to(device)
    valid = batch["valid_mask"].to(device)
    noise = PoseNoiseConfig(**config["noise"])
    observed = corrupt_pose_sequence(target, valid, noise).detach()
    model_config = PoseStabilizerConfig(**config["model"])
    model = PoseTemporalStabilizer(model_config).to(device)
    loss_config = dict(config["loss"])
    loss_config.setdefault("max_blend", model_config.max_blend)
    criterion = PoseStabilizerLoss(**loss_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optim"]["lr"]), weight_decay=float(config["optim"].get("weight_decay", 0.0)))
    amp_enabled = bool(config["optim"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _evaluate(model, criterion, target, observed, valid)
    summary: dict[str, Any] = {"device": str(device), "dataset_summary": dataset.summary(), "fixed_batch_source_counts": source_counts, "noise": asdict(noise), "initial": initial}
    print("========== Pose Stabilizer V2 E0 fixed-batch test ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    wandb_run = None
    wandb_config = config.get("logging", {}).get("wandb", {})
    if bool(wandb_config.get("enabled", True)):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_config.get("project", "vggt-human")),
            entity=str(wandb_config.get("entity", "")) or None,
            name=str(wandb_config.get("name", config["experiment"]["name"])),
            group=str(wandb_config.get("group", "smpl_temporal_stabilizer_v2")),
            mode=str(wandb_config.get("mode", "online")),
            config=config,
            tags=list(wandb_config.get("tags", ["smpl", "pose-stabilizer", "e0-overfit"])),
        )
        wandb_run.log({f"initial/{key}": value for key, value in initial.items()}, step=0)

    steps = int(config["optim"]["steps"])
    interval = int(config["optim"].get("log_interval", 25))
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            losses = criterion(model(observed, valid), target, observed, valid)
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optim"].get("grad_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % interval == 0 or step == steps:
            metrics = _floats(losses)
            print(f"[step {step:04d}/{steps}] total={metrics['loss_total']:.6f} base={metrics['metric_base_geodesic_rad']:.5f} final={metrics['metric_final_geodesic_rad']:.5f} improve={metrics['metric_improvement_rad']:.5f} blend={metrics['metric_blend_mean']:.3f}")
            if wandb_run is not None:
                wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)
    final = _evaluate(model, criterion, target, observed, valid)
    passed = final["metric_final_geodesic_rad"] < initial["metric_base_geodesic_rad"] and final["metric_improvement_rad"] > 0.01
    summary.update({"final": final, "passed": passed, "pass_rule": "final geodesic < base geodesic and mean improvement > 0.01 rad"})
    (output_dir / "e0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"format": "smpl_temporal_stabilizer_v2_pose_e0", "model_config": asdict(model_config), "model_state": model.state_dict(), "initial": initial, "final": final, "passed": passed}, output_dir / "checkpoint_e0.pt")
    print(f"[E0 {'PASS' if passed else 'FAIL'}] final={final['metric_final_geodesic_rad']:.5f}rad base={initial['metric_base_geodesic_rad']:.5f}rad")
    if wandb_run is not None:
        final_log = {f"final/{key}": value for key, value in final.items()}
        final_log["e0/passed"] = int(passed)
        wandb_run.log(final_log, step=steps)
        wandb_run.finish()


if __name__ == "__main__":
    main()
