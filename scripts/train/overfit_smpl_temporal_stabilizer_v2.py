#!/usr/bin/env python3
"""E0: prove the V2 translation stabilizer on one fixed real GT batch.

This is intentionally not a full training program.  It freezes both the
selected EMDB/3DPW windows and their injected single-frame translation noise,
then repeatedly fits the same batch.  If this cannot improve the base error,
full-data training must not be started.
"""

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
from vggt_omega.models import TranslationStabilizerConfig, TranslationStabilizerLoss, TranslationTemporalStabilizer
from vggt_omega.training import TranslationNoiseConfig, corrupt_translation_sequence
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


def _get_dotted(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing config field: {dotted}")
        value = value[part]
    return value


def _resolve_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config["data"].get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must be a non-empty list")
    result = []
    for source in sources:
        item = dict(source)
        if not str(item.get("root", "")).strip():
            item["root"] = _get_dotted(config, str(item["root_key"]))
        result.append(item)
    return result


def _build_fixed_batch(dataset: SMPLTemporalPickleDataset, batch_size: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    candidates: dict[str, list[int]] = {}
    for dataset_index, (record_index, _) in enumerate(dataset.index):
        candidates.setdefault(dataset.records[record_index].dataset_name, []).append(dataset_index)
    if not candidates:
        raise RuntimeError("Dataset contains no windows")
    rng = random.Random(seed)
    source_names = sorted(candidates)
    selected: list[int] = []
    base_count, remainder = divmod(batch_size, len(source_names))
    for source_offset, name in enumerate(source_names):
        count = base_count + int(source_offset < remainder)
        if len(candidates[name]) < count:
            raise RuntimeError(f"Not enough windows in {name}: {len(candidates[name])} < {count}")
        selected.extend(rng.sample(candidates[name], count))
    rng.shuffle(selected)
    samples = [dataset[index] for index in selected]
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    return batch, {name: base_count + int(offset < remainder) for offset, name in enumerate(source_names)}


def _metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


@torch.no_grad()
def _evaluate(
    model: TranslationTemporalStabilizer,
    criterion: TranslationStabilizerLoss,
    target: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    return _metrics_to_float(criterion(model(observed, valid), target, observed, valid))


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

    experiment = config["experiment"]
    seed = int(experiment.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    data_config = config["data"]
    dataset = SMPLTemporalPickleDataset(
        sources=_resolve_sources(config),
        window_size=int(data_config["window_size"]),
        stride=int(data_config.get("stride", 1)),
        partition="train",
        validation_fraction=float(data_config.get("validation_fraction", 0.1)),
        min_valid_frames=int(data_config.get("min_valid_frames", data_config["window_size"])),
    )
    batch, batch_source_counts = _build_fixed_batch(dataset, int(data_config["overfit_batch_size"]), seed)
    target = batch["target_transl"].to(device)
    valid = batch["valid_mask"].to(device)
    noise_config = TranslationNoiseConfig(**config["noise"])
    observed = corrupt_translation_sequence(target, valid, noise_config).detach()

    model_config = TranslationStabilizerConfig(**config["model"])
    if model_config.window_size != target.shape[1]:
        raise ValueError("data.window_size and model.window_size must match")
    model = TranslationTemporalStabilizer(model_config).to(device)
    loss_config = dict(config["loss"])
    loss_config.setdefault("max_blend", model_config.max_blend)
    criterion = TranslationStabilizerLoss(**loss_config)
    optim_config = config["optim"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_config["lr"]), weight_decay=float(optim_config.get("weight_decay", 0.0)))
    amp_enabled = bool(optim_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    initial = _evaluate(model, criterion, target, observed, valid)
    run_summary = {
        "device": str(device),
        "dataset_summary": dataset.summary(),
        "fixed_batch_source_counts": batch_source_counts,
        "initial": initial,
        "noise": asdict(noise_config),
    }
    print("========== Stabilizer V2 E0 fixed-batch test ==========")
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))

    wandb_run = None
    wandb_config = config.get("logging", {}).get("wandb", {})
    if bool(wandb_config.get("enabled", True)):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(wandb_config.get("project", "vggt-human")),
                entity=str(wandb_config.get("entity", "")) or None,
                name=str(wandb_config.get("name", experiment["name"])),
                group=str(wandb_config.get("group", "smpl_temporal_stabilizer_v2")),
                mode=str(wandb_config.get("mode", "online")),
                config=config,
                tags=list(wandb_config.get("tags", ["smpl", "temporal-stabilizer", "e0-overfit"])),
            )
            wandb_run.log({f"initial/{key}": value for key, value in initial.items()}, step=0)
        except ImportError as error:
            raise RuntimeError("W&B is enabled but wandb is unavailable") from error

    steps = int(optim_config["steps"])
    log_interval = int(optim_config.get("log_interval", 25))
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(observed, valid)
            losses = criterion(outputs, target, observed, valid)
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_config.get("grad_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % log_interval == 0 or step == steps:
            metrics = _metrics_to_float(losses)
            print(
                f"[step {step:04d}/{steps}] total={metrics['loss_total']:.6f} "
                f"base={metrics['metric_base_l1_m']:.5f} final={metrics['metric_final_l1_m']:.5f} "
                f"improve={metrics['metric_improvement_m']:.5f} blend={metrics['metric_blend_mean']:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)

    final = _evaluate(model, criterion, target, observed, valid)
    passed = final["metric_final_l1_m"] < initial["metric_base_l1_m"] and final["metric_improvement_m"] > 0.002
    run_summary["final"] = final
    run_summary["passed"] = passed
    run_summary["pass_rule"] = "final_l1 < initial_base_l1 and improvement > 0.002m"
    (output_dir / "e0_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "format": "smpl_temporal_stabilizer_v2_translation_e0",
            "model_config": asdict(model_config),
            "model_state": model.state_dict(),
            "initial": initial,
            "final": final,
            "passed": passed,
        },
        output_dir / "checkpoint_e0.pt",
    )
    print(f"[E0 {'PASS' if passed else 'FAIL'}] final={final['metric_final_l1_m']:.5f}m base={initial['metric_base_l1_m']:.5f}m")
    if wandb_run is not None:
        final_log = {f"final/{key}": value for key, value in final.items()}
        final_log["e0/passed"] = int(passed)
        wandb_run.log(final_log, step=steps)
        wandb_run.finish()


if __name__ == "__main__":
    main()
