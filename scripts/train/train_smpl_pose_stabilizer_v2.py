#!/usr/bin/env python3
"""Train the conservative SMPL pose stabilizer with a mixed error curriculum.

Every minibatch contains clean observations, centre-only flicker, full small
flicker and full medium flicker.  This directly teaches the fusion gate both
when to correct a single bad frame and when to leave a correct frame alone.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from vggt_omega.data import SMPLTemporalPickleDataset
from vggt_omega.models import PoseStabilizerConfig, PoseStabilizerLoss, PoseTemporalStabilizer
from vggt_omega.training import PoseNoiseConfig, corrupt_pose_sequence
from vggt_omega.training.config import deep_update, load_yaml_config


MODE_NAMES = ("clean", "centre_jitter", "small", "medium")


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


def _balanced_sampler(dataset: SMPLTemporalPickleDataset, enabled: bool, seed: int) -> WeightedRandomSampler | None:
    if not enabled:
        return None
    names = [dataset.records[record_index].dataset_name for record_index, _ in dataset.index]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    if len(counts) < 2:
        return None
    weights = torch.as_tensor([1.0 / counts[name] for name in names], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def _balanced_fixed_batch(dataset: SMPLTemporalPickleDataset, batch_size: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    by_source: dict[str, list[int]] = {}
    for sample_index, (record_index, _) in enumerate(dataset.index):
        by_source.setdefault(dataset.records[record_index].dataset_name, []).append(sample_index)
    rng = random.Random(seed)
    names = sorted(by_source)
    per_source, remainder = divmod(batch_size, len(names))
    selected: list[int] = []
    counts: dict[str, int] = {}
    for offset, name in enumerate(names):
        count = per_source + int(offset < remainder)
        selected.extend(rng.sample(by_source[name], count))
        counts[name] = count
    rng.shuffle(selected)
    samples = [dataset[index] for index in selected]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}, counts


def _make_observations(
    target: torch.Tensor,
    valid: torch.Tensor,
    modes: torch.Tensor,
    config: dict[str, Any],
    generator: torch.Generator,
    return_centre_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Mix clean / centre / full small / full medium observations per sample."""
    observed = target.clone()
    centre_mask = torch.zeros(target.shape[:2], device=target.device, dtype=torch.bool)
    small = corrupt_pose_sequence(target, valid, PoseNoiseConfig(**config["small"]), generator=generator)
    medium = corrupt_pose_sequence(target, valid, PoseNoiseConfig(**config["medium"]), generator=generator)
    centre_noise = corrupt_pose_sequence(target, valid, PoseNoiseConfig(**config["centre_jitter"]), generator=generator)
    if bool((modes == 1).any()):
        rows = torch.nonzero(modes == 1, as_tuple=False).squeeze(-1)
        possible_centres = torch.arange(2, target.shape[1] - 2, device=target.device)
        centres = possible_centres[torch.randint(possible_centres.numel(), (rows.numel(),), device=target.device, generator=generator)]
        observed[rows, centres] = centre_noise[rows, centres]
        centre_mask[rows, centres] = True
    observed[modes == 2] = small[modes == 2]
    observed[modes == 3] = medium[modes == 3]
    return (observed, centre_mask) if return_centre_mask else observed


def _mode_ids(batch_size: int, probabilities: list[float], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    values = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    if values.numel() != len(MODE_NAMES) or bool((values < 0).any()) or float(values.sum()) <= 0:
        raise ValueError("mixture probabilities must be four non-negative values with positive sum")
    return torch.multinomial(values / values.sum(), batch_size, replacement=True, generator=generator)


def _to_float(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}


@torch.no_grad()
def _evaluate_fixed_cases(
    model: PoseTemporalStabilizer,
    criterion: PoseStabilizerLoss,
    target: torch.Tensor,
    valid: torch.Tensor,
    noise: dict[str, Any],
    seed: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    generator = torch.Generator(device=target.device)
    results: dict[str, dict[str, float]] = {}
    for mode_index, name in enumerate(MODE_NAMES):
        generator.manual_seed(seed + 1000 + mode_index)
        modes = torch.full((target.shape[0],), mode_index, device=target.device, dtype=torch.long)
        observed = _make_observations(target, valid, modes, noise, generator)
        results[name] = _to_float(criterion(model(observed, valid), target, observed, valid))
    return results


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
    records, file_counts = SMPLTemporalPickleDataset.load_records(_resolve_sources(config))
    common = {
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config.get("stride", 1)),
        "validation_fraction": float(data_config.get("validation_fraction", 0.1)),
        "min_valid_frames": int(data_config.get("min_valid_frames", data_config["window_size"])),
    }
    train_set = SMPLTemporalPickleDataset(partition="train", records=records, source_file_counts=file_counts, **common)
    val_set = SMPLTemporalPickleDataset(partition="val", records=records, source_file_counts=file_counts, **common)
    sampler = _balanced_sampler(train_set, bool(data_config.get("balance_datasets", True)), seed)
    workers = int(data_config.get("num_workers", 8))
    loader_kwargs = {"batch_size": int(config["optim"]["batch_size"]), "num_workers": workers, "pin_memory": bool(data_config.get("pin_memory", True)), "persistent_workers": workers > 0}
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, drop_last=False, **loader_kwargs)
    val_batch, val_sources = _balanced_fixed_batch(val_set, int(data_config["validation_batch_size"]), seed + 1)
    val_target = val_batch["target_pose_6d"].to(device)
    val_valid = val_batch["valid_mask"].to(device)
    data_summary = {"scanned_pickle_files": file_counts, "train": train_set.summary(), "val": val_set.summary(), "fixed_validation_source_counts": val_sources}
    (output_dir / "data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("========== Pose Stabilizer V2 mixed training ==========")
    print(json.dumps(data_summary, ensure_ascii=False, indent=2))

    model_config = PoseStabilizerConfig(**config["model"])
    model = PoseTemporalStabilizer(model_config).to(device)
    loss_config = dict(config["loss"])
    loss_config.setdefault("max_blend", model_config.max_blend)
    criterion = PoseStabilizerLoss(**loss_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optim"]["lr"]), weight_decay=float(config["optim"].get("weight_decay", 0.0)))
    amp_enabled = bool(config["optim"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 100)
    mixture = config["mixture"]
    probabilities = [float(mixture[name]) for name in MODE_NAMES]
    epochs = int(config["optim"]["epochs"])
    log_interval = int(config["optim"].get("log_interval", 50))

    wandb_run = None
    wandb_config = config.get("logging", {}).get("wandb", {})
    if bool(wandb_config.get("enabled", True)):
        try:
            import wandb

            wandb_run = wandb.init(project=str(wandb_config.get("project", "vggt-human")), entity=str(wandb_config.get("entity", "")) or None, name=str(wandb_config.get("name", config["experiment"]["name"])), group=str(wandb_config.get("group", "smpl_temporal_stabilizer_v2")), mode=str(wandb_config.get("mode", "online")), config=config, tags=list(wandb_config.get("tags", ["smpl", "pose-stabilizer", "mixture"])))
        except ImportError as error:
            raise RuntimeError("W&B is enabled but wandb is unavailable") from error

    global_step = 0
    best_score = float("inf")
    for epoch in range(epochs):
        model.train()
        running: dict[str, float] = {}
        mode_total = torch.zeros(len(MODE_NAMES), device=device)
        for step, batch in enumerate(train_loader, start=1):
            target = batch["target_pose_6d"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            mode_ids = _mode_ids(target.shape[0], probabilities, device, generator)
            observed, centre_mask = _make_observations(target, valid, mode_ids, config["noise"], generator, return_centre_mask=True)
            frame_weight = torch.ones(target.shape[:2], dtype=target.dtype, device=device)
            # Only the deliberately corrupted centre is amplified. Other
            # centres remain in the loss with weight one, preserving clean
            # identity supervision inside the same centre-jitter window.
            frame_weight[centre_mask] = float(config["mixture"]["centre_jitter_focus_weight"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                losses = criterion(model(observed, valid), target, observed, valid, frame_weight=frame_weight)
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optim"].get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            mode_total += torch.bincount(mode_ids, minlength=len(MODE_NAMES)).to(dtype=mode_total.dtype)
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())
            if global_step % log_interval == 0:
                metrics = {key: value / step for key, value in running.items()}
                print(f"[epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)}] total={metrics['loss_total']:.5f} final={metrics['metric_final_geodesic_rad']:.5f} base={metrics['metric_base_geodesic_rad']:.5f} improve={metrics['metric_improvement_rad']:.5f}")
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=global_step)
        train_metrics = {key: value / max(len(train_loader), 1) for key, value in running.items()}
        val_cases = _evaluate_fixed_cases(model, criterion, val_target, val_valid, config["noise"], seed)
        # Ensure clean preservation has priority in selection, while requiring
        # all three noisy modes to be useful.
        score = 4.0 * val_cases["clean"]["metric_final_geodesic_rad"] + sum(val_cases[name]["metric_final_geodesic_rad"] for name in MODE_NAMES[1:])
        print("[val] " + " ".join(f"{name}: final={item['metric_final_geodesic_rad']:.5f} improve={item['metric_improvement_rad']:.5f} blend={item['metric_blend_mean']:.3f}" for name, item in val_cases.items()) + f" score={score:.5f}")
        checkpoint = {"format": "smpl_temporal_stabilizer_v2_pose_mixture", "epoch": epoch, "model_config": asdict(model_config), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "mixture": dict(mixture), "train_metrics": train_metrics, "val_cases": val_cases, "selection_score": score}
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        if wandb_run is not None:
            log = {"epoch": epoch + 1, "selection_score": score, **{f"train_epoch/{key}": value for key, value in train_metrics.items()}}
            for mode_name, values in val_cases.items():
                log.update({f"val/{mode_name}/{key}": value for key, value in values.items()})
            log.update({f"mixture/{name}": float(mode_total[index].item() / mode_total.sum().clamp_min(1).item()) for index, name in enumerate(MODE_NAMES)})
            wandb_run.log(log, step=global_step)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
