#!/usr/bin/env python3
"""Train the standalone offline SMPL temporal refiner on EMDB + 3DPW pkls."""

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
from vggt_omega.models import TemporalRefinerConfig, TemporalSMPLRefiner, TemporalSMPLRefinerLoss
from vggt_omega.training import TemporalSMPLNoiseConfig, corrupt_smpl_sequence
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
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _get_dotted(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _resolve_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sources = config["data"].get("sources", [])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("data.sources must be a non-empty YAML list")
    sources = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise TypeError("Each data.sources item must be a mapping")
        source = dict(raw)
        if not str(source.get("root", "")).strip():
            root_key = str(source.get("root_key", "")).strip()
            if not root_key:
                raise ValueError(f"Source {source.get('name')!r} needs root or root_key")
            source["root"] = _get_dotted(config, root_key)
        sources.append(source)
    return sources


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mean_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


def _build_train_sampler(
    dataset: SMPLTemporalPickleDataset,
    mode: str,
    samples_per_epoch: int,
    seed: int,
) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    """Optionally prevent one large source from dominating overlapping windows."""
    per_dataset_windows: dict[str, int] = {}
    names: list[str] = []
    for record_index, _ in dataset.index:
        name = dataset.records[record_index].dataset_name
        names.append(name)
        per_dataset_windows[name] = per_dataset_windows.get(name, 0) + 1
    mode = mode.lower()
    if mode == "natural":
        return None, {"mode": mode, "per_dataset_windows": per_dataset_windows, "samples_per_epoch": len(dataset)}
    if mode != "balanced_dataset":
        raise ValueError("data.sampling_mode must be 'natural' or 'balanced_dataset'")
    if len(per_dataset_windows) < 2:
        return None, {"mode": "natural_single_source", "per_dataset_windows": per_dataset_windows, "samples_per_epoch": len(dataset)}
    weights = torch.as_tensor([1.0 / per_dataset_windows[name] for name in names], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    epoch_samples = int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(dataset)
    sampler = WeightedRandomSampler(weights, num_samples=epoch_samples, replacement=True, generator=generator)
    return sampler, {
        "mode": mode,
        "per_dataset_windows": per_dataset_windows,
        "samples_per_epoch": epoch_samples,
        "expected_dataset_fraction": {name: 1.0 / len(per_dataset_windows) for name in sorted(per_dataset_windows)},
    }


def _make_noise_config(raw: dict[str, Any]) -> TemporalSMPLNoiseConfig:
    if not isinstance(raw, dict):
        raise TypeError("noise configuration must be a mapping")
    fields = TemporalSMPLNoiseConfig.__dataclass_fields__
    return TemporalSMPLNoiseConfig(**{key: raw[key] for key in fields if key in raw})


def _select_training_noise(config: dict[str, Any], epoch: int) -> tuple[str, TemporalSMPLNoiseConfig]:
    """Use only the requested small -> medium two-stage curriculum."""
    base = dict(config.get("noise", {}))
    curriculum = config.get("noise_curriculum", {})
    if not bool(curriculum.get("enabled", False)):
        return "static", _make_noise_config(base)
    small_epochs = int(curriculum.get("small_epochs", 10))
    if small_epochs <= 0:
        raise ValueError("noise_curriculum.small_epochs must be positive")
    stage_name = "small" if epoch < small_epochs else "medium"
    stage = curriculum.get(stage_name)
    if not isinstance(stage, dict):
        raise ValueError(f"noise_curriculum.{stage_name} must be a mapping")
    base.update(stage)
    return stage_name, _make_noise_config(base)


def _select_validation_noise(config: dict[str, Any]) -> TemporalSMPLNoiseConfig:
    """Keep validation corruption fixed so epoch metrics remain comparable."""
    base = dict(config.get("noise", {}))
    curriculum = config.get("noise_curriculum", {})
    stage_name = str(curriculum.get("validation_stage", "medium"))
    if bool(curriculum.get("enabled", False)):
        stage = curriculum.get(stage_name)
        if not isinstance(stage, dict):
            raise ValueError(f"noise_curriculum.validation_stage={stage_name!r} is not configured")
        base.update(stage)
    return _make_noise_config(base)


@torch.no_grad()
def _evaluate(
    model: TemporalSMPLRefiner,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: TemporalSMPLRefinerLoss,
    noise_config: TemporalSMPLNoiseConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        noisy = corrupt_smpl_sequence(batch["target_pose_6d"], batch["target_transl"], batch["valid_mask"], noise_config)
        outputs = model(
            noisy["base_pose_6d"],
            noisy["base_transl"],
            betas=batch["target_betas"],
            valid_mask=batch["valid_mask"],
            confidence=noisy["confidence"],
        )
        losses = criterion(
            outputs,
            batch["target_pose_6d"],
            batch["target_transl"],
            batch["valid_mask"],
            noisy["base_pose_6d"],
            noisy["base_transl"],
            noisy["clean_clip_mask"],
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    return _mean_metrics(totals, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Temporal-refiner training YAML")
    parser.add_argument("--path-config", default="configs/path.yaml", help="Project path YAML used by data source root_key")
    parser.add_argument("--override", action="append", default=[], help="dotted.path=<JSON value>; can be passed repeatedly")
    parser.add_argument("--resume", default="", help="Optional standalone temporal-refiner checkpoint")
    args = parser.parse_args()

    path_config = load_yaml_config(args.path_config) if args.path_config else {}
    config = deep_update(path_config, load_yaml_config(args.config))
    for override in args.override:
        if "=" not in override:
            raise ValueError(f"Invalid --override {override!r}; expected dotted.path=value")
        key, value = override.split("=", 1)
        _set_dotted(config, key, _parse_value(value))

    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    optim_config = config["optim"]
    loss_config = config["loss"]
    logging_config = config.get("logging", {}).get("wandb", {})
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    seed = int(experiment.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = _resolve_sources(config)
    common_data = {
        "sources": sources,
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config.get("stride", 1)),
        "validation_fraction": float(data_config.get("validation_fraction", 0.1)),
        "min_valid_frames": int(data_config.get("min_valid_frames", data_config["window_size"])),
    }
    # Native pkl parsing is intentionally performed once.  Train/validation
    # datasets then share immutable person-track records but have disjoint
    # indices, preventing duplicate scans and window leakage.
    all_records, source_file_counts = SMPLTemporalPickleDataset.load_records(sources)
    train_set = SMPLTemporalPickleDataset(
        partition="train", records=all_records, source_file_counts=source_file_counts, **{key: value for key, value in common_data.items() if key != "sources"}
    )
    val_set = SMPLTemporalPickleDataset(
        partition="val", records=all_records, source_file_counts=source_file_counts, **{key: value for key, value in common_data.items() if key != "sources"}
    )
    raw_by_dataset: dict[str, dict[str, int]] = {}
    for record in all_records:
        item = raw_by_dataset.setdefault(record.dataset_name, {"person_tracks": 0, "frames_total": 0, "frames_valid": 0, "frames_invalid": 0})
        item["person_tracks"] += 1
        item["frames_total"] += int(record.valid.size)
        item["frames_valid"] += int(record.valid.sum())
        item["frames_invalid"] += int((~record.valid).sum())
    for name, item in raw_by_dataset.items():
        item["pickle_files"] = int(source_file_counts.get(name, 0))
    data_summary = {
        "scanned_pickle_files": source_file_counts,
        "raw_person_tracks": len(all_records),
        "raw_by_dataset": raw_by_dataset,
    }
    train_sampler, sampling_summary = _build_train_sampler(
        train_set,
        mode=str(data_config.get("sampling_mode", "balanced_dataset")),
        samples_per_epoch=int(data_config.get("samples_per_epoch", 0)),
        seed=seed,
    )
    data_summary["sampling"] = sampling_summary
    data_summary["train"] = train_set.summary()
    data_summary["val"] = val_set.summary()
    (output_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("========== Temporal-refiner data summary ==========")
    print(json.dumps(data_summary, indent=2, ensure_ascii=False))
    loader_kwargs = {
        "batch_size": int(optim_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 4)),
        "pin_memory": bool(data_config.get("pin_memory", True)),
        "persistent_workers": int(data_config.get("num_workers", 4)) > 0,
    }
    train_loader = DataLoader(train_set, shuffle=train_sampler is None, sampler=train_sampler, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    architecture = TemporalRefinerConfig(**{key: model_config[key] for key in TemporalRefinerConfig.__dataclass_fields__ if key in model_config})
    if architecture.window_size != common_data["window_size"]:
        raise ValueError("model.window_size and data.window_size must match")
    model = TemporalSMPLRefiner(architecture).to(device)
    criterion = TemporalSMPLRefinerLoss(**{key: loss_config[key] for key in TemporalSMPLRefinerLoss.__init__.__annotations__ if key in loss_config})
    validation_noise_config = _select_validation_noise(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_config["lr"]), weight_decay=float(optim_config.get("weight_decay", 0.01)))
    amp_enabled = bool(optim_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))

    wandb_run = None
    if bool(logging_config.get("enabled", True)):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(logging_config.get("project", "vggt-human")),
                entity=str(logging_config.get("entity", "")) or None,
                name=str(logging_config.get("name", experiment.get("name", "smpl_temporal_refiner"))),
                group=str(logging_config.get("group", "smpl_temporal_refiner")),
                mode=str(logging_config.get("mode", "online")),
                config=config,
                tags=list(logging_config.get("tags", ["smpl", "temporal-refiner", "offline"])),
            )
            data_scalars: dict[str, float] = {
                "data/raw_person_tracks": float(len(all_records)),
                "data/train_person_tracks": float(len(train_set.records)),
                "data/val_person_tracks": float(len(val_set.records)),
                "data/train_windows": float(len(train_set)),
                "data/val_windows": float(len(val_set)),
            }
            for dataset_name, values in raw_by_dataset.items():
                for key, value in values.items():
                    data_scalars[f"data/{dataset_name}/{key}"] = float(value)
            wandb_run.log(data_scalars, step=0)
        except ImportError as error:
            raise RuntimeError("logging.wandb.enabled=true but wandb is not installed") from error

    epochs = int(optim_config["epochs"])
    grad_clip = float(optim_config.get("grad_clip_norm", 1.0))
    log_interval = int(optim_config.get("log_interval", 20))
    global_step = 0
    print(f"[temporal-refiner] device={device} train_windows={len(train_set)} val_windows={len(val_set)}")
    for epoch in range(start_epoch, epochs):
        noise_stage, train_noise_config = _select_training_noise(config, epoch)
        print(f"[epoch {epoch + 1}/{epochs}] noise_stage={noise_stage} {asdict(train_noise_config)}")
        model.train()
        running: dict[str, float] = {}
        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            noisy = corrupt_smpl_sequence(batch["target_pose_6d"], batch["target_transl"], batch["valid_mask"], train_noise_config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    noisy["base_pose_6d"], noisy["base_transl"], batch["target_betas"], batch["valid_mask"], noisy["confidence"]
                )
                losses = criterion(
                    outputs, batch["target_pose_6d"], batch["target_transl"], batch["valid_mask"], noisy["base_pose_6d"], noisy["base_transl"], noisy["clean_clip_mask"]
                )
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())
            if global_step % log_interval == 0:
                current = _mean_metrics(running, step + 1)
                print(f"[epoch {epoch + 1}/{epochs} step {step + 1}] loss={current['loss_total']:.5f} trans={current['loss_transl']:.5f} pose={current['loss_pose']:.5f}")
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in current.items()}, step=global_step)

        train_metrics = _mean_metrics(running, max(len(train_loader), 1))
        val_metrics = _evaluate(model, val_loader, criterion, validation_noise_config, device)
        summary = {
            "epoch": epoch + 1,
            "noise/stage": noise_stage,
            "noise/translation_drift_std_m": train_noise_config.translation_drift_std_m,
            "noise/translation_jitter_std_m": train_noise_config.translation_jitter_std_m,
            "noise/translation_outlier_std_m": train_noise_config.translation_outlier_std_m,
            "noise/pose_drift_std_rad": train_noise_config.pose_drift_std_rad,
            "noise/pose_jitter_std_rad": train_noise_config.pose_jitter_std_rad,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        print(f"[epoch {epoch + 1}] val_loss={val_metrics['loss_total']:.5f} val_trans={val_metrics['metric_transl_l1_m']:.5f}m")
        if wandb_run is not None:
            wandb_run.log(summary, step=global_step)
        checkpoint = {
            "format": "smpl_temporal_refiner_v1",
            "epoch": epoch,
            "best_val_loss": best_val,
            "noise_stage": noise_stage,
            "train_noise_config": asdict(train_noise_config),
            "validation_noise_config": asdict(validation_noise_config),
            "model_config": asdict(architecture),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            checkpoint["best_val_loss"] = best_val
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
