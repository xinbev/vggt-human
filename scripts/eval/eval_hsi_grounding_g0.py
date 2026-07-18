from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import (
    apply_overrides,
    build_loader,
    build_model,
    forward_model,
    load_initial_checkpoint,
    move_to_device,
    set_seed,
)
from vggt_omega.training.config import deep_update, load_yaml_config


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    set_seed(int(config.get("experiment", {}).get("seed", 42)))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(config, split=config["data"]["val_split"], shuffle=False, role="val")
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    model.eval()

    values: dict[str, list[torch.Tensor]] = {key: [] for key in ("base", "candidate", "clean_delta", "valid")}
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if args.max_batches > 0 and step >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            predictions = forward_model(model, batch, config, epoch=0)
            target = batch["gt_transl_cam"].float()
            base = predictions["hsi_grounding_base_pred_transl_cam"].float()
            candidate = predictions["hsi_grounding_candidate_pred_transl_cam"].float()
            provider_valid = predictions["gt_smpl_provider_mask"].bool()
            geometry_valid = predictions["hsi_grounding_candidate_valid"][..., 0] > 0.5
            noise = predictions["contact_noise_signed_m"][..., 0].abs()
            active = provider_valid & (noise > 0.005)
            valid = active & geometry_valid
            values["valid"].append(torch.stack([active.float().sum(), valid.float().sum()]).cpu())
            if valid.any():
                values["base"].append(torch.linalg.norm(base[valid] - target[valid], dim=-1).cpu())
                values["candidate"].append(torch.linalg.norm(candidate[valid] - target[valid], dim=-1).cpu())
            teacher_contact = batch["contact_teacher_valid"].bool() & batch["contact_label"].bool()
            clean = provider_valid & (noise <= 0.005) & teacher_contact.any(dim=-1)
            if clean.any():
                values["clean_delta"].append(torch.linalg.norm(candidate[clean] - base[clean], dim=-1).cpu())
            if (step + 1) % 20 == 0:
                print(f"[g0] batches={step + 1}/{len(loader)}", flush=True)

    base = _cat(values["base"])
    candidate = _cat(values["candidate"])
    clean_delta = _cat(values["clean_delta"])
    counts = torch.stack(values["valid"]).sum(dim=0) if values["valid"] else torch.zeros(2)
    base_p95 = _quantile(base, 0.95)
    candidate_p95 = _quantile(candidate, 0.95)
    reduction = 1.0 - candidate_p95 / max(base_p95, 1e-8)
    coverage = float(counts[1] / counts[0].clamp(min=1.0))
    clean_p95 = _quantile(clean_delta, 0.95)
    improvement = float((candidate < base).float().mean()) if base.numel() else 0.0
    passed = (
        reduction >= args.min_p95_reduction
        and coverage >= args.min_valid_coverage
        and clean_p95 <= args.max_clean_displacement
    )
    report = {
        "gate": "pass" if passed else "fail",
        "num_active": int(counts[0]),
        "num_geometry_valid": int(counts[1]),
        "geometry_valid_coverage": coverage,
        "base_translation_p95_m": base_p95,
        "analytic_candidate_translation_p95_m": candidate_p95,
        "p95_reduction": reduction,
        "candidate_improvement_rate": improvement,
        "clean_candidate_displacement_p95_m": clean_p95,
        "thresholds": {
            "min_p95_reduction": args.min_p95_reduction,
            "min_valid_coverage": args.min_valid_coverage,
            "max_clean_displacement_m": args.max_clean_displacement,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def _cat(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(items) if items else torch.empty(0)


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q)) if values.numel() else float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G0 analytic grounding audit")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_stage3_grounding_gt.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-p95-reduction", type=float, default=0.70)
    parser.add_argument("--min-valid-coverage", type=float, default=0.80)
    parser.add_argument("--max-clean-displacement", type=float, default=0.001)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
