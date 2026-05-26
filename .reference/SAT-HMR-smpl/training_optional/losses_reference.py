"""Loss references for SMPL regression training.

SAT-HMR source reference:
- models/criterion.py

This is a compact reference rather than a full training framework. It assumes
matching indices have already been computed by `HungarianMatcher`.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for confidence scores already passed through sigmoid."""
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * loss).mean()


class SMPLRegressionLosses(nn.Module):
    """Reference loss module for matched SMPL query predictions."""

    def __init__(self, j2ds_norm_scale: float = 518.0) -> None:
        super().__init__()
        self.j2ds_norm_scale = j2ds_norm_scale
        self.register_buffer(
            "betas_weight",
            torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).view(1, -1),
        )

    @staticmethod
    def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_confidence(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        pred_confs = outputs["pred_confs"]
        labels = torch.zeros_like(pred_confs)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        labels[batch_idx, src_idx] = 1.0
        return focal_loss(pred_confs, labels)

    def matched_l1_loss(
        self,
        loss_name: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute matched L1 loss for poses, betas, j3ds, j2ds, or depths."""
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        src = outputs[f"pred_{loss_name}"][batch_idx, src_idx]
        target = torch.cat([target[loss_name][target_idx] for target, (_, target_idx) in zip(targets, indices)], dim=0)

        if src.numel() == 0:
            return src.sum() * 0.0
        if src.shape != target.shape:
            raise ValueError(f"Shape mismatch for {loss_name}: {src.shape} vs {target.shape}")

        loss_mask = None
        if loss_name == "j3ds":
            src = src - src[..., [0], :].clone()
            target = target - target[..., [0], :].clone()
            src = src[:, :45, :]
            target = target[:, :45, :]
        elif loss_name == "j2ds":
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            loss_mask = torch.cat(
                [target_dict["j2ds_mask"][target_idx] for target_dict, (_, target_idx) in zip(targets, indices)],
                dim=0,
            )
            src = src[:, :45, :]
            target = target[:, :45, :]
            loss_mask = loss_mask[:, :45, :]

        valid_loss = torch.abs(src - target)
        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss_name == "betas":
            valid_loss = valid_loss * self.betas_weight.to(valid_loss.device)

        num_instances = max(sum(len(target_idx) for _, target_idx in indices), 1)
        return valid_loss.flatten(1).mean(-1).sum() / num_instances

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        enabled_losses: Tuple[str, ...] = ("poses", "betas", "j3ds", "j2ds", "depths"),
    ) -> Dict[str, torch.Tensor]:
        losses = {"confs": self.loss_confidence(outputs, targets, indices)}
        for loss_name in enabled_losses:
            if f"pred_{loss_name}" in outputs and all(loss_name in target for target in targets):
                losses[loss_name] = self.matched_l1_loss(loss_name, outputs, targets, indices)
        return losses
