"""Hungarian matcher reference for multi-query SMPL training.

SAT-HMR source reference:
- models/matcher.py

This file is optional. Use it when your model predicts many person queries and
you need to match predictions to ground-truth people before computing losses.
"""

from typing import Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt_enclose = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclose = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    area_enclose = wh_enclose[:, :, 0] * wh_enclose[:, :, 1]

    return iou - (area_enclose - union) / (area_enclose + 1e-6)


class HungarianMatcher(nn.Module):
    """Assign predicted person queries to ground-truth people."""

    def __init__(
        self,
        cost_conf: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
        cost_kpts: float = 10.0,
        j2ds_norm_scale: float = 518.0,
    ) -> None:
        super().__init__()
        self.cost_conf = cost_conf
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_kpts = cost_kpts
        self.j2ds_norm_scale = j2ds_norm_scale
        if cost_conf == 0 and cost_bbox == 0 and cost_giou == 0 and cost_kpts == 0:
            raise ValueError("At least one matching cost must be non-zero.")

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_queries, _ = outputs["pred_confs"].shape

        out_conf = outputs["pred_confs"].flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        out_kpts = outputs["pred_j2ds"][..., :22, :].flatten(2).flatten(0, 1) / self.j2ds_norm_scale

        tgt_bbox = torch.cat([target["boxes"] for target in targets])
        tgt_kpts = torch.cat([target["j2ds"][:, :22, :].flatten(1) for target in targets]) / self.j2ds_norm_scale
        tgt_kpts_mask = torch.cat([target["j2ds_mask"][:, :22, :].flatten(1) for target in targets])
        tgt_kpts_vis_count = tgt_kpts_mask.sum(-1).clamp(min=1.0)

        alpha = 0.25
        gamma = 2.0
        cost_conf = alpha * ((1.0 - out_conf) ** gamma) * (-(out_conf + 1e-8).log())
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        all_dist = torch.abs(out_kpts[:, None, :] - tgt_kpts[None, :, :])
        cost_kpts = (all_dist * tgt_kpts_mask[None, :, :]).sum(-1) / tgt_kpts_vis_count[None, :]

        cost_matrix = (
            self.cost_conf * cost_conf
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
            + self.cost_kpts * cost_kpts
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(target["boxes"]) for target in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, dim=-1))]
        return [
            (torch.as_tensor(src, dtype=torch.int64), torch.as_tensor(tgt, dtype=torch.int64))
            for src, tgt in indices
        ]
