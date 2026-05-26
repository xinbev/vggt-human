import torch
import torch.nn as nn


class HungarianSMPLMatcher(nn.Module):
    """Hungarian matcher for multi-query SMPL/person predictions.

    Boxes are expected as normalized cxcywh tensors. 2D keypoints are expected
    in pixel units unless j2ds_norm_scale is changed by the caller.
    """

    def __init__(
        self,
        cost_conf: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
        cost_kpts: float = 10.0,
        j2ds_norm_scale: float = 518.0,
        require_boxes: bool = True,
        require_j2ds: bool = True,
    ) -> None:
        super().__init__()
        self.cost_conf = cost_conf
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_kpts = cost_kpts
        self.j2ds_norm_scale = j2ds_norm_scale
        self.require_boxes = require_boxes
        self.require_j2ds = require_j2ds
        if cost_conf == 0 and cost_bbox == 0 and cost_giou == 0 and cost_kpts == 0:
            raise ValueError("At least one matching cost must be enabled")

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError(
                "HungarianSMPLMatcher requires scipy for training-time matching. "
                "Install the optional demo/training dependencies or add scipy to your environment."
            ) from exc

        pred_confs = _flatten_prediction(_require_output(outputs, "pred_confs"), unframed_ndim=3)
        batch_size, num_queries, _ = pred_confs.shape
        if len(targets) != batch_size:
            raise ValueError(f"Expected {batch_size} target items after flattening, got {len(targets)}")

        pred_boxes = _flatten_prediction(_require_output(outputs, "pred_boxes"), unframed_ndim=3) if self._uses_boxes else None
        pred_j2ds = _flatten_prediction(_require_output(outputs, "pred_j2ds"), unframed_ndim=4) if self.cost_kpts != 0 else None

        indices = []
        for batch_idx in range(batch_size):
            num_targets = _num_targets(targets[batch_idx])
            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_confs.device)
                indices.append((empty, empty))
                continue

            cost = pred_confs.new_zeros((num_queries, num_targets))
            if self.cost_conf != 0:
                cost = cost + self.cost_conf * _confidence_cost(pred_confs[batch_idx])[:, None].expand(-1, num_targets)
            if self.cost_bbox != 0:
                target_boxes = _require_target(targets[batch_idx], "boxes", self.require_boxes).to(pred_confs.device)
                cost = cost + self.cost_bbox * torch.cdist(pred_boxes[batch_idx], target_boxes, p=1)
            if self.cost_giou != 0:
                target_boxes = _require_target(targets[batch_idx], "boxes", self.require_boxes).to(pred_confs.device)
                cost = cost - self.cost_giou * generalized_box_iou(
                    cxcywh_to_xyxy(pred_boxes[batch_idx]),
                    cxcywh_to_xyxy(target_boxes),
                )
            if self.cost_kpts != 0:
                target_j2ds = _require_target(targets[batch_idx], "j2ds", self.require_j2ds).to(pred_confs.device)
                target_mask = _require_target(targets[batch_idx], "j2ds_mask", self.require_j2ds).to(pred_confs.device)
                cost = cost + self.cost_kpts * _keypoint_cost(pred_j2ds[batch_idx], target_j2ds, target_mask, self.j2ds_norm_scale)

            src_idx, tgt_idx = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=pred_confs.device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_confs.device),
                )
            )

        return indices

    @property
    def _uses_boxes(self) -> bool:
        return self.cost_bbox != 0 or self.cost_giou != 0


def _flatten_prediction(tensor: torch.Tensor, unframed_ndim: int) -> torch.Tensor:
    if tensor.ndim == unframed_ndim:
        return tensor
    if tensor.ndim == unframed_ndim + 1:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    raise ValueError(f"Expected prediction with {unframed_ndim} or {unframed_ndim + 1} dims, got {tensor.shape}")


def _require_output(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in outputs:
        raise ValueError(f"outputs missing required key: {key}")
    return outputs[key]


def _require_target(target: dict[str, torch.Tensor], key: str, required: bool) -> torch.Tensor:
    if key not in target:
        if required:
            raise ValueError(f"target missing required key: {key}")
        raise ValueError(f"Cost using optional target key {key!r} is enabled, but the key is missing")
    return target[key]


def _num_targets(target: dict[str, torch.Tensor]) -> int:
    for key in ("boxes", "j2ds"):
        value = target.get(key)
        if value is not None:
            return value.shape[0]
    return 0


def _confidence_cost(pred_confs: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    pred_confs = pred_confs.squeeze(-1).clamp(min=1e-6, max=1.0 - 1e-6)
    return alpha * (1.0 - pred_confs).pow(gamma) * (-pred_confs.log())


def _keypoint_cost(pred_j2ds: torch.Tensor, target_j2ds: torch.Tensor, target_mask: torch.Tensor, norm_scale: float) -> torch.Tensor:
    pred_j2ds = pred_j2ds[:, :22]
    target_j2ds = target_j2ds[:, :22]
    target_mask = target_mask[:, :22]
    diff = (pred_j2ds[:, None] - target_j2ds[None]).abs() * target_mask[None]
    visible = target_mask.sum(dim=(1, 2)).clamp(min=1.0)
    return diff.sum(dim=(2, 3)) / visible[None] / norm_scale


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None] - inter
    iou = inter / union.clamp(min=1e-6)

    enclosing_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)
