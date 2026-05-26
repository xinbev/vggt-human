from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher, cxcywh_to_xyxy, generalized_box_iou


class HungarianSMPLLoss(nn.Module):
    def __init__(
        self,
        matcher: HungarianSMPLMatcher,
        pose_weight: float = 1.0,
        betas_weight: float = 0.1,
        cam_weight: float = 0.1,
        conf_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        id_weight: float = 0.0,
        id_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.pose_weight = pose_weight
        self.betas_weight = betas_weight
        self.cam_weight = cam_weight
        self.conf_weight = conf_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.id_weight = id_weight
        self.id_temperature = id_temperature
        self.register_buffer(
            "betas_dim_weight",
            torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).view(1, 10),
            persistent=False,
        )

    def forward(self, predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_confs = _flatten_prediction(_require_prediction(predictions, "pred_confs"), unframed_ndim=3)
        pred_boxes = _flatten_prediction(_require_prediction(predictions, "pred_boxes"), unframed_ndim=3)
        pred_pose = _flatten_prediction(_require_prediction(predictions, "pred_pose_6d"), unframed_ndim=3)
        pred_betas = _flatten_prediction(_require_prediction(predictions, "pred_betas"), unframed_ndim=3)
        pred_cam = _flatten_prediction(_require_prediction(predictions, "pred_cam"), unframed_ndim=3)
        pred_id_embed = predictions.get("pred_id_embed")
        flat_id_embed = _flatten_prediction(pred_id_embed, unframed_ndim=3) if pred_id_embed is not None else None

        targets = flatten_smpl_targets(batch, device=pred_confs.device)
        indices = self.matcher({"pred_confs": pred_confs, "pred_boxes": pred_boxes}, targets)
        losses: dict[str, torch.Tensor] = {}

        conf_target = torch.zeros_like(pred_confs)
        for frame_idx, (src_idx, _) in enumerate(indices):
            if src_idx.numel() > 0:
                conf_target[frame_idx, src_idx, 0] = 1.0
        losses["loss_conf"] = F.binary_cross_entropy(pred_confs.clamp(1e-6, 1.0 - 1e-6), conf_target)

        matched = _collect_matches(indices, targets, pred_confs.device)
        if matched["frame_idx"].numel() == 0:
            zero = pred_confs.sum() * 0.0
            losses.update({"loss_bbox": zero, "loss_giou": zero, "loss_pose": zero, "loss_betas": zero, "loss_cam": zero, "loss_id": zero})
        else:
            frame_idx = matched["frame_idx"]
            src_idx = matched["src_idx"]
            target_boxes = matched["boxes"].to(dtype=pred_boxes.dtype)
            losses["loss_bbox"] = F.l1_loss(pred_boxes[frame_idx, src_idx], target_boxes)
            giou = generalized_box_iou(cxcywh_to_xyxy(pred_boxes[frame_idx, src_idx]), cxcywh_to_xyxy(target_boxes))
            losses["loss_giou"] = (1.0 - giou.diag()).mean()
            losses["loss_pose"] = F.l1_loss(pred_pose[frame_idx, src_idx], matched["pose_6d"].to(dtype=pred_pose.dtype))
            beta_diff = (pred_betas[frame_idx, src_idx] - matched["betas"].to(dtype=pred_betas.dtype)).abs()
            losses["loss_betas"] = (beta_diff * self.betas_dim_weight.to(dtype=pred_betas.dtype, device=pred_betas.device)).mean()
            losses["loss_cam"] = F.l1_loss(pred_cam[frame_idx, src_idx], matched["cam_trans"].to(dtype=pred_cam.dtype))
            losses["loss_id"] = self._identity_loss(flat_id_embed, frame_idx, src_idx, matched)

        losses["loss_total"] = (
            self.conf_weight * losses["loss_conf"]
            + self.bbox_weight * losses["loss_bbox"]
            + self.giou_weight * losses["loss_giou"]
            + self.pose_weight * losses["loss_pose"]
            + self.betas_weight * losses["loss_betas"]
            + self.cam_weight * losses["loss_cam"]
            + self.id_weight * losses["loss_id"]
        )
        return losses

    def _identity_loss(
        self,
        pred_id_embed: torch.Tensor | None,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.id_weight == 0.0 or pred_id_embed is None:
            return frame_idx.new_zeros((), dtype=torch.float32).to(device=frame_idx.device)
        valid = matched["person_id_mask"].bool()
        if valid.sum() < 2:
            return pred_id_embed.sum() * 0.0
        embeds = pred_id_embed[frame_idx[valid], src_idx[valid]]
        ids = matched["person_ids"][valid]
        positive = ids[:, None] == ids[None, :]
        positive.fill_diagonal_(False)
        if not positive.any():
            return embeds.sum() * 0.0
        logits = embeds @ embeds.t() / max(self.id_temperature, 1e-6)
        logits = logits.masked_fill(torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device), float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        denom = positive.sum(dim=1).clamp(min=1)
        per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / denom)
        active = positive.any(dim=1)
        return per_anchor[active].mean() if active.any() else embeds.sum() * 0.0


def flatten_smpl_targets(batch: dict[str, torch.Tensor], device: torch.device) -> list[dict[str, torch.Tensor]]:
    smpl_mask = batch["smpl_mask"].to(device=device).bool()
    boxes_mask = batch["boxes_mask"].to(device=device).bool()
    gt_boxes = batch["gt_boxes"].to(device=device)
    gt_pose = batch["gt_pose_6d"].to(device=device)
    gt_betas = batch["gt_betas"].to(device=device)
    gt_cam = batch["gt_cam_trans"].to(device=device)
    person_ids = batch.get("person_ids")
    person_id_mask = batch.get("person_id_mask")
    if person_ids is not None:
        person_ids = person_ids.to(device=device)
    if person_id_mask is not None:
        person_id_mask = person_id_mask.to(device=device).bool()

    targets = []
    batch_size, num_frames, _ = smpl_mask.shape
    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            valid = smpl_mask[batch_idx, frame_idx] & boxes_mask[batch_idx, frame_idx]
            target = {
                "boxes": gt_boxes[batch_idx, frame_idx, valid],
                "pose_6d": gt_pose[batch_idx, frame_idx, valid],
                "betas": gt_betas[batch_idx, frame_idx, valid],
                "cam_trans": gt_cam[batch_idx, frame_idx, valid],
            }
            if person_ids is not None and person_id_mask is not None:
                target["person_ids"] = person_ids[batch_idx, frame_idx, valid]
                target["person_id_mask"] = person_id_mask[batch_idx, frame_idx, valid]
            else:
                target["person_ids"] = torch.full((int(valid.sum()),), -1, dtype=torch.long, device=device)
                target["person_id_mask"] = torch.zeros(int(valid.sum()), dtype=torch.bool, device=device)
            targets.append(target)
    return targets


def _collect_matches(indices, targets: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    frame_indices = []
    src_indices = []
    target_parts: dict[str, list[torch.Tensor]] = {"boxes": [], "pose_6d": [], "betas": [], "cam_trans": [], "person_ids": [], "person_id_mask": []}
    for frame_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        frame_indices.append(torch.full_like(src_idx, frame_idx))
        src_indices.append(src_idx)
        target = targets[frame_idx]
        for key in target_parts:
            target_parts[key].append(target[key][tgt_idx])
    if not frame_indices:
        return {"frame_idx": torch.empty(0, dtype=torch.long, device=device), "src_idx": torch.empty(0, dtype=torch.long, device=device)}
    out = {"frame_idx": torch.cat(frame_indices), "src_idx": torch.cat(src_indices)}
    out.update({key: torch.cat(values) for key, values in target_parts.items()})
    return out


def _flatten_prediction(tensor: torch.Tensor | None, unframed_ndim: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == unframed_ndim:
        return tensor
    if tensor.ndim == unframed_ndim + 1:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    raise ValueError(f"Expected prediction with {unframed_ndim} or {unframed_ndim + 1} dims, got {tensor.shape}")


def _require_prediction(predictions: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in predictions:
        raise ValueError(f"Model predictions missing required key for Hungarian SMPL training: {key}")
    return predictions[key]
