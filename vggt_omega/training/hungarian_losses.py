from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher, cxcywh_to_xyxy, generalized_box_iou
from vggt_omega.utils.pose_enc import encoding_to_camera


class HungarianSMPLLoss(nn.Module):
    def __init__(
        self,
        matcher: HungarianSMPLMatcher,
        pose_weight: float = 1.0,
        betas_weight: float = 0.1,
        transl_cam_weight: float = 0.1,
        cam_weight: float | None = None,
        conf_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        id_weight: float = 0.0,
        id_temperature: float = 0.07,
        conf_loss_type: str = "bce",
        conf_focal_alpha: float = 0.25,
        conf_focal_gamma: float = 2.0,
        conf_target_type: str = "binary",
        conf_iou_min: float = 0.0,
        conf_iou_power: float = 1.0,
        duplicate_conf_weight: float = 0.0,
        duplicate_iou_threshold: float = 0.5,
        aux_weight: float = 0.0,
        aux_conf_weight: float | None = None,
        aux_bbox_weight: float | None = None,
        aux_giou_weight: float | None = None,
        projected_bbox_weight: float = 0.0,
        projected_giou_weight: float = 0.0,
        projected_bbox_source: str = "joints",
        use_vggt_camera_projection: bool = False,
        smpl_model_dir: str = "",
        projection_image_size: int = 518,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.pose_weight = pose_weight
        self.betas_weight = betas_weight
        self.transl_cam_weight = transl_cam_weight if cam_weight is None else cam_weight
        self.conf_weight = conf_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.id_weight = id_weight
        self.id_temperature = id_temperature
        self.conf_loss_type = conf_loss_type
        self.conf_focal_alpha = conf_focal_alpha
        self.conf_focal_gamma = conf_focal_gamma
        self.conf_target_type = conf_target_type
        self.conf_iou_min = conf_iou_min
        self.conf_iou_power = conf_iou_power
        self.duplicate_conf_weight = duplicate_conf_weight
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.aux_weight = aux_weight
        self.aux_conf_weight = conf_weight if aux_conf_weight is None else aux_conf_weight
        self.aux_bbox_weight = bbox_weight if aux_bbox_weight is None else aux_bbox_weight
        self.aux_giou_weight = giou_weight if aux_giou_weight is None else aux_giou_weight
        self.projected_bbox_weight = projected_bbox_weight
        self.projected_giou_weight = projected_giou_weight
        self.projected_bbox_source = projected_bbox_source
        self.use_vggt_camera_projection = use_vggt_camera_projection
        self.smpl_model_dir = smpl_model_dir
        self.projection_image_size = projection_image_size
        self._smpl_layer: SMPLLayer | None = None
        if self.conf_loss_type not in {"bce", "focal"}:
            raise ValueError(f"Unsupported conf_loss_type: {self.conf_loss_type}")
        if self.conf_target_type not in {"binary", "matched_iou"}:
            raise ValueError(f"Unsupported conf_target_type: {self.conf_target_type}")
        if self.projected_bbox_source not in {"joints", "vertices"}:
            raise ValueError(f"Unsupported projected_bbox_source: {self.projected_bbox_source}")
        if self._uses_projected_bbox and not self.smpl_model_dir:
            raise ValueError("Projected SMPL bbox loss requires loss.smpl_model_dir or assets.smpl_model_dir")
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
        pred_transl_cam = _flatten_prediction(_require_prediction(predictions, "pred_transl_cam"), unframed_ndim=3)
        pred_id_embed = predictions.get("pred_id_embed")
        flat_id_embed = _flatten_prediction(pred_id_embed, unframed_ndim=3) if pred_id_embed is not None else None

        targets = flatten_smpl_targets(batch, device=pred_confs.device)
        indices = self.matcher({"pred_confs": pred_confs, "pred_boxes": pred_boxes}, targets)
        losses: dict[str, torch.Tensor] = {}

        matched = _collect_matches(indices, targets, pred_confs.device)
        conf_target = torch.zeros_like(pred_confs)
        matched_iou = None
        if matched["frame_idx"].numel() > 0:
            matched_iou = _matched_box_iou(pred_boxes, matched)
        self._fill_confidence_target(conf_target, matched, matched_iou)
        losses["loss_conf"] = self._confidence_loss(pred_confs, conf_target)
        losses.update(_confidence_metrics(pred_confs, conf_target, indices, targets, matched_mask=_matched_mask(pred_confs, indices)))
        losses.update(self._duplicate_confidence_loss(pred_confs, pred_boxes, indices, targets))
        losses.update(self._auxiliary_detection_losses(predictions, targets, indices, matched, matched_iou))

        if matched["frame_idx"].numel() == 0:
            zero = pred_confs.sum() * 0.0
            losses.update(
                {
                    "loss_bbox": zero,
                    "loss_giou": zero,
                    "loss_pose": zero,
                    "loss_betas": zero,
                    "loss_transl_cam": zero,
                    "loss_id": zero,
                    "loss_projected_bbox": zero,
                    "loss_projected_giou": zero,
                    "metric_bbox_iou_mean": zero.detach(),
                    "metric_projected_bbox_iou_mean": zero.detach(),
                    "metric_conf_target_pos_mean": zero.detach(),
                    "metric_conf_target_pos_min": zero.detach(),
                    "metric_conf_target_pos_max": zero.detach(),
                }
            )
        else:
            frame_idx = matched["frame_idx"]
            src_idx = matched["src_idx"]
            target_boxes = matched["boxes"].to(dtype=pred_boxes.dtype)
            losses["loss_bbox"] = F.l1_loss(pred_boxes[frame_idx, src_idx], target_boxes)
            giou = generalized_box_iou(cxcywh_to_xyxy(pred_boxes[frame_idx, src_idx]), cxcywh_to_xyxy(target_boxes))
            losses["loss_giou"] = (1.0 - giou.diag()).mean()
            matched_iou = _require_matched_iou(matched_iou, pred_boxes, matched)
            losses["metric_bbox_iou_mean"] = matched_iou.detach().mean()
            target_values = conf_target[frame_idx, src_idx, 0].detach()
            losses["metric_conf_target_pos_mean"] = target_values.mean()
            losses["metric_conf_target_pos_min"] = target_values.min()
            losses["metric_conf_target_pos_max"] = target_values.max()
            losses["loss_pose"] = F.l1_loss(pred_pose[frame_idx, src_idx], matched["pose_6d"].to(dtype=pred_pose.dtype))
            beta_diff = (pred_betas[frame_idx, src_idx] - matched["betas"].to(dtype=pred_betas.dtype)).abs()
            losses["loss_betas"] = (beta_diff * self.betas_dim_weight.to(dtype=pred_betas.dtype, device=pred_betas.device)).mean()
            losses["loss_transl_cam"] = F.l1_loss(
                pred_transl_cam[frame_idx, src_idx],
                matched["transl_cam"].to(dtype=pred_transl_cam.dtype),
            )
            losses["loss_id"] = self._identity_loss(flat_id_embed, frame_idx, src_idx, matched)
            projected = self._projected_bbox_losses(predictions, pred_betas, pred_transl_cam, frame_idx, src_idx, target_boxes)
            losses.update(projected)

        losses["loss_total"] = (
            self.conf_weight * losses["loss_conf"]
            + self.bbox_weight * losses["loss_bbox"]
            + self.giou_weight * losses["loss_giou"]
            + self.pose_weight * losses["loss_pose"]
            + self.betas_weight * losses["loss_betas"]
            + self.transl_cam_weight * losses["loss_transl_cam"]
            + self.id_weight * losses["loss_id"]
            + self.projected_bbox_weight * losses["loss_projected_bbox"]
            + self.projected_giou_weight * losses["loss_projected_giou"]
            + self.duplicate_conf_weight * losses["loss_duplicate_conf"]
            + self.aux_weight * losses["loss_aux_total"]
        )
        return losses

    @property
    def _uses_projected_bbox(self) -> bool:
        return self.use_vggt_camera_projection and (self.projected_bbox_weight != 0.0 or self.projected_giou_weight != 0.0)

    def _confidence_loss(self, pred_confs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_confs = pred_confs.clamp(1e-6, 1.0 - 1e-6)
        if self.conf_loss_type == "bce":
            return F.binary_cross_entropy(pred_confs, target)
        ce = F.binary_cross_entropy(pred_confs, target, reduction="none")
        p_t = pred_confs * target + (1.0 - pred_confs) * (1.0 - target)
        alpha_t = self.conf_focal_alpha * target + (1.0 - self.conf_focal_alpha) * (1.0 - target)
        return (alpha_t * (1.0 - p_t).pow(self.conf_focal_gamma) * ce).mean()

    def _fill_confidence_target(
        self,
        conf_target: torch.Tensor,
        matched: dict[str, torch.Tensor],
        matched_iou: torch.Tensor | None,
    ) -> None:
        if matched["frame_idx"].numel() == 0:
            return
        frame_idx = matched["frame_idx"]
        src_idx = matched["src_idx"]
        if self.conf_target_type == "binary":
            conf_target[frame_idx, src_idx, 0] = 1.0
            return
        if matched_iou is None:
            raise RuntimeError("matched_iou confidence target requires matched IoU values")
        target = matched_iou.detach().clamp(min=self.conf_iou_min, max=1.0).pow(self.conf_iou_power)
        conf_target[frame_idx, src_idx, 0] = target.to(dtype=conf_target.dtype)

    def _duplicate_confidence_loss(
        self,
        pred_confs: torch.Tensor,
        pred_boxes: torch.Tensor,
        indices,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        duplicate_confs = []
        num_duplicates = pred_confs.new_zeros(())
        for frame_idx, (src_idx, _) in enumerate(indices):
            target_boxes = targets[frame_idx].get("boxes")
            if target_boxes is None or target_boxes.numel() == 0:
                continue
            unmatched = torch.ones(pred_confs.shape[1], dtype=torch.bool, device=pred_confs.device)
            unmatched[src_idx] = False
            if not unmatched.any():
                continue
            iou = _box_iou_pairwise(
                cxcywh_to_xyxy(pred_boxes[frame_idx, unmatched].clamp(0.0, 1.0)),
                cxcywh_to_xyxy(target_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype).clamp(0.0, 1.0)),
            )
            duplicate_mask = iou.max(dim=1).values > self.duplicate_iou_threshold
            if duplicate_mask.any():
                confs = pred_confs[frame_idx, unmatched, 0][duplicate_mask]
                duplicate_confs.append(confs)
                num_duplicates = num_duplicates + confs.new_tensor(float(confs.numel()))

        if not duplicate_confs:
            zero = pred_confs.sum() * 0.0
            return {
                "loss_duplicate_conf": zero,
                "metric_duplicate_unmatched_count": zero.detach(),
                "metric_duplicate_unmatched_conf_mean": zero.detach(),
            }
        duplicate_confs_tensor = torch.cat(duplicate_confs).clamp(1e-6, 1.0 - 1e-6)
        return {
            "loss_duplicate_conf": F.binary_cross_entropy(duplicate_confs_tensor, torch.zeros_like(duplicate_confs_tensor)),
            "metric_duplicate_unmatched_count": num_duplicates.detach(),
            "metric_duplicate_unmatched_conf_mean": duplicate_confs_tensor.detach().mean(),
        }

    def _auxiliary_detection_losses(
        self,
        predictions: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices,
        matched: dict[str, torch.Tensor],
        final_matched_iou: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        aux_outputs = predictions.get("aux_outputs")
        if self.aux_weight == 0.0 or aux_outputs is None:
            zero = _require_prediction(predictions, "pred_confs").sum() * 0.0
            return {
                "loss_aux_conf": zero,
                "loss_aux_bbox": zero,
                "loss_aux_giou": zero,
                "loss_aux_total": zero,
            }
        if not isinstance(aux_outputs, dict) or "pred_confs" not in aux_outputs or "pred_boxes" not in aux_outputs:
            raise ValueError("Auxiliary detection loss requires aux_outputs with pred_confs and pred_boxes")

        aux_confs = _flatten_aux_prediction(aux_outputs["pred_confs"], unframed_ndim=4)
        aux_boxes = _flatten_aux_prediction(aux_outputs["pred_boxes"], unframed_ndim=4)
        aux_confs = aux_confs[:-1]
        aux_boxes = aux_boxes[:-1]
        num_aux_layers = aux_confs.shape[0]
        if num_aux_layers == 0:
            zero = _require_prediction(predictions, "pred_confs").sum() * 0.0
            return {
                "loss_aux_conf": zero,
                "loss_aux_bbox": zero,
                "loss_aux_giou": zero,
                "loss_aux_total": zero,
            }
        loss_conf_parts = []
        loss_bbox_parts = []
        loss_giou_parts = []
        for layer_idx in range(num_aux_layers):
            conf_target = torch.zeros_like(aux_confs[layer_idx])
            aux_matched_iou = None
            if matched["frame_idx"].numel() > 0:
                aux_matched_iou = _matched_box_iou(aux_boxes[layer_idx], matched)
            self._fill_confidence_target(conf_target, matched, aux_matched_iou if self.conf_target_type == "matched_iou" else final_matched_iou)
            loss_conf_parts.append(self._confidence_loss(aux_confs[layer_idx], conf_target))
            if matched["frame_idx"].numel() == 0:
                zero = aux_confs[layer_idx].sum() * 0.0
                loss_bbox_parts.append(zero)
                loss_giou_parts.append(zero)
                continue
            frame_idx = matched["frame_idx"]
            src_idx = matched["src_idx"]
            target_boxes = matched["boxes"].to(dtype=aux_boxes.dtype, device=aux_boxes.device)
            pred_boxes = aux_boxes[layer_idx, frame_idx, src_idx]
            loss_bbox_parts.append(F.l1_loss(pred_boxes, target_boxes))
            giou = generalized_box_iou(cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(target_boxes))
            loss_giou_parts.append((1.0 - giou.diag()).mean())

        loss_aux_conf = torch.stack(loss_conf_parts).mean()
        loss_aux_bbox = torch.stack(loss_bbox_parts).mean()
        loss_aux_giou = torch.stack(loss_giou_parts).mean()
        loss_aux_total = (
            self.aux_conf_weight * loss_aux_conf
            + self.aux_bbox_weight * loss_aux_bbox
            + self.aux_giou_weight * loss_aux_giou
        )
        return {
            "loss_aux_conf": loss_aux_conf,
            "loss_aux_bbox": loss_aux_bbox,
            "loss_aux_giou": loss_aux_giou,
            "loss_aux_total": loss_aux_total,
        }

    def _projected_bbox_losses(
        self,
        predictions: dict[str, torch.Tensor],
        pred_betas: torch.Tensor,
        pred_transl_cam: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self._uses_projected_bbox:
            zero = pred_betas.sum() * 0.0
            return {
                "loss_projected_bbox": zero,
                "loss_projected_giou": zero,
                "metric_projected_bbox_iou_mean": zero.detach(),
            }
        if "pose_enc" not in predictions or "pred_poses" not in predictions:
            raise ValueError("Projected SMPL bbox loss requires model outputs pose_enc and pred_poses; set model.enable_camera=true")

        pred_poses = _flatten_prediction(_require_prediction(predictions, "pred_poses"), unframed_ndim=3)
        pose_enc = _require_prediction(predictions, "pose_enc")
        intrinsics = _flatten_intrinsics(pose_enc, self.projection_image_size)
        smpl = self._get_smpl_layer(pred_betas.device)

        poses = pred_poses[frame_idx, src_idx].reshape(-1, 72)
        betas = pred_betas[frame_idx, src_idx]
        transl_cam = pred_transl_cam[frame_idx, src_idx]
        vertices, joints = smpl(poses.float(), betas.float())
        points = joints[:, :24] if self.projected_bbox_source == "joints" else vertices
        points_cam = points.to(dtype=pred_betas.dtype) + transl_cam[:, None, :]
        projected = _project_points(points_cam, intrinsics[frame_idx].to(dtype=points_cam.dtype))
        projected_boxes = _points_to_normalized_cxcywh(projected, self.projection_image_size)
        projected_boxes = projected_boxes.to(dtype=target_boxes.dtype)

        giou = generalized_box_iou(cxcywh_to_xyxy(projected_boxes), cxcywh_to_xyxy(target_boxes))
        return {
            "loss_projected_bbox": F.l1_loss(projected_boxes, target_boxes),
            "loss_projected_giou": (1.0 - giou.diag()).mean(),
            "metric_projected_bbox_iou_mean": _box_iou_diag(cxcywh_to_xyxy(projected_boxes), cxcywh_to_xyxy(target_boxes)).detach().mean(),
        }

    def _get_smpl_layer(self, device: torch.device) -> SMPLLayer:
        if self._smpl_layer is None:
            self._smpl_layer = SMPLLayer(self.smpl_model_dir).to(device=device).eval()
            for param in self._smpl_layer.parameters():
                param.requires_grad = False
        return self._smpl_layer

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
    gt_transl_cam = batch.get("gt_transl_cam", batch["gt_cam_trans"]).to(device=device)
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
                "transl_cam": gt_transl_cam[batch_idx, frame_idx, valid],
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
    target_parts: dict[str, list[torch.Tensor]] = {"boxes": [], "pose_6d": [], "betas": [], "transl_cam": [], "person_ids": [], "person_id_mask": []}
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


def _matched_mask(pred_confs: torch.Tensor, indices) -> torch.Tensor:
    mask = torch.zeros_like(pred_confs, dtype=torch.bool)
    for frame_idx, (src_idx, _) in enumerate(indices):
        if src_idx.numel() > 0:
            mask[frame_idx, src_idx, 0] = True
    return mask


def _matched_box_iou(pred_boxes: torch.Tensor, matched: dict[str, torch.Tensor]) -> torch.Tensor:
    frame_idx = matched["frame_idx"]
    src_idx = matched["src_idx"]
    target_boxes = matched["boxes"].to(device=pred_boxes.device, dtype=pred_boxes.dtype)
    pred_xyxy = cxcywh_to_xyxy(pred_boxes[frame_idx, src_idx].clamp(0.0, 1.0))
    target_xyxy = cxcywh_to_xyxy(target_boxes.clamp(0.0, 1.0))
    return _box_iou_diag(pred_xyxy, target_xyxy)


def _require_matched_iou(
    matched_iou: torch.Tensor | None,
    pred_boxes: torch.Tensor,
    matched: dict[str, torch.Tensor],
) -> torch.Tensor:
    if matched_iou is not None:
        return matched_iou
    return _matched_box_iou(pred_boxes, matched)


def _confidence_metrics(
    pred_confs: torch.Tensor,
    conf_target: torch.Tensor,
    indices,
    targets: list[dict[str, torch.Tensor]],
    matched_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        pos_mask = matched_mask if matched_mask is not None else conf_target.bool()
        neg_mask = ~pos_mask
        pos_mean = pred_confs[pos_mask].mean() if pos_mask.any() else pred_confs.new_zeros(())
        neg_mean = pred_confs[neg_mask].mean() if neg_mask.any() else pred_confs.new_zeros(())
        num_targets = sum(_target_count(target) for target in targets)
        num_matched = sum(int(src_idx.numel()) for src_idx, _ in indices)
        return {
            "metric_num_targets": pred_confs.new_tensor(float(num_targets)),
            "metric_num_matched": pred_confs.new_tensor(float(num_matched)),
            "metric_conf_pos_mean": pos_mean.detach(),
            "metric_conf_neg_mean": neg_mean.detach(),
            "metric_conf_gap": (pos_mean - neg_mean).detach(),
            "metric_pred_count_025": (pred_confs >= 0.25).to(dtype=pred_confs.dtype).sum(dim=(1, 2)).mean().detach(),
            "metric_pred_count_030": (pred_confs >= 0.30).to(dtype=pred_confs.dtype).sum(dim=(1, 2)).mean().detach(),
            "metric_pred_count_050": (pred_confs >= 0.50).to(dtype=pred_confs.dtype).sum(dim=(1, 2)).mean().detach(),
        }


def _target_count(target: dict[str, torch.Tensor]) -> int:
    boxes = target.get("boxes")
    if boxes is not None:
        return int(boxes.shape[0])
    return 0


def _flatten_intrinsics(pose_enc: torch.Tensor, image_size: int) -> torch.Tensor:
    if pose_enc.ndim == 2:
        pose_enc = pose_enc[:, None]
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=(image_size, image_size), build_intrinsics=True)
    if intrinsics is None:
        raise RuntimeError("encoding_to_camera did not return intrinsics")
    return intrinsics.reshape(-1, 3, 3)


def _project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[:, None, 0, 0] * points_cam[..., 0] / z + intrinsics[:, None, 0, 2]
    y = intrinsics[:, None, 1, 1] * points_cam[..., 1] / z + intrinsics[:, None, 1, 2]
    return torch.stack([x, y], dim=-1)


def _points_to_normalized_cxcywh(points: torch.Tensor, image_size: int) -> torch.Tensor:
    points = torch.nan_to_num(points, nan=0.0, posinf=float(image_size), neginf=0.0)
    points = points.clamp(min=0.0, max=float(image_size))
    xy_min = points.amin(dim=1)
    xy_max = points.amax(dim=1)
    center = 0.5 * (xy_min + xy_max) / float(image_size)
    size = (xy_max - xy_min) / float(image_size)
    return torch.cat([center, size.clamp(min=1e-6)], dim=-1).clamp(min=0.0, max=1.0)


def _box_iou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area_diag(boxes1) + box_area_diag(boxes2) - inter
    return inter / union.clamp(min=1e-6)


def _box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area_diag(boxes1)[:, None] + box_area_diag(boxes2)[None] - inter
    return inter / union.clamp(min=1e-6)


def box_area_diag(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def _flatten_prediction(tensor: torch.Tensor | None, unframed_ndim: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == unframed_ndim:
        return tensor
    if tensor.ndim == unframed_ndim + 1:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    raise ValueError(f"Expected prediction with {unframed_ndim} or {unframed_ndim + 1} dims, got {tensor.shape}")


def _flatten_aux_prediction(tensor: torch.Tensor, unframed_ndim: int) -> torch.Tensor:
    if tensor.ndim == unframed_ndim:
        return tensor
    if tensor.ndim == unframed_ndim + 1:
        return tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2], *tensor.shape[3:])
    raise ValueError(f"Expected auxiliary prediction with {unframed_ndim} or {unframed_ndim + 1} dims, got {tensor.shape}")


def _require_prediction(predictions: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in predictions:
        raise ValueError(f"Model predictions missing required key for Hungarian SMPL training: {key}")
    return predictions[key]
