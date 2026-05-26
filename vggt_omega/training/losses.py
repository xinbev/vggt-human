import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPLSlotLoss(nn.Module):
    """Loss for padded SMPL slots aligned by dataset slot order.

    This is the minimal supervised loss compatible with the current SMPL head,
    which predicts pose, betas, confidence, and camera translation but not boxes
    or projected 2D joints yet.
    """

    def __init__(
        self,
        pose_weight: float = 1.0,
        betas_weight: float = 0.1,
        cam_weight: float = 0.1,
        conf_weight: float = 1.0,
        depth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.pose_weight = pose_weight
        self.betas_weight = betas_weight
        self.cam_weight = cam_weight
        self.conf_weight = conf_weight
        self.depth_weight = depth_weight
        self.register_buffer(
            "betas_dim_weight",
            torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).view(1, 1, 1, 10),
            persistent=False,
        )

    def forward(self, predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        pred_pose = _require_prediction(predictions, "pred_pose_6d")
        pred_betas = _require_prediction(predictions, "pred_betas")
        pred_cam = _require_prediction(predictions, "pred_cam")
        pred_confs = _require_prediction(predictions, "pred_confs")

        gt_pose = batch["gt_pose_6d"].to(device=pred_pose.device, dtype=pred_pose.dtype)
        gt_betas = batch["gt_betas"].to(device=pred_betas.device, dtype=pred_betas.dtype)
        gt_cam = batch["gt_cam_trans"].to(device=pred_cam.device, dtype=pred_cam.dtype)
        smpl_mask = batch["smpl_mask"].to(device=pred_confs.device).bool()

        if pred_pose.shape[:3] != gt_pose.shape[:3]:
            raise ValueError(
                "SMPL prediction slots must match target slots for SMPLSlotLoss: "
                f"pred={tuple(pred_pose.shape[:3])}, target={tuple(gt_pose.shape[:3])}. "
                "Set training.model.num_smpl_queries equal to data.max_humans, or add matcher-based losses."
            )

        valid = smpl_mask[..., None]
        losses["loss_pose"] = _masked_l1(pred_pose, gt_pose, valid)
        losses["loss_betas"] = _masked_l1(pred_betas, gt_betas, valid, self.betas_dim_weight.to(pred_betas))
        losses["loss_cam"] = _masked_l1(pred_cam, gt_cam, valid)
        losses["loss_conf"] = F.binary_cross_entropy(
            pred_confs.clamp(min=1e-6, max=1.0 - 1e-6),
            smpl_mask[..., None].to(dtype=pred_confs.dtype),
        )

        total = (
            self.pose_weight * losses["loss_pose"]
            + self.betas_weight * losses["loss_betas"]
            + self.cam_weight * losses["loss_cam"]
            + self.conf_weight * losses["loss_conf"]
        )

        if self.depth_weight != 0.0 and "depth" in predictions and "gt_depth" in batch:
            pred_depth = predictions["depth"]
            gt_depth = batch["gt_depth"].to(device=pred_depth.device, dtype=pred_depth.dtype).permute(0, 1, 3, 4, 2)
            valid_depth = gt_depth > 0
            losses["loss_depth"] = torch.abs(pred_depth - gt_depth)[valid_depth.expand_as(pred_depth)].mean() if valid_depth.any() else pred_depth.sum() * 0.0
            total = total + self.depth_weight * losses["loss_depth"]

        losses["loss_total"] = total
        return losses


def _require_prediction(predictions: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in predictions:
        raise ValueError(f"Model predictions missing required key for SMPL training: {key}")
    return predictions[key]


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    dim_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = torch.abs(prediction - target)
    if dim_weight is not None:
        diff = diff * dim_weight
    diff = diff * valid.to(dtype=diff.dtype)
    denom = valid.to(dtype=diff.dtype).sum().clamp(min=1.0)
    return diff.flatten(-1).mean(-1).sum() / denom
