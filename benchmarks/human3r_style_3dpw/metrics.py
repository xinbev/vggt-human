"""Human3R-style 3DPW matching and camera-coordinate metric functions."""

from __future__ import annotations

import torch


# OpenPose-18 index -> SMPL-24 index.  Facial joints are deliberately omitted:
# the two body models do not share a reliable facial-joint convention.
OPENPOSE_TO_SMPL = (
    (1, 12),  # neck
    (2, 17), (3, 19), (4, 21),  # right arm
    (5, 16), (6, 18), (7, 20),  # left arm
    (8, 2), (9, 5), (10, 8),  # right leg
    (11, 1), (12, 4), (13, 7),  # left leg
)


def project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2:3].clamp_min(1e-6)
    xy = points[..., :2] / z
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    return torch.stack((xy[..., 0] * fx + cx, xy[..., 1] * fy + cy), dim=-1)


def match_by_2d_joints(
    pred_joints_cam: torch.Tensor,
    gt_openpose: torch.Tensor,
    intrinsics: torch.Tensor,
    min_keypoints: int = 5,
    min_confidence: float = 0.2,
    min_bbox_iou: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-to-one greedy matching using normalized GT 2D joint distance.

    Returns matched prediction indices, matched GT person indices, and the
    number of unmatched predictions.  It mirrors Human3R's projection-first,
    2D-joint association principle while using an explicit common-joint map.
    """
    if pred_joints_cam.ndim != 3 or gt_openpose.ndim != 3:
        raise ValueError("pred_joints_cam=[P,24,3], gt_openpose=[G,18,3] required")
    if pred_joints_cam.numel() == 0 or gt_openpose.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_joints_cam.device)
        return empty, empty, torch.tensor(int(pred_joints_cam.shape[0]), device=pred_joints_cam.device)
    pred_2d = project(pred_joints_cam, intrinsics)
    costs = torch.full((gt_openpose.shape[0], pred_joints_cam.shape[0]), float("inf"), device=pred_joints_cam.device)
    for gt_index in range(gt_openpose.shape[0]):
        gt_xyc = gt_openpose[gt_index]
        valid_pairs = [(op, smpl) for op, smpl in OPENPOSE_TO_SMPL if float(gt_xyc[op, 2]) > min_confidence]
        if len(valid_pairs) < min_keypoints:
            continue
        op_idx = torch.tensor([pair[0] for pair in valid_pairs], device=pred_joints_cam.device)
        smpl_idx = torch.tensor([pair[1] for pair in valid_pairs], device=pred_joints_cam.device)
        gt_xy = gt_xyc[op_idx, :2]
        extent = (gt_xy.amax(dim=0) - gt_xy.amin(dim=0)).norm().clamp_min(20.0)
        pred_common = pred_2d[:, smpl_idx]
        distance = (pred_common - gt_xy.unsqueeze(0)).norm(dim=-1).mean(dim=-1) / extent
        iou = _bbox_iou(pred_common, gt_xy)
        costs[gt_index] = distance.masked_fill(iou < float(min_bbox_iou), float("inf"))
    pairs: list[tuple[float, int, int]] = []
    for gt_index in range(costs.shape[0]):
        for pred_index in range(costs.shape[1]):
            value = float(costs[gt_index, pred_index])
            if value != float("inf"):
                pairs.append((value, pred_index, gt_index))
    pairs.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched_pred: list[int] = []
    matched_gt: list[int] = []
    for _, pred_index, gt_index in pairs:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matched_pred.append(pred_index)
        matched_gt.append(gt_index)
    return (
        torch.tensor(matched_pred, dtype=torch.long, device=pred_joints_cam.device),
        torch.tensor(matched_gt, dtype=torch.long, device=pred_joints_cam.device),
        torch.tensor(int(pred_joints_cam.shape[0] - len(used_pred)), device=pred_joints_cam.device),
    )


def pelvis_align(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_pelvis = pred_joints[:, [1, 2]].mean(dim=1, keepdim=True)
    gt_pelvis = gt_joints[:, [1, 2]].mean(dim=1, keepdim=True)
    return (
        pred_joints - pred_pelvis,
        gt_joints - gt_pelvis,
        pred_vertices - pred_pelvis,
        gt_vertices - gt_pelvis,
        pred_pelvis,
        gt_pelvis,
    )


def similarity_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Similarity Procrustes alignment from Human3R/GVHMR-style protocol."""
    pred_center, target_center = pred.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)
    pred0, target0 = pred - pred_center, target - target_center
    pred_norm = torch.linalg.norm(pred0.reshape(pred.shape[0], -1), dim=1).clamp_min(1e-8)
    target_norm = torch.linalg.norm(target0.reshape(target.shape[0], -1), dim=1).clamp_min(1e-8)
    x, y = pred0 / pred_norm[:, None, None], target0 / target_norm[:, None, None]
    u, singular, vh = torch.linalg.svd(x.transpose(1, 2) @ y)
    # Points are stored as row vectors. With C=X^T Y=U S V^T, the
    # least-squares row-vector map is R=U V^T. The previous V U^T used the
    # inverse rotation and made PA-MPJPE spuriously worse than MPJPE.
    rotation = u @ vh
    det = torch.det(rotation)
    if bool((det < 0).any()):
        u, singular = u.clone(), singular.clone()
        u[det < 0, :, -1] *= -1.0
        singular[det < 0, -1] *= -1.0
        rotation = u @ vh
    scale = singular.sum(dim=1) * target_norm / pred_norm
    return scale[:, None, None] * (pred0 @ rotation) + target_center


def _bbox_iou(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """IoU of predicted/GT common-joint 2D bounding boxes, Human3R style."""
    pred_min, pred_max = pred_points.amin(dim=1), pred_points.amax(dim=1)
    gt_min, gt_max = gt_points.amin(dim=0), gt_points.amax(dim=0)
    inter_min = torch.maximum(pred_min, gt_min)
    inter_max = torch.minimum(pred_max, gt_max)
    inter_wh = (inter_max - inter_min + 1.0).clamp_min(0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_wh = (pred_max - pred_min + 1.0).clamp_min(0.0)
    gt_wh = (gt_max - gt_min + 1.0).clamp_min(0.0)
    return inter / (pred_wh[:, 0] * pred_wh[:, 1] + gt_wh[0] * gt_wh[1] - inter).clamp_min(1e-8)


def human3r_camera_metrics(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return Human3R names: CA=pelvis aligned, ME=metric/un-aligned."""
    ca_pred_joints, ca_gt_joints, ca_pred_vertices, ca_gt_vertices, pred_pelvis, gt_pelvis = pelvis_align(
        pred_joints, gt_joints, pred_vertices, gt_vertices
    )
    pa_pred_joints = similarity_align(ca_pred_joints, ca_gt_joints)
    pa_pred_vertices = similarity_align(ca_pred_vertices, ca_gt_vertices)
    jpe = lambda a, b: (a - b).norm(dim=-1).mean(dim=-1) * 1000.0
    return {
        "mpjpe_mm": jpe(ca_pred_joints, ca_gt_joints),
        "pve_mm": jpe(ca_pred_vertices, ca_gt_vertices),
        "pa_mpjpe_mm": jpe(pa_pred_joints, ca_gt_joints),
        "pa_pve_mm": jpe(pa_pred_vertices, ca_gt_vertices),
        "metric_mpjpe_mm": jpe(pred_joints, gt_joints),
        "metric_pve_mm": jpe(pred_vertices, gt_vertices),
        "root_error_mm": jpe(pred_pelvis, gt_pelvis),
    }
