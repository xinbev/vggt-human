"""Build slot-based SMPL supervision from loaded BEDLAM batches."""

from __future__ import annotations

from typing import Tuple

import torch


def perspective_projection(points3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project 3D camera-space points to 2D pixel coordinates."""
    normalized = points3d / points3d[..., 2:3].clamp(min=1e-6)
    return torch.einsum("bij,bnj->bni", K, normalized)[..., :2]


def resize_K(K: torch.Tensor, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> torch.Tensor:
    """Scale camera intrinsics from ``src_hw`` to ``dst_hw``."""
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K_r = K.clone().float()
    if K_r.ndim == 3:
        K_r[:, 0, 0] *= sx
        K_r[:, 0, 2] *= sx
        K_r[:, 1, 1] *= sy
        K_r[:, 1, 2] *= sy
    else:
        K_r[0, 0] *= sx
        K_r[0, 2] *= sx
        K_r[1, 1] *= sy
        K_r[1, 2] *= sy
    return K_r


def _pixel_to_uv_norm(px: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    H, W = hw
    uv = px.clone().float()
    uv[..., 0] = uv[..., 0] / max(W - 1, 1)
    uv[..., 1] = uv[..., 1] / max(H - 1, 1)
    return uv.clamp(0.0, 1.0)


def _build_score_map_and_filter(
    pk_idx: torch.Tensor,
    smpl_mask: torch.Tensor,
    frame_idx: torch.Tensor,
    human_idx: torch.Tensor,
    patch_hw: Tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a binary score heatmap and invalidate out-of-bound/colliding slots."""
    B = smpl_mask.shape[0]
    n_h, n_w = patch_hw
    device = smpl_mask.device
    nhv = pk_idx.shape[0]
    scores = torch.zeros(B, n_h, n_w, device=device)
    visible_flag = torch.ones(nhv, dtype=torch.bool, device=device)

    for idx in range(nhv):
        frame = int(frame_idx[idx])
        human = int(human_idx[idx])
        px = int(pk_idx[idx, 0])
        py = int(pk_idx[idx, 1])
        if 0 <= py < n_h and 0 <= px < n_w:
            if scores[frame, py, px] == 1.0:
                smpl_mask[frame, human] = False
                visible_flag[idx] = False
            else:
                scores[frame, py, px] = 1.0
        else:
            smpl_mask[frame, human] = False
            visible_flag[idx] = False
    return scores, visible_flag


class SMPLGTBuilder:
    """Generate slot UVs and patch-level score maps from SMPL joints."""

    def __init__(
        self,
        mhmr_img_res: int = 896,
        mhmr_patch_size: int = 14,
        scal3r_patch_size: int = 14,
        center_joint_idx: int = 0,
    ):
        self.mhmr_img_res = mhmr_img_res
        self.mhmr_patch_size = mhmr_patch_size
        self.scal3r_patch_size = scal3r_patch_size
        self.center_joint_idx = center_joint_idx

    def build(
        self,
        joints3d_cam: torch.Tensor,
        K_scal3r: torch.Tensor,
        scal3r_hw: Tuple[int, int],
        smpl_mask: torch.Tensor,
        K_mhmr: torch.Tensor | None = None,
        mhmr_hw: Tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build supervision dict consumed by Scal3RHuman training/loss code."""
        B, M, _, _ = joints3d_cam.shape
        device = joints3d_cam.device
        mhmr_hw = mhmr_hw or (self.mhmr_img_res, self.mhmr_img_res)
        if K_mhmr is None:
            K_mhmr = resize_K(K_scal3r, scal3r_hw, mhmr_hw)

        center_j3d = joints3d_cam[:, :, self.center_joint_idx, :]
        smpl_mask = smpl_mask.clone().bool()
        frame_idx, human_idx = torch.where(smpl_mask)

        if len(frame_idx) == 0:
            n_hs = scal3r_hw[0] // self.scal3r_patch_size
            n_ws = scal3r_hw[1] // self.scal3r_patch_size
            n_hm = mhmr_hw[0] // self.mhmr_patch_size
            n_wm = mhmr_hw[1] // self.mhmr_patch_size
            return {
                "smpl_mask": smpl_mask,
                "smpl_uv_scal3r": torch.zeros(B, M, 2, device=device),
                "smpl_uv_mhmr": torch.zeros(B, M, 2, device=device),
                "smpl_scores_scal3r": torch.zeros(B, n_hs, n_ws, device=device),
                "smpl_scores_mhmr": torch.zeros(B, n_hm, n_wm, device=device),
                "center_j3d": center_j3d,
                "smpl_j3d": joints3d_cam,
            }

        center_valid = center_j3d[frame_idx, human_idx].unsqueeze(1)
        K_scal3r_valid = K_scal3r[frame_idx]
        px_scal3r = perspective_projection(center_valid, K_scal3r_valid).squeeze(1)

        ps_s = self.scal3r_patch_size
        n_hs = scal3r_hw[0] // ps_s
        n_ws = scal3r_hw[1] // ps_s
        pk_idx_scal3r = (px_scal3r / ps_s).int()
        scores_scal3r, _ = _build_score_map_and_filter(
            pk_idx_scal3r, smpl_mask, frame_idx, human_idx, (n_hs, n_ws)
        )

        ps_m = self.mhmr_patch_size
        n_hm = mhmr_hw[0] // ps_m
        n_wm = mhmr_hw[1] // ps_m
        frame_idx2, human_idx2 = torch.where(smpl_mask)

        if len(frame_idx2) == 0:
            scores_mhmr = torch.zeros(B, n_hm, n_wm, device=device)
            uv_scal3r = torch.zeros(B, M, 2, device=device)
            uv_mhmr = torch.zeros(B, M, 2, device=device)
        else:
            center_valid2 = center_j3d[frame_idx2, human_idx2].unsqueeze(1)
            px_mhmr2 = perspective_projection(center_valid2, K_mhmr[frame_idx2]).squeeze(1)
            pk_idx_mhmr = (px_mhmr2 / ps_m).int()
            scores_mhmr, _ = _build_score_map_and_filter(
                pk_idx_mhmr, smpl_mask, frame_idx2, human_idx2, (n_hm, n_wm)
            )

            frame_idx3, human_idx3 = torch.where(smpl_mask)
            uv_scal3r = torch.zeros(B, M, 2, device=device)
            uv_mhmr = torch.zeros(B, M, 2, device=device)
            if len(frame_idx3) > 0:
                center_valid3 = center_j3d[frame_idx3, human_idx3].unsqueeze(1)
                px_s3 = perspective_projection(center_valid3, K_scal3r[frame_idx3]).squeeze(1)
                px_m3 = perspective_projection(center_valid3, K_mhmr[frame_idx3]).squeeze(1)
                uv_scal3r[frame_idx3, human_idx3] = _pixel_to_uv_norm(px_s3, scal3r_hw)
                uv_mhmr[frame_idx3, human_idx3] = _pixel_to_uv_norm(px_m3, mhmr_hw)

        return {
            "smpl_mask": smpl_mask,
            "smpl_uv_scal3r": uv_scal3r,
            "smpl_uv_mhmr": uv_mhmr,
            "smpl_scores_scal3r": scores_scal3r,
            "smpl_scores_mhmr": scores_mhmr,
            "center_j3d": center_j3d,
            "smpl_j3d": joints3d_cam,
        }
