from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher, cxcywh_to_xyxy, generalized_box_iou
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices


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
        joints3d_weight: float = 0.0,
        local_joints3d_weight: float = 0.0,
        local_vertices_weight: float = 0.0,
        projected_joints2d_weight: float = 0.0,
        transl_refine_delta_reg_weight: float = 0.0,
        transl_refine_ray_depth_weight: float = 0.0,
        transl_refine_tangent_weight: float = 0.0,
        transl_hard_topk_weight: float = 0.0,
        transl_hard_severe_weight: float = 0.0,
        transl_hard_topk_fraction: float = 0.25,
        transl_hard_min_k: int = 1,
        transl_hard_error_threshold_m: float = 0.20,
        transl_temporal_velocity_weight: float = 0.0,
        transl_temporal_acceleration_weight: float = 0.0,
        transl_temporal_no_worse_weight: float = 0.0,
        transl_temporal_no_worse_margin_m: float = 0.002,
        transl_temporal_no_worse_accel_margin_m: float = 0.003,
        projected_bbox_weight: float = 0.0,
        projected_giou_weight: float = 0.0,
        projected_bbox_source: str = "joints",
        use_vggt_camera_projection: bool = False,
        projection_camera_source: str = "vggt",
        smpl_model_dir: str = "",
        projection_image_size: int = 518,
        hsi_pose_weight: float = 0.0,
        hsi_betas_weight: float = 0.0,
        hsi_transl_cam_weight: float = 0.0,
        hsi_ray_delta_weight: float = 0.0,
        hsi_tangent_delta_weight: float = 0.0,
        hsi_align_point_weight: float = 0.0,
        hsi_align_delta_reg_weight: float = 0.0,
        hsi_align_no_worse_weight: float = 0.0,
        hsi_transl_clean_identity_weight: float = 0.0,
        hsi_transl_noise_gate_weight: float = 0.0,
        hsi_joints3d_weight: float = 0.0,
        hsi_vertices_weight: float = 0.0,
        hsi_projected_joints2d_weight: float = 0.0,
        hsi_depth_teacher_weight: float = 0.0,
        hsi_depth_teacher_max_m: float = 0.0,
        hsi_depth_teacher_error_clip_m: float = 0.0,
        hsi_depth_teacher_use_human_roi: bool = False,
        hsi_depth_teacher_roi_expand: float = 0.35,
        hsi_depth_teacher_min_valid_pixels: int = 256,
        hsi_smpl_scale_teacher_weight: float = 0.0,
        hsi_smpl_scale_teacher_source: str = "vertices",
        hsi_smpl_scale_teacher_use_bias: bool = False,
        hsi_smpl_scale_teacher_visibility_tolerance_m: float = 0.20,
        hsi_smpl_scale_teacher_window: int = 3,
        hsi_smpl_scale_teacher_max_points_per_person: int = 512,
        hsi_smpl_scale_teacher_min_points_per_person: int = 32,
        hsi_smpl_scale_teacher_min_visible_points: int = 128,
        hsi_smpl_scale_teacher_mad_multiplier: float = 2.5,
        hsi_smpl_scale_teacher_log_loss: bool = True,
        hsi_smpl_scale_teacher_bias_reg_weight: float = 0.05,
        hsi_smpl_scale_teacher_max_z_m: float = 0.0,
        hsi_anchor_depth_weight: float = 0.0,
        hsi_anchor_scene_xyz_weight: float = 0.0,
        hsi_anchor_scene_window: int = 5,
        hsi_delta_reg_weight: float = 0.0,
        hsi_no_worse_weight: float = 0.0,
        hsi_no_worse_margin_m: float = 0.02,
        hsi_temporal_no_worse_weight: float = 0.0,
        hsi_temporal_no_worse_margin_m: float = 0.002,
        hsi_temporal_no_worse_accel_margin_m: float = 0.003,
        hsi_gate_reg_weight: float = 0.0,
        hsi_foot_contact_weight: float = 0.0,
        hsi_foot_contact_threshold_m: float = 0.12,
        hsi_foot_float_margin_m: float = 0.05,
        hsi_foot_penetration_margin_m: float = 0.02,
        hsi_foot_sole_contact_weight: float = 0.0,
        hsi_foot_sole_num_vertices: int = 80,
        hsi_foot_sole_contact_threshold_m: float = 0.08,
        hsi_foot_sole_float_margin_m: float = 0.04,
        hsi_foot_sole_penetration_margin_m: float = 0.015,
        hsi_foot_sole_float_weight: float = 0.25,
        hsi_foot_sole_penetration_weight: float = 3.0,
        hsi_support_plane_contact_weight: float = 0.0,
        hsi_support_plane_num_vertices: int = 80,
        hsi_support_plane_window: int = 9,
        hsi_support_plane_min_points: int = 6,
        hsi_support_plane_contact_threshold_m: float = 0.08,
        hsi_support_plane_float_margin_m: float = 0.04,
        hsi_support_plane_penetration_margin_m: float = 0.015,
        hsi_support_plane_float_weight: float = 0.25,
        hsi_support_plane_penetration_weight: float = 4.0,
        hsi_teacher_pose_weight: float = 0.0,
        hsi_teacher_betas_weight: float = 0.0,
        hsi_teacher_transl_weight: float = 0.0,
        hsi_teacher_joints_weight: float = 0.0,
        hsi_teacher_vertices_weight: float = 0.0,
        hsi_teacher_scene_affine_weight: float = 0.0,
        hsi_pose_velocity_weight: float = 0.0,
        hsi_betas_velocity_weight: float = 0.0,
        hsi_transl_velocity_weight: float = 0.0,
        hsi_joints_velocity_weight: float = 0.0,
        hsi_joints_acceleration_weight: float = 0.0,
        hsi_foot_sliding_weight: float = 0.0,
        hsi_foot_sliding_contact_threshold_m: float = 0.08,
        hsi_scene_scale_temporal_weight: float = 0.0,
        hsi_scene_scale_sequence_weight: float = 0.0,
        hsi_scene_bias_temporal_weight: float = 0.0,
        hsi_scene_bias_sequence_weight: float = 0.0,
        hsi_contact_weight: float = 0.0,
        hsi_contact_threshold: float = 0.08,
        hsi_contact_teacher_camera_source: str = "prediction",
        hsi_contact_refine_plane_weight: float = 0.0,
        hsi_contact_refine_pose_weight: float = 0.0,
        hsi_contact_refine_class_weight: float = 0.0,
        hsi_contact_refine_no_worse_weight: float = 0.0,
        hsi_contact_refine_swing_no_pull_weight: float = 0.0,
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
        self.joints3d_weight = joints3d_weight
        self.local_joints3d_weight = local_joints3d_weight
        self.local_vertices_weight = local_vertices_weight
        self.projected_joints2d_weight = projected_joints2d_weight
        self.transl_refine_delta_reg_weight = transl_refine_delta_reg_weight
        self.transl_refine_ray_depth_weight = transl_refine_ray_depth_weight
        self.transl_refine_tangent_weight = transl_refine_tangent_weight
        self.transl_hard_topk_weight = transl_hard_topk_weight
        self.transl_hard_severe_weight = transl_hard_severe_weight
        self.transl_hard_topk_fraction = transl_hard_topk_fraction
        self.transl_hard_min_k = transl_hard_min_k
        self.transl_hard_error_threshold_m = transl_hard_error_threshold_m
        self.transl_temporal_velocity_weight = transl_temporal_velocity_weight
        self.transl_temporal_acceleration_weight = transl_temporal_acceleration_weight
        self.transl_temporal_no_worse_weight = transl_temporal_no_worse_weight
        self.transl_temporal_no_worse_margin_m = transl_temporal_no_worse_margin_m
        self.transl_temporal_no_worse_accel_margin_m = transl_temporal_no_worse_accel_margin_m
        self.projected_bbox_weight = projected_bbox_weight
        self.projected_giou_weight = projected_giou_weight
        self.projected_bbox_source = projected_bbox_source
        self.use_vggt_camera_projection = use_vggt_camera_projection
        self.projection_camera_source = str(projection_camera_source or "vggt")
        self.smpl_model_dir = smpl_model_dir
        self.projection_image_size = projection_image_size
        self.hsi_pose_weight = hsi_pose_weight
        self.hsi_betas_weight = hsi_betas_weight
        self.hsi_transl_cam_weight = hsi_transl_cam_weight
        self.hsi_ray_delta_weight = hsi_ray_delta_weight
        self.hsi_tangent_delta_weight = hsi_tangent_delta_weight
        self.hsi_align_point_weight = hsi_align_point_weight
        self.hsi_align_delta_reg_weight = hsi_align_delta_reg_weight
        self.hsi_align_no_worse_weight = hsi_align_no_worse_weight
        self.hsi_transl_clean_identity_weight = hsi_transl_clean_identity_weight
        self.hsi_transl_noise_gate_weight = hsi_transl_noise_gate_weight
        self.hsi_joints3d_weight = hsi_joints3d_weight
        self.hsi_vertices_weight = hsi_vertices_weight
        self.hsi_projected_joints2d_weight = hsi_projected_joints2d_weight
        self.hsi_depth_teacher_weight = hsi_depth_teacher_weight
        self.hsi_depth_teacher_max_m = hsi_depth_teacher_max_m
        self.hsi_depth_teacher_error_clip_m = hsi_depth_teacher_error_clip_m
        self.hsi_depth_teacher_use_human_roi = hsi_depth_teacher_use_human_roi
        self.hsi_depth_teacher_roi_expand = hsi_depth_teacher_roi_expand
        self.hsi_depth_teacher_min_valid_pixels = hsi_depth_teacher_min_valid_pixels
        self.hsi_smpl_scale_teacher_weight = hsi_smpl_scale_teacher_weight
        self.hsi_smpl_scale_teacher_source = str(hsi_smpl_scale_teacher_source or "vertices")
        self.hsi_smpl_scale_teacher_use_bias = hsi_smpl_scale_teacher_use_bias
        self.hsi_smpl_scale_teacher_visibility_tolerance_m = hsi_smpl_scale_teacher_visibility_tolerance_m
        self.hsi_smpl_scale_teacher_window = hsi_smpl_scale_teacher_window
        self.hsi_smpl_scale_teacher_max_points_per_person = hsi_smpl_scale_teacher_max_points_per_person
        self.hsi_smpl_scale_teacher_min_points_per_person = hsi_smpl_scale_teacher_min_points_per_person
        self.hsi_smpl_scale_teacher_min_visible_points = hsi_smpl_scale_teacher_min_visible_points
        self.hsi_smpl_scale_teacher_mad_multiplier = hsi_smpl_scale_teacher_mad_multiplier
        self.hsi_smpl_scale_teacher_log_loss = hsi_smpl_scale_teacher_log_loss
        self.hsi_smpl_scale_teacher_bias_reg_weight = hsi_smpl_scale_teacher_bias_reg_weight
        self.hsi_smpl_scale_teacher_max_z_m = hsi_smpl_scale_teacher_max_z_m
        self.hsi_anchor_depth_weight = hsi_anchor_depth_weight
        self.hsi_anchor_scene_xyz_weight = hsi_anchor_scene_xyz_weight
        self.hsi_anchor_scene_window = hsi_anchor_scene_window
        self.hsi_delta_reg_weight = hsi_delta_reg_weight
        self.hsi_no_worse_weight = hsi_no_worse_weight
        self.hsi_no_worse_margin_m = hsi_no_worse_margin_m
        self.hsi_temporal_no_worse_weight = hsi_temporal_no_worse_weight
        self.hsi_temporal_no_worse_margin_m = hsi_temporal_no_worse_margin_m
        self.hsi_temporal_no_worse_accel_margin_m = hsi_temporal_no_worse_accel_margin_m
        self.hsi_gate_reg_weight = hsi_gate_reg_weight
        self.hsi_foot_contact_weight = hsi_foot_contact_weight
        self.hsi_foot_contact_threshold_m = hsi_foot_contact_threshold_m
        self.hsi_foot_float_margin_m = hsi_foot_float_margin_m
        self.hsi_foot_penetration_margin_m = hsi_foot_penetration_margin_m
        self.hsi_foot_sole_contact_weight = hsi_foot_sole_contact_weight
        self.hsi_foot_sole_num_vertices = hsi_foot_sole_num_vertices
        self.hsi_foot_sole_contact_threshold_m = hsi_foot_sole_contact_threshold_m
        self.hsi_foot_sole_float_margin_m = hsi_foot_sole_float_margin_m
        self.hsi_foot_sole_penetration_margin_m = hsi_foot_sole_penetration_margin_m
        self.hsi_foot_sole_float_weight = hsi_foot_sole_float_weight
        self.hsi_foot_sole_penetration_weight = hsi_foot_sole_penetration_weight
        self.hsi_support_plane_contact_weight = hsi_support_plane_contact_weight
        self.hsi_support_plane_num_vertices = hsi_support_plane_num_vertices
        self.hsi_support_plane_window = hsi_support_plane_window
        self.hsi_support_plane_min_points = hsi_support_plane_min_points
        self.hsi_support_plane_contact_threshold_m = hsi_support_plane_contact_threshold_m
        self.hsi_support_plane_float_margin_m = hsi_support_plane_float_margin_m
        self.hsi_support_plane_penetration_margin_m = hsi_support_plane_penetration_margin_m
        self.hsi_support_plane_float_weight = hsi_support_plane_float_weight
        self.hsi_support_plane_penetration_weight = hsi_support_plane_penetration_weight
        self.hsi_teacher_pose_weight = hsi_teacher_pose_weight
        self.hsi_teacher_betas_weight = hsi_teacher_betas_weight
        self.hsi_teacher_transl_weight = hsi_teacher_transl_weight
        self.hsi_teacher_joints_weight = hsi_teacher_joints_weight
        self.hsi_teacher_vertices_weight = hsi_teacher_vertices_weight
        self.hsi_teacher_scene_affine_weight = hsi_teacher_scene_affine_weight
        self.hsi_pose_velocity_weight = hsi_pose_velocity_weight
        self.hsi_betas_velocity_weight = hsi_betas_velocity_weight
        self.hsi_transl_velocity_weight = hsi_transl_velocity_weight
        self.hsi_joints_velocity_weight = hsi_joints_velocity_weight
        self.hsi_joints_acceleration_weight = hsi_joints_acceleration_weight
        self.hsi_foot_sliding_weight = hsi_foot_sliding_weight
        self.hsi_foot_sliding_contact_threshold_m = hsi_foot_sliding_contact_threshold_m
        self.hsi_scene_scale_temporal_weight = hsi_scene_scale_temporal_weight
        self.hsi_scene_scale_sequence_weight = hsi_scene_scale_sequence_weight
        self.hsi_scene_bias_temporal_weight = hsi_scene_bias_temporal_weight
        self.hsi_scene_bias_sequence_weight = hsi_scene_bias_sequence_weight
        self.hsi_contact_weight = hsi_contact_weight
        self.hsi_contact_threshold = hsi_contact_threshold
        self.hsi_contact_teacher_camera_source = str(hsi_contact_teacher_camera_source or "prediction").lower()
        self.hsi_contact_refine_plane_weight = float(hsi_contact_refine_plane_weight)
        self.hsi_contact_refine_pose_weight = float(hsi_contact_refine_pose_weight)
        self.hsi_contact_refine_class_weight = float(hsi_contact_refine_class_weight)
        self.hsi_contact_refine_no_worse_weight = float(hsi_contact_refine_no_worse_weight)
        self.hsi_contact_refine_swing_no_pull_weight = float(hsi_contact_refine_swing_no_pull_weight)
        self._smpl_layer: SMPLLayer | None = None
        self._foot_sole_indices: torch.Tensor | None = None
        self._foot_sole_indices_by_count: dict[int, torch.Tensor] = {}
        if self.conf_loss_type not in {"bce", "focal"}:
            raise ValueError(f"Unsupported conf_loss_type: {self.conf_loss_type}")
        if self.conf_target_type not in {"binary", "matched_iou"}:
            raise ValueError(f"Unsupported conf_target_type: {self.conf_target_type}")
        if self.projected_bbox_source not in {"joints", "vertices"}:
            raise ValueError(f"Unsupported projected_bbox_source: {self.projected_bbox_source}")
        if self.hsi_smpl_scale_teacher_source not in {"vertices", "joints"}:
            raise ValueError(f"Unsupported hsi_smpl_scale_teacher_source: {self.hsi_smpl_scale_teacher_source}")
        if self.projection_camera_source not in {"vggt", "gt", "auto"}:
            raise ValueError(f"Unsupported projection_camera_source: {self.projection_camera_source}")
        if self._uses_projected_bbox and not self.smpl_model_dir:
            raise ValueError("Projected SMPL bbox loss requires loss.smpl_model_dir or assets.smpl_model_dir")
        self.register_buffer(
            "betas_dim_weight",
            torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).view(1, 10),
            persistent=False,
        )

    def forward(self, predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._active_projection_image_hw = _infer_projection_image_hw(predictions, batch, self.projection_image_size)
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
                    "loss_joints3d": zero,
                    "loss_local_joints3d": zero,
                    "loss_local_vertices": zero,
                    "loss_projected_joints2d": zero,
                    "loss_transl_refine_delta_reg": zero,
                    "loss_transl_refine_ray_depth": zero,
                    "loss_transl_refine_tangent": zero,
                    "loss_transl_hard_topk": zero,
                    "loss_transl_hard_severe": zero,
                    "loss_transl_temporal_velocity": zero,
                    "loss_transl_temporal_acceleration": zero,
                    "loss_transl_temporal_no_worse": zero,
                    "loss_projected_bbox": zero,
                    "loss_projected_giou": zero,
                    "loss_hsi_ray_delta": zero,
                    "loss_hsi_tangent_delta": zero,
                    "loss_hsi_align_point": zero,
                    "loss_hsi_align_delta_reg": zero,
                    "loss_hsi_align_no_worse": zero,
                    "loss_hsi_contact_refine_plane": zero,
                    "loss_hsi_contact_refine_pose": zero,
                    "loss_hsi_contact_refine_class": zero,
                    "loss_hsi_contact_refine_no_worse": zero,
                    "loss_hsi_contact_refine_swing_no_pull": zero,
                    "metric_joints3d_l1": zero.detach(),
                    "metric_local_joints3d_l1": zero.detach(),
                    "metric_local_vertices_l1": zero.detach(),
                    "metric_projected_joints2d_l1": zero.detach(),
                    "metric_transl_l2_mean": zero.detach(),
                    "metric_transl_l2_topk_mean": zero.detach(),
                    "metric_transl_l2_max": zero.detach(),
                    "metric_transl_l2_over_threshold_rate": zero.detach(),
                    "metric_transl_hard_count": zero.detach(),
                    "metric_transl_temporal_pair_count": zero.detach(),
                    "metric_transl_temporal_triple_count": zero.detach(),
                    "metric_transl_temporal_velocity_l1": zero.detach(),
                    "metric_transl_temporal_acceleration_l1": zero.detach(),
                    "metric_transl_temporal_no_worse_ratio": zero.detach(),
                    "metric_transl_temporal_no_worse_l1": zero.detach(),
                    "metric_base_transl_l1": zero.detach(),
                    "metric_refined_transl_l1": zero.detach(),
                    "metric_transl_refine_l1_delta": zero.detach(),
                    "metric_transl_box_prior_weight_abs": zero.detach(),
                    "metric_base_transl_ray_depth_l1": zero.detach(),
                    "metric_refined_transl_ray_depth_l1": zero.detach(),
                    "metric_transl_ray_depth_l1_delta": zero.detach(),
                    "metric_base_transl_tangent_l1": zero.detach(),
                    "metric_refined_transl_tangent_l1": zero.detach(),
                    "metric_transl_tangent_l1_delta": zero.detach(),
                    "metric_hsi_ray_delta_l1": zero.detach(),
                    "metric_hsi_ray_delta_base_l1": zero.detach(),
                    "metric_hsi_ray_delta_refined_l1": zero.detach(),
                    "metric_hsi_ray_delta_l1_delta": zero.detach(),
                    "metric_hsi_ray_delta_expected_abs": zero.detach(),
                    "metric_hsi_ray_delta_pred_abs": zero.detach(),
                    "metric_hsi_ray_delta_sign_acc": zero.detach(),
                    "metric_hsi_align_base_point_l1": zero.detach(),
                    "metric_hsi_align_refined_point_l1": zero.detach(),
                    "metric_hsi_align_point_l1_delta": zero.detach(),
                    "metric_hsi_align_delta_l1": zero.detach(),
                    "metric_hsi_align_gate_mean": zero.detach(),
                    "metric_hsi_align_valid_ratio": zero.detach(),
                    "metric_bbox_iou_mean": zero.detach(),
                    "metric_projected_bbox_iou_mean": zero.detach(),
                    "metric_conf_target_pos_mean": zero.detach(),
                    "metric_conf_target_pos_min": zero.detach(),
                    "metric_conf_target_pos_max": zero.detach(),
                }
            )
            losses.update(self._zero_hsi_losses(predictions))
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
            losses.update(self._translation_hard_tail_losses(pred_transl_cam, frame_idx, src_idx, matched))
            losses["loss_id"] = self._identity_loss(flat_id_embed, frame_idx, src_idx, matched)
            joint_losses = self._smpl_joint_losses(predictions, batch, pred_betas, pred_transl_cam, frame_idx, src_idx, matched)
            losses.update(joint_losses)
            losses.update(self._smpl_local_losses(predictions, pred_betas, frame_idx, src_idx, matched))
            losses.update(self._translation_refine_losses(predictions, pred_transl_cam, frame_idx, src_idx, matched))
            losses.update(self._translation_temporal_losses(predictions, batch, pred_transl_cam, frame_idx, src_idx, matched))
            projected = self._projected_bbox_losses(predictions, batch, pred_betas, pred_transl_cam, frame_idx, src_idx, target_boxes)
            losses.update(projected)
            losses.update(self._hsi_refined_losses(predictions, batch, frame_idx, src_idx, matched))

        losses["loss_total"] = (
            self.conf_weight * losses["loss_conf"]
            + self.bbox_weight * losses["loss_bbox"]
            + self.giou_weight * losses["loss_giou"]
            + self.pose_weight * losses["loss_pose"]
            + self.betas_weight * losses["loss_betas"]
            + self.transl_cam_weight * losses["loss_transl_cam"]
            + self.id_weight * losses["loss_id"]
            + self.joints3d_weight * losses["loss_joints3d"]
            + self.local_joints3d_weight * losses["loss_local_joints3d"]
            + self.local_vertices_weight * losses["loss_local_vertices"]
            + self.projected_joints2d_weight * losses["loss_projected_joints2d"]
            + self.transl_refine_delta_reg_weight * losses["loss_transl_refine_delta_reg"]
            + self.transl_refine_ray_depth_weight * losses["loss_transl_refine_ray_depth"]
            + self.transl_refine_tangent_weight * losses["loss_transl_refine_tangent"]
            + self.transl_hard_topk_weight * losses["loss_transl_hard_topk"]
            + self.transl_hard_severe_weight * losses["loss_transl_hard_severe"]
            + self.transl_temporal_velocity_weight * losses["loss_transl_temporal_velocity"]
            + self.transl_temporal_acceleration_weight * losses["loss_transl_temporal_acceleration"]
            + self.transl_temporal_no_worse_weight * losses["loss_transl_temporal_no_worse"]
            + self.projected_bbox_weight * losses["loss_projected_bbox"]
            + self.projected_giou_weight * losses["loss_projected_giou"]
            + self.duplicate_conf_weight * losses["loss_duplicate_conf"]
            + self.aux_weight * losses["loss_aux_total"]
            + self.hsi_pose_weight * losses["loss_hsi_pose"]
            + self.hsi_betas_weight * losses["loss_hsi_betas"]
            + self.hsi_transl_cam_weight * losses["loss_hsi_transl_cam"]
            + self.hsi_ray_delta_weight * losses["loss_hsi_ray_delta"]
            + self.hsi_tangent_delta_weight * losses["loss_hsi_tangent_delta"]
            + self.hsi_align_point_weight * losses["loss_hsi_align_point"]
            + self.hsi_align_delta_reg_weight * losses["loss_hsi_align_delta_reg"]
            + self.hsi_align_no_worse_weight * losses["loss_hsi_align_no_worse"]
            + self.hsi_transl_clean_identity_weight * losses["loss_hsi_transl_clean_identity"]
            + self.hsi_transl_noise_gate_weight * losses["loss_hsi_transl_noise_gate"]
            + self.hsi_joints3d_weight * losses["loss_hsi_joints3d"]
            + self.hsi_vertices_weight * losses["loss_hsi_vertices"]
            + self.hsi_projected_joints2d_weight * losses["loss_hsi_projected_joints2d"]
            + self.hsi_depth_teacher_weight * losses["loss_hsi_depth_teacher"]
            + self.hsi_smpl_scale_teacher_weight * losses["loss_hsi_smpl_scale_teacher"]
            + self.hsi_anchor_depth_weight * losses["loss_hsi_anchor_depth"]
            + self.hsi_anchor_scene_xyz_weight * losses["loss_hsi_anchor_scene_xyz"]
            + self.hsi_delta_reg_weight * losses["loss_hsi_delta_reg"]
            + self.hsi_no_worse_weight * losses["loss_hsi_no_worse"]
            + self.hsi_gate_reg_weight * losses["loss_hsi_gate_reg"]
            + self.hsi_foot_contact_weight * losses["loss_hsi_foot_contact"]
            + self.hsi_foot_sole_contact_weight * losses["loss_hsi_foot_sole_contact"]
            + self.hsi_support_plane_contact_weight * losses["loss_hsi_support_plane_contact"]
            + self.hsi_teacher_pose_weight * losses["loss_hsi_teacher_pose"]
            + self.hsi_teacher_betas_weight * losses["loss_hsi_teacher_betas"]
            + self.hsi_teacher_transl_weight * losses["loss_hsi_teacher_transl"]
            + self.hsi_teacher_joints_weight * losses["loss_hsi_teacher_joints"]
            + self.hsi_teacher_vertices_weight * losses["loss_hsi_teacher_vertices"]
            + self.hsi_teacher_scene_affine_weight * losses["loss_hsi_teacher_scene_affine"]
            + self.hsi_pose_velocity_weight * losses["loss_hsi_pose_velocity"]
            + self.hsi_betas_velocity_weight * losses["loss_hsi_betas_velocity"]
            + self.hsi_transl_velocity_weight * losses["loss_hsi_transl_velocity"]
            + self.hsi_joints_velocity_weight * losses["loss_hsi_joints_velocity"]
            + self.hsi_joints_acceleration_weight * losses["loss_hsi_joints_acceleration"]
            + self.hsi_temporal_no_worse_weight * losses["loss_hsi_temporal_no_worse"]
            + self.hsi_foot_sliding_weight * losses["loss_hsi_foot_sliding"]
            + self.hsi_scene_scale_temporal_weight * losses["loss_hsi_scene_scale_temporal"]
            + self.hsi_scene_scale_sequence_weight * losses["loss_hsi_scene_scale_sequence"]
            + self.hsi_scene_bias_temporal_weight * losses["loss_hsi_scene_bias_temporal"]
            + self.hsi_scene_bias_sequence_weight * losses["loss_hsi_scene_bias_sequence"]
            + self.hsi_contact_weight * losses["loss_hsi_contact"]
            + self.hsi_contact_refine_plane_weight * losses["loss_hsi_contact_refine_plane"]
            + self.hsi_contact_refine_pose_weight * losses["loss_hsi_contact_refine_pose"]
            + self.hsi_contact_refine_class_weight * losses["loss_hsi_contact_refine_class"]
            + self.hsi_contact_refine_no_worse_weight * losses["loss_hsi_contact_refine_no_worse"]
            + self.hsi_contact_refine_swing_no_pull_weight * losses["loss_hsi_contact_refine_swing_no_pull"]
        )
        return losses

    @property
    def _uses_projected_bbox(self) -> bool:
        return self.use_vggt_camera_projection and (self.projected_bbox_weight != 0.0 or self.projected_giou_weight != 0.0)

    def _projection_intrinsics(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.projection_camera_source in {"gt", "auto"}:
            batch_intrinsics = batch.get("gt_intrinsics", batch.get("K_scal3r"))
            if batch_intrinsics is not None:
                return _flatten_batch_intrinsics(batch_intrinsics, device=device, dtype=dtype)
            if self.projection_camera_source == "gt":
                raise ValueError("projection_camera_source='gt' requires batch['gt_intrinsics'] or batch['K_scal3r']")
        if "pose_enc" not in predictions:
            raise ValueError("VGGT projection camera requires model output pose_enc; set model.enable_camera=true")
        return _flatten_intrinsics(_require_prediction(predictions, "pose_enc"), self._projection_image_hw()).to(device=device, dtype=dtype)

    def _contact_teacher_intrinsics(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fallback_intrinsics: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source = self.hsi_contact_teacher_camera_source
        if source in {"prediction", "pred", "vggt", ""}:
            return fallback_intrinsics.to(device=device, dtype=dtype)
        if source == "gt":
            batch_intrinsics = batch.get("gt_intrinsics", batch.get("K_scal3r"))
            if batch_intrinsics is None:
                raise ValueError("hsi_contact_teacher_camera_source='gt' requires batch['gt_intrinsics'] or batch['K_scal3r']")
            return _flatten_batch_intrinsics(batch_intrinsics, device=device, dtype=dtype)
        if source == "auto":
            batch_intrinsics = batch.get("gt_intrinsics", batch.get("K_scal3r"))
            if batch_intrinsics is not None:
                return _flatten_batch_intrinsics(batch_intrinsics, device=device, dtype=dtype)
            return fallback_intrinsics.to(device=device, dtype=dtype)
        raise ValueError(f"Unsupported hsi_contact_teacher_camera_source: {source!r}")

    def _projection_image_hw(self) -> tuple[int, int]:
        return getattr(self, "_active_projection_image_hw", (int(self.projection_image_size), int(self.projection_image_size)))

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
        batch: dict[str, torch.Tensor],
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
        if "pred_poses" not in predictions:
            raise ValueError("Projected SMPL bbox loss requires model output pred_poses")

        pred_poses = _flatten_prediction(_require_prediction(predictions, "pred_poses"), unframed_ndim=3)
        intrinsics = self._projection_intrinsics(predictions, batch, device=pred_betas.device, dtype=pred_betas.dtype)
        smpl = self._get_smpl_layer(pred_betas.device)

        poses = pred_poses[frame_idx, src_idx].reshape(-1, 72)
        betas = pred_betas[frame_idx, src_idx]
        transl_cam = pred_transl_cam[frame_idx, src_idx]
        vertices, joints = smpl(poses.float(), betas.float())
        points = joints[:, :24] if self.projected_bbox_source == "joints" else vertices
        points_cam = points.to(dtype=pred_betas.dtype) + transl_cam[:, None, :]
        projected = _project_points(points_cam, intrinsics[frame_idx].to(dtype=points_cam.dtype))
        projected_boxes = _points_to_normalized_cxcywh(projected, self._projection_image_hw())
        projected_boxes = projected_boxes.to(dtype=target_boxes.dtype)

        giou = generalized_box_iou(cxcywh_to_xyxy(projected_boxes), cxcywh_to_xyxy(target_boxes))
        return {
            "loss_projected_bbox": F.l1_loss(projected_boxes, target_boxes),
            "loss_projected_giou": (1.0 - giou.diag()).mean(),
            "metric_projected_bbox_iou_mean": _box_iou_diag(cxcywh_to_xyxy(projected_boxes), cxcywh_to_xyxy(target_boxes)).detach().mean(),
        }

    def _smpl_joint_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        pred_betas: torch.Tensor,
        pred_transl_cam: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.joints3d_weight == 0.0 and self.projected_joints2d_weight == 0.0:
            zero = pred_betas.sum() * 0.0
            return {
                "loss_joints3d": zero,
                "loss_projected_joints2d": zero,
                "metric_joints3d_l1": zero.detach(),
                "metric_projected_joints2d_l1": zero.detach(),
            }
        if not self.smpl_model_dir:
            raise ValueError("SMPL joint losses require loss.smpl_model_dir or assets.smpl_model_dir")
        if "pred_poses" not in predictions:
            raise ValueError("SMPL joint losses require model output pred_poses")

        pred_poses = _flatten_prediction(_require_prediction(predictions, "pred_poses"), unframed_ndim=3)
        pred_poses_matched = pred_poses[frame_idx, src_idx].reshape(-1, 72)
        pred_betas_matched = pred_betas[frame_idx, src_idx]
        pred_transl = pred_transl_cam[frame_idx, src_idx]
        gt_poses = rot6d_to_axis_angle(matched["pose_6d"].to(device=pred_betas.device, dtype=pred_betas.dtype)).reshape(-1, 72)
        gt_betas = matched["betas"].to(device=pred_betas.device, dtype=pred_betas.dtype)
        gt_transl = matched["transl_cam"].to(device=pred_betas.device, dtype=pred_betas.dtype)

        smpl = self._get_smpl_layer(pred_betas.device)
        _, pred_joints = smpl(pred_poses_matched.float(), pred_betas_matched.float())
        _, gt_joints = smpl(gt_poses.float(), gt_betas.float())
        pred_joints_cam = pred_joints[:, :24].to(dtype=pred_betas.dtype) + pred_transl[:, None, :]
        gt_joints_cam = gt_joints[:, :24].to(dtype=pred_betas.dtype) + gt_transl[:, None, :]
        joints3d_l1 = F.l1_loss(pred_joints_cam, gt_joints_cam)

        if self.projected_joints2d_weight == 0.0:
            zero = pred_betas.sum() * 0.0
            projected_l1 = zero
        else:
            intrinsics = self._projection_intrinsics(predictions, batch, device=pred_betas.device, dtype=pred_betas.dtype)
            pred_2d = _normalize_points_2d(_project_points(pred_joints_cam, intrinsics[frame_idx].to(dtype=pred_joints_cam.dtype)), self._projection_image_hw())
            gt_2d = _normalize_points_2d(_project_points(gt_joints_cam, intrinsics[frame_idx].to(dtype=gt_joints_cam.dtype)), self._projection_image_hw())
            valid = (pred_joints_cam[..., 2] > 1e-4) & (gt_joints_cam[..., 2] > 1e-4)
            if valid.any():
                projected_l1 = F.l1_loss(pred_2d[valid], gt_2d[valid])
            else:
                projected_l1 = pred_betas.sum() * 0.0

        return {
            "loss_joints3d": joints3d_l1,
            "loss_projected_joints2d": projected_l1,
            "metric_joints3d_l1": joints3d_l1.detach(),
            "metric_projected_joints2d_l1": projected_l1.detach(),
        }

    def _smpl_local_losses(
        self,
        predictions: dict[str, torch.Tensor],
        pred_betas: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = pred_betas.sum() * 0.0
        if self.local_joints3d_weight == 0.0 and self.local_vertices_weight == 0.0:
            return {
                "loss_local_joints3d": zero,
                "loss_local_vertices": zero,
                "metric_local_joints3d_l1": zero.detach(),
                "metric_local_vertices_l1": zero.detach(),
            }
        if not self.smpl_model_dir:
            raise ValueError("Local SMPL losses require loss.smpl_model_dir or assets.smpl_model_dir")
        if "pred_poses" not in predictions:
            raise ValueError("Local SMPL losses require model output pred_poses")

        pred_poses = _flatten_prediction(_require_prediction(predictions, "pred_poses"), unframed_ndim=3)
        pred_poses_matched = pred_poses[frame_idx, src_idx].reshape(-1, 72)
        pred_betas_matched = pred_betas[frame_idx, src_idx]
        gt_poses = rot6d_to_axis_angle(matched["pose_6d"].to(device=pred_betas.device, dtype=pred_betas.dtype)).reshape(-1, 72)
        gt_betas = matched["betas"].to(device=pred_betas.device, dtype=pred_betas.dtype)

        smpl = self._get_smpl_layer(pred_betas.device)
        pred_vertices, pred_joints = smpl(pred_poses_matched.float(), pred_betas_matched.float())
        gt_vertices, gt_joints = smpl(gt_poses.float(), gt_betas.float())
        pred_joints = pred_joints[:, :24].to(dtype=pred_betas.dtype)
        gt_joints = gt_joints[:, :24].to(dtype=pred_betas.dtype)
        pred_vertices = pred_vertices.to(dtype=pred_betas.dtype)
        gt_vertices = gt_vertices.to(dtype=pred_betas.dtype)
        pred_root = pred_joints[:, :1]
        gt_root = gt_joints[:, :1]
        pred_joints_local = pred_joints - pred_root
        gt_joints_local = gt_joints - gt_root
        pred_vertices_local = pred_vertices - pred_root
        gt_vertices_local = gt_vertices - gt_root
        joints_l1 = F.l1_loss(pred_joints_local, gt_joints_local)
        vertices_l1 = F.l1_loss(pred_vertices_local, gt_vertices_local)
        return {
            "loss_local_joints3d": joints_l1,
            "loss_local_vertices": vertices_l1,
            "metric_local_joints3d_l1": joints_l1.detach(),
            "metric_local_vertices_l1": vertices_l1.detach(),
        }

    def _translation_hard_tail_losses(
        self,
        pred_transl_cam: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_transl = matched["transl_cam"].to(device=pred_transl_cam.device, dtype=pred_transl_cam.dtype)
        pred_transl = pred_transl_cam[frame_idx, src_idx]
        errors = torch.linalg.norm(pred_transl - target_transl, dim=-1)
        if errors.numel() == 0:
            zero = pred_transl_cam.sum() * 0.0
            return {
                "loss_transl_hard_topk": zero,
                "loss_transl_hard_severe": zero,
                "metric_transl_l2_mean": zero.detach(),
                "metric_transl_l2_topk_mean": zero.detach(),
                "metric_transl_l2_max": zero.detach(),
                "metric_transl_l2_over_threshold_rate": zero.detach(),
                "metric_transl_hard_count": zero.detach(),
            }

        min_k = max(int(self.transl_hard_min_k), 0)
        fraction = max(float(self.transl_hard_topk_fraction), 0.0)
        fraction_k = int(errors.numel() * fraction + 0.999999) if fraction > 0.0 else 0
        hard_k = min(max(min_k, fraction_k), int(errors.numel()))
        if hard_k > 0:
            hard_values = torch.topk(errors, k=hard_k, largest=True).values
            hard_topk = hard_values.mean()
            hard_topk_metric = hard_topk
        else:
            hard_topk = errors.sum() * 0.0
            hard_topk_metric = errors.mean()

        threshold = float(self.transl_hard_error_threshold_m)
        if threshold > 0.0:
            over_threshold = errors > threshold
            severe = F.relu(errors - threshold).mean()
            over_rate = over_threshold.to(dtype=errors.dtype).mean()
        else:
            over_threshold = torch.ones_like(errors, dtype=torch.bool)
            severe = errors.mean()
            over_rate = torch.ones((), device=errors.device, dtype=errors.dtype)

        return {
            "loss_transl_hard_topk": hard_topk,
            "loss_transl_hard_severe": severe,
            "metric_transl_l2_mean": errors.detach().mean(),
            "metric_transl_l2_topk_mean": hard_topk_metric.detach(),
            "metric_transl_l2_max": errors.detach().max(),
            "metric_transl_l2_over_threshold_rate": over_rate.detach(),
            "metric_transl_hard_count": over_threshold.to(dtype=errors.dtype).detach().sum(),
        }

    def _translation_refine_losses(
        self,
        predictions: dict[str, torch.Tensor],
        pred_transl_cam: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_transl = matched["transl_cam"].to(device=pred_transl_cam.device, dtype=pred_transl_cam.dtype)
        pred_transl = pred_transl_cam[frame_idx, src_idx]
        delta = predictions.get("pred_transl_cam_delta")
        if delta is not None:
            flat_delta = _flatten_prediction(delta, unframed_ndim=3)
            matched_delta = flat_delta[frame_idx, src_idx]
            delta_reg = F.smooth_l1_loss(matched_delta, torch.zeros_like(matched_delta))
        else:
            delta_reg = pred_transl.sum() * 0.0

        base_transl = predictions.get("base_pred_transl_cam")
        flat_base_transl = None
        if base_transl is not None:
            flat_base_transl = _flatten_prediction(base_transl, unframed_ndim=3)
            base_l1 = F.l1_loss(flat_base_transl[frame_idx, src_idx], target_transl)
        else:
            base_l1 = F.l1_loss(pred_transl, target_transl).detach()
        refined_l1 = F.l1_loss(pred_transl, target_transl)
        box_prior_weight = predictions.get("pred_transl_box_prior_weight")
        if box_prior_weight is not None:
            flat_weight = _flatten_prediction(box_prior_weight, unframed_ndim=3)
            matched_weight = flat_weight[frame_idx, src_idx]
            box_prior_weight_abs = matched_weight.abs().mean()
        else:
            box_prior_weight_abs = pred_transl.sum() * 0.0

        ray_depth_loss = pred_transl.sum() * 0.0
        tangent_loss = pred_transl.sum() * 0.0
        base_ray_l1 = pred_transl.sum() * 0.0
        refined_ray_l1 = pred_transl.sum() * 0.0
        base_tangent_l1 = pred_transl.sum() * 0.0
        refined_tangent_l1 = pred_transl.sum() * 0.0
        ray_dir = predictions.get("pred_transl_ray_dir")
        tangent_x = predictions.get("pred_transl_tangent_x")
        tangent_y = predictions.get("pred_transl_tangent_y")
        if ray_dir is not None and tangent_x is not None and tangent_y is not None:
            flat_ray = _flatten_prediction(ray_dir, unframed_ndim=3)
            flat_tx = _flatten_prediction(tangent_x, unframed_ndim=3)
            flat_ty = _flatten_prediction(tangent_y, unframed_ndim=3)
            matched_ray = flat_ray[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            matched_tx = flat_tx[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            matched_ty = flat_ty[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            target_ray_depth = (target_transl * matched_ray).sum(dim=-1, keepdim=True)
            target_tangent = torch.cat(
                [
                    (target_transl * matched_tx).sum(dim=-1, keepdim=True),
                    (target_transl * matched_ty).sum(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            pred_ray_depth = predictions.get("pred_transl_ray_depth")
            pred_tangent = predictions.get("pred_transl_tangent")
            base_ray_depth = predictions.get("base_pred_transl_ray_depth")
            base_tangent = predictions.get("base_pred_transl_tangent")
            if pred_ray_depth is not None:
                flat_pred_ray_depth = _flatten_prediction(pred_ray_depth, unframed_ndim=3)
                matched_pred_ray_depth = flat_pred_ray_depth[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            else:
                matched_pred_ray_depth = (pred_transl * matched_ray).sum(dim=-1, keepdim=True)
            if pred_tangent is not None:
                flat_pred_tangent = _flatten_prediction(pred_tangent, unframed_ndim=3)
                matched_pred_tangent = flat_pred_tangent[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            else:
                matched_pred_tangent = torch.cat(
                    [
                        (pred_transl * matched_tx).sum(dim=-1, keepdim=True),
                        (pred_transl * matched_ty).sum(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
            if base_ray_depth is not None:
                flat_base_ray_depth = _flatten_prediction(base_ray_depth, unframed_ndim=3)
                matched_base_ray_depth = flat_base_ray_depth[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            else:
                matched_base_ray_depth = (
                    (flat_base_transl[frame_idx, src_idx] * matched_ray).sum(dim=-1, keepdim=True)
                    if flat_base_transl is not None
                    else matched_pred_ray_depth
                )
            if base_tangent is not None:
                flat_base_tangent = _flatten_prediction(base_tangent, unframed_ndim=3)
                matched_base_tangent = flat_base_tangent[frame_idx, src_idx].to(dtype=pred_transl.dtype)
            else:
                base_values = flat_base_transl[frame_idx, src_idx] if flat_base_transl is not None else pred_transl
                matched_base_tangent = torch.cat(
                    [
                        (base_values * matched_tx).sum(dim=-1, keepdim=True),
                        (base_values * matched_ty).sum(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
            ray_depth_loss = F.l1_loss(matched_pred_ray_depth, target_ray_depth)
            tangent_loss = F.l1_loss(matched_pred_tangent, target_tangent)
            base_ray_l1 = F.l1_loss(matched_base_ray_depth, target_ray_depth)
            refined_ray_l1 = ray_depth_loss
            base_tangent_l1 = F.l1_loss(matched_base_tangent, target_tangent)
            refined_tangent_l1 = tangent_loss
        return {
            "loss_transl_refine_delta_reg": delta_reg,
            "loss_transl_refine_ray_depth": ray_depth_loss,
            "loss_transl_refine_tangent": tangent_loss,
            "metric_base_transl_l1": base_l1.detach(),
            "metric_refined_transl_l1": refined_l1.detach(),
            "metric_transl_refine_l1_delta": (refined_l1 - base_l1).detach(),
            "metric_transl_box_prior_weight_abs": box_prior_weight_abs.detach(),
            "metric_base_transl_ray_depth_l1": base_ray_l1.detach(),
            "metric_refined_transl_ray_depth_l1": refined_ray_l1.detach(),
            "metric_transl_ray_depth_l1_delta": (refined_ray_l1 - base_ray_l1).detach(),
            "metric_base_transl_tangent_l1": base_tangent_l1.detach(),
            "metric_refined_transl_tangent_l1": refined_tangent_l1.detach(),
            "metric_transl_tangent_l1_delta": (refined_tangent_l1 - base_tangent_l1).detach(),
        }

    def _translation_temporal_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        pred_transl_cam: torch.Tensor,
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = pred_transl_cam.sum() * 0.0
        out = {
            "loss_transl_temporal_velocity": zero,
            "loss_transl_temporal_acceleration": zero,
            "loss_transl_temporal_no_worse": zero,
            "metric_transl_temporal_pair_count": zero.detach(),
            "metric_transl_temporal_triple_count": zero.detach(),
            "metric_transl_temporal_velocity_l1": zero.detach(),
            "metric_transl_temporal_acceleration_l1": zero.detach(),
            "metric_transl_temporal_no_worse_ratio": zero.detach(),
            "metric_transl_temporal_no_worse_l1": zero.detach(),
        }
        num_frames = _infer_sequence_length(batch, predictions)
        if num_frames <= 1 or frame_idx.numel() == 0:
            return out

        pred_transl = pred_transl_cam[frame_idx, src_idx]
        target_transl = matched["transl_cam"].to(device=pred_transl_cam.device, dtype=pred_transl_cam.dtype)
        seed = predictions.get("seed_pred_transl_cam")
        if seed is None:
            seed = predictions.get("base_pred_transl_cam")
        flat_seed = _flatten_prediction(seed, unframed_ndim=3) if seed is not None else None
        seed_transl = flat_seed[frame_idx, src_idx].detach() if flat_seed is not None else pred_transl.detach()

        batch_ids = torch.div(frame_idx, int(num_frames), rounding_mode="floor")
        seq_ids = frame_idx % int(num_frames)
        person_ids = matched.get("person_ids")
        person_id_mask = matched.get("person_id_mask")
        if person_ids is None:
            person_ids = torch.full_like(src_idx, -1)
        if person_id_mask is None:
            person_id_mask = torch.zeros_like(src_idx, dtype=torch.bool)

        groups: dict[tuple[int, int], dict[int, int]] = {}
        for item_idx in range(int(frame_idx.numel())):
            batch_id = int(batch_ids[item_idx].detach().cpu())
            seq_id = int(seq_ids[item_idx].detach().cpu())
            track_valid = bool(person_id_mask[item_idx].detach().cpu())
            track_id = int(person_ids[item_idx].detach().cpu()) if track_valid else -(int(src_idx[item_idx].detach().cpu()) + 1)
            groups.setdefault((batch_id, track_id), {}).setdefault(seq_id, item_idx)

        pair_prev: list[int] = []
        pair_next: list[int] = []
        triple_prev: list[int] = []
        triple_mid: list[int] = []
        triple_next: list[int] = []
        for seq_map in groups.values():
            for seq in range(1, int(num_frames)):
                if seq - 1 in seq_map and seq in seq_map:
                    pair_prev.append(seq_map[seq - 1])
                    pair_next.append(seq_map[seq])
            for seq in range(1, int(num_frames) - 1):
                if seq - 1 in seq_map and seq in seq_map and seq + 1 in seq_map:
                    triple_prev.append(seq_map[seq - 1])
                    triple_mid.append(seq_map[seq])
                    triple_next.append(seq_map[seq + 1])

        no_worse_terms: list[torch.Tensor] = []
        no_worse_flags: list[torch.Tensor] = []
        if pair_prev:
            prev = torch.tensor(pair_prev, dtype=torch.long, device=pred_transl.device)
            curr = torch.tensor(pair_next, dtype=torch.long, device=pred_transl.device)
            velocity_delta = _velocity_residual(pred_transl, target_transl, prev, curr)
            seed_velocity_delta = _velocity_residual(seed_transl, target_transl, prev, curr)
            out["loss_transl_temporal_velocity"] = _smooth_l1_abs(velocity_delta.abs()).mean()
            pred_vel_err = torch.linalg.norm(velocity_delta, dim=-1)
            seed_vel_err = torch.linalg.norm(seed_velocity_delta, dim=-1).detach()
            velocity_excess = F.relu(pred_vel_err - seed_vel_err - float(self.transl_temporal_no_worse_margin_m))
            no_worse_terms.append(velocity_excess)
            no_worse_flags.append(pred_vel_err > seed_vel_err + float(self.transl_temporal_no_worse_margin_m))
            out["metric_transl_temporal_pair_count"] = pred_transl.new_tensor(float(len(pair_prev))).detach()
            out["metric_transl_temporal_velocity_l1"] = velocity_delta.abs().mean().detach()

        if triple_prev:
            prev = torch.tensor(triple_prev, dtype=torch.long, device=pred_transl.device)
            mid = torch.tensor(triple_mid, dtype=torch.long, device=pred_transl.device)
            nxt = torch.tensor(triple_next, dtype=torch.long, device=pred_transl.device)
            acceleration_delta = _acceleration_residual(pred_transl, target_transl, prev, mid, nxt)
            seed_acceleration_delta = _acceleration_residual(seed_transl, target_transl, prev, mid, nxt)
            out["loss_transl_temporal_acceleration"] = _smooth_l1_abs(acceleration_delta.abs()).mean()
            pred_acc_err = torch.linalg.norm(acceleration_delta, dim=-1)
            seed_acc_err = torch.linalg.norm(seed_acceleration_delta, dim=-1).detach()
            acc_excess = F.relu(pred_acc_err - seed_acc_err - float(self.transl_temporal_no_worse_accel_margin_m))
            no_worse_terms.append(acc_excess)
            no_worse_flags.append(pred_acc_err > seed_acc_err + float(self.transl_temporal_no_worse_accel_margin_m))
            out["metric_transl_temporal_triple_count"] = pred_transl.new_tensor(float(len(triple_prev))).detach()
            out["metric_transl_temporal_acceleration_l1"] = acceleration_delta.abs().mean().detach()

        if no_worse_terms:
            no_worse = torch.cat([term.reshape(-1) for term in no_worse_terms], dim=0)
            flags = torch.cat([flag.reshape(-1).to(dtype=pred_transl.dtype) for flag in no_worse_flags], dim=0)
            out["loss_transl_temporal_no_worse"] = _smooth_l1_abs(no_worse, beta=0.01).mean()
            out["metric_transl_temporal_no_worse_ratio"] = flags.mean().detach()
            out["metric_transl_temporal_no_worse_l1"] = no_worse.mean().detach()
        return out

    def _get_smpl_layer(self, device: torch.device) -> SMPLLayer:
        if self._smpl_layer is None:
            self._smpl_layer = SMPLLayer(self.smpl_model_dir).to(device=device).eval()
            for param in self._smpl_layer.parameters():
                param.requires_grad = False
        return self._smpl_layer

    def _zero_hsi_losses(self, predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        anchor = next((value for value in predictions.values() if isinstance(value, torch.Tensor)), None)
        if anchor is None:
            anchor = torch.zeros(())
        zero = anchor.sum() * 0.0
        return {
            "loss_hsi_pose": zero,
            "loss_hsi_betas": zero,
            "loss_hsi_transl_cam": zero,
            "loss_hsi_ray_delta": zero,
            "loss_hsi_tangent_delta": zero,
            "loss_hsi_align_point": zero,
            "loss_hsi_align_delta_reg": zero,
            "loss_hsi_align_no_worse": zero,
            "loss_hsi_transl_clean_identity": zero,
            "loss_hsi_transl_noise_gate": zero,
            "loss_hsi_contact_refine_plane": zero,
            "loss_hsi_contact_refine_pose": zero,
            "loss_hsi_contact_refine_class": zero,
            "loss_hsi_contact_refine_no_worse": zero,
            "loss_hsi_contact_refine_swing_no_pull": zero,
            "loss_hsi_joints3d": zero,
            "loss_hsi_vertices": zero,
            "loss_hsi_projected_joints2d": zero,
            "loss_hsi_depth_teacher": zero,
            "loss_hsi_smpl_scale_teacher": zero,
            "loss_hsi_anchor_depth": zero,
            "loss_hsi_anchor_scene_xyz": zero,
            "loss_hsi_delta_reg": zero,
            "loss_hsi_no_worse": zero,
            "loss_hsi_gate_reg": zero,
            "loss_hsi_foot_contact": zero,
            "loss_hsi_foot_sole_contact": zero,
            "loss_hsi_support_plane_contact": zero,
            "loss_hsi_teacher_pose": zero,
            "loss_hsi_teacher_betas": zero,
            "loss_hsi_teacher_transl": zero,
            "loss_hsi_teacher_joints": zero,
            "loss_hsi_teacher_vertices": zero,
            "loss_hsi_teacher_scene_affine": zero,
            "loss_hsi_pose_velocity": zero,
            "loss_hsi_betas_velocity": zero,
            "loss_hsi_transl_velocity": zero,
            "loss_hsi_joints_velocity": zero,
            "loss_hsi_joints_acceleration": zero,
            "loss_hsi_temporal_no_worse": zero,
            "loss_hsi_foot_sliding": zero,
            "loss_hsi_scene_scale_temporal": zero,
            "loss_hsi_scene_scale_sequence": zero,
            "loss_hsi_scene_bias_temporal": zero,
            "loss_hsi_scene_bias_sequence": zero,
            "loss_hsi_contact": zero,
            "metric_hsi_joints3d_l1": zero.detach(),
            "metric_hsi_vertices_l1": zero.detach(),
            "metric_hsi_base_transl_l1": zero.detach(),
            "metric_hsi_refined_transl_l1": zero.detach(),
            "metric_hsi_transl_l1_delta": zero.detach(),
            "metric_hsi_ray_delta_l1": zero.detach(),
            "metric_hsi_ray_delta_base_l1": zero.detach(),
            "metric_hsi_ray_delta_refined_l1": zero.detach(),
            "metric_hsi_ray_delta_l1_delta": zero.detach(),
            "metric_hsi_ray_delta_expected_abs": zero.detach(),
            "metric_hsi_ray_delta_pred_abs": zero.detach(),
            "metric_hsi_ray_delta_sign_acc": zero.detach(),
            "metric_hsi_tangent_delta_base_l1": zero.detach(),
            "metric_hsi_tangent_delta_refined_l1": zero.detach(),
            "metric_hsi_tangent_delta_l1_delta": zero.detach(),
            "metric_hsi_align_base_point_l1": zero.detach(),
            "metric_hsi_align_refined_point_l1": zero.detach(),
            "metric_hsi_align_point_l1_delta": zero.detach(),
            "metric_hsi_align_delta_l1": zero.detach(),
            "metric_hsi_align_gate_mean": zero.detach(),
            "metric_hsi_align_valid_ratio": zero.detach(),
            "metric_hsi_transl_l2_median": zero.detach(),
            "metric_hsi_base_transl_l2_median": zero.detach(),
            "metric_hsi_transl_l2_p90": zero.detach(),
            "metric_hsi_transl_improvement_rate": zero.detach(),
            "metric_hsi_transl_noisy_fraction": zero.detach(),
            "metric_hsi_base_transl_noisy_l2_median": zero.detach(),
            "metric_hsi_transl_noisy_l2_median": zero.detach(),
            "metric_hsi_transl_noisy_improvement_rate": zero.detach(),
            "metric_hsi_transl_clean_displacement_mean_m": zero.detach(),
            "metric_hsi_transl_clean_gate_mean": zero.detach(),
            "metric_hsi_transl_noisy_gate_mean": zero.detach(),
            "metric_hsi_contact_float_p95_m": zero.detach(),
            "metric_hsi_contact_penetration_p95_m": zero.detach(),
            "metric_hsi_contact_false_pull_rate": zero.detach(),
            "metric_hsi_contact_base_abs_p95_m": zero.detach(),
            "metric_hsi_contact_refined_abs_p95_m": zero.detach(),
            "metric_hsi_contact_swing_displacement_mean_m": zero.detach(),
            "metric_hsi_contact_contact_gate_mean": zero.detach(),
            "metric_hsi_contact_swing_gate_mean": zero.detach(),
            "metric_stage2_selection": zero.detach(),
            "metric_stage3_selection": zero.detach(),
            "metric_hsi_no_worse_ratio": zero.detach(),
            "metric_hsi_joint_error_delta": zero.detach(),
            "metric_hsi_gate_mean": zero.detach(),
            "metric_hsi_foot_float_m": zero.detach(),
            "metric_hsi_foot_penetration_m": zero.detach(),
            "metric_hsi_foot_contact_count": zero.detach(),
            "metric_hsi_foot_sole_float_m": zero.detach(),
            "metric_hsi_foot_sole_penetration_m": zero.detach(),
            "metric_hsi_foot_sole_contact_count": zero.detach(),
            "metric_hsi_support_plane_float_m": zero.detach(),
            "metric_hsi_support_plane_penetration_m": zero.detach(),
            "metric_hsi_support_plane_signed_m": zero.detach(),
            "metric_hsi_support_plane_contact_count": zero.detach(),
            "metric_hsi_teacher_transl_l1": zero.detach(),
            "metric_hsi_teacher_vertices_l1": zero.detach(),
            "metric_hsi_teacher_scene_affine_l1": zero.detach(),
            "metric_hsi_temporal_pair_count": zero.detach(),
            "metric_hsi_temporal_triple_count": zero.detach(),
            "metric_hsi_pose_velocity_l1": zero.detach(),
            "metric_hsi_betas_velocity_l1": zero.detach(),
            "metric_hsi_transl_velocity_l1": zero.detach(),
            "metric_hsi_joints_velocity_l1": zero.detach(),
            "metric_hsi_joints_acceleration_l1": zero.detach(),
            "metric_hsi_temporal_no_worse_ratio": zero.detach(),
            "metric_hsi_temporal_no_worse_l1": zero.detach(),
            "metric_hsi_foot_sliding_l1": zero.detach(),
            "metric_hsi_foot_sliding_contact_count": zero.detach(),
            "metric_hsi_scene_log_scale_delta": zero.detach(),
            "metric_hsi_scene_log_scale_seq_abs": zero.detach(),
            "metric_hsi_scene_bias_delta": zero.detach(),
            "metric_hsi_scene_bias_seq_abs": zero.detach(),
            "metric_hsi_anchor_depth_l1": zero.detach(),
            "metric_hsi_anchor_scene_xyz_l1": zero.detach(),
            "metric_hsi_delta_reg": zero.detach(),
            "metric_hsi_depth_teacher_l1": zero.detach(),
            "metric_hsi_depth_teacher_valid_pixels": zero.detach(),
            "metric_hsi_depth_teacher_roi_used": zero.detach(),
            "metric_hsi_smpl_scale_teacher_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_valid_points": zero.detach(),
            "metric_hsi_smpl_scale_teacher_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_log_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_rel_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_bias": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_bias": zero.detach(),
            "metric_hsi_contact_pos_frac": zero.detach(),
        }

    def _hsi_refined_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if "hsi_refined_pred_pose_6d" not in predictions:
            return self._zero_hsi_losses(predictions)
        refined_pose6d = _flatten_prediction(_require_prediction(predictions, "hsi_refined_pred_pose_6d"), unframed_ndim=3)
        refined_poses = _flatten_prediction(_require_prediction(predictions, "hsi_refined_pred_poses"), unframed_ndim=3)
        refined_betas = _flatten_prediction(_require_prediction(predictions, "hsi_refined_pred_betas"), unframed_ndim=3)
        refined_transl = _flatten_prediction(_require_prediction(predictions, "hsi_refined_pred_transl_cam"), unframed_ndim=3)

        pred_pose = refined_pose6d[frame_idx, src_idx]
        pred_betas = refined_betas[frame_idx, src_idx]
        pred_transl = refined_transl[frame_idx, src_idx]
        target_pose = matched["pose_6d"].to(device=pred_pose.device, dtype=pred_pose.dtype)
        target_betas = matched["betas"].to(device=pred_betas.device, dtype=pred_betas.dtype)
        target_transl = matched["transl_cam"].to(device=pred_transl.device, dtype=pred_transl.dtype)

        losses: dict[str, torch.Tensor] = {
            "loss_hsi_pose": F.l1_loss(pred_pose, target_pose),
            "loss_hsi_betas": F.l1_loss(pred_betas, target_betas),
            "loss_hsi_transl_cam": F.l1_loss(pred_transl, target_transl),
            "loss_hsi_transl_clean_identity": pred_transl.sum() * 0.0,
            "loss_hsi_transl_noise_gate": pred_transl.sum() * 0.0,
        }
        base_pose_value = predictions.get("hsi_contact_base_pred_pose_6d", predictions.get("pred_pose_6d"))
        base_transl_value = predictions.get("hsi_contact_base_pred_transl_cam", predictions.get("pred_transl_cam"))
        base_pose6d = _flatten_prediction(base_pose_value, unframed_ndim=3)
        base_betas = _flatten_prediction(_require_prediction(predictions, "pred_betas"), unframed_ndim=3)
        base_transl = _flatten_prediction(base_transl_value, unframed_ndim=3)
        delta_reg = (
            F.smooth_l1_loss(pred_transl, base_transl[frame_idx, src_idx].detach())
            + 0.01 * F.smooth_l1_loss(pred_pose, base_pose6d[frame_idx, src_idx].detach())
            + 0.01 * F.smooth_l1_loss(pred_betas, base_betas[frame_idx, src_idx].detach())
        )
        losses["loss_hsi_delta_reg"] = delta_reg
        losses["metric_hsi_delta_reg"] = delta_reg.detach()
        smpl = self._get_smpl_layer(pred_betas.device)
        pred_aa = refined_poses[frame_idx, src_idx].reshape(-1, 72)
        gt_aa = rot6d_to_axis_angle(target_pose).reshape(-1, 72)
        base_aa = rot6d_to_axis_angle(base_pose6d[frame_idx, src_idx].detach()).reshape(-1, 72)
        base_betas_matched = base_betas[frame_idx, src_idx].detach()
        base_transl_matched = base_transl[frame_idx, src_idx].detach()
        pred_vertices, pred_joints = smpl(pred_aa.float(), pred_betas.float())
        gt_vertices, gt_joints = smpl(gt_aa.float(), target_betas.float())
        base_vertices, base_joints = smpl(base_aa.float(), base_betas_matched.float())
        pred_joints_cam = pred_joints[:, :24].to(dtype=pred_betas.dtype) + pred_transl[:, None, :]
        gt_joints_cam = gt_joints[:, :24].to(dtype=pred_betas.dtype) + target_transl[:, None, :]
        base_joints_cam = base_joints[:, :24].to(dtype=pred_betas.dtype) + base_transl_matched[:, None, :]
        pred_vertices_cam = pred_vertices.to(dtype=pred_betas.dtype) + pred_transl[:, None, :]
        gt_vertices_cam = gt_vertices.to(dtype=pred_betas.dtype) + target_transl[:, None, :]
        base_vertices_cam = base_vertices.to(dtype=pred_betas.dtype) + base_transl_matched[:, None, :]
        losses["loss_hsi_joints3d"] = F.l1_loss(pred_joints_cam, gt_joints_cam)
        losses["metric_hsi_joints3d_l1"] = losses["loss_hsi_joints3d"].detach()
        losses["loss_hsi_vertices"] = F.l1_loss(pred_vertices_cam, gt_vertices_cam)
        losses["metric_hsi_vertices_l1"] = losses["loss_hsi_vertices"].detach()
        base_transl_l1 = F.l1_loss(base_transl_matched, target_transl)
        refined_transl_l1 = F.l1_loss(pred_transl, target_transl)
        losses["metric_hsi_base_transl_l1"] = base_transl_l1.detach()
        losses["metric_hsi_refined_transl_l1"] = refined_transl_l1.detach()
        losses["metric_hsi_transl_l1_delta"] = (refined_transl_l1 - base_transl_l1).detach()
        base_transl_l2_items = torch.linalg.norm(base_transl_matched - target_transl, dim=-1)
        refined_transl_l2_items = torch.linalg.norm(pred_transl - target_transl, dim=-1)
        losses["metric_hsi_transl_l2_median"] = refined_transl_l2_items.median().detach()
        losses["metric_hsi_base_transl_l2_median"] = base_transl_l2_items.median().detach()
        losses["metric_hsi_transl_l2_p90"] = torch.quantile(refined_transl_l2_items.float(), 0.90).to(dtype=pred_transl.dtype).detach()
        losses["metric_hsi_transl_improvement_rate"] = (
            refined_transl_l2_items < base_transl_l2_items
        ).to(dtype=pred_transl.dtype).mean().detach()
        clean_value = predictions.get("transl_noise_is_clean")
        clean_flat = _flatten_prediction(clean_value, unframed_ndim=3) if isinstance(clean_value, torch.Tensor) else None
        if clean_flat is not None:
            clean_items = clean_flat[frame_idx, src_idx, 0] > 0.5
            noisy_items = ~clean_items
            align_gate_value = predictions.get("hsi_align_gate")
            align_gate_flat = (
                _flatten_prediction(align_gate_value, unframed_ndim=3)
                if isinstance(align_gate_value, torch.Tensor)
                else None
            )
            matched_align_gate = align_gate_flat[frame_idx, src_idx, 0] if align_gate_flat is not None else None
            if matched_align_gate is not None:
                gate_target = noisy_items.to(dtype=matched_align_gate.dtype)
                losses["loss_hsi_transl_noise_gate"] = F.binary_cross_entropy(
                    matched_align_gate.clamp(min=1e-6, max=1.0 - 1e-6), gate_target
                )
            losses["metric_hsi_transl_noisy_fraction"] = noisy_items.to(dtype=pred_transl.dtype).mean().detach()
            if noisy_items.any():
                noisy_base = base_transl_l2_items[noisy_items]
                noisy_refined = refined_transl_l2_items[noisy_items]
                losses["metric_hsi_base_transl_noisy_l2_median"] = noisy_base.median().detach()
                losses["metric_hsi_transl_noisy_l2_median"] = noisy_refined.median().detach()
                losses["metric_hsi_transl_noisy_improvement_rate"] = (
                    noisy_refined < noisy_base
                ).to(dtype=pred_transl.dtype).mean().detach()
                if matched_align_gate is not None:
                    losses["metric_hsi_transl_noisy_gate_mean"] = matched_align_gate[noisy_items].mean().detach()
            if clean_items.any():
                clean_displacement = torch.linalg.norm(
                    pred_transl[clean_items] - base_transl_matched[clean_items], dim=-1
                )
                losses["loss_hsi_transl_clean_identity"] = F.smooth_l1_loss(
                    pred_transl[clean_items], base_transl_matched[clean_items], beta=0.005
                )
                losses["metric_hsi_transl_clean_displacement_mean_m"] = clean_displacement.mean().detach()
                if matched_align_gate is not None:
                    losses["metric_hsi_transl_clean_gate_mean"] = matched_align_gate[clean_items].mean().detach()
        ray = F.normalize(base_transl_matched, dim=-1, eps=1e-6)
        expected_ray_delta = ((target_transl - base_transl_matched) * ray).sum(dim=-1, keepdim=True)
        predicted_ray_delta = ((pred_transl - base_transl_matched) * ray).sum(dim=-1, keepdim=True)
        ray_delta_err = (predicted_ray_delta - expected_ray_delta).abs()
        ray_base_l1 = expected_ray_delta.abs().mean()
        ray_refined_l1 = ray_delta_err.mean()
        losses["loss_hsi_ray_delta"] = F.smooth_l1_loss(predicted_ray_delta, expected_ray_delta)
        losses["metric_hsi_ray_delta_l1"] = ray_refined_l1.detach()
        losses["metric_hsi_ray_delta_base_l1"] = ray_base_l1.detach()
        losses["metric_hsi_ray_delta_refined_l1"] = ray_refined_l1.detach()
        losses["metric_hsi_ray_delta_l1_delta"] = (ray_refined_l1 - ray_base_l1).detach()
        losses["metric_hsi_ray_delta_expected_abs"] = ray_base_l1.detach()
        losses["metric_hsi_ray_delta_pred_abs"] = predicted_ray_delta.abs().mean().detach()
        sign_valid = expected_ray_delta.abs() > 1e-6
        if sign_valid.any():
            sign_match = torch.sign(predicted_ray_delta[sign_valid]) == torch.sign(expected_ray_delta[sign_valid])
            losses["metric_hsi_ray_delta_sign_acc"] = sign_match.to(dtype=pred_transl.dtype).mean().detach()
        else:
            losses["metric_hsi_ray_delta_sign_acc"] = pred_transl.sum().detach() * 0.0

        expected_delta = target_transl - base_transl_matched
        predicted_delta = pred_transl - base_transl_matched
        expected_tangent = expected_delta - (expected_delta * ray).sum(dim=-1, keepdim=True) * ray
        predicted_tangent = predicted_delta - (predicted_delta * ray).sum(dim=-1, keepdim=True) * ray
        tangent_error = predicted_tangent - expected_tangent
        losses["loss_hsi_tangent_delta"] = F.smooth_l1_loss(
            predicted_tangent, expected_tangent, beta=0.01
        )
        tangent_base_l1 = expected_tangent.abs().mean()
        tangent_refined_l1 = tangent_error.abs().mean()
        losses["metric_hsi_tangent_delta_base_l1"] = tangent_base_l1.detach()
        losses["metric_hsi_tangent_delta_refined_l1"] = tangent_refined_l1.detach()
        losses["metric_hsi_tangent_delta_l1_delta"] = (tangent_refined_l1 - tangent_base_l1).detach()

        losses["loss_hsi_align_point"] = _optional_prediction_loss(predictions, "loss_hsi_align_point", pred_transl)
        losses["loss_hsi_align_delta_reg"] = _optional_prediction_loss(predictions, "loss_hsi_align_delta_reg", pred_transl)
        losses["loss_hsi_align_no_worse"] = _optional_prediction_loss(predictions, "loss_hsi_align_no_worse", pred_transl)
        losses["metric_hsi_align_base_point_l1"] = _optional_prediction_metric(
            predictions, "hsi_align_base_point_l1", pred_transl
        )
        losses["metric_hsi_align_refined_point_l1"] = _optional_prediction_metric(
            predictions, "hsi_align_refined_point_l1", pred_transl
        )
        losses["metric_hsi_align_point_l1_delta"] = _optional_prediction_metric(
            predictions, "hsi_align_point_l1_delta", pred_transl
        )
        align_delta = predictions.get("hsi_align_delta_transl_cam")
        if isinstance(align_delta, torch.Tensor):
            losses["metric_hsi_align_delta_l1"] = align_delta.abs().mean().detach()
        else:
            losses["metric_hsi_align_delta_l1"] = pred_transl.sum().detach() * 0.0
        align_gate = predictions.get("hsi_align_gate")
        if isinstance(align_gate, torch.Tensor):
            losses["metric_hsi_align_gate_mean"] = align_gate.float().mean().detach()
        else:
            losses["metric_hsi_align_gate_mean"] = pred_transl.sum().detach() * 0.0
        align_valid = predictions.get("hsi_align_valid_ratio")
        if isinstance(align_valid, torch.Tensor):
            losses["metric_hsi_align_valid_ratio"] = align_valid.float().mean().detach()
        else:
            losses["metric_hsi_align_valid_ratio"] = pred_transl.sum().detach() * 0.0

        hsi_joint_err = torch.linalg.norm(pred_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1)
        base_joint_err = torch.linalg.norm(base_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1).detach()
        hsi_vert_err = torch.linalg.norm(pred_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1)
        base_vert_err = torch.linalg.norm(base_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1).detach()
        margin = float(self.hsi_no_worse_margin_m)
        no_worse = F.relu(hsi_joint_err - base_joint_err - margin) + 0.5 * F.relu(hsi_vert_err - base_vert_err - margin)
        losses["loss_hsi_no_worse"] = no_worse.mean()
        losses["metric_hsi_no_worse_ratio"] = (hsi_joint_err > (base_joint_err + margin)).to(dtype=pred_betas.dtype).mean().detach()
        losses["metric_hsi_joint_error_delta"] = (hsi_joint_err - base_joint_err).mean().detach()
        losses["metric_stage2_selection"] = (
            losses["metric_hsi_transl_l2_median"]
            + 0.25 * losses["metric_hsi_transl_l2_p90"]
            + 2.0 * F.relu(losses["metric_hsi_joint_error_delta"])
        ).detach()
        contact_losses = self._hsi_contact_refine_losses(
            predictions=predictions,
            frame_idx=frame_idx,
            src_idx=src_idx,
            matched=matched,
            pred_pose=pred_pose,
            target_pose=target_pose,
            pred_vertices_cam=pred_vertices_cam,
            base_vertices_cam=base_vertices_cam,
        )
        losses.update(contact_losses)
        losses["metric_stage3_selection"] = (
            losses["metric_stage3_selection"]
            + 2.0 * F.relu(losses["metric_hsi_joint_error_delta"])
            + 2.0 * F.relu(losses["metric_hsi_transl_l1_delta"])
        ).detach()
        gate = predictions.get("hsi_refine_gate")
        if gate is not None:
            flat_gate = _flatten_prediction(gate, unframed_ndim=3)
            matched_gate = flat_gate[frame_idx, src_idx, 0] if flat_gate is not None else pred_transl.new_zeros(pred_transl.shape[0])
            losses["loss_hsi_gate_reg"] = matched_gate.mean()
            losses["metric_hsi_gate_mean"] = matched_gate.detach().mean()
        else:
            losses["loss_hsi_gate_reg"] = pred_transl.sum() * 0.0
            losses["metric_hsi_gate_mean"] = losses["loss_hsi_gate_reg"].detach()

        if "hsi_intrinsics_override" in predictions or "pose_enc" in predictions:
            if "hsi_intrinsics_override" in predictions:
                intrinsics = _flatten_batch_intrinsics(
                    _require_prediction(predictions, "hsi_intrinsics_override"),
                    device=pred_joints_cam.device,
                    dtype=pred_joints_cam.dtype,
                )
            else:
                intrinsics = _flatten_intrinsics(_require_prediction(predictions, "pose_enc"), self._projection_image_hw())
            pred_2d = _normalize_points_2d(_project_points(pred_joints_cam, intrinsics[frame_idx].to(dtype=pred_joints_cam.dtype)), self._projection_image_hw())
            gt_2d = _normalize_points_2d(_project_points(gt_joints_cam, intrinsics[frame_idx].to(dtype=gt_joints_cam.dtype)), self._projection_image_hw())
            valid = (pred_joints_cam[..., 2] > 1e-4) & (gt_joints_cam[..., 2] > 1e-4)
            losses["loss_hsi_projected_joints2d"] = F.l1_loss(pred_2d[valid], gt_2d[valid]) if valid.any() else pred_joints_cam.sum() * 0.0
        else:
            losses["loss_hsi_projected_joints2d"] = pred_joints_cam.sum() * 0.0

        depth_losses = self._hsi_depth_losses(
            predictions,
            batch,
            frame_idx,
            src_idx,
            pred_joints_cam,
            gt_joints_cam,
            pred_vertices_cam,
            gt_vertices_cam,
        )
        losses.update(depth_losses)
        losses.update(
            self._hsi_teacher_losses(
                predictions=predictions,
                frame_idx=frame_idx,
                src_idx=src_idx,
                pred_pose=pred_pose,
                pred_betas=pred_betas,
                pred_transl=pred_transl,
                pred_joints_cam=pred_joints_cam,
                pred_vertices_cam=pred_vertices_cam,
                smpl=smpl,
            )
        )
        losses.update(
            self._hsi_temporal_losses(
                predictions=predictions,
                batch=batch,
                frame_idx=frame_idx,
                matched=matched,
                pred_pose=pred_pose,
                target_pose=target_pose,
                pred_betas=pred_betas,
                target_betas=target_betas,
                pred_transl=pred_transl,
                target_transl=target_transl,
                pred_joints_cam=pred_joints_cam,
                gt_joints_cam=gt_joints_cam,
                base_transl=base_transl_matched,
                base_joints_cam=base_joints_cam,
            )
        )
        return losses

    def _hsi_contact_refine_losses(
        self,
        predictions: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor,
        pred_vertices_cam: torch.Tensor,
        base_vertices_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_vertices_cam.sum() * 0.0
        out = {
            "loss_hsi_contact_refine_plane": zero,
            "loss_hsi_contact_refine_pose": zero,
            "loss_hsi_contact_refine_class": zero,
            "loss_hsi_contact_refine_no_worse": zero,
            "loss_hsi_contact_refine_swing_no_pull": zero,
            "metric_hsi_contact_float_p95_m": zero.detach(),
            "metric_hsi_contact_penetration_p95_m": zero.detach(),
            "metric_hsi_contact_false_pull_rate": zero.detach(),
            "metric_hsi_contact_base_abs_p95_m": zero.detach(),
            "metric_hsi_contact_refined_abs_p95_m": zero.detach(),
            "metric_hsi_contact_swing_displacement_mean_m": zero.detach(),
            "metric_hsi_contact_contact_gate_mean": zero.detach(),
            "metric_hsi_contact_swing_gate_mean": zero.detach(),
            "metric_stage3_selection": zero.detach(),
        }
        if "contact_teacher_valid" not in matched:
            return out
        teacher_valid = matched["contact_teacher_valid"].bool()
        contact_label = matched["contact_label"].bool()
        plane_center = matched["contact_plane_center_cam"].to(device=pred_vertices_cam.device, dtype=pred_vertices_cam.dtype)
        plane_normal = matched["contact_plane_normal_cam"].to(device=pred_vertices_cam.device, dtype=pred_vertices_cam.dtype)
        target_signed = matched["contact_signed_distance_m"].to(device=pred_vertices_cam.device, dtype=pred_vertices_cam.dtype)
        sole_lr = self._get_foot_sole_indices_lr(pred_vertices_cam.device, count_per_foot=48)
        pred_foot = pred_vertices_cam[:, sole_lr].mean(dim=-2)
        base_foot = base_vertices_cam[:, sole_lr].mean(dim=-2)
        pred_signed = ((pred_foot - plane_center) * plane_normal).sum(dim=-1)
        base_signed = ((base_foot - plane_center) * plane_normal).sum(dim=-1)
        contact_valid = teacher_valid & contact_label & torch.isfinite(pred_signed)
        if contact_valid.any():
            pred_error = pred_signed - target_signed
            base_error = base_signed.detach() - target_signed
            out["loss_hsi_contact_refine_plane"] = F.smooth_l1_loss(
                pred_error[contact_valid], torch.zeros_like(pred_error[contact_valid]), beta=0.01
            )
            out["loss_hsi_contact_refine_no_worse"] = F.relu(
                pred_error[contact_valid].abs() - base_error[contact_valid].abs() - 0.002
            ).mean()
            float_amount = F.relu(pred_error[contact_valid])
            penetration_amount = F.relu(-pred_error[contact_valid])
            out["metric_hsi_contact_float_p95_m"] = torch.quantile(float_amount.float(), 0.95).to(dtype=pred_error.dtype).detach()
            out["metric_hsi_contact_penetration_p95_m"] = torch.quantile(
                penetration_amount.float(), 0.95
            ).to(dtype=pred_error.dtype).detach()
            out["metric_hsi_contact_base_abs_p95_m"] = torch.quantile(
                base_error[contact_valid].abs().float(), 0.95
            ).to(dtype=pred_error.dtype).detach()
            out["metric_hsi_contact_refined_abs_p95_m"] = torch.quantile(
                pred_error[contact_valid].abs().float(), 0.95
            ).to(dtype=pred_error.dtype).detach()
        lower = torch.tensor([1, 2, 4, 5, 7, 8], device=pred_pose.device, dtype=torch.long)
        pred_lower = pred_pose.reshape(-1, 24, 6)[:, lower]
        target_lower = target_pose.reshape(-1, 24, 6)[:, lower]
        out["loss_hsi_contact_refine_pose"] = F.smooth_l1_loss(pred_lower, target_lower, beta=0.02)
        logits = predictions.get("hsi_contact_foot_logits")
        if isinstance(logits, torch.Tensor):
            flat_logits = _flatten_prediction(logits, unframed_ndim=3)[frame_idx, src_idx]
            if teacher_valid.any():
                out["loss_hsi_contact_refine_class"] = F.binary_cross_entropy_with_logits(
                    flat_logits[teacher_valid], contact_label[teacher_valid].to(dtype=flat_logits.dtype)
                )
            probability = torch.sigmoid(flat_logits)
            if contact_valid.any():
                out["metric_hsi_contact_contact_gate_mean"] = probability[contact_valid].mean().detach()
        swing_valid = teacher_valid & ~contact_label
        if swing_valid.any():
            swing_displacement = torch.linalg.norm(pred_foot - base_foot.detach(), dim=-1)
            out["loss_hsi_contact_refine_swing_no_pull"] = F.smooth_l1_loss(
                swing_displacement[swing_valid],
                torch.zeros_like(swing_displacement[swing_valid]),
                beta=0.005,
            )
            out["metric_hsi_contact_swing_displacement_mean_m"] = swing_displacement[swing_valid].mean().detach()
            out["metric_hsi_contact_false_pull_rate"] = (
                swing_displacement[swing_valid] > 0.02
            ).to(dtype=pred_vertices_cam.dtype).mean().detach()
            if isinstance(logits, torch.Tensor):
                out["metric_hsi_contact_swing_gate_mean"] = torch.sigmoid(flat_logits)[swing_valid].mean().detach()
        out["metric_stage3_selection"] = (
            out["metric_hsi_contact_float_p95_m"]
            + 2.0 * out["metric_hsi_contact_penetration_p95_m"]
            + 0.5 * out["metric_hsi_contact_false_pull_rate"]
        ).detach()
        return out

    def _hsi_temporal_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_betas: torch.Tensor,
        pred_transl: torch.Tensor,
        target_transl: torch.Tensor,
        pred_joints_cam: torch.Tensor,
        gt_joints_cam: torch.Tensor,
        base_transl: torch.Tensor,
        base_joints_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_transl.sum() * 0.0
        out = {
            "loss_hsi_pose_velocity": zero,
            "loss_hsi_betas_velocity": zero,
            "loss_hsi_transl_velocity": zero,
            "loss_hsi_joints_velocity": zero,
            "loss_hsi_joints_acceleration": zero,
            "loss_hsi_temporal_no_worse": zero,
            "loss_hsi_foot_sliding": zero,
            "loss_hsi_scene_scale_temporal": zero,
            "loss_hsi_scene_scale_sequence": zero,
            "loss_hsi_scene_bias_temporal": zero,
            "loss_hsi_scene_bias_sequence": zero,
            "metric_hsi_temporal_pair_count": zero.detach(),
            "metric_hsi_temporal_triple_count": zero.detach(),
            "metric_hsi_pose_velocity_l1": zero.detach(),
            "metric_hsi_betas_velocity_l1": zero.detach(),
            "metric_hsi_transl_velocity_l1": zero.detach(),
            "metric_hsi_joints_velocity_l1": zero.detach(),
            "metric_hsi_joints_acceleration_l1": zero.detach(),
            "metric_hsi_temporal_no_worse_ratio": zero.detach(),
            "metric_hsi_temporal_no_worse_l1": zero.detach(),
            "metric_hsi_foot_sliding_l1": zero.detach(),
            "metric_hsi_foot_sliding_contact_count": zero.detach(),
            "metric_hsi_scene_log_scale_delta": zero.detach(),
            "metric_hsi_scene_log_scale_seq_abs": zero.detach(),
            "metric_hsi_scene_bias_delta": zero.detach(),
            "metric_hsi_scene_bias_seq_abs": zero.detach(),
        }
        num_frames = _infer_sequence_length(batch, predictions)
        if num_frames >= 2:
            out.update(self._hsi_person_temporal_losses(
                predictions=predictions,
                batch=batch,
                frame_idx=frame_idx,
                matched=matched,
                num_frames=num_frames,
                pred_pose=pred_pose,
                target_pose=target_pose,
                pred_betas=pred_betas,
                target_betas=target_betas,
                pred_transl=pred_transl,
                target_transl=target_transl,
                pred_joints_cam=pred_joints_cam,
                gt_joints_cam=gt_joints_cam,
                base_transl=base_transl,
                base_joints_cam=base_joints_cam,
                zero=zero,
            ))
        out.update(self._hsi_scene_temporal_losses(predictions, zero))
        return out

    def _hsi_person_temporal_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        matched: dict[str, torch.Tensor],
        num_frames: int,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_betas: torch.Tensor,
        pred_transl: torch.Tensor,
        target_transl: torch.Tensor,
        pred_joints_cam: torch.Tensor,
        gt_joints_cam: torch.Tensor,
        base_transl: torch.Tensor,
        base_joints_cam: torch.Tensor,
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {
            "loss_hsi_pose_velocity": zero,
            "loss_hsi_betas_velocity": zero,
            "loss_hsi_transl_velocity": zero,
            "loss_hsi_joints_velocity": zero,
            "loss_hsi_joints_acceleration": zero,
            "loss_hsi_temporal_no_worse": zero,
            "loss_hsi_foot_sliding": zero,
            "metric_hsi_temporal_pair_count": zero.detach(),
            "metric_hsi_temporal_triple_count": zero.detach(),
            "metric_hsi_pose_velocity_l1": zero.detach(),
            "metric_hsi_betas_velocity_l1": zero.detach(),
            "metric_hsi_transl_velocity_l1": zero.detach(),
            "metric_hsi_joints_velocity_l1": zero.detach(),
            "metric_hsi_joints_acceleration_l1": zero.detach(),
            "metric_hsi_temporal_no_worse_ratio": zero.detach(),
            "metric_hsi_temporal_no_worse_l1": zero.detach(),
            "metric_hsi_foot_sliding_l1": zero.detach(),
            "metric_hsi_foot_sliding_contact_count": zero.detach(),
        }
        person_ids = matched.get("person_ids")
        person_mask = matched.get("person_id_mask")
        if person_ids is None or person_mask is None or frame_idx.numel() == 0:
            return out
        valid = person_mask.bool() & (person_ids >= 0)
        if not valid.any():
            return out

        batch_ids = torch.div(frame_idx, int(num_frames), rounding_mode="floor")
        seq_ids = frame_idx % int(num_frames)
        groups: dict[tuple[int, int], dict[int, int]] = {}
        for item_idx in torch.nonzero(valid, as_tuple=False).flatten().detach().cpu().tolist():
            key = (int(batch_ids[item_idx].detach().cpu()), int(person_ids[item_idx].detach().cpu()))
            seq = int(seq_ids[item_idx].detach().cpu())
            groups.setdefault(key, {}).setdefault(seq, item_idx)

        pair_prev: list[int] = []
        pair_next: list[int] = []
        triple_prev: list[int] = []
        triple_mid: list[int] = []
        triple_next: list[int] = []
        for seq_map in groups.values():
            for seq in range(1, int(num_frames)):
                if seq - 1 in seq_map and seq in seq_map:
                    pair_prev.append(seq_map[seq - 1])
                    pair_next.append(seq_map[seq])
            for seq in range(1, int(num_frames) - 1):
                if seq - 1 in seq_map and seq in seq_map and seq + 1 in seq_map:
                    triple_prev.append(seq_map[seq - 1])
                    triple_mid.append(seq_map[seq])
                    triple_next.append(seq_map[seq + 1])

        temporal_no_worse_terms: list[torch.Tensor] = []
        temporal_no_worse_flags: list[torch.Tensor] = []
        velocity_margin = float(self.hsi_temporal_no_worse_margin_m)
        accel_margin = float(self.hsi_temporal_no_worse_accel_margin_m)

        if pair_prev:
            prev = torch.tensor(pair_prev, dtype=torch.long, device=pred_transl.device)
            curr = torch.tensor(pair_next, dtype=torch.long, device=pred_transl.device)
            pose_delta = _velocity_residual(pred_pose, target_pose, prev, curr)
            betas_delta = _velocity_residual(pred_betas, target_betas, prev, curr)
            transl_delta = _velocity_residual(pred_transl, target_transl, prev, curr)
            joints_delta = _velocity_residual(pred_joints_cam, gt_joints_cam, prev, curr)
            base_transl_delta = _velocity_residual(base_transl.detach(), target_transl, prev, curr)
            base_joints_delta = _velocity_residual(base_joints_cam.detach(), gt_joints_cam, prev, curr)
            out["loss_hsi_pose_velocity"] = _smooth_l1_abs(pose_delta.abs()).mean()
            out["loss_hsi_betas_velocity"] = _smooth_l1_abs(betas_delta.abs()).mean()
            out["loss_hsi_transl_velocity"] = _smooth_l1_abs(transl_delta.abs()).mean()
            out["loss_hsi_joints_velocity"] = _smooth_l1_abs(joints_delta.abs()).mean()

            hsi_transl_vel_err = torch.linalg.norm(transl_delta, dim=-1)
            base_transl_vel_err = torch.linalg.norm(base_transl_delta, dim=-1).detach()
            transl_vel_excess = F.relu(hsi_transl_vel_err - base_transl_vel_err - velocity_margin)
            temporal_no_worse_terms.append(transl_vel_excess)
            temporal_no_worse_flags.append(hsi_transl_vel_err > (base_transl_vel_err + velocity_margin))

            hsi_joints_vel_err = torch.linalg.norm(joints_delta, dim=-1).mean(dim=-1)
            base_joints_vel_err = torch.linalg.norm(base_joints_delta, dim=-1).mean(dim=-1).detach()
            joints_vel_excess = F.relu(hsi_joints_vel_err - base_joints_vel_err - velocity_margin)
            temporal_no_worse_terms.append(joints_vel_excess)
            temporal_no_worse_flags.append(hsi_joints_vel_err > (base_joints_vel_err + velocity_margin))

            foot_contact = self._matched_foot_contact_mask(
                predictions=predictions,
                batch=batch,
                frame_idx=frame_idx,
                gt_joints_cam=gt_joints_cam,
            )
            if foot_contact is not None:
                foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=pred_joints_cam.device)
                foot_residual = _velocity_residual(pred_joints_cam[:, foot_idx], gt_joints_cam[:, foot_idx], prev, curr)
                pair_contact = foot_contact[prev] & foot_contact[curr]
                if pair_contact.any():
                    foot_abs = foot_residual.abs()
                    out["loss_hsi_foot_sliding"] = _smooth_l1_abs(foot_abs[pair_contact]).mean()
                    out["metric_hsi_foot_sliding_l1"] = foot_abs[pair_contact].mean().detach()
                    out["metric_hsi_foot_sliding_contact_count"] = pred_transl.new_tensor(float(pair_contact.sum().detach().cpu())).detach()
            out["metric_hsi_temporal_pair_count"] = pred_transl.new_tensor(float(len(pair_prev))).detach()
            out["metric_hsi_pose_velocity_l1"] = pose_delta.abs().mean().detach()
            out["metric_hsi_betas_velocity_l1"] = betas_delta.abs().mean().detach()
            out["metric_hsi_transl_velocity_l1"] = transl_delta.abs().mean().detach()
            out["metric_hsi_joints_velocity_l1"] = joints_delta.abs().mean().detach()

        if triple_prev:
            prev = torch.tensor(triple_prev, dtype=torch.long, device=pred_transl.device)
            mid = torch.tensor(triple_mid, dtype=torch.long, device=pred_transl.device)
            nxt = torch.tensor(triple_next, dtype=torch.long, device=pred_transl.device)
            joints_acc_delta = _acceleration_residual(pred_joints_cam, gt_joints_cam, prev, mid, nxt)
            base_joints_acc_delta = _acceleration_residual(base_joints_cam.detach(), gt_joints_cam, prev, mid, nxt)
            out["loss_hsi_joints_acceleration"] = _smooth_l1_abs(joints_acc_delta.abs()).mean()
            hsi_joints_acc_err = torch.linalg.norm(joints_acc_delta, dim=-1).mean(dim=-1)
            base_joints_acc_err = torch.linalg.norm(base_joints_acc_delta, dim=-1).mean(dim=-1).detach()
            joints_acc_excess = F.relu(hsi_joints_acc_err - base_joints_acc_err - accel_margin)
            temporal_no_worse_terms.append(joints_acc_excess)
            temporal_no_worse_flags.append(hsi_joints_acc_err > (base_joints_acc_err + accel_margin))
            out["metric_hsi_temporal_triple_count"] = pred_transl.new_tensor(float(len(triple_prev))).detach()
            out["metric_hsi_joints_acceleration_l1"] = joints_acc_delta.abs().mean().detach()

        if temporal_no_worse_terms:
            temporal_no_worse = torch.cat([term.reshape(-1) for term in temporal_no_worse_terms], dim=0)
            temporal_no_worse_flags_tensor = torch.cat(
                [flag.reshape(-1).to(dtype=pred_transl.dtype) for flag in temporal_no_worse_flags],
                dim=0,
            )
            out["loss_hsi_temporal_no_worse"] = _smooth_l1_abs(temporal_no_worse, beta=0.01).mean()
            out["metric_hsi_temporal_no_worse_ratio"] = temporal_no_worse_flags_tensor.mean().detach()
            out["metric_hsi_temporal_no_worse_l1"] = temporal_no_worse.mean().detach()
        return out

    def _matched_foot_contact_mask(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        gt_joints_cam: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.hsi_foot_sliding_weight == 0.0:
            return None
        if "pose_enc" not in predictions or "gt_depth" not in batch:
            return None
        gt_depth = _canonical_depth(batch["gt_depth"].to(device=gt_joints_cam.device, dtype=gt_joints_cam.dtype))
        pred_depth = predictions.get("depth")
        if pred_depth is not None:
            pred_depth_hw = _canonical_depth(pred_depth)
            if gt_depth.shape[-2:] != pred_depth_hw.shape[-2:]:
                gt_depth = F.interpolate(
                    gt_depth.reshape(-1, 1, *gt_depth.shape[-2:]),
                    size=pred_depth_hw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(*gt_depth.shape[:2], *pred_depth_hw.shape[-2:])
        intrinsics = _flatten_intrinsics(_require_prediction(predictions, "pose_enc"), self._projection_image_hw())
        teacher_intrinsics = self._contact_teacher_intrinsics(
            predictions,
            batch,
            fallback_intrinsics=intrinsics,
            device=gt_joints_cam.device,
            dtype=gt_joints_cam.dtype,
        )
        foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=gt_joints_cam.device)
        gt_foot = gt_joints_cam[:, foot_idx]
        gt_projected = _project_points(gt_foot, teacher_intrinsics[frame_idx].to(dtype=gt_foot.dtype))
        gt_projected = _scale_points_to_depth(gt_projected, self._projection_image_hw(), gt_depth.shape[-2], gt_depth.shape[-1])
        sampled_gt, gt_valid = _sample_depth_at_points(gt_depth.reshape(-1, *gt_depth.shape[-2:]), gt_projected, frame_idx)
        return (torch.abs(sampled_gt - gt_foot[..., 2].to(dtype=sampled_gt.dtype)) < float(self.hsi_foot_sliding_contact_threshold_m)) & gt_valid

    def _hsi_scene_temporal_losses(
        self,
        predictions: dict[str, torch.Tensor],
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {
            "loss_hsi_scene_scale_temporal": zero,
            "loss_hsi_scene_scale_sequence": zero,
            "loss_hsi_scene_bias_temporal": zero,
            "loss_hsi_scene_bias_sequence": zero,
            "metric_hsi_scene_log_scale_delta": zero.detach(),
            "metric_hsi_scene_log_scale_seq_abs": zero.detach(),
            "metric_hsi_scene_bias_delta": zero.detach(),
            "metric_hsi_scene_bias_seq_abs": zero.detach(),
        }
        scale = predictions.get("hsi_scene_scale")
        bias = predictions.get("hsi_scene_depth_bias")
        if scale is None or scale.ndim < 3 or scale.shape[1] < 2:
            return out
        log_scale = torch.log(scale.float().clamp(min=1e-6))
        log_delta = log_scale[:, 1:] - log_scale[:, :-1]
        out["loss_hsi_scene_scale_temporal"] = _smooth_l1_abs(log_delta.abs()).mean()
        out["metric_hsi_scene_log_scale_delta"] = log_delta.abs().mean().detach()
        seq_log_scale = log_scale.median(dim=1, keepdim=True).values.detach()
        seq_log_abs = (log_scale - seq_log_scale).abs()
        out["loss_hsi_scene_scale_sequence"] = _smooth_l1_abs(seq_log_abs).mean()
        out["metric_hsi_scene_log_scale_seq_abs"] = seq_log_abs.mean().detach()
        if bias is not None and bias.ndim >= 3 and bias.shape[1] >= 2:
            bias_float = bias.float()
            bias_delta = bias_float[:, 1:] - bias_float[:, :-1]
            out["loss_hsi_scene_bias_temporal"] = _smooth_l1_abs(bias_delta.abs()).mean()
            out["metric_hsi_scene_bias_delta"] = bias_delta.abs().mean().detach()
            seq_bias = bias_float.median(dim=1, keepdim=True).values.detach()
            seq_bias_abs = (bias_float - seq_bias).abs()
            out["loss_hsi_scene_bias_sequence"] = _smooth_l1_abs(seq_bias_abs).mean()
            out["metric_hsi_scene_bias_seq_abs"] = seq_bias_abs.mean().detach()
        return out

    def _hsi_teacher_losses(
        self,
        predictions: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        pred_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        pred_transl: torch.Tensor,
        pred_joints_cam: torch.Tensor,
        pred_vertices_cam: torch.Tensor,
        smpl: SMPLLayer,
    ) -> dict[str, torch.Tensor]:
        zero = pred_transl.sum() * 0.0
        out = {
            "loss_hsi_teacher_pose": zero,
            "loss_hsi_teacher_betas": zero,
            "loss_hsi_teacher_transl": zero,
            "loss_hsi_teacher_joints": zero,
            "loss_hsi_teacher_vertices": zero,
            "loss_hsi_teacher_scene_affine": zero,
            "metric_hsi_teacher_transl_l1": zero.detach(),
            "metric_hsi_teacher_vertices_l1": zero.detach(),
            "metric_hsi_teacher_scene_affine_l1": zero.detach(),
        }
        required = (
            "teacher_hsi_refined_pred_pose_6d",
            "teacher_hsi_refined_pred_poses",
            "teacher_hsi_refined_pred_betas",
            "teacher_hsi_refined_pred_transl_cam",
        )
        if any(key not in predictions for key in required):
            return out

        teacher_pose6d = _flatten_prediction(predictions["teacher_hsi_refined_pred_pose_6d"], unframed_ndim=3)
        teacher_poses = _flatten_prediction(predictions["teacher_hsi_refined_pred_poses"], unframed_ndim=3)
        teacher_betas = _flatten_prediction(predictions["teacher_hsi_refined_pred_betas"], unframed_ndim=3)
        teacher_transl = _flatten_prediction(predictions["teacher_hsi_refined_pred_transl_cam"], unframed_ndim=3)
        if teacher_pose6d is None or teacher_poses is None or teacher_betas is None or teacher_transl is None:
            return out

        teacher_pose = teacher_pose6d[frame_idx, src_idx].to(device=pred_pose.device, dtype=pred_pose.dtype).detach()
        teacher_aa = teacher_poses[frame_idx, src_idx].reshape(-1, 72).to(device=pred_pose.device).detach()
        teacher_betas_matched = teacher_betas[frame_idx, src_idx].to(device=pred_betas.device, dtype=pred_betas.dtype).detach()
        teacher_transl_matched = teacher_transl[frame_idx, src_idx].to(device=pred_transl.device, dtype=pred_transl.dtype).detach()

        out["loss_hsi_teacher_pose"] = F.smooth_l1_loss(pred_pose, teacher_pose)
        out["loss_hsi_teacher_betas"] = F.smooth_l1_loss(pred_betas, teacher_betas_matched)
        teacher_transl_loss = F.smooth_l1_loss(pred_transl, teacher_transl_matched)
        out["loss_hsi_teacher_transl"] = teacher_transl_loss
        out["metric_hsi_teacher_transl_l1"] = torch.abs(pred_transl - teacher_transl_matched).mean().detach()

        if self.hsi_teacher_joints_weight != 0.0 or self.hsi_teacher_vertices_weight != 0.0:
            teacher_vertices, teacher_joints = smpl(teacher_aa.float(), teacher_betas_matched.float())
            teacher_joints_cam = teacher_joints[:, :24].to(dtype=pred_betas.dtype) + teacher_transl_matched[:, None, :]
            teacher_vertices_cam = teacher_vertices.to(dtype=pred_betas.dtype) + teacher_transl_matched[:, None, :]
            out["loss_hsi_teacher_joints"] = F.smooth_l1_loss(pred_joints_cam, teacher_joints_cam)
            teacher_vertices_loss = F.smooth_l1_loss(pred_vertices_cam, teacher_vertices_cam)
            out["loss_hsi_teacher_vertices"] = teacher_vertices_loss
            out["metric_hsi_teacher_vertices_l1"] = torch.abs(pred_vertices_cam - teacher_vertices_cam).mean().detach()

        if "teacher_hsi_scene_scale" in predictions and "teacher_hsi_scene_depth_bias" in predictions:
            scale = _flatten_prediction(predictions.get("hsi_scene_scale"), unframed_ndim=2)
            bias = _flatten_prediction(predictions.get("hsi_scene_depth_bias"), unframed_ndim=2)
            teacher_scale = _flatten_prediction(predictions.get("teacher_hsi_scene_scale"), unframed_ndim=2)
            teacher_bias = _flatten_prediction(predictions.get("teacher_hsi_scene_depth_bias"), unframed_ndim=2)
            if scale is not None and bias is not None and teacher_scale is not None and teacher_bias is not None:
                unique_frames = torch.unique(frame_idx)
                student_affine = torch.cat([scale[unique_frames], bias[unique_frames]], dim=-1)
                teacher_affine = torch.cat(
                    [
                        teacher_scale[unique_frames].to(device=scale.device, dtype=scale.dtype),
                        teacher_bias[unique_frames].to(device=bias.device, dtype=bias.dtype),
                    ],
                    dim=-1,
                ).detach()
                affine_l1 = F.smooth_l1_loss(student_affine, teacher_affine)
                out["loss_hsi_teacher_scene_affine"] = affine_l1
                out["metric_hsi_teacher_scene_affine_l1"] = torch.abs(student_affine - teacher_affine).mean().detach()
        return out

    def _hsi_depth_losses(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        src_idx: torch.Tensor,
        pred_joints_cam: torch.Tensor,
        gt_joints_cam: torch.Tensor,
        pred_vertices_cam: torch.Tensor,
        gt_vertices_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_joints_cam.sum() * 0.0
        out = {
            "loss_hsi_depth_teacher": zero,
            "loss_hsi_anchor_depth": zero,
            "loss_hsi_anchor_scene_xyz": zero,
            "loss_hsi_smpl_scale_teacher": zero,
            "loss_hsi_foot_contact": zero,
            "loss_hsi_foot_sole_contact": zero,
            "loss_hsi_support_plane_contact": zero,
            "loss_hsi_teacher_pose": zero,
            "loss_hsi_teacher_betas": zero,
            "loss_hsi_teacher_transl": zero,
            "loss_hsi_teacher_joints": zero,
            "loss_hsi_teacher_vertices": zero,
            "loss_hsi_teacher_scene_affine": zero,
            "loss_hsi_contact": zero,
            "metric_hsi_anchor_depth_l1": zero.detach(),
            "metric_hsi_anchor_scene_xyz_l1": zero.detach(),
            "metric_hsi_foot_float_m": zero.detach(),
            "metric_hsi_foot_penetration_m": zero.detach(),
            "metric_hsi_foot_contact_count": zero.detach(),
            "metric_hsi_foot_sole_float_m": zero.detach(),
            "metric_hsi_foot_sole_penetration_m": zero.detach(),
            "metric_hsi_foot_sole_contact_count": zero.detach(),
            "metric_hsi_support_plane_float_m": zero.detach(),
            "metric_hsi_support_plane_penetration_m": zero.detach(),
            "metric_hsi_support_plane_signed_m": zero.detach(),
            "metric_hsi_support_plane_contact_count": zero.detach(),
            "metric_hsi_teacher_transl_l1": zero.detach(),
            "metric_hsi_teacher_vertices_l1": zero.detach(),
            "metric_hsi_teacher_scene_affine_l1": zero.detach(),
            "metric_hsi_depth_teacher_l1": zero.detach(),
            "metric_hsi_depth_teacher_valid_pixels": zero.detach(),
            "metric_hsi_depth_teacher_roi_used": zero.detach(),
            "metric_hsi_smpl_scale_teacher_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_valid_points": zero.detach(),
            "metric_hsi_smpl_scale_teacher_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_log_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_rel_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_bias": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_bias": zero.detach(),
            "metric_hsi_contact_pos_frac": zero.detach(),
        }
        if "depth" not in predictions or "hsi_scene_scale" not in predictions or "hsi_scene_depth_bias" not in predictions:
            return out
        if "gt_depth" not in batch:
            return out
        depth = _canonical_depth(_require_prediction(predictions, "depth"))
        gt_depth = _canonical_depth(batch["gt_depth"].to(device=depth.device, dtype=depth.dtype))
        if gt_depth.shape[-2:] != depth.shape[-2:]:
            gt_depth = F.interpolate(
                gt_depth.reshape(-1, 1, *gt_depth.shape[-2:]),
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(*gt_depth.shape[:2], *depth.shape[-2:])
        scale = predictions["hsi_scene_scale"].to(device=depth.device, dtype=depth.dtype)
        bias = predictions["hsi_scene_depth_bias"].to(device=depth.device, dtype=depth.dtype)
        aligned_depth = depth * scale.squeeze(-1)[..., None, None] + bias.squeeze(-1)[..., None, None]
        valid_depth = torch.isfinite(gt_depth) & torch.isfinite(aligned_depth) & (gt_depth > 1e-6)
        if self.hsi_depth_teacher_max_m > 0:
            valid_depth = valid_depth & (gt_depth <= float(self.hsi_depth_teacher_max_m))
        roi_used = False
        if self.hsi_depth_teacher_use_human_roi and "gt_boxes" in batch and "boxes_mask" in batch:
            roi_mask = _human_roi_depth_mask(
                batch["gt_boxes"].to(device=depth.device, dtype=depth.dtype),
                batch["boxes_mask"].to(device=depth.device).bool(),
                depth.shape[-2],
                depth.shape[-1],
                expand=float(self.hsi_depth_teacher_roi_expand),
            )
            roi_valid = valid_depth & roi_mask
            if int(roi_valid.sum().detach().cpu()) >= int(self.hsi_depth_teacher_min_valid_pixels):
                valid_depth = roi_valid
                roi_used = True
        if valid_depth.any():
            abs_err = torch.abs(aligned_depth[valid_depth] - gt_depth[valid_depth])
            loss_err = abs_err
            if self.hsi_depth_teacher_error_clip_m > 0:
                loss_err = loss_err.clamp(max=float(self.hsi_depth_teacher_error_clip_m))
            depth_l1 = _smooth_l1_abs(loss_err).mean()
            out["loss_hsi_depth_teacher"] = depth_l1
            out["metric_hsi_depth_teacher_l1"] = abs_err.mean().detach()
            out["metric_hsi_depth_teacher_valid_pixels"] = depth.new_tensor(float(valid_depth.sum().detach().cpu())).detach()
            out["metric_hsi_depth_teacher_roi_used"] = depth.new_tensor(float(roi_used)).detach()

        if self.hsi_smpl_scale_teacher_weight != 0.0:
            out.update(
                self._hsi_smpl_scale_teacher_losses(
                    depth=depth,
                    gt_depth=gt_depth,
                    scale=scale,
                    bias=bias,
                    batch=batch,
                    frame_idx=frame_idx,
                    gt_joints_cam=gt_joints_cam,
                    gt_vertices_cam=gt_vertices_cam,
                )
            )

        if "pose_enc" not in predictions:
            return out
        intrinsics = _flatten_intrinsics(_require_prediction(predictions, "pose_enc"), self._projection_image_hw())
        contact_teacher_intrinsics = self._contact_teacher_intrinsics(
            predictions,
            batch,
            fallback_intrinsics=intrinsics,
            device=pred_joints_cam.device,
            dtype=pred_joints_cam.dtype,
        )
        projected = _project_points(pred_joints_cam, intrinsics[frame_idx].to(dtype=pred_joints_cam.dtype))
        projected = _scale_points_to_depth(projected, self._projection_image_hw(), aligned_depth.shape[-2], aligned_depth.shape[-1])
        sampled_aligned, valid = _sample_depth_at_points(aligned_depth.reshape(-1, *aligned_depth.shape[-2:]), projected, frame_idx)
        if valid.any():
            anchor_l1 = F.smooth_l1_loss(sampled_aligned[valid], pred_joints_cam[..., 2][valid].to(dtype=sampled_aligned.dtype))
            out["loss_hsi_anchor_depth"] = anchor_l1
            out["metric_hsi_anchor_depth_l1"] = anchor_l1.detach()

        if self.hsi_foot_contact_weight != 0.0:
            foot_out = self._hsi_foot_contact_losses(
                aligned_depth=aligned_depth,
                gt_depth=gt_depth,
                intrinsics=intrinsics,
                teacher_intrinsics=contact_teacher_intrinsics,
                frame_idx=frame_idx,
                pred_joints_cam=pred_joints_cam,
                gt_joints_cam=gt_joints_cam,
            )
            out.update(foot_out)

        if self.hsi_foot_sole_contact_weight != 0.0:
            sole_out = self._hsi_foot_sole_contact_losses(
                aligned_depth=aligned_depth,
                gt_depth=gt_depth,
                intrinsics=intrinsics,
                teacher_intrinsics=contact_teacher_intrinsics,
                frame_idx=frame_idx,
                pred_vertices_cam=pred_vertices_cam,
                gt_vertices_cam=gt_vertices_cam,
            )
            out.update(sole_out)

        if self.hsi_support_plane_contact_weight != 0.0:
            plane_out = self._hsi_support_plane_contact_losses(
                aligned_depth=aligned_depth,
                gt_depth=gt_depth,
                intrinsics=intrinsics,
                teacher_intrinsics=contact_teacher_intrinsics,
                frame_idx=frame_idx,
                pred_vertices_cam=pred_vertices_cam,
                gt_vertices_cam=gt_vertices_cam,
            )
            out.update(plane_out)

        logits = predictions.get("hsi_contact_logits")
        if logits is None:
            return out
        flat_logits = _flatten_prediction(logits, unframed_ndim=4)
        if flat_logits is None:
            return out
        matched_logits = flat_logits[frame_idx, src_idx, :, 0]
        gt_projected = _project_points(gt_joints_cam, contact_teacher_intrinsics[frame_idx].to(dtype=gt_joints_cam.dtype))
        gt_projected = _scale_points_to_depth(gt_projected, self._projection_image_hw(), gt_depth.shape[-2], gt_depth.shape[-1])
        sampled_gt, gt_valid = _sample_depth_at_points(gt_depth.reshape(-1, *gt_depth.shape[-2:]), gt_projected, frame_idx)
        contact_target = (torch.abs(sampled_gt - gt_joints_cam[..., 2].to(dtype=sampled_gt.dtype)) < self.hsi_contact_threshold) & gt_valid
        if self.hsi_anchor_scene_xyz_weight != 0.0 and contact_target.any():
            local_dist, local_valid = _sample_local_scene_distance(
                aligned_depth.reshape(-1, *aligned_depth.shape[-2:]),
                projected,
                pred_joints_cam,
                intrinsics[frame_idx].to(dtype=pred_joints_cam.dtype),
                frame_idx,
                window_size=int(self.hsi_anchor_scene_window),
                image_size_hw=self._projection_image_hw(),
            )
            scene_valid = contact_target & local_valid & torch.isfinite(local_dist)
            if scene_valid.any():
                scene_xyz = F.smooth_l1_loss(local_dist[scene_valid], torch.zeros_like(local_dist[scene_valid]))
                out["loss_hsi_anchor_scene_xyz"] = scene_xyz
                out["metric_hsi_anchor_scene_xyz_l1"] = local_dist[scene_valid].mean().detach()
        if contact_target.any() or gt_valid.any():
            out["loss_hsi_contact"] = F.binary_cross_entropy_with_logits(
                matched_logits[gt_valid],
                contact_target[gt_valid].to(dtype=matched_logits.dtype),
            ) if gt_valid.any() else zero
            out["metric_hsi_contact_pos_frac"] = contact_target.to(dtype=matched_logits.dtype).mean().detach()
        return out

    def _hsi_smpl_scale_teacher_losses(
        self,
        *,
        depth: torch.Tensor,
        gt_depth: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        batch: dict[str, torch.Tensor],
        frame_idx: torch.Tensor,
        gt_joints_cam: torch.Tensor,
        gt_vertices_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = depth.sum() * 0.0
        out = {
            "loss_hsi_smpl_scale_teacher": zero,
            "metric_hsi_smpl_scale_teacher_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_valid_points": zero.detach(),
            "metric_hsi_smpl_scale_teacher_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_scale": zero.detach(),
            "metric_hsi_smpl_scale_teacher_log_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_rel_l1": zero.detach(),
            "metric_hsi_smpl_scale_teacher_bias": zero.detach(),
            "metric_hsi_smpl_scale_teacher_pred_bias": zero.detach(),
        }
        if "K_scal3r" not in batch or frame_idx.numel() == 0:
            return out

        flat_scale = _flatten_prediction(scale, unframed_ndim=2)
        flat_bias = _flatten_prediction(bias, unframed_ndim=2)
        if flat_scale is None or flat_bias is None:
            return out

        dataset_intrinsics = _flatten_batch_intrinsics(batch["K_scal3r"], device=depth.device, dtype=depth.dtype)
        points_cam = gt_vertices_cam if self.hsi_smpl_scale_teacher_source == "vertices" else gt_joints_cam
        points_cam = _subsample_points_per_person(points_cam, int(self.hsi_smpl_scale_teacher_max_points_per_person))
        projected = _project_points(points_cam, dataset_intrinsics[frame_idx].to(dtype=points_cam.dtype))
        projected = _scale_points_to_depth(projected, self._projection_image_hw(), depth.shape[-2], depth.shape[-1])
        raw_sampled, gt_sampled, valid = _sample_depth_pair_nearest_to_point_z(
            raw_depth=depth.reshape(-1, *depth.shape[-2:]),
            gt_depth=gt_depth.reshape(-1, *gt_depth.shape[-2:]),
            points_2d=projected,
            points_z=points_cam[..., 2],
            frame_idx=frame_idx,
            window=int(self.hsi_smpl_scale_teacher_window),
            tolerance_m=float(self.hsi_smpl_scale_teacher_visibility_tolerance_m),
        )
        point_z = points_cam[..., 2].to(dtype=depth.dtype)
        valid = valid & torch.isfinite(raw_sampled) & torch.isfinite(gt_sampled) & (raw_sampled > 1e-6) & (point_z > 1e-6)
        if float(self.hsi_smpl_scale_teacher_max_z_m) > 0.0:
            valid = valid & (point_z <= float(self.hsi_smpl_scale_teacher_max_z_m))
        if int(self.hsi_smpl_scale_teacher_min_points_per_person) > 0:
            per_person_valid = valid.sum(dim=-1) >= int(self.hsi_smpl_scale_teacher_min_points_per_person)
            valid = valid & per_person_valid[:, None]
        if not valid.any():
            return out

        frame_losses: list[torch.Tensor] = []
        teacher_scales: list[torch.Tensor] = []
        pred_scales: list[torch.Tensor] = []
        teacher_biases: list[torch.Tensor] = []
        pred_biases: list[torch.Tensor] = []
        valid_counts: list[torch.Tensor] = []
        log_l1_values: list[torch.Tensor] = []
        rel_l1_values: list[torch.Tensor] = []
        scale_values = point_z / raw_sampled.clamp(min=1e-6)

        for flat_frame_tensor in torch.unique(frame_idx):
            flat_frame = int(flat_frame_tensor.detach().cpu())
            frame_mask = frame_idx == flat_frame
            frame_valid = valid[frame_mask]
            if not frame_valid.any():
                continue
            raw_values = raw_sampled[frame_mask][frame_valid]
            z_values = point_z[frame_mask][frame_valid]
            scale_candidates = scale_values[frame_mask][frame_valid]
            robust = _robust_median_filter(scale_candidates, float(self.hsi_smpl_scale_teacher_mad_multiplier))
            if int(robust.sum().detach().cpu()) < int(self.hsi_smpl_scale_teacher_min_visible_points):
                continue
            raw_values = raw_values[robust]
            z_values = z_values[robust]
            scale_candidates = scale_candidates[robust]
            teacher_scale = scale_candidates.median().detach().clamp(min=1e-6)
            if self.hsi_smpl_scale_teacher_use_bias:
                teacher_bias = (z_values - teacher_scale * raw_values).median().detach()
            else:
                teacher_bias = torch.zeros((), device=depth.device, dtype=depth.dtype)
            pred_scale = flat_scale[flat_frame, 0].to(dtype=depth.dtype).clamp(min=1e-6)
            pred_bias = flat_bias[flat_frame, 0].to(dtype=depth.dtype)
            if self.hsi_smpl_scale_teacher_log_loss:
                scale_loss = F.smooth_l1_loss(torch.log(pred_scale), torch.log(teacher_scale))
            else:
                scale_loss = F.smooth_l1_loss(pred_scale, teacher_scale)
            bias_loss = F.smooth_l1_loss(pred_bias, teacher_bias)
            frame_losses.append(scale_loss + float(self.hsi_smpl_scale_teacher_bias_reg_weight) * bias_loss)
            teacher_scales.append(teacher_scale)
            pred_scales.append(pred_scale.detach())
            teacher_biases.append(teacher_bias)
            pred_biases.append(pred_bias.detach())
            valid_counts.append(depth.new_tensor(float(robust.sum().detach().cpu())))
            log_l1_values.append(torch.abs(torch.log(pred_scale.detach()) - torch.log(teacher_scale.detach())))
            rel_l1_values.append(torch.abs(pred_scale.detach() - teacher_scale.detach()) / teacher_scale.detach().clamp(min=1e-6))

        if not frame_losses:
            return out
        out["loss_hsi_smpl_scale_teacher"] = torch.stack(frame_losses).mean()
        out["metric_hsi_smpl_scale_teacher_l1"] = torch.stack(
            [torch.abs(pred - teacher).detach() for pred, teacher in zip(pred_scales, teacher_scales)]
        ).mean()
        out["metric_hsi_smpl_scale_teacher_valid_points"] = torch.stack(valid_counts).mean().detach()
        out["metric_hsi_smpl_scale_teacher_scale"] = torch.stack(teacher_scales).mean().detach()
        out["metric_hsi_smpl_scale_teacher_pred_scale"] = torch.stack(pred_scales).mean().detach()
        out["metric_hsi_smpl_scale_teacher_log_l1"] = torch.stack(log_l1_values).mean().detach()
        out["metric_hsi_smpl_scale_teacher_rel_l1"] = torch.stack(rel_l1_values).mean().detach()
        out["metric_hsi_smpl_scale_teacher_bias"] = torch.stack(teacher_biases).mean().detach()
        out["metric_hsi_smpl_scale_teacher_pred_bias"] = torch.stack(pred_biases).mean().detach()
        return out

    def _hsi_foot_contact_losses(
        self,
        aligned_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        teacher_intrinsics: torch.Tensor,
        frame_idx: torch.Tensor,
        pred_joints_cam: torch.Tensor,
        gt_joints_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_joints_cam.sum() * 0.0
        out = {
            "loss_hsi_foot_contact": zero,
            "metric_hsi_foot_float_m": zero.detach(),
            "metric_hsi_foot_penetration_m": zero.detach(),
            "metric_hsi_foot_contact_count": zero.detach(),
        }
        foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=pred_joints_cam.device)
        pred_foot = pred_joints_cam[:, foot_idx]
        gt_foot = gt_joints_cam[:, foot_idx]
        pred_projected = _project_points(pred_foot, intrinsics[frame_idx].to(dtype=pred_foot.dtype))
        pred_projected = _scale_points_to_depth(pred_projected, self._projection_image_hw(), aligned_depth.shape[-2], aligned_depth.shape[-1])
        sampled_aligned, pred_valid = _sample_depth_at_points(aligned_depth.reshape(-1, *aligned_depth.shape[-2:]), pred_projected, frame_idx)

        gt_projected = _project_points(gt_foot, teacher_intrinsics[frame_idx].to(dtype=gt_foot.dtype))
        gt_projected = _scale_points_to_depth(gt_projected, self._projection_image_hw(), gt_depth.shape[-2], gt_depth.shape[-1])
        sampled_gt, gt_valid = _sample_depth_at_points(gt_depth.reshape(-1, *gt_depth.shape[-2:]), gt_projected, frame_idx)
        contact_target = (torch.abs(sampled_gt - gt_foot[..., 2].to(dtype=sampled_gt.dtype)) < float(self.hsi_foot_contact_threshold_m)) & gt_valid
        valid = contact_target & pred_valid
        if not valid.any():
            return out

        depth_delta = sampled_aligned - pred_foot[..., 2].to(dtype=sampled_aligned.dtype)
        float_amt = F.relu(depth_delta - float(self.hsi_foot_float_margin_m))
        penetration_amt = F.relu(-depth_delta - float(self.hsi_foot_penetration_margin_m))
        contact_loss = (_smooth_l1_abs(float_amt[valid]) + _smooth_l1_abs(penetration_amt[valid])).mean()
        out["loss_hsi_foot_contact"] = contact_loss
        out["metric_hsi_foot_float_m"] = float_amt[valid].mean().detach()
        out["metric_hsi_foot_penetration_m"] = penetration_amt[valid].mean().detach()
        out["metric_hsi_foot_contact_count"] = pred_joints_cam.new_tensor(float(valid.sum().detach().cpu())).detach()
        return out

    def _hsi_foot_sole_contact_losses(
        self,
        aligned_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        teacher_intrinsics: torch.Tensor,
        frame_idx: torch.Tensor,
        pred_vertices_cam: torch.Tensor,
        gt_vertices_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_vertices_cam.sum() * 0.0
        out = {
            "loss_hsi_foot_sole_contact": zero,
            "metric_hsi_foot_sole_float_m": zero.detach(),
            "metric_hsi_foot_sole_penetration_m": zero.detach(),
            "metric_hsi_foot_sole_contact_count": zero.detach(),
        }
        foot_idx = self._get_foot_sole_indices(pred_vertices_cam.device)
        pred_sole = pred_vertices_cam[:, foot_idx]
        gt_sole = gt_vertices_cam[:, foot_idx]

        gt_projected = _project_points(gt_sole, teacher_intrinsics[frame_idx].to(dtype=gt_sole.dtype))
        gt_projected = _scale_points_to_depth(gt_projected, self._projection_image_hw(), gt_depth.shape[-2], gt_depth.shape[-1])
        sampled_gt, gt_valid = _sample_depth_at_points(gt_depth.reshape(-1, *gt_depth.shape[-2:]), gt_projected, frame_idx)
        contact_target = (torch.abs(sampled_gt - gt_sole[..., 2].to(dtype=sampled_gt.dtype)) < float(self.hsi_foot_sole_contact_threshold_m)) & gt_valid

        pred_projected = _project_points(pred_sole, intrinsics[frame_idx].to(dtype=pred_sole.dtype))
        pred_projected = _scale_points_to_depth(pred_projected, self._projection_image_hw(), aligned_depth.shape[-2], aligned_depth.shape[-1])
        sampled_aligned, pred_valid = _sample_depth_at_points(aligned_depth.reshape(-1, *aligned_depth.shape[-2:]), pred_projected, frame_idx)
        valid = contact_target & pred_valid
        if not valid.any():
            return out

        depth_delta = sampled_aligned - pred_sole[..., 2].to(dtype=sampled_aligned.dtype)
        float_amt = F.relu(depth_delta - float(self.hsi_foot_sole_float_margin_m))
        penetration_amt = F.relu(-depth_delta - float(self.hsi_foot_sole_penetration_margin_m))
        contact_loss = (
            float(self.hsi_foot_sole_float_weight) * _smooth_l1_abs(float_amt[valid])
            + float(self.hsi_foot_sole_penetration_weight) * _smooth_l1_abs(penetration_amt[valid])
        ).mean()
        out["loss_hsi_foot_sole_contact"] = contact_loss
        out["metric_hsi_foot_sole_float_m"] = float_amt[valid].mean().detach()
        out["metric_hsi_foot_sole_penetration_m"] = penetration_amt[valid].mean().detach()
        out["metric_hsi_foot_sole_contact_count"] = pred_vertices_cam.new_tensor(float(valid.sum().detach().cpu())).detach()
        return out

    def _hsi_support_plane_contact_losses(
        self,
        aligned_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        teacher_intrinsics: torch.Tensor,
        frame_idx: torch.Tensor,
        pred_vertices_cam: torch.Tensor,
        gt_vertices_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = pred_vertices_cam.sum() * 0.0
        out = {
            "loss_hsi_support_plane_contact": zero,
            "metric_hsi_support_plane_float_m": zero.detach(),
            "metric_hsi_support_plane_penetration_m": zero.detach(),
            "metric_hsi_support_plane_signed_m": zero.detach(),
            "metric_hsi_support_plane_contact_count": zero.detach(),
        }
        foot_idx = self._get_foot_sole_indices(pred_vertices_cam.device, count=int(self.hsi_support_plane_num_vertices))
        pred_sole = pred_vertices_cam[:, foot_idx]
        gt_sole = gt_vertices_cam[:, foot_idx]

        gt_projected = _project_points(gt_sole, teacher_intrinsics[frame_idx].to(dtype=gt_sole.dtype))
        gt_projected = _scale_points_to_depth(gt_projected, self._projection_image_hw(), gt_depth.shape[-2], gt_depth.shape[-1])
        sampled_gt, gt_valid = _sample_depth_at_points(gt_depth.reshape(-1, *gt_depth.shape[-2:]), gt_projected, frame_idx)
        contact_target = (torch.abs(sampled_gt - gt_sole[..., 2].to(dtype=sampled_gt.dtype)) < float(self.hsi_support_plane_contact_threshold_m)) & gt_valid
        if not contact_target.any():
            return out

        pred_projected = _project_points(pred_sole, intrinsics[frame_idx].to(dtype=pred_sole.dtype))
        pred_projected = _scale_points_to_depth(pred_projected, self._projection_image_hw(), aligned_depth.shape[-2], aligned_depth.shape[-1])
        signed_delta, plane_valid = _sample_local_support_plane_signed_delta(
            aligned_depth.reshape(-1, *aligned_depth.shape[-2:]),
            pred_projected,
            pred_sole,
            intrinsics[frame_idx].to(dtype=pred_sole.dtype),
            frame_idx,
            window_size=int(self.hsi_support_plane_window),
            min_points=int(self.hsi_support_plane_min_points),
            image_size_hw=self._projection_image_hw(),
        )
        valid = contact_target & plane_valid & torch.isfinite(signed_delta)
        if not valid.any():
            return out

        float_amt = F.relu(signed_delta - float(self.hsi_support_plane_float_margin_m))
        penetration_amt = F.relu(-signed_delta - float(self.hsi_support_plane_penetration_margin_m))
        contact_loss = (
            float(self.hsi_support_plane_float_weight) * _smooth_l1_abs(float_amt[valid])
            + float(self.hsi_support_plane_penetration_weight) * _smooth_l1_abs(penetration_amt[valid])
        ).mean()
        out["loss_hsi_support_plane_contact"] = contact_loss
        out["metric_hsi_support_plane_float_m"] = float_amt[valid].mean().detach()
        out["metric_hsi_support_plane_penetration_m"] = penetration_amt[valid].mean().detach()
        out["metric_hsi_support_plane_signed_m"] = signed_delta[valid].mean().detach()
        out["metric_hsi_support_plane_contact_count"] = pred_vertices_cam.new_tensor(float(valid.sum().detach().cpu())).detach()
        return out

    def _get_foot_sole_indices(self, device: torch.device, count: int | None = None) -> torch.Tensor:
        count = min(max(int(self.hsi_foot_sole_num_vertices if count is None else count), 1), 6890)
        cached = self._foot_sole_indices_by_count.get(count)
        if cached is None:
            smpl = self._get_smpl_layer(device)
            template = smpl.layer.v_template.detach().float().reshape(-1, 3)
            count = min(count, int(template.shape[0]))
            cached = torch.argsort(template[:, 1])[:count].long().cpu()
            self._foot_sole_indices_by_count[count] = cached
            self._foot_sole_indices = cached
        return cached.to(device=device)

    def _get_foot_sole_indices_lr(self, device: torch.device, count_per_foot: int = 48) -> torch.Tensor:
        cache_name = f"_foot_sole_lr_{int(count_per_foot)}"
        cached = getattr(self, cache_name, None)
        if cached is None:
            smpl = self._get_smpl_layer(device)
            cached = build_sole_vertex_indices(smpl.layer.v_template.detach().float(), count_per_foot).cpu()
            setattr(self, cache_name, cached)
        return cached.to(device=device)

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
    person_ids = batch.get("gt_track_ids", batch.get("person_ids"))
    person_id_mask = batch.get("gt_track_mask", batch.get("person_id_mask"))
    track_source = batch.get("gt_track_source")
    track_quality = batch.get("gt_track_quality")
    contact_keys = (
        "contact_plane_center_cam",
        "contact_plane_normal_cam",
        "contact_plane_rmse_m",
        "contact_signed_distance_m",
        "contact_foot_velocity_m",
        "contact_label",
        "contact_teacher_valid",
    )
    contact_values = {key: batch.get(key) for key in contact_keys}
    if person_ids is not None:
        person_ids = person_ids.to(device=device)
    if person_id_mask is not None:
        person_id_mask = person_id_mask.to(device=device).bool()
    if track_source is not None:
        track_source = track_source.to(device=device)
    if track_quality is not None:
        track_quality = track_quality.to(device=device)

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
            if track_source is not None:
                target["gt_track_source"] = track_source[batch_idx, frame_idx, valid]
            else:
                target["gt_track_source"] = torch.full((int(valid.sum()),), -1, dtype=torch.long, device=device)
            if track_quality is not None:
                target["gt_track_quality"] = track_quality[batch_idx, frame_idx, valid]
            else:
                target["gt_track_quality"] = torch.zeros(int(valid.sum()), dtype=torch.float32, device=device)
            for key, value in contact_values.items():
                if value is not None:
                    target[key] = value.to(device=device)[batch_idx, frame_idx, valid]
                elif key in {"contact_plane_center_cam", "contact_plane_normal_cam"}:
                    target[key] = torch.zeros(int(valid.sum()), 2, 3, dtype=torch.float32, device=device)
                elif key in {"contact_label", "contact_teacher_valid"}:
                    target[key] = torch.zeros(int(valid.sum()), 2, dtype=torch.bool, device=device)
                else:
                    target[key] = torch.zeros(int(valid.sum()), 2, dtype=torch.float32, device=device)
            targets.append(target)
    return targets


def _collect_matches(indices, targets: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    frame_indices = []
    src_indices = []
    target_parts: dict[str, list[torch.Tensor]] = {
        "boxes": [],
        "pose_6d": [],
        "betas": [],
        "transl_cam": [],
        "person_ids": [],
        "person_id_mask": [],
        "gt_track_source": [],
        "gt_track_quality": [],
        "contact_plane_center_cam": [],
        "contact_plane_normal_cam": [],
        "contact_plane_rmse_m": [],
        "contact_signed_distance_m": [],
        "contact_foot_velocity_m": [],
        "contact_label": [],
        "contact_teacher_valid": [],
    }
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


def _coerce_image_size_hw(image_size_hw: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size_hw, int):
        return int(image_size_hw), int(image_size_hw)
    return int(image_size_hw[0]), int(image_size_hw[1])


def _infer_projection_image_hw(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    fallback_image_size: int,
) -> tuple[int, int]:
    for source in (predictions.get("images"), batch.get("images")):
        if isinstance(source, torch.Tensor) and source.ndim >= 4:
            return int(source.shape[-2]), int(source.shape[-1])
    image_hw = batch.get("image_hw")
    if isinstance(image_hw, torch.Tensor) and image_hw.numel() >= 2:
        flat = image_hw.reshape(-1, 2)
        return int(flat[0, 0].item()), int(flat[0, 1].item())
    return int(fallback_image_size), int(fallback_image_size)


def _flatten_intrinsics(pose_enc: torch.Tensor, image_size_hw: int | tuple[int, int]) -> torch.Tensor:
    if pose_enc.ndim == 2:
        pose_enc = pose_enc[:, None]
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=_coerce_image_size_hw(image_size_hw), build_intrinsics=True)
    if intrinsics is None:
        raise RuntimeError("encoding_to_camera did not return intrinsics")
    return intrinsics.reshape(-1, 3, 3)


def _flatten_batch_intrinsics(intrinsics: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if intrinsics.ndim == 4:
        flat = intrinsics.reshape(intrinsics.shape[0] * intrinsics.shape[1], 3, 3)
    elif intrinsics.ndim == 3:
        flat = intrinsics
    else:
        raise ValueError(f"Unsupported intrinsics shape: {tuple(intrinsics.shape)}")
    if flat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics ending in (3, 3), got {tuple(intrinsics.shape)}")
    return flat.to(device=device, dtype=dtype)


def _canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _sample_depth_at_points(
    depth_flat: torch.Tensor,
    points_2d: torch.Tensor,
    frame_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth_flat.shape[-2:]
    px = points_2d[..., 0].round().long()
    py = points_2d[..., 1].round().long()
    valid = (
        torch.isfinite(points_2d).all(dim=-1)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    px = px.clamp(0, width - 1)
    py = py.clamp(0, height - 1)
    sampled = depth_flat[frame_idx[:, None], py, px]
    valid = valid & torch.isfinite(sampled) & (sampled > 1e-6)
    return sampled, valid


def _human_roi_depth_mask(
    boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    depth_height: int,
    depth_width: int,
    expand: float = 0.35,
) -> torch.Tensor:
    mask = torch.zeros(*boxes.shape[:2], depth_height, depth_width, dtype=torch.bool, device=boxes.device)
    if boxes.numel() == 0:
        return mask
    expand = max(float(expand), 0.0)
    batch_size, num_frames, num_boxes = boxes.shape[:3]
    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            valid_indices = torch.nonzero(boxes_mask[batch_idx, frame_idx], as_tuple=False).flatten()
            for box_idx in valid_indices[:num_boxes]:
                cx, cy, width, height = boxes[batch_idx, frame_idx, box_idx].unbind(dim=-1)
                width = width * (1.0 + expand)
                height = height * (1.0 + expand)
                x1 = int(torch.floor((cx - 0.5 * width).clamp(0.0, 1.0) * depth_width).item())
                x2 = int(torch.ceil((cx + 0.5 * width).clamp(0.0, 1.0) * depth_width).item())
                y1 = int(torch.floor((cy - 0.5 * height).clamp(0.0, 1.0) * depth_height).item())
                y2 = int(torch.ceil((cy + 0.5 * height).clamp(0.0, 1.0) * depth_height).item())
                if x2 > x1 and y2 > y1:
                    mask[batch_idx, frame_idx, y1:y2, x1:x2] = True
    return mask


def _smooth_l1_abs(abs_err: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    beta = max(float(beta), 1e-6)
    return torch.where(abs_err < beta, 0.5 * abs_err.square() / beta, abs_err - 0.5 * beta)


def _subsample_points_per_person(points: torch.Tensor, max_points: int) -> torch.Tensor:
    max_points = int(max_points)
    if max_points <= 0 or int(points.shape[-2]) <= max_points:
        return points
    indices = torch.linspace(0, int(points.shape[-2]) - 1, steps=max_points, device=points.device).round().long()
    return points.index_select(dim=-2, index=indices)


def _robust_median_filter(values: torch.Tensor, mad_multiplier: float) -> torch.Tensor:
    finite = torch.isfinite(values) & (values > 1e-6)
    if not finite.any():
        return finite
    valid_values = values[finite]
    median = valid_values.median()
    abs_dev = torch.abs(valid_values - median)
    mad = abs_dev.median()
    threshold = torch.clamp(mad * float(mad_multiplier), min=1e-6)
    valid_filtered = abs_dev <= threshold
    if not valid_filtered.any():
        valid_filtered = torch.ones_like(valid_values, dtype=torch.bool)
    out = torch.zeros_like(finite)
    out[finite] = valid_filtered
    return out


def _sample_depth_pair_nearest_to_point_z(
    raw_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    points_2d: torch.Tensor,
    points_z: torch.Tensor,
    frame_idx: torch.Tensor,
    window: int,
    tolerance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = gt_depth.shape[-2:]
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    radius = window // 2

    center_x = points_2d[..., 0].round().long()
    center_y = points_2d[..., 1].round().long()
    point_valid = (
        torch.isfinite(points_2d).all(dim=-1)
        & torch.isfinite(points_z)
        & (points_z > 1e-6)
        & (center_x >= 0)
        & (center_x < width)
        & (center_y >= 0)
        & (center_y < height)
    )

    offsets = torch.arange(-radius, radius + 1, device=points_2d.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, -1)
    oy = oy.reshape(1, 1, -1)
    xs = center_x[..., None] + ox
    ys = center_y[..., None] + oy
    local_valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs.clamp(0, width - 1)
    ys = ys.clamp(0, height - 1)

    sampled_gt = gt_depth[frame_idx[:, None, None], ys, xs]
    sampled_raw = raw_depth[frame_idx[:, None, None], ys, xs]
    local_valid = local_valid & torch.isfinite(sampled_gt) & torch.isfinite(sampled_raw) & (sampled_gt > 1e-6) & (sampled_raw > 1e-6)
    z = points_z[..., None].to(dtype=sampled_gt.dtype)
    abs_delta = torch.abs(sampled_gt - z)
    inf = torch.full_like(abs_delta, float("inf"))
    best_idx = torch.where(local_valid, abs_delta, inf).argmin(dim=-1)
    best_gt = sampled_gt.gather(dim=-1, index=best_idx[..., None]).squeeze(-1)
    best_raw = sampled_raw.gather(dim=-1, index=best_idx[..., None]).squeeze(-1)
    best_delta = abs_delta.gather(dim=-1, index=best_idx[..., None]).squeeze(-1)
    valid = point_valid & local_valid.any(dim=-1) & torch.isfinite(best_gt) & torch.isfinite(best_raw)
    if float(tolerance_m) > 0.0:
        valid = valid & (best_delta <= float(tolerance_m))
    best_gt = torch.where(valid, best_gt, torch.zeros_like(best_gt))
    best_raw = torch.where(valid, best_raw, torch.zeros_like(best_raw))
    return best_raw, best_gt, valid


def _infer_sequence_length(batch: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]) -> int:
    smpl_mask = batch.get("smpl_mask")
    if smpl_mask is not None and smpl_mask.ndim >= 3:
        return int(smpl_mask.shape[1])
    images = batch.get("images")
    if images is not None and images.ndim >= 5:
        return int(images.shape[1])
    for key in ("hsi_refined_pred_transl_cam", "pred_transl_cam", "hsi_scene_scale"):
        value = predictions.get(key)
        if value is not None and value.ndim >= 4:
            return int(value.shape[1])
        if value is not None and key == "hsi_scene_scale" and value.ndim == 3:
            return int(value.shape[1])
    return 1


def _velocity_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    curr: torch.Tensor,
) -> torch.Tensor:
    pred_vel = pred[curr] - pred[prev]
    target_vel = target[curr].to(device=pred.device, dtype=pred.dtype) - target[prev].to(device=pred.device, dtype=pred.dtype)
    return pred_vel - target_vel


def _acceleration_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    mid: torch.Tensor,
    nxt: torch.Tensor,
) -> torch.Tensor:
    pred_acc = pred[nxt] - 2.0 * pred[mid] + pred[prev]
    target = target.to(device=pred.device, dtype=pred.dtype)
    target_acc = target[nxt] - 2.0 * target[mid] + target[prev]
    return pred_acc - target_acc


def _sample_local_scene_distance(
    depth_flat: torch.Tensor,
    points_2d: torch.Tensor,
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_idx: torch.Tensor,
    window_size: int = 5,
    image_size_hw: int | tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth_flat.shape[-2:]
    window_size = max(int(window_size), 1)
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2

    center_x = points_2d[..., 0].round().long()
    center_y = points_2d[..., 1].round().long()
    point_valid = (
        torch.isfinite(points_2d).all(dim=-1)
        & torch.isfinite(points_cam).all(dim=-1)
        & (points_cam[..., 2] > 1e-6)
        & (center_x >= 0)
        & (center_x < width)
        & (center_y >= 0)
        & (center_y < height)
    )

    offsets = torch.arange(-radius, radius + 1, device=points_2d.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, -1)
    oy = oy.reshape(1, 1, -1)
    xs = center_x[..., None] + ox
    ys = center_y[..., None] + oy
    local_valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs.clamp(0, width - 1)
    ys = ys.clamp(0, height - 1)

    sampled_depth = depth_flat[frame_idx[:, None, None], ys, xs]
    local_valid = local_valid & torch.isfinite(sampled_depth) & (sampled_depth > 1e-6)

    fx = intrinsics[:, 0, 0].reshape(-1, 1, 1).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].reshape(-1, 1, 1).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].reshape(-1, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(-1, 1, 1)
    pixel_x = xs.to(dtype=sampled_depth.dtype)
    pixel_y = ys.to(dtype=sampled_depth.dtype)
    if image_size_hw is not None:
        image_h, image_w = _coerce_image_size_hw(image_size_hw)
        pixel_x = pixel_x * (float(image_w) / float(width))
        pixel_y = pixel_y * (float(image_h) / float(height))
    scene_x = (pixel_x - cx.to(dtype=sampled_depth.dtype)) * sampled_depth / fx.to(dtype=sampled_depth.dtype)
    scene_y = (pixel_y - cy.to(dtype=sampled_depth.dtype)) * sampled_depth / fy.to(dtype=sampled_depth.dtype)
    scene_xyz = torch.stack([scene_x, scene_y, sampled_depth], dim=-1)

    dist = torch.linalg.norm(scene_xyz - points_cam[..., None, :].to(dtype=scene_xyz.dtype), dim=-1)
    inf = torch.full_like(dist, float("inf"))
    nearest = torch.where(local_valid, dist, inf).amin(dim=-1)
    valid = point_valid & local_valid.any(dim=-1) & torch.isfinite(nearest)
    nearest = torch.where(valid, nearest, torch.zeros_like(nearest))
    return nearest, valid


def _sample_local_support_plane_signed_delta(
    depth_flat: torch.Tensor,
    points_2d: torch.Tensor,
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_idx: torch.Tensor,
    window_size: int = 9,
    min_points: int = 6,
    image_size_hw: int | tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth_flat.shape[-2:]
    window_size = max(int(window_size), 1)
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    min_points = max(int(min_points), 3)

    center_x = points_2d[..., 0].round().long()
    center_y = points_2d[..., 1].round().long()
    point_valid = (
        torch.isfinite(points_2d).all(dim=-1)
        & torch.isfinite(points_cam).all(dim=-1)
        & (points_cam[..., 2] > 1e-6)
        & (center_x >= 0)
        & (center_x < width)
        & (center_y >= 0)
        & (center_y < height)
    )

    offsets = torch.arange(-radius, radius + 1, device=points_2d.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, -1)
    oy = oy.reshape(1, 1, -1)
    xs = center_x[..., None] + ox
    ys = center_y[..., None] + oy
    local_valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs.clamp(0, width - 1)
    ys = ys.clamp(0, height - 1)

    sampled_depth = depth_flat[frame_idx[:, None, None], ys, xs]
    local_valid = local_valid & torch.isfinite(sampled_depth) & (sampled_depth > 1e-6)

    fx = intrinsics[:, 0, 0].reshape(-1, 1, 1).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].reshape(-1, 1, 1).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].reshape(-1, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(-1, 1, 1)
    pixel_x = xs.to(dtype=sampled_depth.dtype)
    pixel_y = ys.to(dtype=sampled_depth.dtype)
    if image_size_hw is not None:
        image_h, image_w = _coerce_image_size_hw(image_size_hw)
        pixel_x = pixel_x * (float(image_w) / float(width))
        pixel_y = pixel_y * (float(image_h) / float(height))
    scene_x = (pixel_x - cx.to(dtype=sampled_depth.dtype)) * sampled_depth / fx.to(dtype=sampled_depth.dtype)
    scene_y = (pixel_y - cy.to(dtype=sampled_depth.dtype)) * sampled_depth / fy.to(dtype=sampled_depth.dtype)
    scene_xyz = torch.stack([scene_x, scene_y, sampled_depth], dim=-1)

    weights = local_valid.to(dtype=scene_xyz.dtype)
    valid_count = local_valid.sum(dim=-1)
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    center = (scene_xyz * weights[..., None]).sum(dim=-2) / denom
    centered = (scene_xyz - center[..., None, :]) * weights[..., None]
    cov = torch.matmul(centered.transpose(-1, -2), centered) / denom[..., None].clamp(min=1.0)
    eye = torch.eye(3, dtype=cov.dtype, device=cov.device).reshape(1, 1, 3, 3)
    cov = cov + eye * 1e-6

    _, evecs = torch.linalg.eigh(cov.float())
    normal = evecs[..., 0].to(dtype=points_cam.dtype)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.where(normal[..., 2:3] > 0, -normal, normal)

    center = center.to(dtype=points_cam.dtype).detach()
    normal = normal.detach()
    signed = ((points_cam - center) * normal).sum(dim=-1)
    valid = point_valid & (valid_count >= min_points) & torch.isfinite(signed)
    signed = torch.where(valid, signed, torch.zeros_like(signed))
    return signed, valid


def _scale_points_to_depth(points_2d: torch.Tensor, image_size_hw: int | tuple[int, int], depth_height: int, depth_width: int) -> torch.Tensor:
    image_h, image_w = _coerce_image_size_hw(image_size_hw)
    scale = points_2d.new_tensor(
        [
            float(depth_width) / float(image_w),
            float(depth_height) / float(image_h),
        ]
    )
    return points_2d * scale


def _project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[:, None, 0, 0] * points_cam[..., 0] / z + intrinsics[:, None, 0, 2]
    y = intrinsics[:, None, 1, 1] * points_cam[..., 1] / z + intrinsics[:, None, 1, 2]
    return torch.stack([x, y], dim=-1)


def _normalize_points_2d(points: torch.Tensor, image_size_hw: int | tuple[int, int]) -> torch.Tensor:
    image_h, image_w = _coerce_image_size_hw(image_size_hw)
    scale = points.new_tensor([max(float(image_w), 1.0), max(float(image_h), 1.0)])
    return points / scale


def _points_to_normalized_cxcywh(points: torch.Tensor, image_size_hw: int | tuple[int, int]) -> torch.Tensor:
    image_h, image_w = _coerce_image_size_hw(image_size_hw)
    max_xy = points.new_tensor([float(image_w), float(image_h)])
    points = torch.nan_to_num(points, nan=0.0, posinf=float(max(image_w, image_h)), neginf=0.0)
    points = torch.maximum(torch.minimum(points, max_xy), torch.zeros_like(points))
    xy_min = points.amin(dim=1)
    xy_max = points.amax(dim=1)
    scale = points.new_tensor([max(float(image_w), 1.0), max(float(image_h), 1.0)])
    center = 0.5 * (xy_min + xy_max) / scale
    size = (xy_max - xy_min) / scale
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


def _optional_prediction_loss(
    predictions: dict[str, torch.Tensor],
    key: str,
    anchor: torch.Tensor,
) -> torch.Tensor:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return anchor.sum() * 0.0
    return value.to(device=anchor.device, dtype=anchor.dtype).reshape(-1).mean()


def _optional_prediction_metric(
    predictions: dict[str, torch.Tensor],
    key: str,
    anchor: torch.Tensor,
) -> torch.Tensor:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        return anchor.sum().detach() * 0.0
    return value.to(device=anchor.device, dtype=anchor.dtype).reshape(-1).mean().detach()
