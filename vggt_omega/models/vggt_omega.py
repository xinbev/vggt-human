# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.integrations import NLFSMPLProvider
from vggt_omega.models.heads import AggregatorSMPLHead, CameraHead, DenseHead, HSIRefinementHead, TextAlignmentHead
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner
from vggt_omega.utils.hsi_affine import apply_hsi_scene_affine_mode


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_smpl: bool = False,
        num_smpl_queries: int = 0,
        smpl_num_layers: int = 4,
        smpl_intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        smpl_predict_boxes: bool = False,
        smpl_bbox_mode: str = "direct",
        smpl_predict_id_embed: bool = False,
        smpl_id_embed_dim: int = 256,
        smpl_return_aux: bool = False,
        smpl_translation_output_mode: str = "direct",
        smpl_translation_decode_hidden_dim: int = 512,
        smpl_translation_decode_max_log_depth_delta: float = 1.00,
        smpl_translation_decode_max_ray_delta_m: float = 1.50,
        smpl_translation_decode_max_tangent_offset_m: float = 1.00,
        smpl_translation_decode_human_height_prior_m: float = 1.70,
        smpl_enable_translation_refine: bool = False,
        smpl_translation_refine_hidden_dim: int = 512,
        smpl_translation_refine_max_ray_delta_m: float = 0.60,
        smpl_translation_refine_max_tangent_delta_m: float = 0.35,
        smpl_translation_refine_max_log_depth_delta: float = 0.50,
        smpl_translation_refine_max_box_prior_weight: float = 0.50,
        smpl_translation_refine_human_height_prior_m: float = 1.70,
        smpl_translation_refine_use_log_depth: bool = True,
        smpl_query_box_prior: bool = False,
        smpl_query_patch_pool: bool = False,
        smpl_query_patch_pool_expand: float = 0.10,
        smpl_query_patch_pool_mode: str = "box",
        smpl_query_mask_min_patch_count: int = 4,
        smpl_query_mask_fallback_to_box: bool = True,
        smpl_track_assignment_mode: str = "gt",
        smpl_use_external_track_prior: bool = True,
        smpl_enable_post_track_temporal_translation: bool = False,
        smpl_track_assign_max_age: int = 90,
        smpl_track_assign_min_quality: float = 0.25,
        smpl_track_assign_max_center_distance_norm: float = 0.25,
        smpl_track_assign_max_transl_distance_m: float = 1.50,
        smpl_track_assign_max_beta_l1: float = 0.30,
        smpl_track_assign_external_iou_min: float = 0.50,
        smpl_enable_temporal_translation: bool = False,
        smpl_temporal_translation_hidden_dim: int = 512,
        smpl_temporal_translation_max_velocity_delta_m: float = 0.25,
        smpl_temporal_translation_gate_bias: float = 2.5,
        smpl_temporal_translation_use_world: bool = True,
        enable_hsi_refine: bool = False,
        hsi_hidden_dim: int = 512,
        hsi_num_layers: int = 5,
        hsi_num_heads: int = 8,
        hsi_num_iters: int = 3,
        hsi_scene_window: int = 3,
        hsi_probe_mode: str = "projected",
        hsi_affine_probe_mode: str = "projected",
        hsi_probe_window: int = 9,
        hsi_probe_blend: float = 1.0,
        hsi_use_delta_gate: bool = False,
        hsi_enable_temporal_momentum: bool = False,
        hsi_temporal_momentum_decay: float = 0.7,
        hsi_temporal_momentum_detach: bool = True,
        hsi_temporal_momentum_use_track_ids: bool = True,
        hsi_track_quality_min: float = 0.25,
        hsi_track_gap_max: int = 30,
        hsi_scene_affine_mode: str = "per_frame",
        hsi_scene_affine_ema_alpha: float = 0.25,
        hsi_scene_log_scale_min: float = -5.0,
        hsi_scene_log_scale_max: float = 5.0,
        hsi_transl_delta_scale: float = 0.05,
        smpl_model_dir: str = "",
        smpl_provider: str = "internal",
        nlf_model_path: str = "",
        nlf_third_party_root: str = "third_party/nlf",
        nlf_model_name: str = "smpl",
        nlf_use_detector: bool = False,
        nlf_require_boxes: bool = True,
        nlf_internal_batch_size: int = 64,
        nlf_num_aug: int = 1,
        nlf_detector_threshold: float = 0.3,
        nlf_detector_nms_iou_threshold: float = 0.7,
        nlf_max_detections: int = 150,
        image_size: int = 518,
        freeze_dense_head: bool = False,
        freeze_aggregator_forward: bool = False,
    ) -> None:
        super().__init__()
        if enable_smpl and num_smpl_queries <= 0:
            raise ValueError("enable_smpl=True requires num_smpl_queries > 0")
        if smpl_enable_translation_refine and not enable_camera:
            raise ValueError("smpl_enable_translation_refine=True requires enable_camera=True")
        self.smpl_provider = str(smpl_provider or "internal").lower()
        if self.smpl_provider not in {"internal", "nlf", "gt_perturbed"}:
            raise ValueError(f"Unsupported smpl_provider: {self.smpl_provider}")
        if self.smpl_provider == "nlf" and enable_smpl and not enable_camera:
            raise ValueError("smpl_provider='nlf' requires enable_camera=True")

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_smpl_queries=num_smpl_queries if enable_smpl else 0,
            smpl_query_box_prior=smpl_query_box_prior if enable_smpl else False,
            smpl_query_patch_pool=smpl_query_patch_pool if enable_smpl else False,
            smpl_query_patch_pool_expand=smpl_query_patch_pool_expand,
            smpl_query_patch_pool_mode=smpl_query_patch_pool_mode,
            smpl_query_mask_min_patch_count=smpl_query_mask_min_patch_count,
            smpl_query_mask_fallback_to_box=smpl_query_mask_fallback_to_box,
        )
        self.freeze_aggregator_forward = freeze_aggregator_forward
        self.hsi_scene_affine_mode = str(hsi_scene_affine_mode or "per_frame")
        self.hsi_scene_affine_ema_alpha = float(hsi_scene_affine_ema_alpha)
        self.image_size = int(image_size)
        self.smpl_track_assignment_mode = str(smpl_track_assignment_mode or "gt")
        self.smpl_use_external_track_prior = bool(smpl_use_external_track_prior)
        self.smpl_enable_post_track_temporal_translation = bool(smpl_enable_post_track_temporal_translation)
        if self.smpl_track_assignment_mode not in {"none", "gt", "external_prior", "base_smpl"}:
            raise ValueError(f"Unsupported smpl_track_assignment_mode: {self.smpl_track_assignment_mode}")
        self.smpl_track_assigner = (
            BaseSMPLTrackAssigner(
                max_age=smpl_track_assign_max_age,
                min_track_quality=smpl_track_assign_min_quality,
                max_center_distance_norm=smpl_track_assign_max_center_distance_norm,
                max_transl_distance_m=smpl_track_assign_max_transl_distance_m,
                max_beta_l1=smpl_track_assign_max_beta_l1,
                external_prior_iou_min=smpl_track_assign_external_iou_min,
            )
            if enable_smpl
            else None
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.smpl_head = (
            AggregatorSMPLHead(
                dim_in=2 * embed_dim,
                num_layers=smpl_num_layers,
                intermediate_layer_idx=smpl_intermediate_layer_idx,
                predict_boxes=smpl_predict_boxes,
                bbox_mode=smpl_bbox_mode,
                predict_id_embed=smpl_predict_id_embed,
                id_embed_dim=smpl_id_embed_dim,
                return_aux=smpl_return_aux,
                translation_output_mode=smpl_translation_output_mode,
                translation_decode_hidden_dim=smpl_translation_decode_hidden_dim,
                translation_decode_max_log_depth_delta=smpl_translation_decode_max_log_depth_delta,
                translation_decode_max_ray_delta_m=smpl_translation_decode_max_ray_delta_m,
                translation_decode_max_tangent_offset_m=smpl_translation_decode_max_tangent_offset_m,
                translation_decode_human_height_prior_m=smpl_translation_decode_human_height_prior_m,
                enable_translation_refine=smpl_enable_translation_refine,
                translation_refine_hidden_dim=smpl_translation_refine_hidden_dim,
                translation_refine_max_ray_delta_m=smpl_translation_refine_max_ray_delta_m,
                translation_refine_max_tangent_delta_m=smpl_translation_refine_max_tangent_delta_m,
                translation_refine_max_log_depth_delta=smpl_translation_refine_max_log_depth_delta,
                translation_refine_max_box_prior_weight=smpl_translation_refine_max_box_prior_weight,
                translation_refine_human_height_prior_m=smpl_translation_refine_human_height_prior_m,
                translation_refine_use_log_depth=smpl_translation_refine_use_log_depth,
                enable_temporal_translation=smpl_enable_temporal_translation or smpl_enable_post_track_temporal_translation,
                temporal_translation_hidden_dim=smpl_temporal_translation_hidden_dim,
                temporal_translation_max_velocity_delta_m=smpl_temporal_translation_max_velocity_delta_m,
                temporal_translation_gate_bias=smpl_temporal_translation_gate_bias,
                temporal_translation_use_world=smpl_temporal_translation_use_world,
                image_size=image_size,
            )
            if enable_smpl and self.smpl_provider == "internal"
            else None
        )
        self.nlf_smpl_provider = (
            NLFSMPLProvider(
                model_path=nlf_model_path,
                third_party_root=nlf_third_party_root,
                model_name=nlf_model_name,
                use_detector=nlf_use_detector,
                require_boxes=nlf_require_boxes,
                internal_batch_size=nlf_internal_batch_size,
                num_aug=nlf_num_aug,
                detector_threshold=nlf_detector_threshold,
                detector_nms_iou_threshold=nlf_detector_nms_iou_threshold,
                max_detections=nlf_max_detections,
            )
            if enable_smpl and self.smpl_provider == "nlf"
            else None
        )
        self.hsi_refinement_head = (
            HSIRefinementHead(
                dim_in=2 * embed_dim,
                hidden_dim=hsi_hidden_dim,
                num_layers=hsi_num_layers,
                num_heads=hsi_num_heads,
                num_iters=hsi_num_iters,
                scene_window=hsi_scene_window,
                probe_mode=hsi_probe_mode,
                affine_probe_mode=hsi_affine_probe_mode,
                probe_window=hsi_probe_window,
                probe_blend=hsi_probe_blend,
                use_delta_gate=hsi_use_delta_gate,
                enable_temporal_momentum=hsi_enable_temporal_momentum,
                temporal_momentum_decay=hsi_temporal_momentum_decay,
                temporal_momentum_detach=hsi_temporal_momentum_detach,
                temporal_momentum_use_track_ids=hsi_temporal_momentum_use_track_ids,
                track_quality_min=hsi_track_quality_min,
                track_gap_max=hsi_track_gap_max,
                scene_log_scale_min=hsi_scene_log_scale_min,
                scene_log_scale_max=hsi_scene_log_scale_max,
                transl_delta_scale=hsi_transl_delta_scale,
                smpl_model_dir=smpl_model_dir,
                image_size=image_size,
            )
            if enable_hsi_refine
            else None
        )
        has_runtime_smpl_provider = self.smpl_provider == "gt_perturbed"
        if self.hsi_refinement_head is not None and (
            (self.smpl_head is None and self.nlf_smpl_provider is None and not has_runtime_smpl_provider)
            or self.dense_head is None
            or self.camera_head is None
        ):
            raise ValueError("enable_hsi_refine=True requires a SMPL provider, enable_depth, and enable_camera")

    def forward(
        self,
        images: torch.Tensor,
        smpl_query_boxes: torch.Tensor | None = None,
        smpl_query_boxes_mask: torch.Tensor | None = None,
        smpl_query_patch_masks: torch.Tensor | None = None,
        smpl_track_ids: torch.Tensor | None = None,
        smpl_track_mask: torch.Tensor | None = None,
        external_track_ids: torch.Tensor | None = None,
        external_track_mask: torch.Tensor | None = None,
        external_track_confidence: torch.Tensor | None = None,
        smpl_override_outputs: dict[str, torch.Tensor] | None = None,
        hsi_intrinsics_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        image_size_hw = (int(images.shape[-2]), int(images.shape[-1]))

        amp_enabled = images.device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=images.device.type, dtype=amp_dtype, enabled=amp_enabled):
            if self.freeze_aggregator_forward:
                with torch.no_grad():
                    aggregated_tokens_list, token_layout, smpl_reference_boxes = self.aggregator(
                        images,
                        smpl_query_boxes,
                        smpl_query_boxes_mask,
                        smpl_query_patch_masks,
                    )
            else:
                aggregated_tokens_list, token_layout, smpl_reference_boxes = self.aggregator(
                    images,
                    smpl_query_boxes,
                    smpl_query_boxes_mask,
                    smpl_query_patch_masks,
                )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, : token_layout.register_end].contiguous(),
        }
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    token_layout=token_layout,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=token_layout.patch_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        token_layout=token_layout,
                    )
                )

            if smpl_override_outputs is not None:
                predictions.update(smpl_override_outputs)
            elif self.smpl_head is not None:
                predictions.update(
                    self.smpl_head(
                        aggregated_tokens_list,
                        token_layout=token_layout,
                        reference_boxes=smpl_reference_boxes,
                        pose_enc=predictions.get("pose_enc"),
                        image_size_hw=image_size_hw,
                        track_ids=None,
                        track_mask=None,
                        run_temporal_translation=False,
                        return_temporal_hidden=self.smpl_enable_post_track_temporal_translation,
                    )
                )
            elif self.nlf_smpl_provider is not None:
                predictions.update(
                    self.nlf_smpl_provider(
                        images=images,
                        pose_enc=predictions.get("pose_enc"),
                        smpl_query_boxes=smpl_reference_boxes if smpl_query_boxes is not None else None,
                        smpl_query_boxes_mask=smpl_query_boxes_mask,
                        max_humans=token_layout.num_smpl_queries,
                    )
                )
            elif self.smpl_provider == "gt_perturbed":
                raise ValueError("smpl_provider='gt_perturbed' requires smpl_override_outputs")
            if self.smpl_head is not None or self.nlf_smpl_provider is not None or smpl_override_outputs is not None:
                assigned = self._assign_smpl_tracks(
                    predictions=predictions,
                    reference_boxes=smpl_reference_boxes,
                    query_mask=smpl_query_boxes_mask,
                    smpl_track_ids=smpl_track_ids,
                    smpl_track_mask=smpl_track_mask,
                    external_track_ids=external_track_ids,
                    external_track_mask=external_track_mask,
                    external_track_confidence=external_track_confidence,
                )
                predictions.update(assigned)
                if self.smpl_enable_post_track_temporal_translation and getattr(self.smpl_head, "temporal_translation_refiner", None) is not None:
                    temporal_hidden = predictions.pop("smpl_temporal_hidden", None)
                    if not isinstance(temporal_hidden, torch.Tensor):
                        raise RuntimeError("Post-track temporal translation requires smpl_temporal_hidden")
                    temporal_outputs = self.smpl_head.refine_translation_with_tracks(
                        smpl_outputs=predictions,
                        temporal_hidden=temporal_hidden,
                        pose_enc=predictions.get("pose_enc"),
                        image_size_hw=image_size_hw,
                        track_ids=predictions.get("assigned_track_ids"),
                        track_mask=predictions.get("assigned_track_mask"),
                    )
                    if isinstance(temporal_outputs.get("pred_transl_cam"), torch.Tensor):
                        base_transl = predictions["pred_transl_cam"]
                        refined_transl = temporal_outputs["pred_transl_cam"]
                        predictions.update(temporal_outputs)
                        predictions["base_pred_transl_cam_before_track_temporal"] = base_transl
                        predictions["track_refined_pred_transl_cam"] = refined_transl
            if self.hsi_refinement_head is not None:
                if hsi_intrinsics_override is not None:
                    predictions["hsi_intrinsics_override"] = hsi_intrinsics_override
                predictions.update(
                    self.hsi_refinement_head(
                        aggregated_tokens_list,
                        token_layout=token_layout,
                        smpl_outputs=predictions,
                        depth=predictions["depth"],
                        pose_enc=predictions["pose_enc"],
                        image_size_hw=image_size_hw,
                        track_ids=predictions.get("assigned_track_ids"),
                        track_mask=predictions.get("assigned_track_mask"),
                        track_quality=predictions.get("assigned_track_quality"),
                        track_gap=predictions.get("assigned_track_gap"),
                        intrinsics_override=hsi_intrinsics_override,
                    )
                )
                apply_hsi_scene_affine_mode(
                    predictions,
                    mode=self.hsi_scene_affine_mode,
                    ema_alpha=self.hsi_scene_affine_ema_alpha,
                )

        if not self.training:
            predictions["images"] = images
        return predictions

    def _assign_smpl_tracks(
        self,
        predictions: dict[str, torch.Tensor],
        reference_boxes: torch.Tensor | None,
        query_mask: torch.Tensor | None,
        smpl_track_ids: torch.Tensor | None,
        smpl_track_mask: torch.Tensor | None,
        external_track_ids: torch.Tensor | None,
        external_track_mask: torch.Tensor | None,
        external_track_confidence: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        pred_transl = predictions.get("pred_transl_cam")
        pred_betas = predictions.get("pred_betas")
        pred_confs = predictions.get("pred_confs")
        if not isinstance(pred_transl, torch.Tensor) or not isinstance(pred_betas, torch.Tensor) or not isinstance(pred_confs, torch.Tensor):
            return {}
        batch_size, num_frames, num_queries = pred_transl.shape[:3]
        device = pred_transl.device
        if query_mask is None:
            query_mask = torch.ones(batch_size, num_frames, num_queries, dtype=torch.bool, device=device)
        else:
            query_mask = query_mask.to(device=device).bool()
        mode = self.smpl_track_assignment_mode
        if mode == "none":
            ids = torch.full((batch_size, num_frames, num_queries), -1, dtype=torch.long, device=device)
            return {
                "assigned_track_ids": ids,
                "assigned_track_mask": torch.zeros_like(query_mask),
                "assigned_track_quality": torch.zeros(batch_size, num_frames, num_queries, dtype=torch.float32, device=device),
                "assigned_track_gap": torch.zeros(batch_size, num_frames, num_queries, dtype=torch.long, device=device),
                "assigned_track_source": torch.full((batch_size, num_frames, num_queries), -1, dtype=torch.long, device=device),
            }
        if mode == "gt" and smpl_track_ids is not None:
            mask = smpl_track_mask.to(device=device).bool() if smpl_track_mask is not None else query_mask
            return {
                "assigned_track_ids": smpl_track_ids.to(device=device).long(),
                "assigned_track_mask": mask & query_mask,
                "assigned_track_quality": (mask & query_mask).to(dtype=torch.float32),
                "assigned_track_gap": torch.zeros(batch_size, num_frames, num_queries, dtype=torch.long, device=device),
                "assigned_track_source": torch.full((batch_size, num_frames, num_queries), 3, dtype=torch.long, device=device),
            }
        if mode == "external_prior" and external_track_ids is not None:
            mask = external_track_mask.to(device=device).bool() if external_track_mask is not None else query_mask
            quality = external_track_confidence.to(device=device).float() if external_track_confidence is not None else mask.to(dtype=torch.float32)
            return {
                "assigned_track_ids": external_track_ids.to(device=device).long(),
                "assigned_track_mask": mask & query_mask,
                "assigned_track_quality": quality * (mask & query_mask).to(dtype=quality.dtype),
                "assigned_track_gap": torch.zeros(batch_size, num_frames, num_queries, dtype=torch.long, device=device),
                "assigned_track_source": torch.full((batch_size, num_frames, num_queries), 2, dtype=torch.long, device=device),
            }
        boxes = predictions.get("pred_boxes")
        if not isinstance(boxes, torch.Tensor):
            boxes = reference_boxes
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.zeros(batch_size, num_frames, num_queries, 4, dtype=pred_transl.dtype, device=device)
        if self.smpl_track_assigner is None:
            raise RuntimeError("SMPL track assigner is not initialized")
        return self.smpl_track_assigner.assign(
            boxes=boxes.to(device=device),
            pred_betas=pred_betas,
            pred_transl_cam=pred_transl,
            pred_confs=pred_confs,
            query_mask=query_mask,
            external_track_ids=external_track_ids if self.smpl_use_external_track_prior else None,
            external_track_mask=external_track_mask if self.smpl_use_external_track_prior else None,
            external_track_confidence=external_track_confidence if self.smpl_use_external_track_prior else None,
        )


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
