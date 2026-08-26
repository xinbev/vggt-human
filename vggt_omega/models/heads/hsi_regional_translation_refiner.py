from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.models.geometry import RegionalSceneProbe, SMPLRegionBank
from vggt_omega.models.geometry.smpl_region_bank import DEFAULT_REGION_COUNTS
from vggt_omega.tracking.hsi_trstr_memory import HSITRSTRTrackMemory, TRSTRTrackState
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class HSIRegionalTranslationRefiner(nn.Module):
    """Translation-only regional surface refiner.

    Each valid person owns an independent region bank. Regions vote for one
    shared camera-space translation correction; pose and betas are read-only.
    """

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 256,
        region_embedding_dim: int = 32,
        num_regions: int = 96,
        representative_vertices: int = 8,
        num_iters: int = 2,
        patch_sizes: tuple[int, ...] = (3, 7),
        probe_token_dim: int = 16,
        adaptive_radius_max: int = 5,
        annulus_width: int = 2,
        human_depth_tolerance_m: float = 0.15,
        human_depth_dilation_px: int = 2,
        enable_temporal: bool = False,
        temporal_quality_min: float = 0.25,
        temporal_gap_max: int = 8,
        temporal_use_world: bool = False,
        min_valid_ratio: float = 0.25,
        max_ray_delta_m: float = 0.35,
        max_tangent_delta_m: float = 0.20,
        max_person_delta_m: float = 0.50,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSIRegionalTranslationRefiner requires smpl_model_dir")
        if int(num_regions) not in {48, 72, 96}:
            raise ValueError("TRSTR num_regions must be one of the ablation budgets: 48, 72, 96")
        patch_sizes = tuple(int(size) for size in patch_sizes)
        if not patch_sizes or any(size <= 0 or size % 2 == 0 for size in patch_sizes):
            raise ValueError("patch_sizes must contain positive odd window sizes")

        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        self.region_bank = SMPLRegionBank(
            self.smpl.layer,
            region_counts=_scaled_region_counts(int(num_regions)),
            representative_vertices=representative_vertices,
        )
        self.num_regions = int(num_regions)
        if self.region_bank.num_regions != self.num_regions:
            raise RuntimeError("TRSTR region bank size does not match num_regions")
        self.num_iters = max(int(num_iters), 1)
        self.patch_sizes = patch_sizes
        self.min_valid_ratio = min(max(float(min_valid_ratio), 0.0), 1.0)
        self.max_ray_delta_m = float(max_ray_delta_m)
        self.max_tangent_delta_m = float(max_tangent_delta_m)
        self.max_person_delta_m = max(float(max_person_delta_m), 1e-4)
        self.image_size = int(image_size)
        self.enable_temporal = bool(enable_temporal)
        self.temporal_quality_min = float(temporal_quality_min)
        self.temporal_gap_max = max(int(temporal_gap_max), 1)
        self.temporal_use_world = bool(temporal_use_world)

        self.scene_probe = RegionalSceneProbe(
            token_dim=probe_token_dim,
            fixed_patch_sizes=patch_sizes,
            adaptive_radius_max=adaptive_radius_max,
            annulus_width=annulus_width,
            human_depth_tolerance_m=human_depth_tolerance_m,
            human_depth_dilation_px=human_depth_dilation_px,
            min_valid_ratio=self.min_valid_ratio,
        )

        self.region_embedding = nn.Embedding(self.num_regions, int(region_embedding_dim))
        feature_dim = (
            3
            + 3
            + 2
            + self.scene_probe.num_tokens * self.scene_probe.token_dim
            + self.scene_probe.num_tokens
            + 2
            + int(region_embedding_dim)
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.vote_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)))
        self.gate_head = _biased_last_linear(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)),
            bias=2.0,
        )
        self.logvar_head = _zero_last_linear(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.person_gate_head = _biased_last_linear(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)),
            bias=2.0,
        )
        self.temporal_gate_head = (
            _biased_last_linear(
                nn.Sequential(
                    nn.Linear(hidden_dim + 10, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                ),
                bias=2.0,
            )
            if self.enable_temporal
            else None
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        image_size_hw: tuple[int, int] | None = None,
        intrinsics_override: torch.Tensor | None = None,
        depth_is_metric: bool = False,
        person_valid: torch.Tensor | None = None,
        track_ids: torch.Tensor | None = None,
        track_quality: torch.Tensor | None = None,
        track_gap: torch.Tensor | None = None,
        track_memory: object | None = None,
        frame_offset: int = 0,
    ) -> dict[str, torch.Tensor]:
        pose6d = _require(predictions, "pred_pose_6d").float()
        betas = _require(predictions, "pred_betas").float()
        transl = _require(predictions, "pred_transl_cam").float()
        if pose6d.ndim != 4 or betas.ndim != 4 or transl.ndim != 4:
            raise ValueError("TRSTR expects pose6d/betas/transl with [B,S,Q,*] shapes")
        batch_size, num_frames, num_queries = transl.shape[:3]
        if person_valid is None:
            conf = predictions.get("pred_confs")
            person_valid = torch.ones(batch_size, num_frames, num_queries, dtype=torch.bool, device=transl.device)
            if isinstance(conf, torch.Tensor):
                person_valid = conf[..., 0] > 0.0
        person_valid = person_valid.to(device=transl.device).bool()

        depth_hw = _canonical_depth(depth).float()
        if not depth_is_metric:
            scale = predictions.get("hsi_scene_scale")
            bias = predictions.get("hsi_scene_depth_bias")
            if isinstance(scale, torch.Tensor) and isinstance(bias, torch.Tensor):
                depth_hw = depth_hw * _expand_frame_value(scale, depth_hw) + _expand_frame_value(bias, depth_hw)
        height, width = depth_hw.shape[-2:]
        image_size_hw = image_size_hw or (height, width)
        intrinsics = _resolve_intrinsics(
            pose_enc,
            image_size_hw=image_size_hw,
            batch_size=batch_size,
            num_frames=num_frames,
            dtype=transl.dtype,
            device=transl.device,
            override=intrinsics_override,
        )

        current = transl
        iteration_transl = [current]
        iteration_votes = []
        iteration_gates = []
        iteration_valid = []
        last = None
        for _ in range(self.num_iters):
            vertices = self._decode_vertices(pose6d, betas, current)
            pooled, _ = self.region_bank.pool_vertices(vertices)
            representatives = self.region_bank.representative_points(vertices)
            probe = self.scene_probe(
                centers=pooled.reshape(batch_size * num_frames, num_queries, self.num_regions, 3),
                representatives=representatives.reshape(
                    batch_size * num_frames,
                    num_queries,
                    self.num_regions,
                    representatives.shape[2],
                    3,
                ),
                vertices_by_frame=vertices.reshape(batch_size * num_frames, num_queries, vertices.shape[1], 3),
                depth_by_frame=depth_hw.reshape(batch_size * num_frames, height, width),
                intrinsics_by_frame=intrinsics,
                person_valid=person_valid.reshape(batch_size * num_frames, num_queries),
                image_size_hw=image_size_hw,
            )
            probe_tokens = probe["tokens"].reshape(pooled.shape[0], self.num_regions, -1)
            region_group = self.region_bank.region_group_ids.to(device=transl.device)
            group_embedding = self.region_embedding(region_group).unsqueeze(0).expand(pooled.shape[0], -1, -1)
            features = torch.cat(
                [
                    pooled.clamp(-20.0, 20.0) / 20.0,
                    representatives.std(dim=2, unbiased=False).clamp(max=5.0) / 5.0,
                    probe["projected_norm"],
                    probe_tokens,
                    probe["valid_ratios"],
                    probe["other_human_ratio"].unsqueeze(-1),
                    probe["adaptive_radius"].to(dtype=pooled.dtype).unsqueeze(-1)
                    / float(self.scene_probe.adaptive_radius_max),
                    group_embedding,
                ],
                dim=-1,
            )
            if features.shape[-1] != self.feature_mlp[0].in_features:
                raise RuntimeError(
                    f"TRSTR feature dimension mismatch: got {features.shape[-1]}, "
                    f"expected {self.feature_mlp[0].in_features}"
                )
            hidden = self.feature_mlp(features)
            vote = self._basis_vote(hidden, pooled)
            gate = torch.sigmoid(self.gate_head(hidden))
            logvar = self.logvar_head(hidden).clamp(-4.0, 4.0)
            valid = probe["region_valid"] & person_valid.reshape(-1, 1)
            weight = valid.to(dtype=vote.dtype) * gate * torch.exp(-logvar)
            denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-5)
            aggregate = (vote * weight).sum(dim=1) / denom
            aggregate = aggregate.clamp(min=-self.max_person_delta_m, max=self.max_person_delta_m)
            person_hidden = (hidden * valid.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
            person_gate = torch.sigmoid(self.person_gate_head(person_hidden))
            person_gate = person_gate * person_valid.reshape(-1, 1).to(person_gate.dtype)
            update = person_gate * aggregate
            current = current.reshape(-1, 3) + update
            current = current.reshape(batch_size, num_frames, num_queries, 3)
            iteration_transl.append(current)
            iteration_votes.append(vote.reshape(batch_size, num_frames, num_queries, self.num_regions, 3))
            iteration_gates.append(gate.reshape(batch_size, num_frames, num_queries, self.num_regions, 1))
            iteration_valid.append(valid.reshape(batch_size, num_frames, num_queries, self.num_regions))
            last = {
                "region_vote": vote,
                "region_gate": gate,
                "region_logvar": logvar,
                "region_valid": valid,
                "person_gate": person_gate.reshape(batch_size, num_frames, num_queries, 1),
                "valid_ratios": probe["valid_ratios"],
                "other_human_ratio": probe["other_human_ratio"],
                "self_surface_ratio": probe["self_surface_ratio"],
                "environment_ratio": probe["environment_ratio"],
                "person_hidden": person_hidden.reshape(batch_size, num_frames, num_queries, -1),
            }

        if last is None:
            raise RuntimeError("TRSTR produced no refinement iteration")
        spatial_refined = current
        temporal_gate = current.new_ones(batch_size, num_frames, num_queries, 1)
        temporal_valid = torch.zeros_like(temporal_gate, dtype=torch.bool)
        temporal_velocity = current.new_zeros(batch_size, num_frames, num_queries, 3)
        if self.enable_temporal:
            fused_delta, temporal_gate, temporal_valid, temporal_velocity = self._fuse_temporal_update(
                current=transl,
                proposed_update=spatial_refined - transl,
                person_hidden=last["person_hidden"],
                person_valid=person_valid,
                track_ids=track_ids,
                track_quality=track_quality,
                track_gap=track_gap,
                pose_enc=pose_enc,
                predictions=predictions,
                external_memory=track_memory,
                frame_offset=int(frame_offset),
            )
            current = transl + fused_delta
        final_probe = self._final_probe(
            pose6d=pose6d,
            betas=betas,
            transl=current,
            depth_hw=depth_hw,
            intrinsics=intrinsics,
            person_valid=person_valid,
            image_size_hw=image_size_hw,
        )
        if self.enable_temporal and isinstance(track_memory, HSITRSTRTrackMemory):
            self._write_external_memory(
                memory=track_memory,
                translation=current,
                velocity=last["temporal_velocity"],
                person_hidden=last["person_hidden"],
                person_valid=person_valid,
                track_ids=track_ids,
                track_quality=track_quality,
                frame_offset=int(frame_offset),
                pose_enc=pose_enc,
                predictions=predictions,
            )
        outputs = {
            "hsi_trstr_refined_pred_transl_cam": current,
            "hsi_trstr_spatial_refined_pred_transl_cam": spatial_refined,
            "hsi_trstr_delta_transl_cam": current - transl,
            "hsi_trstr_iteration_transl": torch.stack(iteration_transl, dim=0),
            "hsi_trstr_iteration_region_vote": torch.stack(iteration_votes, dim=0),
            "hsi_trstr_iteration_region_gate": torch.stack(iteration_gates, dim=0),
            "hsi_trstr_iteration_region_valid": torch.stack(iteration_valid, dim=0),
            "hsi_trstr_region_vote": last["region_vote"].reshape(batch_size, num_frames, num_queries, self.num_regions, 3),
            "hsi_trstr_region_gate": last["region_gate"].reshape(batch_size, num_frames, num_queries, self.num_regions, 1),
            "hsi_trstr_region_logvar": last["region_logvar"].reshape(batch_size, num_frames, num_queries, self.num_regions, 1),
            "hsi_trstr_region_valid": last["region_valid"].reshape(batch_size, num_frames, num_queries, self.num_regions),
            "hsi_trstr_person_gate": last["person_gate"],
            "hsi_trstr_person_uncertainty": torch.exp(last["region_logvar"].mean(dim=1)).reshape(batch_size, num_frames, num_queries, 1),
            "hsi_trstr_region_valid_ratios": last["valid_ratios"].reshape(
                batch_size, num_frames, num_queries, self.num_regions, self.scene_probe.num_tokens
            ),
            "hsi_trstr_other_human_ratio": last["other_human_ratio"].reshape(
                batch_size, num_frames, num_queries, self.num_regions
            ),
            "hsi_trstr_self_surface_ratio": last["self_surface_ratio"].reshape(
                batch_size, num_frames, num_queries, self.num_regions
            ),
            "hsi_trstr_environment_ratio": last["environment_ratio"].reshape(
                batch_size, num_frames, num_queries, self.num_regions
            ),
            "hsi_trstr_temporal_gate": temporal_gate,
            "hsi_trstr_temporal_valid": temporal_valid,
            "hsi_trstr_temporal_velocity": temporal_velocity,
            "hsi_trstr_final_region_valid": final_probe["region_valid"],
            "hsi_trstr_final_other_human_ratio": final_probe["other_human_ratio"],
        }
        return outputs

    def _decode_vertices(self, pose6d: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        flat_pose = pose6d.reshape(-1, 24, 6)
        flat_betas = betas.reshape(-1, betas.shape[-1])
        with torch.no_grad():
            poses = rot6d_to_axis_angle(flat_pose).reshape(-1, 72)
            vertices, _ = self.smpl(poses.float(), flat_betas.float())
        return vertices.to(device=transl.device, dtype=transl.dtype) + transl.reshape(-1, 1, 3)

    def _basis_vote(self, hidden: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.vote_head(hidden))
        ray = F.normalize(centers, dim=-1, eps=1e-5)
        reference = torch.zeros_like(ray)
        reference[..., 1] = 1.0
        tangent_x = F.normalize(torch.cross(ray, reference, dim=-1), dim=-1, eps=1e-5)
        fallback = torch.zeros_like(ray)
        fallback[..., 0] = 1.0
        tangent_x = torch.where(torch.linalg.norm(tangent_x, dim=-1, keepdim=True) < 1e-4, F.normalize(torch.cross(ray, fallback, dim=-1), dim=-1, eps=1e-5), tangent_x)
        tangent_y = F.normalize(torch.cross(ray, tangent_x, dim=-1), dim=-1, eps=1e-5)
        coeff = raw * centers.new_tensor([self.max_ray_delta_m, self.max_tangent_delta_m, self.max_tangent_delta_m])
        return coeff[..., :1] * ray + coeff[..., 1:2] * tangent_x + coeff[..., 2:3] * tangent_y

    @torch.no_grad()
    def _final_probe(
        self,
        pose6d: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        depth_hw: torch.Tensor,
        intrinsics: torch.Tensor,
        person_valid: torch.Tensor,
        image_size_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch_size, num_frames, num_queries = transl.shape[:3]
        vertices = self._decode_vertices(pose6d, betas, transl)
        centers, _ = self.region_bank.pool_vertices(vertices)
        representatives = self.region_bank.representative_points(vertices)
        probe = self.scene_probe(
            centers=centers.reshape(batch_size * num_frames, num_queries, self.num_regions, 3),
            representatives=representatives.reshape(
                batch_size * num_frames,
                num_queries,
                self.num_regions,
                representatives.shape[2],
                3,
            ),
            vertices_by_frame=vertices.reshape(batch_size * num_frames, num_queries, vertices.shape[1], 3),
            depth_by_frame=depth_hw.reshape(batch_size * num_frames, *depth_hw.shape[-2:]),
            intrinsics_by_frame=intrinsics,
            person_valid=person_valid.reshape(batch_size * num_frames, num_queries),
            image_size_hw=image_size_hw,
        )
        return {
            "region_valid": probe["region_valid"].reshape(
                batch_size, num_frames, num_queries, self.num_regions
            ),
            "other_human_ratio": probe["other_human_ratio"].reshape(
                batch_size, num_frames, num_queries, self.num_regions
            ),
        }

    def _fuse_temporal_update(
        self,
        current: torch.Tensor,
        proposed_update: torch.Tensor,
        person_hidden: torch.Tensor,
        person_valid: torch.Tensor,
        track_ids: torch.Tensor | None,
        track_quality: torch.Tensor | None,
        track_gap: torch.Tensor | None,
        pose_enc: torch.Tensor | None,
        predictions: dict[str, torch.Tensor],
        external_memory: object | None,
        frame_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.temporal_gate_head is None or track_ids is None:
            ones = proposed_update.new_ones(*proposed_update.shape[:-1], 1)
            return proposed_update, ones, torch.zeros_like(ones, dtype=torch.bool), torch.zeros_like(proposed_update)
        track_ids = track_ids.to(device=current.device).long()
        quality = (
            track_quality.to(device=current.device, dtype=current.dtype)
            if isinstance(track_quality, torch.Tensor)
            else torch.ones_like(track_ids, dtype=current.dtype)
        )
        configured_gap = (
            track_gap.to(device=current.device).long()
            if isinstance(track_gap, torch.Tensor)
            else torch.ones_like(track_ids)
        )
        current_work, rotation = _to_temporal_work_coords(
            current,
            pose_enc=pose_enc,
            predictions=predictions,
            use_world=self.temporal_use_world,
        )
        update_work = _camera_delta_to_work(proposed_update, rotation)
        fused_work = update_work.clone()
        gate_out = proposed_update.new_ones(*proposed_update.shape[:-1], 1)
        valid_out = torch.zeros_like(gate_out, dtype=torch.bool)
        velocity_out = torch.zeros_like(proposed_update)
        local_states: dict[tuple[int, int], TRSTRTrackState] = {}
        batch_size, num_frames, num_queries = current.shape[:3]
        for batch_idx in range(batch_size):
            for frame_idx in range(num_frames):
                global_frame = int(frame_offset) + frame_idx
                for query_idx in range(num_queries):
                    if not bool(person_valid[batch_idx, frame_idx, query_idx]):
                        continue
                    track_id = int(track_ids[batch_idx, frame_idx, query_idx].detach().cpu())
                    if track_id < 0:
                        continue
                    key = (batch_idx, track_id)
                    state = local_states.get(key)
                    if state is None and batch_idx == 0 and isinstance(external_memory, HSITRSTRTrackMemory):
                        state = external_memory.get(
                            track_id=track_id,
                            frame_index=global_frame,
                            device=current.device,
                            dtype=current.dtype,
                        )
                    gap = global_frame - int(state.last_frame) if state is not None else int(configured_gap[batch_idx, frame_idx, query_idx])
                    quality_value = quality[batch_idx, frame_idx, query_idx]
                    state_valid = (
                        state is not None
                        and 0 < gap <= self.temporal_gap_max
                        and float(quality_value.detach().cpu()) >= self.temporal_quality_min
                    )
                    current_item = current_work[batch_idx, frame_idx, query_idx]
                    proposal_item = update_work[batch_idx, frame_idx, query_idx]
                    if state_valid:
                        prior_position = state.translation + float(gap) * state.velocity
                        prior_delta = (prior_position - current_item).clamp(
                            min=-self.max_person_delta_m,
                            max=self.max_person_delta_m,
                        )
                        temporal_input = torch.cat(
                            [
                                person_hidden[batch_idx, frame_idx, query_idx],
                                proposal_item,
                                prior_delta,
                                quality_value.reshape(1),
                                proposal_item.new_tensor([state.confidence]),
                                F.cosine_similarity(
                                    person_hidden[batch_idx, frame_idx, query_idx],
                                    state.region_token,
                                    dim=0,
                                ).reshape(1)
                                if state.region_token is not None
                                else proposal_item.new_zeros(1),
                                proposal_item.new_tensor([min(float(gap) / float(self.temporal_gap_max), 1.0)]),
                            ],
                            dim=-1,
                        )
                        current_gate = torch.sigmoid(self.temporal_gate_head(temporal_input))
                        fused_item = current_gate * proposal_item + (1.0 - current_gate) * prior_delta
                        gate_out[batch_idx, frame_idx, query_idx] = current_gate
                        valid_out[batch_idx, frame_idx, query_idx] = True
                        velocity = (current_item + fused_item - state.translation) / float(gap)
                    else:
                        fused_item = proposal_item
                        velocity = torch.zeros_like(fused_item)
                    fused_work[batch_idx, frame_idx, query_idx] = fused_item
                    velocity_out[batch_idx, frame_idx, query_idx] = velocity
                    local_states[key] = TRSTRTrackState(
                        track_id=track_id,
                        last_frame=global_frame,
                        translation=current_item + fused_item,
                        velocity=velocity,
                        confidence=float(quality_value.detach().cpu()),
                        region_token=person_hidden[batch_idx, frame_idx, query_idx],
                    )
        return _work_delta_to_camera(fused_work, rotation), gate_out, valid_out, velocity_out

    @torch.no_grad()
    def _write_external_memory(
        self,
        memory: HSITRSTRTrackMemory,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        person_hidden: torch.Tensor,
        person_valid: torch.Tensor,
        track_ids: torch.Tensor | None,
        track_quality: torch.Tensor | None,
        frame_offset: int,
        pose_enc: torch.Tensor | None,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        if track_ids is None or translation.shape[0] != 1:
            return
        quality = track_quality if isinstance(track_quality, torch.Tensor) else torch.ones_like(track_ids, dtype=translation.dtype)
        translation_work, _ = _to_temporal_work_coords(
            translation,
            pose_enc=pose_enc,
            predictions=predictions,
            use_world=self.temporal_use_world,
        )
        for frame_idx in range(translation.shape[1]):
            for query_idx in range(translation.shape[2]):
                if not bool(person_valid[0, frame_idx, query_idx]):
                    continue
                track_id = int(track_ids[0, frame_idx, query_idx].detach().cpu())
                if track_id < 0:
                    continue
                memory.update(
                    track_id=track_id,
                    frame_index=int(frame_offset) + frame_idx,
                    translation=translation_work[0, frame_idx, query_idx],
                    velocity=velocity[0, frame_idx, query_idx],
                    confidence=float(quality[0, frame_idx, query_idx].detach().cpu()),
                    region_token=person_hidden[0, frame_idx, query_idx],
                )


def _canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported TRSTR depth shape: {tuple(depth.shape)}")


def _resolve_intrinsics(
    pose_enc: torch.Tensor | None,
    image_size_hw: tuple[int, int],
    batch_size: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    override: torch.Tensor | None,
) -> torch.Tensor:
    if override is None:
        if pose_enc is None:
            raise ValueError("TRSTR requires pose_enc when intrinsics_override is absent")
        _, intrinsics = encoding_to_camera(pose_enc.float(), image_size_hw=image_size_hw, build_intrinsics=True)
        return intrinsics.reshape(-1, 3, 3).to(device=device, dtype=dtype)
    intrinsics = override.to(device=device, dtype=dtype)
    if intrinsics.ndim == 4 and tuple(intrinsics.shape[:2]) == (batch_size, num_frames):
        return intrinsics.reshape(-1, 3, 3)
    if intrinsics.ndim == 3 and intrinsics.shape[0] == batch_size * num_frames:
        return intrinsics
    raise ValueError(f"Unsupported TRSTR intrinsics shape: {tuple(intrinsics.shape)}")


def _expand_frame_value(value: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    out = value.to(device=depth.device, dtype=depth.dtype)
    while out.ndim < depth.ndim:
        out = out.unsqueeze(-1)
    return out


def _require(mapping: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"TRSTR predictions missing tensor: {key}")
    return value


def _zero_last_linear(module: nn.Sequential) -> nn.Sequential:
    linear = module[-1]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
    return module


def _biased_last_linear(module: nn.Sequential, bias: float) -> nn.Sequential:
    linear = module[-1]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        nn.init.constant_(linear.bias, float(bias))
    return module


def _scaled_region_counts(num_regions: int) -> dict[str, int]:
    """Scale the 96-region risk budget while preserving every anatomy group."""
    num_regions = int(num_regions)
    if num_regions not in {48, 72, 96}:
        raise ValueError(f"Unsupported TRSTR region budget: {num_regions}")
    names = tuple(DEFAULT_REGION_COUNTS.keys())
    base = torch.tensor([DEFAULT_REGION_COUNTS[name] for name in names], dtype=torch.float32)
    raw = base * (float(num_regions) / 96.0)
    counts = torch.floor(raw).long().clamp(min=1)
    remaining = num_regions - int(counts.sum())
    if remaining > 0:
        order = torch.argsort(raw - counts.float(), descending=True)
        for index in order[:remaining].tolist():
            counts[index] += 1
    elif remaining < 0:
        order = torch.argsort(raw - counts.float(), descending=False)
        for index in order.tolist():
            if remaining == 0:
                break
            if counts[index] > 1:
                counts[index] -= 1
                remaining += 1
    if int(counts.sum()) != num_regions:
        raise RuntimeError(f"Could not allocate region budget {num_regions}: {counts.tolist()}")
    return {name: int(count) for name, count in zip(names, counts.tolist(), strict=True)}


def _to_temporal_work_coords(
    translation: torch.Tensor,
    pose_enc: torch.Tensor | None,
    predictions: dict[str, torch.Tensor],
    use_world: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not use_world or pose_enc is None:
        return translation, None
    extrinsics, _ = encoding_to_camera(pose_enc.float(), image_size_hw=(1, 1), build_intrinsics=False)
    extrinsics = extrinsics.to(device=translation.device, dtype=translation.dtype)
    rotation = extrinsics[..., :3, :3]
    camera_translation = extrinsics[..., :3, 3]
    scale = predictions.get("hsi_scene_scale")
    if isinstance(scale, torch.Tensor):
        scale_value = scale.to(device=translation.device, dtype=translation.dtype)
        while scale_value.ndim > 3:
            scale_value = scale_value.squeeze(-1)
        camera_translation = camera_translation * scale_value.reshape(*camera_translation.shape[:-1], 1)
    centered = translation - camera_translation[:, :, None, :]
    world = torch.einsum("bsij,bsqj->bsqi", rotation.transpose(-1, -2), centered)
    return world, rotation


def _camera_delta_to_work(delta: torch.Tensor, rotation: torch.Tensor | None) -> torch.Tensor:
    if rotation is None:
        return delta
    return torch.einsum("bsij,bsqj->bsqi", rotation.transpose(-1, -2), delta)


def _work_delta_to_camera(delta: torch.Tensor, rotation: torch.Tensor | None) -> torch.Tensor:
    if rotation is None:
        return delta
    return torch.einsum("bsij,bsqj->bsqi", rotation, delta)
