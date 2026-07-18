from __future__ import annotations

from dataclasses import dataclass

import torch


TRACK_SOURCE_NEW = 0
TRACK_SOURCE_MATCHED = 1
TRACK_SOURCE_EXTERNAL = 2


@dataclass(slots=True)
class _TrackState:
    track_id: int
    last_frame: int
    box: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor
    confidence: float
    external_id: int = -1
    id_embed: torch.Tensor | None = None


class BaseSMPLTrackAssigner:
    """No-grad geometry-aware track assignment over base SMPL predictions."""

    def __init__(
        self,
        max_age: int = 90,
        min_track_quality: float = 0.25,
        max_center_distance_norm: float = 0.25,
        max_transl_distance_m: float = 1.50,
        max_beta_l1: float = 0.30,
        external_prior_iou_min: float = 0.50,
        id_weight: float = 0.0,
        max_id_distance: float = 0.70,
    ) -> None:
        self.max_age = int(max_age)
        self.min_track_quality = float(min_track_quality)
        self.max_center_distance_norm = float(max_center_distance_norm)
        self.max_transl_distance_m = float(max_transl_distance_m)
        self.max_beta_l1 = float(max_beta_l1)
        self.external_prior_iou_min = float(external_prior_iou_min)
        self.id_weight = min(max(float(id_weight), 0.0), 1.0)
        self.max_id_distance = max(float(max_id_distance), 0.0)

    @torch.no_grad()
    def assign(
        self,
        boxes: torch.Tensor,
        pred_betas: torch.Tensor,
        pred_transl_cam: torch.Tensor,
        pred_confs: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        external_track_ids: torch.Tensor | None = None,
        external_track_mask: torch.Tensor | None = None,
        external_track_confidence: torch.Tensor | None = None,
        pred_id_embed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes must have shape [B,S,Q,4], got {tuple(boxes.shape)}")
        if pred_betas.shape[:3] != boxes.shape[:3] or pred_transl_cam.shape[:3] != boxes.shape[:3]:
            raise ValueError("pred_betas/pred_transl_cam must share [B,S,Q] with boxes")
        batch_size, num_frames, num_queries, _ = boxes.shape
        device = boxes.device
        if query_mask is None:
            query_mask = torch.ones(batch_size, num_frames, num_queries, dtype=torch.bool, device=device)
        else:
            query_mask = query_mask.to(device=device).bool()
        conf = pred_confs.detach().float()
        while conf.ndim > 3:
            conf = conf.mean(dim=-1)
        conf = conf.to(device=device)

        assigned_ids = torch.full((batch_size, num_frames, num_queries), -1, dtype=torch.long, device=device)
        assigned_mask = torch.zeros((batch_size, num_frames, num_queries), dtype=torch.bool, device=device)
        assigned_quality = torch.zeros((batch_size, num_frames, num_queries), dtype=torch.float32, device=device)
        assigned_gap = torch.zeros((batch_size, num_frames, num_queries), dtype=torch.long, device=device)
        assigned_source = torch.full((batch_size, num_frames, num_queries), -1, dtype=torch.long, device=device)

        boxes_f = boxes.detach().float()
        betas_f = pred_betas.detach().float()
        transl_f = pred_transl_cam.detach().float()
        id_f = None
        if pred_id_embed is not None:
            if pred_id_embed.shape[:3] != boxes.shape[:3]:
                raise ValueError("pred_id_embed must share [B,S,Q] with boxes")
            id_f = torch.nn.functional.normalize(pred_id_embed.detach().float(), dim=-1)
        external_ids = external_track_ids.to(device=device).long() if external_track_ids is not None else None
        external_mask = external_track_mask.to(device=device).bool() if external_track_mask is not None else None
        external_conf = external_track_confidence.to(device=device).float() if external_track_confidence is not None else None

        for batch_idx in range(batch_size):
            tracks: dict[int, _TrackState] = {}
            next_id = 0
            for frame_idx in range(num_frames):
                candidates = []
                for query_idx in range(num_queries):
                    if not bool(query_mask[batch_idx, frame_idx, query_idx]):
                        continue
                    ext_id = self._external_id(external_ids, external_mask, batch_idx, frame_idx, query_idx)
                    ext_conf = self._external_confidence(external_conf, external_mask, batch_idx, frame_idx, query_idx)
                    best_track = None
                    best_score = float("-inf")
                    best_gap = 0
                    best_source = TRACK_SOURCE_MATCHED
                    for track in tracks.values():
                        gap = frame_idx - track.last_frame
                        if gap <= 0 or gap > self.max_age:
                            continue
                        score = self._score(
                            boxes_f[batch_idx, frame_idx, query_idx],
                            betas_f[batch_idx, frame_idx, query_idx],
                            transl_f[batch_idx, frame_idx, query_idx],
                            float(conf[batch_idx, frame_idx, query_idx]),
                            track,
                            ext_id,
                            ext_conf,
                            gap,
                            None if id_f is None else id_f[batch_idx, frame_idx, query_idx],
                        )
                        if score > best_score:
                            best_score = score
                            best_track = track
                            best_gap = gap
                            best_source = TRACK_SOURCE_EXTERNAL if ext_id >= 0 and ext_id == track.external_id else TRACK_SOURCE_MATCHED
                    candidates.append((best_score, query_idx, best_track, best_gap, best_source, ext_id))

                candidates.sort(key=lambda item: item[0], reverse=True)
                used_tracks: set[int] = set()
                for score, query_idx, track, gap, source, ext_id in candidates:
                    if track is None or score < self.min_track_quality or track.track_id in used_tracks:
                        track_id = next_id
                        next_id += 1
                        source = TRACK_SOURCE_NEW if ext_id < 0 else TRACK_SOURCE_EXTERNAL
                        quality = max(float(conf[batch_idx, frame_idx, query_idx]), self.min_track_quality)
                    else:
                        track_id = track.track_id
                        used_tracks.add(track_id)
                        quality = float(max(min(score, 1.0), 0.0))

                    tracks[track_id] = _TrackState(
                        track_id=track_id,
                        last_frame=frame_idx,
                        box=boxes_f[batch_idx, frame_idx, query_idx].detach(),
                        transl=transl_f[batch_idx, frame_idx, query_idx].detach(),
                        betas=betas_f[batch_idx, frame_idx, query_idx].detach(),
                        confidence=float(conf[batch_idx, frame_idx, query_idx]),
                        external_id=ext_id,
                        id_embed=(None if id_f is None else id_f[batch_idx, frame_idx, query_idx].detach()),
                    )
                    assigned_ids[batch_idx, frame_idx, query_idx] = int(track_id)
                    assigned_mask[batch_idx, frame_idx, query_idx] = True
                    assigned_quality[batch_idx, frame_idx, query_idx] = float(quality)
                    assigned_gap[batch_idx, frame_idx, query_idx] = int(max(gap, 0))
                    assigned_source[batch_idx, frame_idx, query_idx] = int(source)

                dead = [track_id for track_id, track in tracks.items() if frame_idx - track.last_frame > self.max_age]
                for track_id in dead:
                    del tracks[track_id]

        return {
            "assigned_track_ids": assigned_ids,
            "assigned_track_mask": assigned_mask,
            "assigned_track_quality": assigned_quality,
            "assigned_track_gap": assigned_gap,
            "assigned_track_source": assigned_source,
        }

    def _score(
        self,
        box: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        confidence: float,
        track: _TrackState,
        external_id: int,
        external_confidence: float,
        gap: int,
        id_embed: torch.Tensor | None,
    ) -> float:
        center_dist = torch.linalg.norm(box[:2] - track.box[:2]).item()
        transl_dist = torch.linalg.norm(transl - track.transl).item()
        beta_l1 = torch.mean(torch.abs(betas - track.betas)).item()
        if center_dist > self.max_center_distance_norm or transl_dist > self.max_transl_distance_m or beta_l1 > self.max_beta_l1:
            return float("-inf")
        bbox_score = max(0.0, 1.0 - center_dist / max(self.max_center_distance_norm, 1e-6))
        transl_score = max(0.0, 1.0 - transl_dist / max(self.max_transl_distance_m, 1e-6))
        beta_score = max(0.0, 1.0 - beta_l1 / max(self.max_beta_l1, 1e-6))
        confidence_score = max(0.0, min(float(confidence), 1.0))
        external_score = max(0.0, min(float(external_confidence), 1.0)) if external_id >= 0 and external_id == track.external_id else 0.0
        id_score = None
        if self.id_weight > 0.0 and id_embed is not None and track.id_embed is not None:
            cosine = float(torch.dot(id_embed, track.id_embed).clamp(-1.0, 1.0).item())
            id_distance = 1.0 - cosine
            if id_distance > self.max_id_distance:
                return float("-inf")
            id_score = max(0.0, min(1.0, 1.0 - id_distance / max(self.max_id_distance, 1e-6)))
        gap_penalty = 0.01 * max(int(gap) - 1, 0)
        geometry_score = (
            0.35 * bbox_score
            + 0.35 * transl_score
            + 0.15 * beta_score
            + 0.10 * confidence_score
            + 0.05 * external_score
        )
        if id_score is not None:
            return (1.0 - self.id_weight) * geometry_score + self.id_weight * id_score - gap_penalty
        return geometry_score - gap_penalty

    @staticmethod
    def _external_id(
        external_ids: torch.Tensor | None,
        external_mask: torch.Tensor | None,
        batch_idx: int,
        frame_idx: int,
        query_idx: int,
    ) -> int:
        if external_ids is None:
            return -1
        if external_mask is not None and not bool(external_mask[batch_idx, frame_idx, query_idx]):
            return -1
        value = int(external_ids[batch_idx, frame_idx, query_idx].detach().cpu())
        return value if value >= 0 else -1

    @staticmethod
    def _external_confidence(
        external_confidence: torch.Tensor | None,
        external_mask: torch.Tensor | None,
        batch_idx: int,
        frame_idx: int,
        query_idx: int,
    ) -> float:
        if external_confidence is None:
            return 1.0
        if external_mask is not None and not bool(external_mask[batch_idx, frame_idx, query_idx]):
            return 0.0
        return float(external_confidence[batch_idx, frame_idx, query_idx].detach().cpu())
