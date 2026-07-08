from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import axis_angle_to_rot6d


class NLFSMPLProvider(nn.Module):
    """Adapt NLF ragged SMPL predictions to VGGTOmega's dense SMPL contract.

    NLF must see the same image plane as the camera intrinsics passed to it.
    This provider therefore consumes the already-processed/padded VGGT input
    tensor and decodes intrinsics with the runtime ``images.shape[-2:]``.
    """

    def __init__(
        self,
        model_path: str = "",
        third_party_root: str = "third_party/nlf",
        model_name: str = "smpl",
        use_detector: bool = False,
        require_boxes: bool = True,
        internal_batch_size: int = 64,
        num_aug: int = 1,
        detector_threshold: float = 0.3,
        detector_nms_iou_threshold: float = 0.7,
        max_detections: int = 150,
        model_loader: Callable[[torch.device], Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = str(model_path or "")
        self.third_party_root = str(third_party_root or "third_party/nlf")
        self.model_name = str(model_name or "smpl")
        self.use_detector = bool(use_detector)
        self.require_boxes = bool(require_boxes)
        self.internal_batch_size = int(internal_batch_size)
        self.num_aug = int(num_aug)
        self.detector_threshold = float(detector_threshold)
        self.detector_nms_iou_threshold = float(detector_nms_iou_threshold)
        self.max_detections = int(max_detections)
        self.__dict__["_nlf_model"] = None
        self.__dict__["_model_loader"] = model_loader

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        pose_enc: torch.Tensor,
        smpl_query_boxes: torch.Tensor | None = None,
        smpl_query_boxes_mask: torch.Tensor | None = None,
        max_humans: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"NLFSMPLProvider expects images [B,S,3,H,W], got {tuple(images.shape)}")
        if pose_enc is None:
            raise ValueError("NLFSMPLProvider requires VGGT pose_enc to decode intrinsics")
        batch_size, num_frames, channels, image_h, image_w = images.shape
        if channels != 3:
            raise ValueError(f"NLFSMPLProvider expects RGB images with 3 channels, got {channels}")
        num_queries = int(max_humans or (smpl_query_boxes.shape[2] if smpl_query_boxes is not None else 0))
        if num_queries <= 0:
            raise ValueError("NLFSMPLProvider requires max_humans/num_smpl_queries > 0")

        flat_images = images.reshape(batch_size * num_frames, channels, image_h, image_w).detach()
        flat_images = flat_images.to(dtype=torch.float32).clamp(0.0, 1.0)
        _, intrinsics = encoding_to_camera(
            pose_enc.detach().float(),
            image_size_hw=(int(image_h), int(image_w)),
            build_intrinsics=True,
        )
        if intrinsics is None:
            raise RuntimeError("encoding_to_camera did not return intrinsics for NLF")
        flat_intrinsics = intrinsics.reshape(batch_size * num_frames, 3, 3).to(device=flat_images.device, dtype=torch.float32)

        slot_indices: list[torch.Tensor] | None = None
        nlf_boxes: list[torch.Tensor] | None = None
        if not self.use_detector:
            if smpl_query_boxes is None:
                if self.require_boxes:
                    raise ValueError(
                        "NLF provider is configured with use_detector=false, but smpl_query_boxes were not provided. "
                        "Pass processed-image query boxes or set nlf_use_detector=true for demo inference."
                    )
            else:
                nlf_boxes, slot_indices = self._build_box_lists(
                    smpl_query_boxes=smpl_query_boxes,
                    smpl_query_boxes_mask=smpl_query_boxes_mask,
                    image_hw=(int(image_h), int(image_w)),
                    device=flat_images.device,
                )

        model = self._load_model(flat_images.device)
        result = self._run_nlf(model, flat_images, flat_intrinsics, nlf_boxes)
        outputs = self._ragged_to_dense(
            result=result,
            batch_size=batch_size,
            num_frames=num_frames,
            num_queries=num_queries,
            image_hw=(int(image_h), int(image_w)),
            slot_indices=slot_indices,
            device=images.device,
            dtype=torch.float32,
        )
        outputs["pred_cam"] = outputs["pred_transl_cam"]
        outputs["base_pred_transl_cam"] = outputs["pred_transl_cam"]
        outputs["nlf_intrinsics"] = flat_intrinsics.reshape(batch_size, num_frames, 3, 3).to(device=images.device)
        outputs["nlf_image_hw"] = torch.tensor([int(image_h), int(image_w)], device=images.device, dtype=torch.long)
        return outputs

    def _load_model(self, device: torch.device) -> Any:
        model = self.__dict__.get("_nlf_model")
        if model is not None:
            if hasattr(model, "to"):
                model = model.to(device)
                self.__dict__["_nlf_model"] = model
            return model

        loader = self.__dict__.get("_model_loader")
        if loader is not None:
            model = loader(device)
        else:
            self._add_third_party_root()
            if not self.model_path:
                raise FileNotFoundError(
                    "NLF checkpoint path is empty. Set model.nlf_model_path or checkpoints.nlf_smpl in the config."
                )
            model_path = Path(self.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path
            if not model_path.is_file():
                raise FileNotFoundError(f"NLF checkpoint not found: {model_path}")
            _ensure_torchvision_nms_registered()
            model = torch.jit.load(str(model_path), map_location=device)
        if hasattr(model, "eval"):
            model = model.eval()
        if hasattr(model, "to"):
            model = model.to(device)
        self.__dict__["_nlf_model"] = model
        return model

    def _add_third_party_root(self) -> None:
        root = Path(self.third_party_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))

    def _run_nlf(
        self,
        model: Any,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        boxes: list[torch.Tensor] | None,
    ) -> dict[str, Any]:
        common = {
            "intrinsic_matrix": intrinsics,
            "distortion_coeffs": None,
            "extrinsic_matrix": None,
            "internal_batch_size": self.internal_batch_size,
            "num_aug": self.num_aug,
            "model_name": self.model_name,
        }
        if self.use_detector:
            if not hasattr(model, "detect_smpl_batched"):
                raise AttributeError("Loaded NLF model does not expose detect_smpl_batched")
            return model.detect_smpl_batched(
                images,
                detector_threshold=self.detector_threshold,
                detector_nms_iou_threshold=self.detector_nms_iou_threshold,
                max_detections=self.max_detections,
                **common,
            )
        if boxes is None:
            boxes = [torch.zeros(0, 4, device=images.device, dtype=torch.float32) for _ in range(images.shape[0])]
        if not hasattr(model, "estimate_smpl_batched"):
            raise AttributeError("Loaded NLF model does not expose estimate_smpl_batched")
        return model.estimate_smpl_batched(images, boxes, **common)

    @staticmethod
    def _build_box_lists(
        smpl_query_boxes: torch.Tensor,
        smpl_query_boxes_mask: torch.Tensor | None,
        image_hw: tuple[int, int],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if smpl_query_boxes.ndim != 4 or smpl_query_boxes.shape[-1] != 4:
            raise ValueError(f"smpl_query_boxes must have shape [B,S,Q,4], got {tuple(smpl_query_boxes.shape)}")
        batch_size, num_frames, num_queries, _ = smpl_query_boxes.shape
        boxes = smpl_query_boxes.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if smpl_query_boxes_mask is None:
            mask = torch.ones(batch_size, num_frames, num_queries, dtype=torch.bool, device=device)
        else:
            mask = smpl_query_boxes_mask.to(device=device).bool()
            if mask.shape != (batch_size, num_frames, num_queries):
                raise ValueError(
                    "smpl_query_boxes_mask must have shape "
                    f"[B,S,Q]={batch_size, num_frames, num_queries}, got {tuple(mask.shape)}"
                )
        flat_xywh = _normalized_cxcywh_to_xywh(boxes, image_hw).reshape(batch_size * num_frames, num_queries, 4)
        flat_mask = mask.reshape(batch_size * num_frames, num_queries)
        box_lists = []
        slot_indices = []
        for frame_idx in range(batch_size * num_frames):
            valid_idx = torch.nonzero(flat_mask[frame_idx], as_tuple=False).reshape(-1)
            slot_indices.append(valid_idx)
            box_lists.append(flat_xywh[frame_idx, valid_idx].contiguous())
        return box_lists, slot_indices

    @staticmethod
    def _ragged_to_dense(
        result: dict[str, Any],
        batch_size: int,
        num_frames: int,
        num_queries: int,
        image_hw: tuple[int, int],
        slot_indices: list[torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        pose_out = torch.zeros(batch_size * num_frames, num_queries, 72, device=device, dtype=dtype)
        pose6d_out = torch.zeros(batch_size * num_frames, num_queries, 144, device=device, dtype=dtype)
        betas_out = torch.zeros(batch_size * num_frames, num_queries, 10, device=device, dtype=dtype)
        transl_out = torch.zeros(batch_size * num_frames, num_queries, 3, device=device, dtype=dtype)
        conf_out = torch.zeros(batch_size * num_frames, num_queries, 1, device=device, dtype=dtype)
        boxes_out = torch.zeros(batch_size * num_frames, num_queries, 4, device=device, dtype=dtype)

        poses = _as_ragged(result.get("pose"), batch_size * num_frames)
        betas = _as_ragged(result.get("betas"), batch_size * num_frames)
        trans = _as_ragged(result.get("trans"), batch_size * num_frames)
        boxes = _as_ragged(result.get("boxes"), batch_size * num_frames)

        for frame_idx in range(batch_size * num_frames):
            pose_i = _coerce_pose(poses[frame_idx], device=device, dtype=dtype)
            betas_i = _coerce_last_dim(betas[frame_idx], 10, device=device, dtype=dtype)
            trans_i = _coerce_last_dim(trans[frame_idx], 3, device=device, dtype=dtype)
            boxes_i = _coerce_last_dim(boxes[frame_idx], 4, device=device, dtype=dtype, allow_extra=True)
            num_people = min(pose_i.shape[0], betas_i.shape[0], trans_i.shape[0])
            if boxes_i.shape[0] > 0:
                num_people = min(num_people, boxes_i.shape[0])
            if num_people <= 0:
                continue
            if slot_indices is None:
                write_slots = torch.arange(min(num_people, num_queries), device=device)
                read_idx = torch.arange(write_slots.numel(), device=device)
            else:
                candidate_slots = slot_indices[frame_idx].to(device=device)
                count = min(num_people, candidate_slots.numel())
                write_slots = candidate_slots[:count]
                read_idx = torch.arange(count, device=device)
            if write_slots.numel() == 0:
                continue

            pose_sel = pose_i[read_idx]
            pose_out[frame_idx, write_slots] = pose_sel
            pose6d_out[frame_idx, write_slots] = axis_angle_to_rot6d(pose_sel.reshape(-1, 24, 3)).reshape(-1, 144)
            betas_out[frame_idx, write_slots] = betas_i[read_idx, :10]
            transl_out[frame_idx, write_slots] = trans_i[read_idx, :3]
            if boxes_i.shape[0] > 0:
                boxes_xywh = boxes_i[read_idx, :4]
                boxes_out[frame_idx, write_slots] = _xywh_to_normalized_cxcywh(boxes_xywh, image_hw)
                if boxes_i.shape[-1] >= 5:
                    conf_out[frame_idx, write_slots, 0] = boxes_i[read_idx, 4].clamp(0.0, 1.0)
                else:
                    conf_out[frame_idx, write_slots, 0] = 1.0
            else:
                conf_out[frame_idx, write_slots, 0] = 1.0

        shape = (batch_size, num_frames, num_queries)
        return {
            "pred_poses": pose_out.reshape(*shape, 72),
            "pred_pose_6d": pose6d_out.reshape(*shape, 144),
            "pred_betas": betas_out.reshape(*shape, 10),
            "pred_transl_cam": transl_out.reshape(*shape, 3),
            "pred_confs": conf_out.reshape(*shape, 1),
            "pred_boxes": boxes_out.reshape(*shape, 4),
            "nlf_valid_mask": (conf_out.reshape(*shape, 1) > 0),
        }


def _as_ragged(value: Any, expected_len: int) -> list[Any]:
    if value is None:
        return [None for _ in range(expected_len)]
    if isinstance(value, (list, tuple)):
        if len(value) != expected_len:
            raise ValueError(f"Expected ragged output length {expected_len}, got {len(value)}")
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.shape[0] != expected_len:
            raise ValueError(f"Expected tensor output first dimension {expected_len}, got {tuple(value.shape)}")
        return [value[i] for i in range(expected_len)]
    raise TypeError(f"Unsupported NLF ragged output type: {type(value)!r}")


def _coerce_pose(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = _to_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 0:
        return torch.zeros(0, 72, device=device, dtype=dtype)
    tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[-1] < 72:
        raise ValueError(f"NLF SMPL pose must have at least 72 values, got {tuple(tensor.shape)}")
    return tensor[:, :72]


def _coerce_last_dim(
    value: Any,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_extra: bool = False,
) -> torch.Tensor:
    tensor = _to_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 0:
        return torch.zeros(0, size, device=device, dtype=dtype)
    tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[-1] < size:
        raise ValueError(f"Expected last dimension at least {size}, got {tuple(tensor.shape)}")
    return tensor if allow_extra else tensor[:, :size]


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(0, device=device, dtype=dtype)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _normalized_cxcywh_to_xywh(boxes: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    cx, cy, bw, bh = boxes.unbind(dim=-1)
    x1 = (cx - 0.5 * bw) * float(image_w)
    y1 = (cy - 0.5 * bh) * float(image_h)
    return torch.stack([x1, y1, bw * float(image_w), bh * float(image_h)], dim=-1)


def _xywh_to_normalized_cxcywh(xywh: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    x1, y1, bw, bh = xywh.unbind(dim=-1)
    bw = bw.clamp(min=0.0)
    bh = bh.clamp(min=0.0)
    return torch.stack(
        [
            (x1 + 0.5 * bw) / max(float(image_w), 1.0),
            (y1 + 0.5 * bh) / max(float(image_h), 1.0),
            bw / max(float(image_w), 1.0),
            bh / max(float(image_h), 1.0),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)


def _ensure_torchvision_nms_registered() -> None:
    """Register torchvision custom ops required by the serialized NLF detector."""
    try:
        import torchvision
        import torchvision.ops  # noqa: F401

        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        torchvision.ops.nms(boxes, scores, 0.5)
    except Exception as exc:
        torch_version = getattr(torch, "__version__", "unknown")
        try:
            import torchvision as _torchvision

            torchvision_version = getattr(_torchvision, "__version__", "unknown")
        except Exception:
            torchvision_version = "unavailable"
        raise RuntimeError(
            "NLF TorchScript requires the torchvision::nms custom op, but it is not available. "
            "Install a torchvision build that matches the active torch/CUDA environment, then rerun the smoke test. "
            f"torch={torch_version}, torchvision={torchvision_version}"
        ) from exc
