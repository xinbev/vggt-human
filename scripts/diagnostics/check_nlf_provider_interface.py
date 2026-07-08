from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.integrations import NLFSMPLProvider


class FakeNLFModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> "FakeNLFModel":
        return self

    def to(self, device: torch.device) -> "FakeNLFModel":
        return self

    def estimate_smpl_batched(
        self,
        images: torch.Tensor,
        boxes: list[torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: Any,
        extrinsic_matrix: Any,
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        self.calls.append(
            {
                "images_shape": tuple(images.shape),
                "boxes": [box.detach().cpu() for box in boxes],
                "intrinsics": intrinsic_matrix.detach().cpu(),
                "distortion_coeffs": distortion_coeffs,
                "extrinsic_matrix": extrinsic_matrix,
                "kwargs": dict(kwargs),
            }
        )
        pose: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        trans: list[torch.Tensor] = []
        out_boxes: list[torch.Tensor] = []
        for frame_idx, frame_boxes in enumerate(boxes):
            count = int(frame_boxes.shape[0])
            pose.append(torch.zeros(count, 24, 3, device=images.device))
            betas.append(torch.full((count, 10), 0.01 * (frame_idx + 1), device=images.device))
            frame_trans = torch.zeros(count, 3, device=images.device)
            if count:
                frame_trans[:, 2] = 2.0 + 0.1 * frame_idx
            trans.append(frame_trans)
            if count:
                conf = torch.linspace(0.95, 0.85, count, device=images.device).unsqueeze(-1)
                out_boxes.append(torch.cat([frame_boxes.to(images.device), conf], dim=-1))
            else:
                out_boxes.append(torch.zeros(0, 5, device=images.device))
        return {"pose": pose, "betas": betas, "trans": trans, "boxes": out_boxes}


def main() -> None:
    torch.manual_seed(7)
    fake = FakeNLFModel()
    provider = NLFSMPLProvider(
        model_loader=lambda device: fake,
        use_detector=False,
        require_boxes=True,
        model_name="smpl",
    )
    batch_size, num_frames, num_queries = 1, 2, 3
    image_h, image_w = 480, 544
    images = torch.rand(batch_size, num_frames, 3, image_h, image_w)
    pose_enc = torch.zeros(batch_size, num_frames, 9)
    pose_enc[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    pose_enc[..., 7] = torch.tensor(2.0 * torch.atan(torch.tensor((image_h / 2.0) / 600.0)))
    pose_enc[..., 8] = torch.tensor(2.0 * torch.atan(torch.tensor((image_w / 2.0) / 700.0)))
    boxes = torch.tensor(
        [
            [
                [
                    [0.50, 0.50, 0.25, 0.40],
                    [0.20, 0.30, 0.10, 0.20],
                    [0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.60, 0.45, 0.30, 0.35],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[True, True, False], [True, False, False]]])

    out = provider(
        images=images,
        pose_enc=pose_enc,
        smpl_query_boxes=boxes,
        smpl_query_boxes_mask=mask,
        max_humans=num_queries,
    )

    assert fake.calls, "Fake NLF model was not called"
    call = fake.calls[0]
    assert call["images_shape"] == (batch_size * num_frames, 3, image_h, image_w)
    assert call["distortion_coeffs"] is None
    assert call["extrinsic_matrix"] is None
    intrinsics = call["intrinsics"]
    assert torch.allclose(intrinsics[:, 0, 2], torch.full((2,), image_w / 2.0))
    assert torch.allclose(intrinsics[:, 1, 2], torch.full((2,), image_h / 2.0))
    assert torch.allclose(intrinsics[:, 0, 0], torch.full((2,), 700.0), atol=1e-4)
    assert torch.allclose(intrinsics[:, 1, 1], torch.full((2,), 600.0), atol=1e-4)

    first_box = call["boxes"][0][0]
    expected_first_box = torch.tensor([204.0, 144.0, 136.0, 192.0])
    assert torch.allclose(first_box, expected_first_box, atol=1e-4), (first_box, expected_first_box)
    assert tuple(out["pred_poses"].shape) == (batch_size, num_frames, num_queries, 72)
    assert tuple(out["pred_pose_6d"].shape) == (batch_size, num_frames, num_queries, 144)
    assert tuple(out["pred_betas"].shape) == (batch_size, num_frames, num_queries, 10)
    assert tuple(out["pred_transl_cam"].shape) == (batch_size, num_frames, num_queries, 3)
    assert tuple(out["pred_confs"].shape) == (batch_size, num_frames, num_queries, 1)
    assert tuple(out["pred_boxes"].shape) == (batch_size, num_frames, num_queries, 4)
    assert tuple(out["nlf_intrinsics"].shape) == (batch_size, num_frames, 3, 3)
    assert tuple(out["nlf_image_hw"].tolist()) == (image_h, image_w)
    assert torch.isfinite(out["pred_transl_cam"]).all()
    assert float(out["pred_confs"][0, 0, 0, 0]) > 0.0
    assert float(out["pred_confs"][0, 0, 2, 0]) == 0.0
    assert bool(out["nlf_valid_mask"][0, 1, 0, 0])
    assert not bool(out["nlf_valid_mask"][0, 1, 1, 0])

    output_dir = Path("outputs/debug/nlf_provider_interface_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "image_hw": [image_h, image_w],
        "intrinsics_cxy": intrinsics[:, :2, 2].tolist(),
        "first_box_xywh": first_box.tolist(),
        "output_shapes": {key: list(value.shape) for key, value in out.items() if isinstance(value, torch.Tensor)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] NLF provider interface smoke passed")
    print(json.dumps({"summary": str(output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
