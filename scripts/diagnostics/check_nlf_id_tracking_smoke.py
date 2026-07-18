from __future__ import annotations

import torch

from vggt_omega.models.heads.smpl_head import SMPLIdentityHead, SMPLROIIdentityHead
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner


def main() -> None:
    check_identity_head()
    check_roi_identity_head()
    check_embedding_aware_assignment()
    print("[ok] NLF ID tracking smoke checks passed")


def check_identity_head() -> None:
    head = SMPLIdentityHead(dim_in=32, hidden_dim=16, id_embed_dim=8)
    output = head(torch.randn(2, 3, 4, 32))
    assert tuple(output.shape) == (2, 3, 4, 8)
    assert torch.allclose(torch.linalg.norm(output, dim=-1), torch.ones(2, 3, 4), atol=1e-5)


def check_roi_identity_head() -> None:
    head = SMPLROIIdentityHead(query_dim=32, roi_dim=70, hidden_dim=16, id_embed_dim=8)
    output = head(torch.randn(2, 3, 4, 32), torch.randn(2, 3, 4, 70))
    assert tuple(output.shape) == (2, 3, 4, 8)
    assert torch.allclose(torch.linalg.norm(output, dim=-1), torch.ones(2, 3, 4), atol=1e-5)


def check_embedding_aware_assignment() -> None:
    assigner = BaseSMPLTrackAssigner(
        max_center_distance_norm=0.5,
        max_transl_distance_m=2.0,
        max_beta_l1=1.0,
        id_weight=0.5,
        max_id_distance=0.4,
    )
    boxes = torch.tensor(
        [[[[0.30, 0.5, 0.2, 0.4], [0.70, 0.5, 0.2, 0.4]],
          [[0.32, 0.5, 0.2, 0.4], [0.68, 0.5, 0.2, 0.4]]]],
        dtype=torch.float32,
    )
    betas = torch.zeros(1, 2, 2, 10)
    transl = torch.zeros(1, 2, 2, 3)
    conf = torch.ones(1, 2, 2, 1)
    embeddings = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    output = assigner.assign(boxes, betas, transl, conf, pred_id_embed=embeddings)
    ids = output["assigned_track_ids"][0]
    assert int(ids[0, 0]) == int(ids[1, 1])
    assert int(ids[0, 1]) == int(ids[1, 0])


if __name__ == "__main__":
    main()
