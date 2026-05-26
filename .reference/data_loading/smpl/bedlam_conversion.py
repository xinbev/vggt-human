from __future__ import annotations

import torch

from .rotation import IDENTITY_6D, aa_to_6d


def smplx_body_to_smpl_joints(
    persons: list[dict],
    max_humans: int = 10,
    smpl_model=None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Convert one frame of BEDLAM SMPL-X person dicts to padded SMPL tensors."""
    max_slots = int(max_humans)
    n_persons = min(len(persons), max_slots)

    joints3d_list: list[torch.Tensor] = []
    pose_6d_list: list[torch.Tensor] = []
    betas_list: list[torch.Tensor] = []
    cam_trans_list: list[torch.Tensor] = []

    for person in persons[:n_persons]:
        root_pose = torch.as_tensor(person["smplx_root_pose"], dtype=torch.float32).reshape(1, 3)
        body_pose = torch.as_tensor(person["smplx_body_pose"], dtype=torch.float32).reshape(21, 3)
        shape = torch.as_tensor(person["smplx_shape"], dtype=torch.float32).reshape(-1)
        transl = torch.as_tensor(person["smplx_transl"], dtype=torch.float32).reshape(3)

        betas = shape[:10]
        cam_trans = transl

        all_aa = torch.cat([root_pose, body_pose], dim=0)
        pose_6d_22 = aa_to_6d(all_aa)
        identity_6d = IDENTITY_6D.unsqueeze(0).expand(2, -1)
        pose_6d = torch.cat([pose_6d_22, identity_6d], dim=0).reshape(144)

        smpl_body_pose_aa = torch.cat([body_pose, torch.zeros(2, 3, dtype=torch.float32)], dim=0)
        if smpl_model is not None:
            smpl_device = next(iter(smpl_model.parameters())).device
            with torch.no_grad():
                smpl_out = smpl_model(
                    betas=betas.unsqueeze(0).to(smpl_device),
                    global_orient=root_pose.to(smpl_device),
                    body_pose=smpl_body_pose_aa.reshape(1, 69).to(smpl_device),
                    transl=transl.unsqueeze(0).to(smpl_device),
                )
                joints3d = smpl_out.joints[0, :24].cpu()
        else:
            joints3d = torch.zeros(24, 3)
            joints3d[0] = cam_trans

        joints3d_list.append(joints3d)
        pose_6d_list.append(pose_6d)
        betas_list.append(betas)
        cam_trans_list.append(cam_trans)

    def pad_slots(values: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
        if not values:
            return torch.zeros(max_slots, *shape)
        stacked = torch.stack(values, dim=0)
        if stacked.shape[0] < max_slots:
            pad = torch.zeros(max_slots - stacked.shape[0], *shape)
            stacked = torch.cat([stacked, pad], dim=0)
        return stacked

    smpl_mask = torch.zeros(max_slots, dtype=torch.bool)
    smpl_mask[:n_persons] = True
    return {
        "joints3d": pad_slots(joints3d_list, (24, 3)).to(device),
        "pose_6d": pad_slots(pose_6d_list, (144,)).to(device),
        "betas": pad_slots(betas_list, (10,)).to(device),
        "cam_trans": pad_slots(cam_trans_list, (3,)).to(device),
        "smpl_mask": smpl_mask.to(device),
    }


def build_smpl_batch_from_persons(
    persons_per_frame: list[list[dict]],
    max_humans: int = 10,
    smpl_model=None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Convert S frames of BEDLAM person dicts into stacked [S, M, ...] tensors."""
    results = [
        smplx_body_to_smpl_joints(persons, max_humans, smpl_model, device)
        for persons in persons_per_frame
    ]
    if not results:
        raise ValueError("persons_per_frame must contain at least one frame")
    return {key: torch.stack([result[key] for result in results], dim=0) for key in results[0]}
