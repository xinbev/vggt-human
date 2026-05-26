"""PyTorch Dataset for preprocessed BEDLAM data.

Expected directory layout:

    processed_bedlam/
      Training/
        <scene>_<seq>/
          rgb/    frame_*.png
          depth/  frame_*.npy
          cam/    frame_*.npz   {intrinsics:[3,3], pose:[4,4] w2c}
          smpl/   frame_*.pkl   list[dict] per-person SMPL-X params
      Test/
        ...

The returned item schema is documented in ``portable_data_loading/README.md``.
"""

from __future__ import annotations

import os
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from portable_data_loading.preprocessing.camera import scale_intrinsics_for_resize
from portable_data_loading.preprocessing.image import load_image_tensor, load_multihmr_letterbox_tensor
from portable_data_loading.preprocessing.letterbox import letterbox_intrinsics
from portable_data_loading.smpl.bedlam_conversion import build_smpl_batch_from_persons

from .indexing import build_sequence_index
from .io import load_depth_tensor, load_intrinsics, load_persons


class BedlamDataset(Dataset):
    """Read preprocessed BEDLAM data and return per-item dicts with S frames."""

    def __init__(
        self,
        root: str,
        split: str = "Training",
        S: int = 2,
        max_humans: int = 10,
        img_size: int = 518,
        mhmr_size: int = 896,
        smpl_model: Any = None,
        stride: int = 1,
        seed: int | None = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.S = int(S)
        self.max_humans = int(max_humans)
        self.img_size = int(img_size)
        self.mhmr_size = int(mhmr_size)
        self.smpl_model = smpl_model
        self.stride = int(stride)
        self.rng = random.Random(seed)

        self._sequences = build_sequence_index(root, split)
        self._index: list[tuple[int, int]] = []
        for seq_idx, (_, frames) in enumerate(self._sequences):
            max_start = len(frames) - (self.S - 1) * self.stride
            for frame_idx in range(max_start):
                self._index.append((seq_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start_frame_idx = self._index[idx]
        seq_dir, frames = self._sequences[seq_idx]
        frame_ids = [frames[start_frame_idx + step * self.stride] for step in range(self.S)]

        images_list: list[torch.Tensor] = []
        mhmr_list: list[torch.Tensor] = []
        depth_list: list[torch.Tensor] = []
        K_scal3r_list: list[torch.Tensor] = []
        K_mhmr_list: list[torch.Tensor] = []
        mhmr_scale_list: list[torch.Tensor] = []
        mhmr_pad_list: list[torch.Tensor] = []
        mhmr_orig_hw_list: list[torch.Tensor] = []
        persons_per_frame: list[list[dict]] = []

        for frame_id in frame_ids:
            rgb_path = os.path.join(seq_dir, "rgb", frame_id + ".png")
            depth_path = os.path.join(seq_dir, "depth", frame_id + ".npy")
            cam_path = os.path.join(seq_dir, "cam", frame_id + ".npz")
            smpl_path = os.path.join(seq_dir, "smpl", frame_id + ".pkl")

            img_scal3r, orig_hw = load_image_tensor(rgb_path, self.img_size)
            img_mhmr, mhmr_meta = load_multihmr_letterbox_tensor(rgb_path, self.mhmr_size)
            if os.path.isfile(depth_path):
                gt_depth = load_depth_tensor(depth_path, self.img_size)
            else:
                gt_depth = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

            K_raw = load_intrinsics(cam_path)
            K_scal3r = torch.from_numpy(scale_intrinsics_for_resize(K_raw, orig_hw, self.img_size))
            K_mhmr = torch.from_numpy(letterbox_intrinsics(K_raw, mhmr_meta))
            persons = load_persons(smpl_path) if os.path.isfile(smpl_path) else []

            images_list.append(img_scal3r)
            mhmr_list.append(img_mhmr)
            depth_list.append(gt_depth)
            K_scal3r_list.append(K_scal3r)
            K_mhmr_list.append(K_mhmr)
            mhmr_scale_list.append(
                torch.tensor([mhmr_meta["scale_x"], mhmr_meta["scale_y"]], dtype=torch.float32)
            )
            mhmr_pad_list.append(
                torch.tensor([mhmr_meta["pad_x"], mhmr_meta["pad_y"]], dtype=torch.float32)
            )
            mhmr_orig_hw_list.append(
                torch.tensor([mhmr_meta["orig_h"], mhmr_meta["orig_w"]], dtype=torch.float32)
            )
            persons_per_frame.append(persons)

        smpl_batch = build_smpl_batch_from_persons(
            persons_per_frame,
            max_humans=self.max_humans,
            smpl_model=self.smpl_model,
        )

        return {
            "images": torch.stack(images_list, dim=0),
            "img_mhmr": torch.stack(mhmr_list, dim=0),
            "gt_depth": torch.stack(depth_list, dim=0),
            "K_scal3r": torch.stack(K_scal3r_list, dim=0),
            "K_mhmr": torch.stack(K_mhmr_list, dim=0),
            "mhmr_letterbox_scale": torch.stack(mhmr_scale_list, dim=0),
            "mhmr_letterbox_pad": torch.stack(mhmr_pad_list, dim=0),
            "mhmr_orig_hw": torch.stack(mhmr_orig_hw_list, dim=0),
            "joints3d_cam": smpl_batch["joints3d"],
            "gt_pose": smpl_batch["pose_6d"],
            "gt_betas": smpl_batch["betas"],
            "gt_cam_trans": smpl_batch["cam_trans"],
            "smpl_mask": smpl_batch["smpl_mask"],
        }
