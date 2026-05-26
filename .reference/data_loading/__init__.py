"""Portable data-loading utilities extracted from the Scal3R workspace.

This package is intentionally independent from ``scal3r.models`` so it can be
copied to another agent/workspace together with its documented dependencies.
"""

from .batching.collate import bedlam_collate_fn
from .bedlam.dataset import BedlamDataset
from .image_folder_dataset import ImageFolderDataset

__all__ = ["BedlamDataset", "ImageFolderDataset", "bedlam_collate_fn"]
