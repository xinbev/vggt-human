from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from os.path import abspath, expanduser, join


@dataclass
class ImageFolderDataset:
    """List image files from a folder using one or more glob patterns."""

    input_dir: str
    image_patterns: tuple[str, ...]
    max_images: int | None = None

    def list_images(self) -> list[str]:
        images: list[str] = []
        for pattern in self.image_patterns:
            images.extend(glob(join(self.input_dir, pattern)))
        resolved = sorted({abspath(expanduser(path)) for path in images})
        if self.max_images:
            return resolved[: self.max_images]
        return resolved
