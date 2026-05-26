from __future__ import annotations

import builtins
import inspect
from pathlib import Path

import numpy as np
import torch


def ensure_legacy_numpy_aliases() -> None:
    """Restore deprecated numpy scalar aliases expected by chumpy/smplx."""
    alias_map = {
        "bool": np.bool_,
        "int": builtins.int,
        "float": builtins.float,
        "complex": builtins.complex,
        "object": builtins.object,
        "unicode": builtins.str,
        "str": builtins.str,
    }
    for name, value in alias_map.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def ensure_legacy_inspect_getargspec() -> None:
    """Restore inspect.getargspec for chumpy on Python 3.11+."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]


def resolve_smpl_model_dir(path: str | Path) -> Path:
    """Normalize an SMPL model path to the parent directory containing ``smpl/``."""
    p = Path(path).expanduser().resolve()
    if (p / "smpl" / "SMPL_NEUTRAL.pkl").is_file():
        return p
    if (p / "SMPL_NEUTRAL.pkl").is_file():
        return p.parent
    raise FileNotFoundError(
        "Could not locate SMPL model files under "
        f"{p}. Expected either <dir>/smpl/SMPL_NEUTRAL.pkl or <dir>/SMPL_NEUTRAL.pkl."
    )


def build_smpl_model(
    model_dir: str | Path,
    device: torch.device | str = "cpu",
    gender: str = "neutral",
    num_betas: int = 10,
):
    """Build an optional ``smplx.SMPL`` model from a local body-model directory."""
    ensure_legacy_numpy_aliases()
    ensure_legacy_inspect_getargspec()
    try:
        import smplx
    except ImportError as exc:
        raise ImportError(
            "The `smplx` package is required to build a real SMPL model. "
            "Install it before enabling SMPL-joint supervision."
        ) from exc

    root = resolve_smpl_model_dir(model_dir)
    return smplx.create(str(root), "smpl", gender=gender, num_betas=num_betas).to(device)
