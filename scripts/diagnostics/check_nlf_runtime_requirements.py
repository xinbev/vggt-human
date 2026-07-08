from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.training.config import deep_update, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NLF/SMPL runtime dependencies and asset paths.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--output-dir", default="outputs/debug/nlf_runtime_requirements")
    parser.add_argument("--nlf-checkpoint", default="")
    parser.add_argument("--nlf-root", default="")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--projdir", default="")
    parser.add_argument("--skip-load-nlf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_cfg = load_yaml_config(args.path_config)
    config = deep_update(path_cfg, {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nlf_root = Path(args.nlf_root or config.get("third_party", {}).get("nlf_root", "third_party/nlf")).expanduser()
    nlf_ckpt = Path(args.nlf_checkpoint or config.get("checkpoints", {}).get("nlf_smpl", "")).expanduser()
    smpl_model_dir = Path(args.smpl_model_dir or config.get("assets", {}).get("smpl_model_dir", "")).expanduser()
    if "DATA_ROOT" not in os.environ:
        inferred_data_root = infer_data_root(config)
        if inferred_data_root:
            os.environ["DATA_ROOT"] = inferred_data_root
    if str(nlf_root) not in sys.path:
        sys.path.insert(0, str(nlf_root))

    summary: dict[str, Any] = {
        "path_config": str(args.path_config),
        "nlf_root": str(nlf_root),
        "nlf_checkpoint": str(nlf_ckpt),
        "smpl_model_dir": str(smpl_model_dir),
        "imports": {},
        "assets": {},
    }

    require_path(nlf_root, "NLF source directory", is_dir=True)
    require_path(nlf_ckpt, "NLF TorchScript checkpoint", is_file=True)
    require_path(smpl_model_dir, "SMPL model directory", is_dir=True)

    for module_name in ("torch", "torchvision", "smplx", "smplfitter.pt", "posepile.paths", "simplepyutils"):
        module = importlib.import_module(module_name)
        summary["imports"][module_name] = str(getattr(module, "__version__", "ok"))

    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")
    importlib.import_module("torchvision.ops")
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    torchvision.ops.nms(boxes, scores, 0.5)
    summary["assets"]["torchvision_nms_registered"] = True

    smplx = importlib.import_module("smplx")
    smpl_layer = smplx.create(
        str(smpl_model_dir),
        model_type="smpl",
        gender="neutral",
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_transl=False,
    )
    summary["assets"]["smplx_vertices"] = int(getattr(smpl_layer, "NUM_VERTS", 0) or smpl_layer.v_template.shape[0])

    posepile_paths = importlib.import_module("posepile.paths")
    data_root = Path(str(getattr(posepile_paths, "DATA_ROOT", ""))).expanduser()
    projdir = Path(args.projdir or os.getenv("PROJDIR", "") or (data_root / "projects/localizerfields")).expanduser()
    summary["assets"]["posepile_DATA_ROOT"] = str(data_root)
    summary["assets"]["nlf_PROJDIR"] = str(projdir)
    summary["assets"]["projdir_required_files"] = check_optional_projdir(projdir)

    if not args.skip_load_nlf:
        model = torch.jit.load(str(nlf_ckpt), map_location="cpu").eval()
        summary["assets"]["nlf_has_estimate_smpl_batched"] = bool(hasattr(model, "estimate_smpl_batched"))
        summary["assets"]["nlf_has_detect_smpl_batched"] = bool(hasattr(model, "detect_smpl_batched"))
        if not summary["assets"]["nlf_has_estimate_smpl_batched"]:
            raise AttributeError("NLF model does not expose estimate_smpl_batched")

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] NLF runtime requirements check passed")
    print(json.dumps({"summary": str(out_json)}, indent=2))


def require_path(path: Path, label: str, is_file: bool = False, is_dir: bool = False) -> None:
    if is_file and not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if is_dir and not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def check_optional_projdir(projdir: Path) -> dict[str, bool]:
    files = (
        "joint_info_866.pkl",
        "canonical_loc_symmetric_init_866.npy",
        "canonical_verts/smpl.npy",
        "canonical_joints/smpl.npy",
        "canonical_verts/smplx.npy",
        "canonical_joints/smplx.npy",
    )
    return {name: (projdir / name).is_file() for name in files}


def infer_data_root(config: dict[str, Any]) -> str:
    for key in ("bedlam_root", "hf_bedlam_images_root", "threedpw_root"):
        raw = str(config.get("datasets", {}).get(key, "") or "")
        if not raw:
            continue
        path = Path(raw).expanduser()
        parts = path.parts
        if len(parts) >= 4 and parts[-2:] == ("bedlam", "processed_bedlam"):
            return str(Path(*parts[:-2]))
        if len(parts) >= 2:
            return str(path.parent)
    return ""


if __name__ == "__main__":
    main()
