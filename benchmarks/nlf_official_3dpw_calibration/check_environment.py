#!/usr/bin/env python3
"""Validate prerequisites for NLF's released 3DPW predictor/fitter protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Official NLF DATA_ROOT; must expose DATA_ROOT/3dpw")
    parser.add_argument("--projdir", required=True, help="Official NLF PROJDIR containing canonical_verts and canonical_joints")
    parser.add_argument("--nlf-root", default="third_party/nlf")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    data_root, projdir, nlf_root, model = map(Path, (args.data_root, args.projdir, args.nlf_root, args.model_path))
    required = {
        "3DPW test pkls": data_root / "3dpw" / "sequenceFiles" / "test",
        "3DPW RGB": data_root / "3dpw" / "imageFiles",
        "canonical SMPL vertices": projdir / "canonical_verts" / "smpl.npy",
        "canonical SMPL joints": projdir / "canonical_joints" / "smpl.npy",
        "SMPL faces": projdir / "smpl_faces.npy",
        "NLF PyTorch predictor": nlf_root / "nlf" / "pt" / "inference_scripts" / "predict_tdpw.py",
        "NLF TensorFlow fitter": nlf_root / "nlf" / "tf" / "inference_scripts" / "fit_tdpw.py",
        "NLF TorchScript model": model,
    }
    status = {name: {"path": str(path), "exists": path.exists()} for name, path in required.items()}
    python_modules = {name: importlib.util.find_spec(name) is not None for name in ("tensorflow", "smplfitter", "posepile", "simplepyutils", "cameralib")}
    ready = all(item["exists"] for item in status.values()) and all(python_modules.values())
    print(json.dumps({"ready": ready, "assets": status, "python_modules": python_modules, "env": {"DATA_ROOT": os.environ.get("DATA_ROOT"), "PROJDIR": os.environ.get("PROJDIR")}}, indent=2, ensure_ascii=False))
    if not ready:
        raise SystemExit("Official NLF 3DPW calibration environment is incomplete. Resolve every false field before running predict/fit.")


if __name__ == "__main__":
    main()
