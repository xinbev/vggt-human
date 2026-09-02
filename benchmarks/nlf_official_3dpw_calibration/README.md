# NLF Official 3DPW Calibration Gate

This directory does not reimplement NLF's predictor or fitter. It runs the released NLF chain unchanged:

```text
official predict_tdpw.py
  (detect_poses_batched, GT 2D association, real intrinsics, detector flip augmentation)
-> official fit_tdpw.py
  (uncertainty-weighted neutral SMPL fitting)
-> fitted SMPL result files
```

The calibration is accepted only when the official NLF result reproduces the published 3DPW numbers within a predeclared tolerance. The target cited by the user is:

| PA-MPJPE | MPJPE | PVE |
| ---: | ---: | ---: |
| 37.3 | 60.3 | 71.4 |

Before running, use `check_environment.py`. The released scripts require a separate official NLF data layout and assets:

```text
DATA_ROOT/3dpw/sequenceFiles/test
DATA_ROOT/3dpw/imageFiles
PROJDIR/canonical_verts/smpl.npy
PROJDIR/canonical_joints/smpl.npy
PROJDIR/smpl_faces.npy
TensorFlow + smplfitter + posepile + NLF auxiliary packages
```

Those paths are intentionally not guessed from this project. The current `zhw_env` is sufficient for our TorchScript wrapper, but it is not evidence that the official TensorFlow fitting environment is installed.

After this gate passes, the project must add V2 after the **fitted official SMPL output**, then calculate baseline and V2 metrics using the same official association/evaluation input. Do not compare V2 against the current direct `detect_smpl_batched` wrapper output as an “official NLF” result.

The launcher finishes with `evaluate_fitted.py`, producing `outputs/benchmarks/nlf_official_3dpw/calibration.json`. It uses raw test pkl, gender-specific camera-space GT and Human3R-style pelvis/PA metrics. The hard calibration condition is every primary delta to `37.3 / 60.3 / 71.4` within 1 mm. If it fails, inspect the JSON before touching V2.
