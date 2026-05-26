"""SMPL asset paths for migration.

Modify these paths in the target project.

SAT-HMR source reference:
- configs/paths.py
"""

from pathlib import Path

# TODO[target-project]: replace this with your own SMPL asset root.
SMPL_MODEL_PATH = Path("/path/to/weights/smpl_data")

# TODO[target-project]: replace this with your own mean parameter file.
SMPL_MEAN_PARAMS_PATH = SMPL_MODEL_PATH / "smpl" / "smpl_mean_params.npz"
