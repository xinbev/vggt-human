# NLF Runtime And Data Plan

## SMPL Assets

There are two SMPL consumers:

- HSI and training losses use `vggt_omega.models.SMPLLayer`, which calls `smplx.create(...)`.
- NLF TorchScript uses `smplfitter.pt.BodyModel(...)` internally.

Configure the project-side SMPL path in `configs/path.yaml`:

```yaml
assets:
  smpl_model_dir: "/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models"
```

Valid layouts are either:

```text
<smpl_model_dir>/smpl/SMPL_NEUTRAL.pkl
<smpl_model_dir>/smpl/SMPL_MALE.pkl
<smpl_model_dir>/smpl/SMPL_FEMALE.pkl
```

or:

```text
<smpl_model_dir>/SMPL_NEUTRAL.pkl
<smpl_model_dir>/SMPL_MALE.pkl
<smpl_model_dir>/SMPL_FEMALE.pkl
```

The smoke script will verify the path with `smplx.create`.

## NLF Checkpoint

Your current path is the right kind of artifact:

```yaml
checkpoints:
  nlf_smpl: "/home/zhw/lab_users/xyb/home/projects/vggt-human/third_party/weights/nlf/nlf_l_multi_0.3.2.torchscript"
```

The adapter expects the TorchScript model to expose:

```text
estimate_smpl_batched
detect_smpl_batched
```

## NLF Dependencies

For inference inside this project, install the minimal runtime dependencies into the same Python environment used by VGGT-Human:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
pip install -e third_party/nlf
pip install smplx
pip install git+https://github.com/isarandi/simplepyutils.git
pip install git+https://github.com/isarandi/posepile.git
pip install git+https://github.com/isarandi/smplfitter.git
```

Also ensure the existing environment has compatible `torch` and `torchvision`. NLF detection and crop warping use `torchvision.ops` and `torchvision.transforms.v2`.

If the server has no network, clone those repos into `third_party/` and install editable copies instead:

```bash
pip install -e third_party/simplepyutils
pip install -e third_party/posepile
pip install -e third_party/smplfitter
```

NLF source imports `posepile.paths.DATA_ROOT`; if your environment complains about `DATA_ROOT`, set it before running:

```bash
export DATA_ROOT=/home/zhw/xyb_space
```

The project smoke scripts now default to this value, but keeping it in your shell profile is still cleaner.

For the TorchScript checkpoint, `PROJDIR` is usually not needed because the serialized model carries the trained buffers. If loading asks for NLF canonical files, set:

```bash
export PROJDIR=/path/to/localizerfields
```

and make sure it contains files such as `joint_info_866.pkl`, `canonical_verts/smpl.npy`, and `canonical_joints/smpl.npy`.

Run:

```bash
bash scripts/smoke/check_nlf_runtime_requirements.sh
```

This writes:

```text
outputs/debug/nlf_runtime_requirements/summary.json
```

## Dataset Redesign

For Stage 1, the core target is depth scale correction and human-scene fusion. Therefore the dataset must have metric depth and camera-space SMPL supervision. A dataset with only RGB and SMPL is not enough for the full HSI objective.

### Required Core Dataset

Preferred: BEDLAM processed layout.

```text
<bedlam_root>/<split>/<sequence>/rgb/<frame>.png
<bedlam_root>/<split>/<sequence>/depth/<frame>.npy
<bedlam_root>/<split>/<sequence>/cam/<frame>.npz
<bedlam_root>/<split>/<sequence>/smpl/<frame>.pkl
```

Required frame fields:

- RGB image.
- Metric depth in meters, aligned with the RGB image.
- Camera intrinsics if available.
- Per-person SMPL pose, betas, and camera-space root translation.
- Per-person boxes or enough joints/vertices to generate boxes.

Current loader expects BEDLAM SMPL person dictionaries with:

```text
smplx_pose or split root/body pose fields
smplx_shape
smplx_transl
bbox/j2d fields or sidecar boxes
person_id if available
```

### Sidecars We Generate

From the core dataset, generate:

```text
outputs/preprocess/bedlam_boxes
```

These provide processed-compatible boxes for NLF slots:

```text
smpl_query_boxes
smpl_query_boxes_mask
```

For NLF-HSI training, these boxes should describe visible trainable people, not every GT SMPL body in the annotation. The preprocessing script writes visibility diagnostics and sets `train_valid=false` for people that should not create an NLF crop, such as fully occluded bodies or tiny boxes. The dataset reader then removes those people from both query boxes and SMPL supervision so slot order remains aligned.

Generate them on the BEDLAM sample with:

```bash
VISIBLE_ONLY=true \
MIN_VISIBLE_JOINTS=4 \
MIN_BOX_AREA=100 \
bash scripts/preprocess/prepare_bedlam_boxes.sh
```

Run the complete sample smoke with:

```bash
bash scripts/smoke/check_nlf_bedlam_sample_flow.sh
```

For Stage 2, generate tracking sidecars:

```text
outputs/preprocess/video_tracks
```

These provide:

```text
external_track_ids
external_track_mask
external_track_confidence
```

### Recommended Dataset Order

1. Use BEDLAM processed data for Stage 0 and Stage 1.
2. Add real video datasets only after geometry is stable.
3. Use 3DPW/RICH/EMDB mainly for evaluation or temporal/ID checks unless metric depth is added.

### What You Need To Provide

Minimum for full Stage 1:

- BEDLAM processed RGB.
- BEDLAM metric depth.
- BEDLAM SMPL annotations.
- Camera files if available.

Minimum for Stage 2:

- Same as Stage 1.
- Video sequences with consistent person IDs, or permission to generate track sidecars using detector/tracker.

HF-BEDLAM images plus NPZ can be useful for SMPL supervision, but the current `hf_bedlam` loader fills `gt_depth` with zeros. It should not be used as the main Stage 1 depth-scale dataset unless we add a real depth source or pseudo-depth teacher.
