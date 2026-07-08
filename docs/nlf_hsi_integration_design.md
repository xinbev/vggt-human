# NLF -> VGGT-Human -> HSI Integration Design

## Goal

This integration adds NLF as an external SMPL base provider for VGGT-Human while preserving the internal SMPL head baseline.

The intended flow is:

```text
processed images -> VGGT camera/depth -> NLF SMPL base -> HSI refinement
```

NLF replaces only the SMPL base fields consumed by HSI. VGGT backbone, camera head, depth head, and HSI head stay on the existing path.

## Provider Switch

`model.smpl_provider` controls the SMPL base source:

- `internal`: existing `AggregatorSMPLHead` path.
- `nlf`: skip the internal SMPL head and call `NLFSMPLProvider`.

The NLF provider lives at `vggt_omega/integrations/nlf_smpl_provider.py`. It is loaded lazily so training checkpoints do not serialize the external NLF TorchScript model.

## Image Plane Contract

NLF must consume the same image plane as the camera intrinsics passed to it.

In `VGGTOmega.forward(images)`, the runtime tensor shape defines the contract:

```text
images: [B, S, 3, H, W]
nlf_images: [B*S, 3, H, W]
K = encoding_to_camera(pose_enc, image_size_hw=(H, W), build_intrinsics=True)
nlf_K: [B*S, 3, 3]
```

This means:

- NLF never reloads original images.
- NLF uses processed or padded VGGT input images.
- VGGT float images stay in `[0, 1]`; PyTorch NLF internally linearizes float inputs with `pow(2.2)`, while its demo path only divides by 255 for `uint8` inputs.
- K is decoded from the same runtime `[H, W]`.
- No fixed `512`, `518`, or config image size is used for NLF intrinsics.
- If collate padding changes the final tensor plane, NLF also sees the padded image plane.

`smpl_query_boxes` are assumed to be normalized `cxcywh` in the current processed/padded image plane. The adapter converts them to NLF's pixel top-left `xywh` with the same runtime `[H, W]`, then converts NLF's returned `xywh` boxes back to normalized `cxcywh` for HSI and the matcher.

## Camera Coordinates

The NLF adapter passes:

```text
intrinsic_matrix=K
distortion_coeffs=None
extrinsic_matrix=None
```

No extrinsics are passed to NLF by default. Therefore NLF `trans` stays in the current camera coordinate system and is written directly as:

```text
pred_transl_cam = NLF trans
base_pred_transl_cam = pred_transl_cam
```

This is the coordinate system HSI expects for human-scene fusion and depth-scale correction.

## HSI Output Contract

NLF ragged multi-person output is padded into dense HSI fields:

```text
pred_poses:      [B, S, Q, 72]
pred_pose_6d:    [B, S, Q, 144]
pred_betas:      [B, S, Q, 10]
pred_transl_cam: [B, S, Q, 3]
pred_confs:      [B, S, Q, 1]
pred_boxes:      [B, S, Q, 4]
```

Diagnostic fields:

```text
pred_cam
base_pred_transl_cam
nlf_intrinsics
nlf_image_hw
nlf_valid_mask
```

Training defaults to sidecar boxes and stable slots. Detector fallback is kept for demo use through `model.nlf_use_detector=true`.

## Training Stages

Stage 0: interface and projection validation.

- Freeze VGGT, NLF, and HSI.
- Run a small batch and export overlays on processed images.
- Confirm K, image plane, translation units, and slot padding.

Stage 1: frozen NLF plus HSI geometry correction.

- Freeze VGGT backbone, camera head, depth head, and NLF.
- Train only HSI residual fields and scene affine depth correction.
- Main target: depth scale, depth bias, and human-environment geometric fusion.

Stage 2: temporal and ID tracking.

- Enable `smpl_track_ids`, `external_track_ids`, or track sidecars.
- Train or validate HSI temporal branch and track-aware refinement.
- Keep this after Stage 1 so ID losses do not destabilize early geometry.

Stage 3: optional calibration.

- Add confidence calibration or a light adapter only if NLF domain gap is large.
- Do not train NLF itself by default.

## Configs And Scripts

Config:

- `configs/train_smpl_hsi_nlf_provider.yaml`
- `configs/path.yaml`

Smoke:

- `scripts/smoke/check_nlf_runtime_requirements.sh`
- `scripts/smoke/check_nlf_provider_interface.sh`
- `scripts/smoke/check_nlf_hsi_forward.sh`

Training:

- `scripts/train/train_smpl_hsi_nlf_provider.sh`

Visualization:

- `scripts/vis/vis_nlf_hsi_depth_smpl_diagnostics.sh`

Server paths:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/smoke/check_nlf_provider_interface.sh
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/smoke/check_nlf_hsi_forward.sh
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/train/train_smpl_hsi_nlf_provider.sh
```

## Pre-Training Checklist

- Put NLF source at `third_party/nlf`.
- Put the NLF TorchScript checkpoint at `checkpoints.nlf_smpl` or override `NLF_CKPT`.
- Confirm NLF imports on the server, including `smplfitter`.
- Confirm `assets.smpl_model_dir` points to valid SMPL model assets.
- Prepare processed-compatible query boxes:
  - `smpl_query_boxes`
  - `smpl_query_boxes_mask`
- For Stage 2, also prepare:
  - `smpl_track_ids` or `external_track_ids`
  - matching masks/confidence fields

Run before Stage 1:

```bash
bash scripts/smoke/check_nlf_runtime_requirements.sh
bash scripts/smoke/check_nlf_provider_interface.sh
bash scripts/smoke/check_nlf_hsi_forward.sh
```

Expected smoke result:

- NLF output fields are complete.
- Non-square input is covered by the provider smoke.
- `pred_transl_cam` has no NaN.
- `nlf_image_hw` equals the actual processed tensor `[H, W]`.
- HSI outputs `hsi_scene_scale` and refined SMPL fields.

Then start Stage 1:

```bash
bash scripts/train/train_smpl_hsi_nlf_provider.sh
```

Outputs are written to:

```text
outputs/debug/nlf_provider_interface_smoke
outputs/debug/nlf_hsi_forward_smoke
outputs/train/smpl_hsi_nlf_provider_stage1
outputs/vis/nlf_hsi_depth_smpl_diagnostics
```

## Main Risks

- NLF `trans` unit or origin may differ on some checkpoints. Stage 0 projection overlays must verify this before training.
- Sidecar boxes must match the processed/padded image plane. Mismatched boxes will corrupt slot order and NLF crops.
- NLF detector fallback is useful for demos but not stable enough for supervised multi-person training slots.
- HSI can refine geometry only after the base SMPL translation is in the same camera-space convention as VGGT depth.
