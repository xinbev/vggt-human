# OmniEraser Preprocess Usage

OmniEraser is used as an optional preprocessing frontend to remove humans or
other masked foreground objects from RGB frames, producing cleaner scene images
for environment reconstruction. This does not change the VGGT baseline path.

## Reference

- Third-party project: `third_party/Omnieraser`
- Paper: OmniEraser, "Remove Objects and Their Effects in Images with Paired
  Video-Frame Data"
- Used module: `ControlNet_version`
- Adaptation type: project wrapper around the third-party inference pipeline,
  not a direct import inside VGGT model code
- Baseline: preserved; this only writes preprocessed images under `outputs/`

The configured checkpoint in `configs/path.yaml` is:

```yaml
third_party:
  Omnieraser_root: third_party/Omnieraser
  Omnieraser_checkpoint: third_party/weights/Omnieraser/controlnet_flux_pytorch_lora_weights.safetensors
```

This matches the ControlNet OmniEraser LoRA file
`controlnet_flux_pytorch_lora_weights.safetensors`, so the project wrapper uses
`third_party/Omnieraser/ControlNet_version`.

## Inputs

The wrapper expects:

- RGB image: `.png`, `.jpg`, `.jpeg`, `.webp`, or `.bmp`
- Mask image with white pixels marking the region to remove

Use `--invert-mask` if black pixels mark the removal region.

The wrapper follows the third-party Gradio behavior: resize the shorter side to
`--resolution` and round dimensions down to multiples of 16. The default
resolution is 1024.

## Server Setup

Install OmniEraser extras in the server environment. Keep these dependencies
separate from the baseline environment if possible, because FLUX/diffusers
versions are heavier than VGGT-Omega itself.

```bash
pip install -r third_party/Omnieraser/ControlNet_version/requirements.txt
```

The first run also needs access to the FLUX and ControlNet base models:

```text
black-forest-labs/FLUX.1-dev
alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta
```

These may already be in the Hugging Face cache on the server. If not, the server
environment needs Hugging Face access and accepted model licenses.

## Run

### Full YOLO -> SAM2 -> OmniEraser Pipeline

Use this when the input is an ordered frame directory and the desired final
artifact is a clean pair of frame-sequence folders:

```bash
bash scripts/preprocess/prepare_human_erased_frames.sh /path/to/input_frames --overwrite
```

Server path after git sync:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/preprocess/prepare_human_erased_frames.sh
```

Default output:

```text
outputs/preprocess/human_erasure/<source_name>/
  original_frames/    # copied input frame sequence
  erased_frames/      # PNG frame sequence after human erasure
  person_masks/       # merged YOLO+SAM2 person masks, white = erase
  tracks/             # YOLO/BoostTrack/SAM2 sidecars
  manifest.json
```

Useful overrides:

```bash
OUTPUT_ROOT=outputs/preprocess/human_erasure \
bash scripts/preprocess/prepare_human_erased_frames.sh /path/to/input_frames \
  --source-name demo_clip \
  --mask-dilate 8 \
  --erase-resolution 1024 \
  --erase-steps 28 \
  --erase-seed 24 \
  --overwrite
```

The pipeline runs:

```text
input frame directory
  -> original_frames/
  -> YOLO person detection
  -> SAM2 box-prompted masks
  -> per-frame union person mask
  -> OmniEraser ControlNet inpainting
  -> erased_frames/
```

Frames with empty masks are copied unchanged into `erased_frames/` without
loading or running OmniEraser for those frames.

### OmniEraser Only

Local repository script:

```bash
scripts/preprocess/omnieraser_remove.sh
```

Server path after git sync:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/preprocess/omnieraser_remove.sh
```

Single image:

```bash
bash scripts/preprocess/omnieraser_remove.sh \
  /path/to/frame.png \
  /path/to/person_mask.png
```

Directory mode, matching images and masks by filename stem:

```bash
bash scripts/preprocess/omnieraser_remove.sh \
  /path/to/frames \
  /path/to/masks
```

Useful overrides:

```bash
OUTPUT_DIR=outputs/preprocess/omnieraser/demo \
bash scripts/preprocess/omnieraser_remove.sh frame.png mask.png \
  --seed 24 \
  --steps 28 \
  --resolution 1024
```

## Outputs

For the full pipeline, the two downstream folders are:

```text
outputs/preprocess/human_erasure/<source_name>/original_frames/
outputs/preprocess/human_erasure/<source_name>/erased_frames/
```

For the OmniEraser-only helper, results are written by default to:

```text
outputs/preprocess/omnieraser/<image_stem>_omnieraser.png
```

For downstream VGGT experiments, use these generated images as an alternate RGB
source or sidecar preprocessing result. Do not overwrite the original dataset
frames.

## Known Risks

- This is a diffusion inpainting prior, so geometry may become visually plausible
  but not metrically faithful in heavily occluded areas.
- Mask quality matters. Overly tight masks may leave body shadows or boundary
  artifacts; overly large masks may hallucinate important scene content.
- The base FLUX and ControlNet models are loaded from Hugging Face identifiers
  unless already cached locally.
- Windows local validation is limited to static checks because the local machine
  does not have the full GPU runtime and ckpt environment.
