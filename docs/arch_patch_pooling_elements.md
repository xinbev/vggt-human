# Architecture Patch Pooling Elements

This script generates paper-figure visual elements for the step:

```text
image -> patch grid -> SAM2 person mask -> person patches -> pooled person query
```

It is meant for architecture diagrams, not for model evaluation. If no mask or
bbox is provided, the script runs the project YOLO TorchScript detector and
SAM2 box-prompt predictor to create a person mask from the image. A precomputed
mask can still be passed to skip that heavier step. A bbox can be used as a
last-resort fallback, but the paper figure should prefer SAM2 masks because
the real project uses mask-aware patch pooling instead of a full rectangular
person box.

## Code

Local Python script:

```text
scripts/vis/create_arch_patch_pooling_elements.py
```

Server wrapper:

```text
scripts/vis/create_arch_patch_pooling_elements.sh
```

## Local Example

Automatic mode:

```text
python scripts/vis/create_arch_patch_pooling_elements.py \
  --image path/to/frame.png \
  --output-dir outputs/vis/paper_arch_patch_pooling_elements \
  --patch-size 32
```

This reads these defaults from `configs/path.yaml`:

```text
checkpoints.yolo8x
third_party.sam2_root
third_party.sam2_checkpoint
```

Automatic mode defaults to the top 2 detected people, which is usually the
right setting for a two-person architecture example. To force a single person,
use:

```text
--auto-top-k 0 --auto-person-index 0
```

Reuse an existing SAM2 mask:

```text
python scripts/vis/create_arch_patch_pooling_elements.py \
  --image path/to/frame.png \
  --mask path/to/sam2_masks.npz \
  --mask-key person_1 \
  --output-dir outputs/vis/paper_arch_patch_pooling_elements \
  --patch-size 32
```

SAM2 masks saved by this project use compressed `.npz` files. The keys usually
look like:

```text
person_1
person_2
det_000001
```

Use `--mask-key` to select one instance. If no key is given for an `.npz`, all
arrays are OR-combined.

When multiple mask keys are provided, each person instance is drawn with a
different color. Patch-token cells are assigned to the instance with the
largest mask coverage in that patch, so `04_patch_token_grid.png/svg` keeps
multi-person query evidence visually separated.

Fallback bbox mode is still available:

```text
python scripts/vis/create_arch_patch_pooling_elements.py \
  --image path/to/frame.png \
  --bbox 120,80,360,500 \
  --output-dir outputs/vis/paper_arch_patch_pooling_elements
```

`--bbox` uses pixel coordinates `x1,y1,x2,y2`. Multiple people can be shown by
repeating `--bbox`, but this draws a rectangular approximation and should not be
used for the main paper mechanism if a SAM2 mask is available.

## Server Example

After syncing to:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human
```

run:

```text
IMAGE_PATH=/path/to/frame.png \
bash scripts/vis/create_arch_patch_pooling_elements.sh
```

To reuse an existing SAM2 mask instead of running YOLO+SAM2:

```text
IMAGE_PATH=/path/to/frame.png \
PERSON_MASK=/path/to/sam2_masks.npz \
MASK_KEY=person_1 \
bash scripts/vis/create_arch_patch_pooling_elements.sh
```

Useful automatic-mode overrides:

```text
DEVICE=cuda
DET_CONF=0.25
AUTO_PERSON_INDEX=0
AUTO_TOP_K=2
YOLO_CHECKPOINT=/path/to/yolov8x.torchscript
SAM2_ROOT=/path/to/sam2
SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
```

Default output:

```text
outputs/vis/paper_arch_patch_pooling_elements/
```

In automatic mode, the script also writes reusable SAM2 mask artifacts:

```text
auto_sam2_mask_original.npz
auto_sam2_mask_original.png
auto_sam2_mask_resized.npz
auto_sam2_mask_resized.png
```

These can be reused later with `--mask ... --mask-key person_auto` to redraw
the same two-person figure assets without rerunning YOLO+SAM2. Automatic mode
also stores per-person keys such as `person_auto_0` and `person_auto_1` in
`auto_sam2_mask_original.npz`, so a single person can be redrawn if needed.

## Outputs

```text
01_image_patch_grid_faded.png
```

Faded input image with a light patch grid. Use this for the "image patches" or
"VGGT patch tokens" input element.

```text
02_person_patch_highlight.png
02b_person_patch_highlight_no_box.png
```

Faded input image with selected person-overlapping patches marked in light red.
When a mask is provided, the red region follows the human silhouette and patch
selection is based on per-patch mask coverage, not bbox coverage. Use this for
"SAM2 person mask" or "person-aware patch selection". The `02b` version omits
the person bbox outline and is usually cleaner for the final architecture
figure.

```text
03_person_patches_extracted.png
```

Only the selected person patches remain visually active. With a mask input,
non-human pixels inside selected patches stay pale, so the element does not
look like full-box pooling. Use this before a pooling arrow.

```text
04_patch_token_grid.png
04_patch_token_grid.svg
```

Abstract patch-token grid with selected tokens highlighted. The SVG version is
useful for final paper layout because it remains editable. Unselected tokens
are omitted: the PNG has a transparent background and the SVG only contains the
selected red token cells.

```text
05_pool_to_person_query.png
```

Selected patch tokens pooled into several query chips. Use this as the visual
element for "pooling person features to construct query".

## Notes

- Keep final text labels in the architecture SVG/PDF, not baked into these
  PNG elements.
- Automatic mode uses the project YOLO TorchScript detector followed by SAM2.
  It requires the configured checkpoints and SAM2 dependency to exist on the
  machine where the script runs.
- Use `--mask` when you already have a SAM2 mask and want a faster deterministic
  redraw. Use `--bbox` only when a mask is not available or when drawing a
  deliberately simplified fallback.
