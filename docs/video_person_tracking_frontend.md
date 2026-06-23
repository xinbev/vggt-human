# Video Person Tracking Frontend

This frontend prepares real-video person observations for VGGT-Omega SMPL query
priors and HSI temporal memory. It does not modify the VGGT baseline model.

## Pipeline

```text
video / frame directory
  -> YOLO TorchScript person detector
  -> BoostTrack++ tracker with active/lost/dead track memory
  -> optional SAM2 box-prompted masks
  -> sidecar files
  -> smpl_query_boxes / smpl_track_ids for VGGTOmega.forward(...)
```

The tracker is driven by detections, and the package also includes
`HSITrackMemory` so inference code can write model outputs back:

```text
SMPLHead base pose/beta/transl
  -> HSI refined pose/beta/transl
  -> HSITrackMemory
  -> future tracklet stitching or HSI-aware matching
```

## Server Setup

Install tracking-only extras in the server environment:

```bash
pip install -r requirements_tracking.txt
SAM2_BUILD_CUDA=0 pip install -e third_party/sam2
```

`configs/path.yaml` contains the default server paths:

```yaml
checkpoints:
  yolo8x: /home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/yolov8x.torchscript
third_party:
  boosttrack_root: /home/zhw/lab_users/xyb/home/projects/vggt-omega/third_party/BoostTrack
  boosttrack_weights_root: /home/zhw/lab_users/xyb/home/projects/vggt-omega/third_party/weights/BoostTrack
  sam2_root: /home/zhw/lab_users/xyb/home/projects/vggt-omega/third_party/sam2
  sam2_checkpoint: /home/zhw/lab_users/xyb/home/projects/vggt-omega/third_party/weights/sam/sam2.1_hiera_large.pt
```

## Run

Video:

```bash
bash scripts/preprocess/prepare_video_person_tracks.sh /path/to/video.mp4
```

Frame directory:

```bash
bash scripts/preprocess/prepare_video_person_tracks.sh /path/to/frames
```

BEDLAM one-sequence smoke test:

```bash
bash scripts/preprocess/test_bedlam_person_tracks.sh
```

Override the selected sequence or frame count:

```bash
SEQ_INDEX=3 MAX_FRAMES=240 SPLIT=Training bash scripts/preprocess/test_bedlam_person_tracks.sh
```

Direct BEDLAM invocation:

```bash
python scripts/preprocess/prepare_video_person_tracks.py \
  --bedlam-sequence-index 0 \
  --bedlam-split Training \
  --path-config configs/path.yaml \
  --output-root outputs/preprocess/video_tracks \
  --overwrite \
  --max-frames 120
```

With SAM2 masks:

```bash
bash scripts/preprocess/prepare_video_person_tracks.sh /path/to/video.mp4 --enable-sam2-masks
```

## Output

For a source named `demo`, outputs are written to:

```text
outputs/preprocess/video_tracks/demo/
  observations.jsonl
  summary.json
  smpl_boxes/<frame_id>.pkl
  masks/<frame_id>.npz        # only with --enable-sam2-masks
```

Visualize tracked IDs:

```bash
python scripts/vis/visualize_video_person_tracks.py \
  --sidecar-root outputs/preprocess/video_tracks/Training/<sequence_name> \
  --output-dir outputs/vis/video_person_tracks/Training/<sequence_name> \
  --write-video
```

Each person observation contains:

```json
{
  "frame_id": "000012",
  "person_id": 3,
  "bbox_xyxy_pixels": [120.0, 44.0, 210.0, 310.0],
  "bbox_cxcywh_norm": [0.32, 0.41, 0.12, 0.35],
  "bbox_valid": true,
  "track_confidence": 0.91,
  "missing_count": 0
}
```

Use `vggt_omega.tracking.build_clip_tensors_from_sidecar(...)` to get:

```python
smpl_query_boxes      # [1, S, Q, 4]
smpl_query_boxes_mask # [1, S, Q]
smpl_track_ids        # [1, S, Q]
smpl_track_mask       # [1, S, Q]
```
