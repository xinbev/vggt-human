# BEDLAM training sample viewer

`scripts/vis/serve_bedlam_training_sample_highschoolgym.sh` visualizes the processed BEDLAM sequence
`20221013_3-10_500_batch01hand_static_highSchoolGym_seq_000000` using `BedlamDataset`, not a parallel raw-file loader.

It merges `configs/path.yaml` with `TRAIN_CONFIG` exactly as training does, then selects the one requested sequence with a temporary manifest under its output directory. The training YAML continues to define window length, stride, resize geometry, SMPL/depth requirements, sidecar boxes, query source, patch-mask settings, contact teachers, and human-slot count.

On the Linux server, run:

```bash
bash scripts/vis/serve_bedlam_training_sample_highschoolgym.sh
```

The default output is `/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/bedlam_training_sample_highschoolgym/`, including a 2D contact sheet and `run_summary.json`. The Viser view at `http://127.0.0.1:8091` is built from the same collated batch tensors: resized input RGB, resized depth, transformed intrinsics, and `gt_*` SMPL targets.

Set `TRAIN_CONFIG` to the exact YAML used for a training run. For example:

```bash
TRAIN_CONFIG=/home/zhw/lab_users/xyb/home/projects/vggt-human/configs/train_smpl_hsi_full_system_restructure.yaml \
WINDOW_INDEX=20 FRAME_OFFSET=0 PORT=8092 \
bash scripts/vis/serve_bedlam_training_sample_highschoolgym.sh
```

Use `SMOKE_ONLY=true` to save and validate the collated training sample without starting Viser.
