# BEDLAM HF selective download

Target dataset:

- Hugging Face repo: `nguyenquivinhquang/BEDLAM`
- Subdirectory: `training_images`
- Default mirror endpoint: `https://hf-mirror.com`

This only handles the raw Hugging Face files. The existing project training config still expects the processed BEDLAM layout in `configs/path.yaml`:

```text
/home/zhw/xyb_space/bedlam/processed_bedlam/
```

To avoid overwriting that processed tree, the download script defaults to:

```text
/home/zhw/xyb_space/bedlam/hf_bedlam/
```

## 1. List files on Windows

Install the lightweight HF client in the Python environment used on Windows if needed:

```powershell
python -m pip install huggingface_hub
```

Then list the `training_images` files without downloading payload data:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/tools/list_hf_dataset_files.py
```

Outputs are written to:

```text
outputs/debug/hf_bedlam_training_images/files.txt
outputs/debug/hf_bedlam_training_images/files.csv
outputs/debug/hf_bedlam_training_images/groups.csv
outputs/debug/hf_bedlam_training_images/selected.txt
```

Use `groups.csv` to see the top-level entries and approximate sizes. Copy the exact paths you want from `files.txt` into `selected.txt`, one path per line.

Useful listing variants:

```powershell
python scripts/tools/list_hf_dataset_files.py --max-files 100
python scripts/tools/list_hf_dataset_files.py --include "*seq_000000*"
python scripts/tools/list_hf_dataset_files.py --group-depth 2
```

If the dataset requires auth, set:

```powershell
$env:HF_TOKEN="hf_xxx"
```

## 2. Download selected files on the Linux server

After syncing the repo and edited `selected.txt` to the server project:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/tools/download_bedlam_training_images_from_hf.sh
```

Dry-run first:

```bash
DRY_RUN=1 bash scripts/tools/download_bedlam_training_images_from_hf.sh
```

Override the target directory if needed:

```bash
LOCAL_DIR=/home/zhw/xyb_space/bedlam/hf_bedlam_subset \
bash scripts/tools/download_bedlam_training_images_from_hf.sh
```

The downloaded files keep their Hugging Face relative paths under `LOCAL_DIR`, for example:

```text
/home/zhw/xyb_space/bedlam/hf_bedlam/training_images/...
```

If server auth is required:

```bash
export HF_TOKEN=hf_xxx
bash scripts/tools/download_bedlam_training_images_from_hf.sh
```
