#!/bin/bash
#SBATCH --job-name=walk_lunges_cache
#SBATCH --partition=general
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=vis/vis_%j.out
#SBATCH --error=vis/vis_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始 GPU 推理并导出 Viser 缓存，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"


INPUT_DIR=outputs/vis/measure_woman_BasementSittingBooth_1112/input
mkdir -p "${INPUT_DIR}"
ln -sfn \
  /home/zhw/lab_users/xyb/home/projects/vggt-human/assets/demo/measure/woman_BasementSittingBooth_1112.jpg \
  "${INPUT_DIR}/woman_BasementSittingBooth_1112.jpg"

SHOW_TRACK_IDS=false \
FILTER_HUMAN_POINTS=false \
FRAMES_DIR="${INPUT_DIR}" \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/measure_woman_BasementSittingBooth_1112 \
FULL_VIEWER_CACHE_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/measure_woman_BasementSittingBooth_1112/full_viewer_cache \
CUDA_VISIBLE_DEVICES_VALUE=0 \
PORT=8080 \
MAX_FRAMES=1 \
VIEWER_MODE=4d \
ENVIRONMENT_DISPLAY=points \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_human_scene_align_walking_id_overlay.sh




echo "GPU 推理和 Viser 缓存导出完成"

