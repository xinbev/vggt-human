#!/bin/bash
#SBATCH --job-name=nlf_pose_eval
#SBATCH --partition=general
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=video/P1_16_outdoor_warmup/out_%j.out
#SBATCH --error=video/P1_16_outdoor_warmup/err_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始评估，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"




FRAMES_DIR=/home/zhw/xyb_space/emdb/P1_16_outdoor_warmup/images/ \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P1_16_outdoor_warmup/outputs \
FULL_VIEWER_CACHE_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P1_16_outdoor_warmup/outputs/full_viewer_cache \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}" \
PORT=8090 \
START_INDEX=36 \
END_INDEX=495 \
FRAME_STRIDE=1 \
INFERENCE_FRAMES=0 \
HUMAN_MASK_DILATION_PX=19 \
MAX_FRAMES=0 \
FRAME_SAMPLING=uniform \
SMPL_DISPLAY_FRAMES=0 \
DISPLAY_PEOPLE=0 \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh


echo "评估完成"



# CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P1_16_outdoor_warmup/outputs/full_viewer_cache \
# SMPL_EDIT_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P1_16_outdoor_warmup/outputs/smpl_edit_offsets.json \
# POINT_SIZE=0.006 \
# FILTER_HUMAN_POINTS=true \
# PORT=8087 \
# bash scripts/vis/serve_full_sequence_viewer_cache.sh