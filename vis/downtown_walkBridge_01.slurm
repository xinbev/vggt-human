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
#SBATCH --output=vis/vis_%j.out
#SBATCH --error=vis/vis_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始评估，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"


# Slurm exposes the allocated GPU through CUDA_VISIBLE_DEVICES (usually as
# logical device 0). Pass that value through instead of using the wrapper's
# interactive default GPU, which may refer to a non-visible physical index.

# CUDA_VISIBLE_DEVICES_VALUE=0 \
# bash scripts/vis/export_stage2_emdb_p4_36_long_walk_viewer_cache.sh

# FRAMES_DIR=/home/zhw/xyb_space/emdb/P3/28_outdoor_walk_lunges/images/ \
# STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
# CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
# SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
# OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_walking_coarse_residual_v3 \
# CUDA_VISIBLE_DEVICES_VALUE=0 \
# PORT=8088 \
# MAX_FRAMES=200 \
# SMOKE_ONLY=false \
# bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh

# cd /home/zhw/lab_users/xyb/home/projects/vggt-human
# MAX_FRAMES=-1 \
# CUDA_VISIBLE_DEVICES_VALUE=0 \
# bash scripts/vis/export_stage2_emdb_p3_28_walk_lunges_full_cache.sh



# FRAMES_DIR=/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color \


FRAMES_DIR=/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01/ \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_walking_coarse_residual_v3 \
FULL_VIEWER_CACHE_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_3dpw_downtown_walkBridge_01/full_viewer_cache  \
PORT=8080 \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}" \
START_INDEX=250 \
END_INDEX=-1 \
FRAME_STRIDE=2 \
MAX_FRAMES=0 \
DISPLAY_PEOPLE=0 \
SMPL_DISPLAY_FRAMES=0 \
CASCADE_EFFECTIVE_AFFINE_MODE=clip_median \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh



# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_3dpw_downtown_walkBridge_01/full_viewer_cache \
# SMPL_EDIT_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_3dpw_downtown_walkBridge_01/smpl_edit_offsets.json \
# PORT=8080 \
# bash scripts/vis/serve_full_sequence_viewer_cache.sh




echo "评估完成"
