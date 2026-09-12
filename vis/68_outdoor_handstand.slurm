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



cd /home/zhw/lab_users/xyb/home/projects/vggt-human

FRAMES_DIR=/home/zhw/xyb_space/emdb/P8/68_outdoor_handstand/images \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f \
FULL_VIEWER_CACHE_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f/full_viewer_cache \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}" \
START_INDEX=300 \
END_INDEX=1200 \
INFERENCE_FRAMES=500 \
NLF_DETECTOR_THRESHOLD=0.30 \
CONF_THRESHOLD=0.05 \
MAX_HUMANS=2 \
COARSE_CONF_THRESHOLD=0.30 \
COARSE_MAX_PEOPLE=2 \
DISPLAY_PEOPLE=2 \
SMPL_DISPLAY_FRAMES=0 \
CASCADE_EFFECTIVE_AFFINE_MODE=clip_median \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh


echo "评估完成"


#  CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f/full_viewer_cache SMPL_EDIT_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f/smpl_edit_offsets.json POINT_SIZE=0.006 HUMAN_MASK_DILATION_PX=20 FILTER_HUMAN_POINTS=true PORT=8083 bash scripts/vis/serve_full_sequence_viewer_cache.sh