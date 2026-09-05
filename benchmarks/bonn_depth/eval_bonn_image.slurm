#!/bin/bash
#SBATCH --job-name=nlf_pose_eval
#SBATCH --partition=general
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=log/nlf_pose_eval_%j.out
#SBATCH --error=log/nlf_pose_eval_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始评估，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"
#================================ VGGT  ====================================



# DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
# PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_curve_predictions \
# STAGE=vggt_traditional_hsi_scale \
# CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
# STAGE2_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
# SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
# PREFIX_LENGTHS="50 100 150 200 250 300 350 400 450 500" \
# START_FRAME=30 \
# ALLOW_SHORT=true \
# IMAGE_RESOLUTION=512 \
# RESIZE_MODE=balanced \
# DEVICE=cuda \
# bash benchmarks/bonn_depth/infer_curve.sh






DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_curve_predictions \
STAGE=vggt_traditional_hsi_scale \
PREFIX_LENGTHS="50 100 150 200 250 300 350 400 450 500" \
START_FRAME=30 \
ALLOW_SHORT=true \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/bonn_depth_curve \
bash benchmarks/bonn_depth/run_curve.sh






echo "评估完成"