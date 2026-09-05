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
#SBATCH --output=nlf_pose_eval_%j.out
#SBATCH --error=nlf_pose_eval_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始评估，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"
#================================ VGGT  ====================================

# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# python benchmarks/bonn_depth/infer.py \
#   --dataset-root /home/zhw/xyb_space/rgbd_bonn_dataset \
#   --output-root /home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
#   --stage pure_vggt \
#   --checkpoint /home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
#   --chunk-size 25 \
#   --device cuda


# DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
# PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
# bash benchmarks/bonn_depth/run.sh
#================================ HSI scale ====================================
# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# python benchmarks/bonn_depth/infer.py \
#   --dataset-root /home/zhw/xyb_space/rgbd_bonn_dataset \
#   --output-root /home/zhw/xyb_space/vggt_bonn_predictions/vggt_traditional_hsi_scale \
#   --stage vggt_traditional_hsi_scale \
#   --checkpoint /home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
#   --stage2-checkpoint /home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
#   --scale-checkpoint /home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
#   --chunk-size 25 \
#   --device cuda

# DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
# PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/vggt_traditional_hsi_scale \
# OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/bonn_depth_traditional_hsi \
# bash benchmarks/bonn_depth/run.sh

#================================ 对比 ====================================

DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PURE_VGGT_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
TRADITIONAL_HSI_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/vggt_traditional_hsi_scale \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/bonn_depth_stages \
ALIGNMENT=scale \
bash benchmarks/bonn_depth/run_stages.sh















echo "评估完成"