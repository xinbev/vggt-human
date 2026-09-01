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

DEVICE=cuda:0 \
DATASET=3dpw \
FRAMES_ROOT=/home/zhw/xyb_space/3DPW/imageFiles \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
CACHE_ROOT=outputs/preprocess/nlf_vggt_temporal_cache_v1 \
bash scripts/preprocess/cache_nlf_vggt_temporal_inputs.sh


echo "评估完成"