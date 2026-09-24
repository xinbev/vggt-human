#!/bin/bash
#SBATCH --job-name=nlf_pose_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=emdb1_SHOW_full_eval_%j.out
#SBATCH --error=emdb1_SHOW_full_eval_%j.err

set -e

# 激活 zhw_env 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate zhw_env

echo "开始评估，作业 ID: ${SLURM_JOB_ID}"
echo "当前环境: ${CONDA_DEFAULT_ENV}"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"


cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CACHE_DIR=outputs/eval/show_emdb1_manual_scale/cache \
SCALE_MODE=manual \
DEVICE=cuda:0 \
bash scripts/eval/evaluate_emdb1_show_manual_scale.sh
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES}"