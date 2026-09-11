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


DEVICE=cuda:0 \
bash scripts/vis/export_paper_architecture_real_assets.sh



echo "评估完成"

# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/55_outdoor_walk2/full_viewer_cache \
# SMPL_EDIT_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/55_outdoor_walk2/smpl_edit_offsets.json \
# PORT=8080 \
# bash scripts/vis/serve_full_sequence_viewer_cache.sh
