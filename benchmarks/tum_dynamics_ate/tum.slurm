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






# RAW_ROOT=/home/zhw/xyb_space/tum_dynamics_raw \
# PREPARED_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
# bash benchmarks/tum_dynamics_ate/prepare_tum_dynamics.sh



DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
PRED_PARENT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/tum_dynamics_predictions \
LENGTHS=50,100,150,200,300,400,500,600,700,800,900,1000 \
MODEL=vggt \
bash benchmarks/tum_dynamics_ate/infer_vggt.sh










echo "评估完成"