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

# DEVICE=cuda:0 \
# THREEDPW_ROOT=/home/zhw/xyb_space/3DPW \
# TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_temporal_stabilizer_v2_pose_mixture_v21_focus/checkpoint_best.pt \
# OUTPUT_DIR=outputs/eval/human3r_style_3dpw_v1 \
# bash benchmarks/human3r_style_3dpw/run.sh


# DEVICE=cuda:0 \
# THREEDPW_ROOT=/home/zhw/xyb_space/3DPW \
# SEQUENCE_FILTER=downtown_arguing_00 \
# MAX_FRAMES=100 \
# COMPONENT_DIAGNOSTICS=true \
# TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_temporal_stabilizer_v2_pose_mixture_v21_focus/checkpoint_best.pt \
# OUTPUT_DIR=outputs/debug/human3r_style_3dpw_downtown_diagnose \
# bash benchmarks/human3r_style_3dpw/run.sh
# CUDA_VISIBLE_DEVICES_VALUE=7 \
DEVICE=cuda:0 \
THREEDPW_ROOT=/home/zhw/xyb_space/3DPW \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_temporal_stabilizer_v2_pose_mixture_v21_focus/checkpoint_best.pt \
OUTPUT_DIR=outputs/eval/human3r_style_3dpw_full_v2 \
bash benchmarks/human3r_style_3dpw/run.sh

echo "评估完成"