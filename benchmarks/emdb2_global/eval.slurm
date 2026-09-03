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

# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# TRSTR_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_stage2_trstr_v3_refine/checkpoint_top_val_epoch_0003_metric_hsi_trstr_v3_selection_1.225058.pt \
# MAX_SEQUENCES=1 \
# bash benchmarks/emdb2_global/run_stride7_full.sh
# cd /home/zhw/lab_users/xyb/home/projects/vggt-human

# TRSTR_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_stage2_trstr_v3_refine \
# PREDICTIONS_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/emdb2_global_stride7_match_v1/predictions \
# OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/emdb2_global_stride7_match_v1/metrics \
# MAX_SEQUENCES=1 \
# bash benchmarks/emdb2_global/run_stride7_full.sh

TRSTR_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_stage2_trstr_v3_refine \
PREDICTIONS_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/emdb2_s7_chunk16_full/predictions \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/emdb2_s7_chunk16_full/metrics \
TRSTR_FRAME_CHUNK=16 \
MAX_SEQUENCES=0 \
bash benchmarks/emdb2_global/run_stride7_full.sh
echo "评估完成"