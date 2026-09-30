#!/bin/bash
#SBATCH --job-name=local_H2
#SBATCH --output=outputs/logs/slurm_local_H2_%j.out
#SBATCH --error=outputs/logs/slurm_local_H2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ugpg_3g47gb:1
#SBATCH --partition=h100
#SBATCH --time=03:59:00
#SBATCH --mem=64G

# Set working directory to project root
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

# Environment initialization
mkdir -p outputs/logs outputs/checkpoints outputs/results

SEED=${1:-17}

echo "=== [Slurm Job: local_H2 | Seed: $SEED | JobID: $SLURM_JOB_ID] ==="
echo "=== Host: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi

# Activate conda environment
source /home/shared/miniconda/etc/profile.d/conda.sh
conda activate camfs

# Execute dedicated runner
python scripts/run2_local_only/run_local_h2.py \
  --partition-seed 1103 \
  --train-seed $SEED \
  --gpu 0
