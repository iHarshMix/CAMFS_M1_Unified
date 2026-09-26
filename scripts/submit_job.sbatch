#!/bin/bash
#SBATCH --job-name=camfs_job
#SBATCH --output=outputs/logs/slurm_%x_%j.out
#SBATCH --error=outputs/logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# Navigate to project root
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

# Environment initialization
mkdir -p outputs/logs outputs/checkpoints outputs/results

echo "=== Slurm Job ID: $SLURM_JOB_ID on $(hostname) ==="
echo "=== GPU Device: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi

# Activate conda environment
source /home/shared/miniconda/etc/profile.d/conda.sh
conda activate camfs

# Execute command passed to sbatch
"$@"
