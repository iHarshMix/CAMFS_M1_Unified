#!/usr/bin/env bash
#SBATCH --job-name=eval_m2
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:phd_4g47gb:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/m2_eval_slurm_%j.log
#SBATCH --error=outputs/logs/m2_eval_slurm_%j.log

set -e
PYTHON_BIN="/home/harshyadav/.conda/envs/camfs/bin/python"
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

echo "Running sacred 50-patient test evaluation on H100..."
${PYTHON_BIN} scripts/evaluate_m2_test_set.py --config configs/m2_cdrd.yaml --gpu 0
