#!/usr/bin/env bash
#SBATCH --job-name=m2_ft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:phd_4g47gb:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/m2_ft_slurm_%j.log
#SBATCH --error=outputs/logs/m2_ft_slurm_%j.log

set -e
PYTHON_BIN="/home/harshyadav/.conda/envs/camfs/bin/python"
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

echo "Running CAMFS M2 CDRD-FT (E4b) Recipient Fine-Tuning Pipeline on H100..."
${PYTHON_BIN} -u scripts/run_m2_cdrd_ft.py --config configs/m2_cdrd.yaml --gpu 0
