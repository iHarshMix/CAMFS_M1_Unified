#!/usr/bin/env bash
#SBATCH --job-name=m2_cdrd
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:phd_4g47gb:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/m2_cdrd_slurm_%j.log
#SBATCH --error=outputs/logs/m2_cdrd_slurm_%j.log

set -e

PYTHON_BIN="/home/harshyadav/.conda/envs/camfs/bin/python"
CONFIG_FILE="configs/m2_cdrd.yaml"

echo "=============================================================================="
echo "  SLURM Job ID: ${SLURM_JOB_ID}"
echo "  Running on Node: $(hostname)"
echo "  Allocated GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "  Date: $(date)"
echo "=============================================================================="

# Move to repository root
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified
mkdir -p outputs/logs outputs/m2_cdrd

# 1. Pre-flight verification tests
echo ""
echo ">>> [Phase 1/3] Running M2 Core Test Suite..."
${PYTHON_BIN} -m unittest tests/test_m2_core.py -v

# 2. 16-Step M2 CDRD Training Pipeline
echo ""
echo ">>> [Phase 2/3] Executing M2 CDRD Training Pipeline on H100..."
${PYTHON_BIN} scripts/run_m2_cdrd.py --config "${CONFIG_FILE}" --gpu 0 --force-retrain

# 3. Sacred 50-Patient Test Set Evaluation
echo ""
echo ">>> [Phase 3/3] Evaluating Sacred 50-Patient Test Set (E0 Base vs E4_aug vs E4_deploy)..."
${PYTHON_BIN} scripts/evaluate_m2_test_set.py --config "${CONFIG_FILE}" --gpu 0

echo ""
echo "=============================================================================="
echo "  Job ${SLURM_JOB_ID} Completed Successfully!"
echo "  Outputs: outputs/m2_cdrd/"
echo "  Ledger:  outputs/provenance/m2_cdrd_ledger.jsonl"
echo "=============================================================================="
