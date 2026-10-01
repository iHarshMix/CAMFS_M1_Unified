#!/bin/bash
# =============================================================================
# CAMFS M1 — RUN-5: Centralized Ceiling Runner for Node V100
# =============================================================================
# Executes all three centralized ceiling configurations sequentially on V100:
#   1. full4:   T1, T1ce, T2, FLAIR (Ceiling for Hospital H1)
#   2. 3mod_a:  T1, T1ce, T2        (Ceiling for Hospitals H2 and H4)
#   3. 2mod:    T1, FLAIR           (Ceiling for Hospital H3)
#
# Usage:
#   bash scripts/run5_centralized/run_all_v100.sh [TRAIN_SEED] [GPU_ID]
# Recommended in detached screen session:
#   screen -dmS oracle_runs bash -c "bash scripts/run5_centralized/run_all_v100.sh 17 0"
# =============================================================================

set -e

SEED=${1:-17}
GPU=${2:-0}

# Navigate to project root
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

# Environment initialization
mkdir -p outputs/logs outputs/checkpoints outputs/results

echo "============================================================================="
echo "=== [RUN-5 Centralized Ceiling: Sequential V100 Suite] ==="
echo "=== Seed: $SEED | GPU: $GPU | Host: $(hostname) | Date: $(date) ==="
echo "============================================================================="
nvidia-smi

# Activate conda environment
source /home/shared/miniconda/etc/profile.d/conda.sh
conda activate camfs

# -----------------------------------------------------------------------------
# 1. Full-4 Modalities (Ceiling for H1)
# -----------------------------------------------------------------------------
echo -e "\n\n>>> [1/3] Launching Centralized Oracle: full4 (T1, T1ce, T2, FLAIR) <<<"
python scripts/run5_centralized/run_centralized.py \
  --modality-config full4 \
  --partition-seed 1103 \
  --train-seed $SEED \
  --batch-size 16 \
  --eval-batch-size 32 \
  --gpu $GPU \
  2>&1 | tee outputs/logs/oracle_full4_stdout.log

# -----------------------------------------------------------------------------
# 2. 3-Modality Subset (Ceiling for H2 and H4)
# -----------------------------------------------------------------------------
echo -e "\n\n>>> [2/3] Launching Centralized Oracle: 3mod_a (T1, T1ce, T2) <<<"
python scripts/run5_centralized/run_centralized.py \
  --modality-config 3mod_a \
  --partition-seed 1103 \
  --train-seed $SEED \
  --batch-size 16 \
  --eval-batch-size 32 \
  --gpu $GPU \
  2>&1 | tee outputs/logs/oracle_3mod_stdout.log

# -----------------------------------------------------------------------------
# 3. 2-Modality Subset (Ceiling for H3)
# -----------------------------------------------------------------------------
echo -e "\n\n>>> [3/3] Launching Centralized Oracle: 2mod (T1, FLAIR) <<<"
python scripts/run5_centralized/run_centralized.py \
  --modality-config 2mod \
  --partition-seed 1103 \
  --train-seed $SEED \
  --batch-size 16 \
  --eval-batch-size 32 \
  --gpu $GPU \
  2>&1 | tee outputs/logs/oracle_2mod_stdout.log

echo -e "\n\n============================================================================="
echo "=== ALL 3 CENTRALIZED CEILING CONFIGURATIONS COMPLETED SUCCESSFULLY! ==="
echo "=== Finished at: $(date) ==="
echo "============================================================================="
