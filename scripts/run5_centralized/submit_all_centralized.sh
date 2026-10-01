#!/bin/bash
# =============================================================================
# CAMFS M1 — Submit All Centralized Oracle Runs to Slurm
# =============================================================================
# Usage:
#   bash scripts/run5_centralized/submit_all_centralized.sh [SEED]
# Default seed is 17.
# =============================================================================

SEED=${1:-17}

echo "Submitting Centralized Oracle runs for seed $SEED to Slurm (partition h100, gres ugpg_3g47gb:1)..."

JOB1=$(sbatch scripts/run5_centralized/submit_centralized_full4.sbatch $SEED | awk '{print $4}')
echo "  [Submitted] full4   (H1 Ceiling)    -> Slurm JobID: $JOB1"

JOB2=$(sbatch scripts/run5_centralized/submit_centralized_3mod.sbatch $SEED | awk '{print $4}')
echo "  [Submitted] 3mod_a  (H2/H4 Ceiling) -> Slurm JobID: $JOB2"

JOB3=$(sbatch scripts/run5_centralized/submit_centralized_2mod.sbatch $SEED | awk '{print $4}')
echo "  [Submitted] 2mod    (H3 Ceiling)    -> Slurm JobID: $JOB3"

echo "All 3 Oracle baseline jobs queued."
