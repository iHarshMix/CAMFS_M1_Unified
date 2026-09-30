#!/bin/bash
# ==============================================================================
# Helper to batch dispatch RUN-2 Local-Only jobs to Slurm
#
# Usage:
#   bash scripts/run2_local_only/submit_all_seeds.sh 17         # Dispatches H1-H4 for seed 17
#   bash scripts/run2_local_only/submit_all_seeds.sh all        # Dispatches H1-H4 for seeds 17, 29, 43
# ==============================================================================

SEED_ARG=${1:-17}

if [ "$SEED_ARG" == "all" ]; then
    SEEDS=(17 29 43)
else
    SEEDS=($SEED_ARG)
fi

for S in "${SEEDS[@]}"; do
    echo "=== Submitting RUN-2 Local-Only Jobs for Seed $S ==="
    sbatch scripts/run2_local_only/submit_h1.sbatch $S
    sbatch scripts/run2_local_only/submit_h2.sbatch $S
    sbatch scripts/run2_local_only/submit_h3.sbatch $S
    sbatch scripts/run2_local_only/submit_h4.sbatch $S
done

echo "Submission complete. Monitor queue with: squeue -u $USER"
