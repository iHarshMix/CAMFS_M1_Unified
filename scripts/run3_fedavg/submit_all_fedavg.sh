#!/bin/bash
# ==============================================================================
# Helper to batch dispatch RUN-3 Compliant FedAvg jobs to Slurm
#
# Usage:
#   bash scripts/run3_fedavg/submit_all_fedavg.sh 17       # Dispatches FedAvg for seed 17
#   bash scripts/run3_fedavg/submit_all_fedavg.sh all      # Dispatches FedAvg for seeds 17, 29, 43
# ==============================================================================

SEED_ARG=${1:-17}

if [ "$SEED_ARG" == "all" ]; then
    SEEDS=(17 29 43)
else
    SEEDS=($SEED_ARG)
fi

for S in "${SEEDS[@]}"; do
    echo "=== Submitting RUN-3 FedAvg Job for Seed $S ==="
    sbatch scripts/run3_fedavg/submit_fedavg_s${S}.sbatch
done

echo "Submission complete. Monitor queue with: squeue -u $USER"
