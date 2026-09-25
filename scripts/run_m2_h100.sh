#!/usr/bin/env bash
# ==============================================================================
# CAMFS M2 CDRD — End-to-End Execution Script (NVIDIA H100)
# ==============================================================================
# Pipeline:
#   1. Pre-flight Unit & Policy Tests (tests/test_m2_core.py)
#   2. Full M2 CDRD Orchestration: Teacher -> Adapter -> Calibrate (scripts/run_m2_cdrd.py)
#   3. Sacred 50-Patient Test Set Evaluation: E0 vs E4_aug vs E4_deploy (scripts/evaluate_m2_test_set.py)
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

PYTHON_BIN="/home/harshyadav/.conda/envs/camfs/bin/python"
CONFIG_FILE="configs/m2_cdrd.yaml"
GPU_ID=0

echo "=============================================================================="
echo "  CAMFS M2: Launching CDRD Knowledge Transfer (H1 -> H3) on H100"
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo "  Python: ${PYTHON_BIN}"
echo "=============================================================================="

# Ensure working directory is repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 1. Run Pre-flight Test Suite
echo ""
echo ">>> [Phase 1/3] Running M2 Core Test Suite (Day-0 Identities, Architecture, P0-P6)..."
${PYTHON_BIN} -m unittest tests/test_m2_core.py -v

# 2. Run M2 CDRD Pipeline
echo ""
echo ">>> [Phase 2/3] Executing M2 CDRD 16-Step Pipeline on GPU ${GPU_ID}..."
${PYTHON_BIN} scripts/run_m2_cdrd.py --config "${CONFIG_FILE}" --gpu ${GPU_ID} --force-retrain

# 3. Run Sacred 50-Patient Test Evaluation
echo ""
echo ">>> [Phase 3/3] Evaluating Sacred 50-Patient Test Set (E0 vs E4)..."
${PYTHON_BIN} scripts/evaluate_m2_test_set.py --config "${CONFIG_FILE}" --gpu ${GPU_ID}

echo ""
echo "=============================================================================="
echo "  🎉 CAMFS M2 EXECUTION AND EVALUATION COMPLETE!"
echo "  Results: outputs/m2_cdrd/"
echo "  Ledger:  outputs/provenance/m2_cdrd_ledger.jsonl"
echo "=============================================================================="
