#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Webilastik 2.0 — Manual SLURM job test
#
# Run this FROM the HPC login node to verify the full pipeline works before
# wiring it up through the allocator + SSH.
#
# Usage:
#   1. Edit the variables in the CONFIG block below.
#   2. sbatch scripts/test_job_manual.sh
#   3. Watch the log: tail -f $SCRATCH/wi2-test-manual.log
#   4. Check sacct: sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
#
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG — edit these ───────────────────────────────────────────────────────

ACCOUNT="ebrains-0000003"
PARTITION="dc-cpu"
CPUS=8
MEM="32G"
TIME="00:30:00"

# Where the predictions go (a directory you can write to in data-proxy)
P_SOURCE="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/zipped_images/"
OUTPUT_DIR="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/segmentationswi2/"

# EBRAINS bearer token — paste the current token here,
# or keep empty to test with a public bucket (no upload).
WI2_TOKEN=""

# Annotations JSON, base64-encoded.
# Generate with:  base64 -w0 annotations_1img.json
# (copy the output, paste below)
ANNOTATIONS_B64="PASTE_BASE64_HERE"

# Project root on HPC (where backend/ package lives)
PROJECT_DIR="/p/project1/ebrains-0000003/wi2/webilastik2"

# Scratch / log dir
SCRATCH="/p/scratch/ebrains-0000003/$USER"

# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=wi2-manual-test
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${SCRATCH}/wi2-test-manual.log
#SBATCH --export=NONE

# ── Environment ───────────────────────────────────────────────────────────────
source /p/project1/ebrains-0000003/miniforge3/etc/profile.d/conda.sh
mamba activate webilastik2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Token injected as env var — never on the command line
export WI2_TOKEN="${WI2_TOKEN}"

echo "=================================================="
echo "  wi2 manual test — $(date)"
echo "  host: $(hostname)"
echo "  cpus: $SLURM_CPUS_PER_TASK"
echo "  project: $PROJECT_DIR"
echo "=================================================="

# Quick smoke test — should print the list of DZIPs
echo ""
echo "── STEP 1: list sources ──"
python -m backend.headless_cli list \
    --url "$P_SOURCE" \
    ${WI2_TOKEN:+--token "$WI2_TOKEN"}

echo ""
echo "── STEP 2: full pipeline ──"
srun --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python -m backend.headless_cli run \
        --annotations-b64 "$ANNOTATIONS_B64" \
        --p-source "$P_SOURCE" \
        --output-dir "$OUTPUT_DIR" \
        --features '{"filters":["gaussianSmoothing","laplacianOfGaussian","gaussianGradientMagnitude","hessianOfGaussianEigenvalues"],"scales":[0.7,1.6,3.5,5.0]}' \
        ${WI2_TOKEN:+--token-env WI2_TOKEN} \
        --workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "  Pipeline exit code: $EXIT_CODE"
echo "  Finished: $(date)"
echo "=================================================="
exit $EXIT_CODE
