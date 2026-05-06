#!/bin/bash -x
#SBATCH --account=ebrains-0000003
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/p/scratch/ebrains-0000003/wi2-run-%j.log
#SBATCH --error=/p/scratch/ebrains-0000003/wi2-run-%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=wi2-run

jutil env activate -p ebrains-0000003

set -euo pipefail

HOME_DIR=/p/project1/ebrains-0000003
CONDA_DIR=${HOME_DIR}/miniforge3
WEBILASTIK_DIR=${HOME_DIR}/webilastik
ANNOTATIONS=${WEBILASTIK_DIR}/annotations_1img.json

P_SOURCE="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/zipped_images/"
OUTPUT_DIR="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/segmentationswi2/"
PREFETCH_DIR=/p/scratch/ebrains-0000003/wi2_cache

# ── PASTE YOUR TOKEN BELOW ────────────────────────────────────────────────────
export WI2_TOKEN="PASTE_TOKEN_HERE"
# ─────────────────────────────────────────────────────────────────────────────

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate webilastik2

export PYTHONUNBUFFERED=1
export PYTHONPATH=${WEBILASTIK_DIR}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# NOTE: DZIPs must already be in PREFETCH_DIR — download them on the login node
# first (batch partition has no internet):
#
#   source /p/project1/ebrains-0000003/miniforge3/etc/profile.d/conda.sh
#   conda activate webilastik2
#   export PYTHONPATH=/p/project1/ebrains-0000003/webilastik
#   export WI2_TOKEN="..."
#   python -m backend.headless_cli run \
#       --annotations /p/project1/ebrains-0000003/webilastik/annotations_1img.json \
#       --p-source    "${P_SOURCE}" \
#       --output-dir  "${OUTPUT_DIR}" \
#       --token-env   WI2_TOKEN \
#       --workers     8 \
#       --prefetch-dir ${PREFETCH_DIR}
#   # (it will fail at upload — that's fine, the cache is what we need)

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --overlap -u \
    python -m backend.headless_cli run \
        --annotations "${ANNOTATIONS}" \
        --p-source    "${P_SOURCE}" \
        --output-dir  "${OUTPUT_DIR}" \
        --token-env   WI2_TOKEN \
        --workers     ${SLURM_CPUS_PER_TASK} \
        --prefetch-dir "${PREFETCH_DIR}"
