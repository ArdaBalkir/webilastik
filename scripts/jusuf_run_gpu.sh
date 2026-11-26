#!/bin/bash
#SBATCH --account=ebrains-0000003
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/p/scratch/ebrains-0000003/wi2-run-%j.log
#SBATCH --error=/p/scratch/ebrains-0000003/wi2-run-%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=wi2-run-gpu
# ─────────────────────────────────────────────────────────────────────────────
# Webilastik 2.0 — GPU-first run on JUSUF, CPU fallback on the same node.
#
# JUSUF "gpus" nodes: 1x NVIDIA V100 16 GB + 2x64 AMD cores, 256 GB RAM.
# They are billed per node like "batch" nodes, so the GPU costs nothing extra;
# the queue is smaller (45 vs 136 nodes), so jobs may wait longer to start.
#
# Flow:
#   1. activate $GPU_ENV and run with --device gpu (CuPy features + cuML RF)
#   2. if the env is missing, no GPU is usable, or anything fails on the GPU,
#      the CLI exits with code 3 → rerun the full pipeline with --device cpu
#      in $CPU_ENV (the prefetch cache means sources aren't downloaded again)
#
# One-time GPU env setup (login node; check the RAPIDS install selector for
# current versions — stay on CUDA 12 builds, CUDA 13 dropped the V100/Volta):
#
#   source /p/project1/ebrains-0000003/miniforge3/etc/profile.d/conda.sh
#   conda create -n webilastik2-gpu -c rapidsai -c conda-forge -c nvidia \
#       cuml cupy 'cuda-version>=12.0,<13' python=3.12 \
#       scipy scikit-learn pillow requests pyjwt cryptography
#   (install the rest from conda too: a pip install -r requirements.txt can
#   replace conda's numpy with one cupy/cuml weren't built against)
#
# Smoke-test it on a dev GPU node (develgpus also has internet):
#   srun -A ebrains-0000003 -p develgpus --gres=gpu:1 -t 00:10:00 \
#       bash -c 'source .../conda.sh; conda activate webilastik2-gpu;
#                cd /p/project1/ebrains-0000003/webilastik;
#                python -c "from backend import device; print(device.gpu_status())"'
#
# Check the budget: jutil user cpuquota
# ─────────────────────────────────────────────────────────────────────────────

jutil env activate -p ebrains-0000003

set -euo pipefail

HOME_DIR=/p/project1/ebrains-0000003
CONDA_DIR=${HOME_DIR}/miniforge3
WEBILASTIK_DIR=${HOME_DIR}/webilastik
ANNOTATIONS=${WEBILASTIK_DIR}/annotations_1img.json
CPU_ENV=webilastik2
GPU_ENV=webilastik2-gpu

P_SOURCE="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/zipped_images/"
OUTPUT_DIR="https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/segmentationswi2/"
PREFETCH_DIR=/p/scratch/ebrains-0000003/wi2_cache

# ── PASTE YOUR TOKEN BELOW ────────────────────────────────────────────────────
export WI2_TOKEN="PASTE_TOKEN_HERE"
# ─────────────────────────────────────────────────────────────────────────────

source ${CONDA_DIR}/etc/profile.d/conda.sh

export PYTHONUNBUFFERED=1
export PYTHONPATH=${WEBILASTIK_DIR}
# Outbound host allowlist (see SECURITY.md): add the storage hosts behind the
# data-proxy's pre-signed URLs, comma-separated, or downloads will be refused.
export WI2_ALLOWED_HTTP_HOSTS="${WI2_ALLOWED_HTTP_HOSTS:-}"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
# Tiles per GPU batch; lower it if the V100 runs out of memory.
export WI2_GPU_BATCH_TILES=${WI2_GPU_BATCH_TILES:-64}

# NOTE: as with jusuf_run.sh, compute nodes have no internet — prefetch the
# DZIPs into PREFETCH_DIR from the login node first (see jusuf_run.sh).

# JSC launch pattern: the GPU is bound per srun step — request it on the step
# with --gres=gpu:1 and Slurm sets CUDA_VISIBLE_DEVICES for that step. CPU
# steps get no GPU. No CUDA module is loaded: cupy/cuml bring their own CUDA
# runtime from conda and only need the node's driver. (--cuda-mps is only for
# several ranks sharing a GPU; this is one process.)
run_pipeline() {
    local gres=""
    if [ "$1" = gpu ]; then gres="--gres=gpu:1"; fi
    srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} ${gres} --overlap -u \
        python -m backend.headless_cli run \
            --annotations "${ANNOTATIONS}" \
            --p-source    "${P_SOURCE}" \
            --output-dir  "${OUTPUT_DIR}" \
            --token-env   WI2_TOKEN \
            --workers     ${SLURM_CPUS_PER_TASK} \
            --prefetch-dir "${PREFETCH_DIR}" \
            --device      "$1"
}

# Exit codes are handled explicitly; conda activation isn't nounset-safe.
set +eu
EXIT_CODE=3
if ! conda activate "${GPU_ENV}"; then
    echo "[wi2] conda env ${GPU_ENV} not available"
elif ! srun --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --overlap -u \
        bash -c 'set -o pipefail; echo "[wi2] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed "s/^/[wi2] GPU: /"'; then
    echo "[wi2] no GPU visible in a job step"
else
    run_pipeline gpu
    EXIT_CODE=$?
fi

if [ "${EXIT_CODE}" -eq 3 ]; then
    echo "[wi2] GPU path unusable — falling back to CPU at $(date)"
    conda activate "${CPU_ENV}"
    run_pipeline cpu
    EXIT_CODE=$?
fi

echo "[wi2] Pipeline finished with exit code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
