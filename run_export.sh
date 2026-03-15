#!/bin/bash -l

set -euo pipefail

# Shared filesystem paths. Override with environment variables if needed.
BASE="${BASE:-/p/home/jusers/webilastic-quint/ebrains-0000003}"
CONDA_SH="${CONDA_SH:-${BASE}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-webilastik2}"
REPO_CHECKOUT="${REPO_CHECKOUT:-${BASE}/webilastik}"

PROJECT_JSON="${PROJECT_JSON:-${BASE}/hpc_testing_oneshot.json}"
DZIP_URL="${DZIP_URL:-https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/zipped_images/79556738_s306.jpg.dzip}"
OUTPUT_URL="${OUTPUT_URL:-https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/predictions}"
WORKERS="${WORKERS:-8}"
LEVEL_OFFSET="${LEVEL_OFFSET:-0}"

if [[ -z "${EBRAINS_TOKEN:-}" ]]; then
  echo "EBRAINS_TOKEN is not set" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_JSON}" ]]; then
  echo "Project JSON not found: ${PROJECT_JSON}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/backend/export_cli.py" ]]; then
  WORKDIR="${SCRIPT_DIR}"
elif [[ -f "${REPO_CHECKOUT}/backend/export_cli.py" ]]; then
  WORKDIR="${REPO_CHECKOUT}"
else
  echo "Could not find backend/export_cli.py in ${SCRIPT_DIR} or ${REPO_CHECKOUT}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${WORKDIR}"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python - <<'PY'
import importlib
import sys

modules = [
    "backend.export_cli",
    "backend.dzi_source",
    "backend.features",
    "backend.classifier",
]

for name in modules:
    importlib.import_module(name)

print(f"Python executable: {sys.executable}")
print("Import smoke test passed")
PY

python -m backend.export_cli \
  --project "${PROJECT_JSON}" \
  --dzip-url "${DZIP_URL}" \
  --output-url "${OUTPUT_URL}" \
  --token "${EBRAINS_TOKEN}" \
  --level-offset "${LEVEL_OFFSET}" \
  --workers "${WORKERS}"