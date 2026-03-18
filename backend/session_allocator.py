"""
Webilastik 2.0 — Session Allocator
====================================
Runs on a machine that can SSH to the HPC login node (typically your laptop
during testing, or a gateway server in production).

The allocator never runs any heavy computation itself.  Its only jobs are:
  1. Accept job requests from the browser
  2. SSH to the HPC login node and submit sbatch scripts
  3. Poll HPC job status via SSH + sacct
  4. Return results

Architecture
------------
Browser ──POST /headless-jobs──► Session Allocator (this process)
                                        |
                            ssh login-node "sbatch < script"
                                        |
                                    HPC SLURM
                                        |
                         backend.headless_cli run (on compute node)
                                        | HTTP
                              data-proxy.ebrains.eu (S3)

Configuration (env vars)
------------------------
  HPC_HOST        login node hostname  (required)
  HPC_USER        SSH username          (required)
  HPC_SSH_KEY     path to private key   (optional, defaults to ~/.ssh/id_rsa)
  HPC_SSH_PORT    SSH port              (default: 22)
  HPC_ACCOUNT     SLURM account/project (required for most HPC sites)
  HPC_PARTITION   SLURM partition       (default: cpu)
  HPC_CPUS        CPUs per task         (default: 64)
  HPC_MEM         memory                (default: 128G)
  HPC_TIME        wall time HH:MM:SS    (default: 08:00:00)
  HPC_ENV_ACTIVATE path to venv activate on HPC (default: ~/wi2/.venv/bin/activate)
  HPC_SCRATCH_DIR  writable dir on HPC   (default: /tmp)
  DISABLE_AUTH    set to 1 to skip EBRAINS JWT validation (dev only)
  ALLOCATOR_PORT  listening port        (default: 8001)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import verify_token, AuthError, get_user_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Webilastik 2.0 Session Allocator")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── HPC SSH config ────────────────────────────────────────────────────────────
HPC_HOST = os.environ.get("HPC_HOST", "")
HPC_USER = os.environ.get("HPC_USER", "")
HPC_SSH_KEY = os.environ.get("HPC_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa"))
HPC_SSH_PORT = int(os.environ.get("HPC_SSH_PORT", "22"))
HPC_ACCOUNT = os.environ.get("HPC_ACCOUNT", "")

# SLURM defaults for the headless pipeline job
HPC_PARTITION = os.environ.get("HPC_PARTITION", "cpu")
HPC_CPUS = int(os.environ.get("HPC_CPUS", "64"))
HPC_MEM = os.environ.get("HPC_MEM", "128G")
HPC_TIME = os.environ.get("HPC_TIME", "08:00:00")
HPC_ENV_ACTIVATE = os.environ.get("HPC_ENV_ACTIVATE", "~/wi2/.venv/bin/activate")
HPC_SCRATCH_DIR = os.environ.get("HPC_SCRATCH_DIR", "/tmp")

# ── In-memory job registry ────────────────────────────────────────────────────
_jobs: Dict[str, dict] = {}

# ── Auth ──────────────────────────────────────────────────────────────────────


def _auth(authorization: Optional[str]) -> str:
    if os.environ.get("DISABLE_AUTH") == "1":
        return "dev-user"
    try:
        payload = verify_token(authorization)
        return get_user_id(payload)
    except AuthError as e:
        raise HTTPException(401, str(e))


def _raw_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    return parts[1] if len(parts) == 2 else authorization


# ── SSH helpers ───────────────────────────────────────────────────────────────


def _ssh_cmd(remote_cmd: str) -> List[str]:
    key_args = ["-i", HPC_SSH_KEY] if HPC_SSH_KEY else []
    return [
        "ssh",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=no",
        "-oCheckHostIP=no",
        f"-p{HPC_SSH_PORT}",
        *key_args,
        f"{HPC_USER}@{HPC_HOST}",
        "--",
        remote_cmd,
    ]


async def _ssh_run(remote_cmd: str, stdin: Optional[str] = None) -> str:
    if not HPC_HOST or not HPC_USER:
        raise RuntimeError(
            "HPC_HOST and HPC_USER env vars must be set. "
            "For local testing without real HPC, set DISABLE_AUTH=1 and "
            "HPC_HOST=localhost (requires passwordless SSH to localhost)."
        )
    cmd = _ssh_cmd(remote_cmd)

    stdin_bytes = stdin.encode() if stdin else None
    stdin_pipe = asyncio.subprocess.PIPE if stdin_bytes else asyncio.subprocess.DEVNULL

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=stdin_pipe,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=stdin_bytes)

    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh command failed (rc={proc.returncode}): {stderr.decode().strip()}"
        )
    return stdout.decode().strip()


# ── sbatch script builder ─────────────────────────────────────────────────────


def _build_sbatch_script(
    *,
    job_id: str,
    annotations_b64: str,
    t_source: Optional[str],
    p_source: str,
    output_dir: str,
    features_json: str,
    level: Optional[int],
    token: Optional[str],
    cpus: int,
    mem: str,
    partition: str,
    time_limit: str,
    account: str,
    log_path: str,
) -> str:
    account_line = f"#SBATCH --account={account}" if account else ""
    level_flag = f"--level {level}" if level is not None else ""
    t_source_flag = f"--t-source '{t_source}'" if t_source else ""
    token_export = f"export WI2_TOKEN='{token}'" if token else "# no token"
    token_flag = "--token-env WI2_TOKEN" if token else ""

    return f"""#!/bin/bash
#SBATCH --job-name=wi2-hl-{job_id[:8]}
#SBATCH --partition={partition}
{account_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}
#SBATCH --export=NONE

# ── Environment ───────────────────────────────────────────────────────────────
source {HPC_ENV_ACTIVATE}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
{token_export}

echo "[wi2] =================================================="
echo "[wi2] Job {job_id} starting on $(hostname) at $(date)"
echo "[wi2] SLURM CPUs: $SLURM_CPUS_PER_TASK  MEM: {mem}"
echo "[wi2] p_source:   {p_source}"
echo "[wi2] output_dir: {output_dir}"
echo "[wi2] =================================================="

srun --cpus-per-task=$SLURM_CPUS_PER_TASK \\
    python -m backend.headless_cli run \\
        --annotations-b64 '{annotations_b64}' \\
        {t_source_flag} \\
        --p-source '{p_source}' \\
        --output-dir '{output_dir}' \\
        --features '{features_json}' \\
        {level_flag} \\
        {token_flag} \\
        --workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?
echo "[wi2] Pipeline finished with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE
"""


# ── SLURM status via sacct ────────────────────────────────────────────────────

_SACCT_DONE_OK = frozenset(["COMPLETED", "COMPLETING"])
_SACCT_FAILED = frozenset(
    [
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    ]
)
_SACCT_PENDING = frozenset(
    [
        "PENDING",
        "CONFIGURING",
        "REQUEUED",
        "RESIZING",
        "SIGNALING",
        "STOPPED",
        "SUSPENDED",
    ]
)


async def _get_slurm_state(slurm_job_id: str) -> str:
    """
    Use sacct to get job state.  sacct covers running AND completed jobs —
    unlike squeue which only shows jobs still in the scheduler queue.
    """
    try:
        raw = await _ssh_run(
            f"sacct -j {slurm_job_id} --noheader --parsable2 "
            f"--format=JobIDRaw,State "
            f"| grep -v '\\.batch\\|\\.extern' | head -1"
        )
        if not raw:
            # Job not in sacct yet — still very newly submitted
            return "PENDING"
        _, state = raw.split("|", 1)
        return state.strip().split()[0].upper()  # "CANCELLED by 1234" -> "CANCELLED"
    except Exception as e:
        logger.warning("sacct query failed for SLURM job %s: %s", slurm_job_id, e)
        return "UNKNOWN"


def _map_state(slurm_state: str) -> str:
    if slurm_state == "RUNNING":
        return "running"
    if slurm_state in _SACCT_DONE_OK:
        return "done"
    if slurm_state in _SACCT_FAILED:
        return "error"
    return "pending"


async def _refresh(job: dict) -> None:
    if job["status"] in ("done", "error", "cancelled"):
        return
    state = await _get_slurm_state(job["slurm_job_id"])
    job["slurm_state"] = state
    job["status"] = _map_state(state)


# ── Request model ─────────────────────────────────────────────────────────────


class HeadlessJobRequest(BaseModel):
    """POST body for /headless-jobs."""

    annotations: List[dict]  # [{dzip_url, strokes:[{label, points}]}]
    t_source: Optional[str] = None  # override dzip_url for all annotations
    features: dict = {  # type: ignore[assignment]
        "filters": [
            "gaussianSmoothing",
            "laplacianOfGaussian",
            "gaussianGradientMagnitude",
            "hessianOfGaussianEigenvalues",
        ],
        "scales": [0.7, 1.6, 3.5, 5.0],
    }
    level: Optional[int] = None
    p_source: str
    output_dir: str
    # SLURM overrides
    partition: Optional[str] = None
    cpus: Optional[int] = None
    mem: Optional[str] = None
    time_limit: Optional[str] = None  # t_source * 5 mins or so, terminates when done
    account: Optional[str] = None  # ebrains-0003


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "hpc_host": HPC_HOST or "(not configured)"}


@app.post("/headless-jobs")
async def create_headless_job(
    req: HeadlessJobRequest,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    token = _raw_token(authorization)
    job_id = str(uuid.uuid4())
    log_path = f"{HPC_SCRATCH_DIR}/wi2-{job_id[:8]}.log"

    ann_b64 = base64.b64encode(
        json.dumps(req.annotations, separators=(",", ":")).encode()
    ).decode()
    features_json = json.dumps(req.features, separators=(",", ":"))

    script = _build_sbatch_script(
        job_id=job_id,
        annotations_b64=ann_b64,
        t_source=req.t_source,
        p_source=req.p_source,
        output_dir=req.output_dir,
        features_json=features_json,
        level=req.level,
        token=token,
        cpus=req.cpus or HPC_CPUS,
        mem=req.mem or HPC_MEM,
        partition=req.partition or HPC_PARTITION,
        time_limit=req.time_limit or HPC_TIME,
        account=req.account or HPC_ACCOUNT,
        log_path=log_path,
    )

    try:
        raw = await _ssh_run("sbatch --parsable", stdin=script)
    except RuntimeError as e:
        logger.error("sbatch failed for job %s: %s", job_id, e)
        raise HTTPException(500, f"sbatch submission failed: {e}")

    slurm_job_id = raw.split(";")[0].strip()
    logger.info("Job %s → SLURM %s (user %s)", job_id, slurm_job_id, user_id)

    _jobs[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "slurm_job_id": slurm_job_id,
        "slurm_state": "PENDING",  # This likely is "PD" in squeue
        "status": "pending",
        "p_source": req.p_source,
        "output_dir": req.output_dir,
        "log_path": log_path,
        "created_at": time.time(),
    }
    return {
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "status": "pending",
        "log_path": log_path,
    }


@app.get("/headless-jobs")
async def list_headless_jobs(authorization: Optional[str] = Header(default=None)):
    user_id = _auth(authorization)
    my_jobs = [j for j in _jobs.values() if j["user_id"] == user_id]
    await asyncio.gather(*[_refresh(j) for j in my_jobs])
    return my_jobs


@app.get("/headless-jobs/{job_id}")
async def get_headless_job(
    job_id: str, authorization: Optional[str] = Header(default=None)
):
    user_id = _auth(authorization)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Not your job")
    await _refresh(job)
    return job


@app.delete("/headless-jobs/{job_id}")
async def cancel_headless_job(
    job_id: str, authorization: Optional[str] = Header(default=None)
):
    user_id = _auth(authorization)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Not your job")
    try:
        await _ssh_run(f"scancel {job['slurm_job_id']}")
    except Exception as e:
        logger.warning("scancel failed: %s", e)
    job["status"] = "cancelled"
    return {"cancelled": job_id}


@app.get("/headless-jobs/{job_id}/log")
async def get_job_log(
    job_id: str,
    tail: int = 100,
    authorization: Optional[str] = Header(default=None),
):
    """Tail the SLURM job log over SSH."""
    user_id = _auth(authorization)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Not your job")
    try:
        log = await _ssh_run(
            f"tail -n {tail} {job['log_path']} 2>/dev/null || echo '(log not yet available)'"
        )
    except Exception as e:
        log = f"(could not fetch log: {e})"
    return {"log": log}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.session_allocator:app",
        host="0.0.0.0",
        port=int(os.environ.get("ALLOCATOR_PORT", "8001")),
        reload=False,
    )
