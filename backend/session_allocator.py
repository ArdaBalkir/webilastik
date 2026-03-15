"""
Session Allocator — thin SLURM job manager for webilastik 2.0.

Endpoints
---------
POST   /sessions            → Submit a SLURM job; returns session_id
GET    /sessions/{id}       → Poll status (pending / running / done / error)
DELETE /sessions/{id}       → Cancel SLURM job
GET    /sessions            → List all sessions for authenticated user
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import verify_token, AuthError, get_user_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Webilastik 2.0 Session Allocator")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Config (from environment) ─────────────────────────────────────────────────
COMPUTE_IMAGE = os.environ.get("COMPUTE_IMAGE", "webilastik2-compute")
SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "gpu")
SLURM_GPUS = os.environ.get("SLURM_GPUS", "1")
SLURM_CPUS = os.environ.get("SLURM_CPUS", "16")
SLURM_MEM = os.environ.get("SLURM_MEM", "64G")
SLURM_TIME = os.environ.get("SLURM_TIME", "02:00:00")
# Base URL prefix where compute servers are reachable (e.g. "https://app.ilastik.org/sessions")
BASE_SESSION_URL = os.environ.get("BASE_SESSION_URL", "http://localhost")
SESSION_PORT_START = int(os.environ.get("SESSION_PORT_START", "9000"))

# Headless job config — separate resource profile (more CPUs, no GPU needed, longer wall time)
SLURM_HL_PARTITION = os.environ.get("SLURM_HL_PARTITION", SLURM_PARTITION)
SLURM_HL_CPUS = int(os.environ.get("SLURM_HL_CPUS", "64"))
SLURM_HL_MEM = os.environ.get("SLURM_HL_MEM", "128G")
SLURM_HL_TIME = os.environ.get("SLURM_HL_TIME", "08:00:00")
# Path to the Python environment activate script on the compute nodes
WI2_ENV_ACTIVATE = os.environ.get(
    "WI2_ENV_ACTIVATE", "/opt/webilastik2/env/bin/activate"
)
# Directory on shared filesystem for job temp files (must be readable by compute nodes)
WI2_SCRATCH_DIR = os.environ.get("WI2_SCRATCH_DIR", "/tmp")

# ── In-memory session registry ────────────────────────────────────────────────
_sessions: Dict[str, dict] = (
    {}
)  # session_id → {user_id, job_id, status, url, port, created_at}

# ── In-memory headless job registry ──────────────────────────────────────────
_headless_jobs: Dict[str, dict] = (
    {}
)  # job_id → {user_id, slurm_job_id, status, log_path, created_at, ...}
_port_counter = SESSION_PORT_START


def _next_port() -> int:
    global _port_counter
    p = _port_counter
    _port_counter += 1
    return p


# ── Auth helper ───────────────────────────────────────────────────────────────


def _auth(authorization: Optional[str]) -> str:
    if os.environ.get("DISABLE_AUTH") == "1":
        return "dev-user"
    try:
        payload = verify_token(authorization)
        return get_user_id(payload)
    except AuthError as e:
        raise HTTPException(401, str(e))


# ── SLURM helpers ─────────────────────────────────────────────────────────────


def _sbatch(session_id: str, port: int, user_id: str) -> str:
    """Submit an sbatch job, return the SLURM job ID."""
    script = f"""#!/bin/bash
#SBATCH --job-name=webilastik-{session_id[:8]}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --gpus={SLURM_GPUS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output=/tmp/webilastik-{session_id}.log

# Activate environment (edit path to match your setup)
source /opt/webilastik2/env/bin/activate

# Start compute server
DISABLE_AUTH=0 PORT={port} \\
    python -m backend.server
"""
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script.encode(),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.decode()}")
    job_id = result.stdout.decode().strip().split(";")[0]
    return job_id


def _squeue_state(job_id: str) -> str:
    """Return SLURM state string for a job, or 'UNKNOWN'."""
    result = subprocess.run(
        ["squeue", "--job", job_id, "--noheader", "--format=%T"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        # Job no longer in queue
        sacct = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--format=State", "--parsable2"],
            capture_output=True,
            timeout=10,
        )
        lines = sacct.stdout.decode().strip().splitlines()
        return lines[0].strip() if lines else "UNKNOWN"
    return result.stdout.decode().strip()


def _scancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], timeout=10, check=False)


# ── SLURM state → our status mapping ──────────────────────────────────────────


def _map_state(slurm_state: str, port: int) -> tuple[str, Optional[str]]:
    """Returns (status, url_or_None)."""
    s = slurm_state.upper()
    if s in ("PENDING", "CONFIGURING"):
        return "pending", None
    if s == "RUNNING":
        url = f"{BASE_SESSION_URL}:{port}"
        return "running", url
    if s in ("COMPLETED", "COMPLETING"):
        return "done", None
    if s in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "UNKNOWN"):
        return "error", None
    return "pending", None  # default


# ── Headless job model ────────────────────────────────────────────────────────


class HeadlessJobRequest(BaseModel):
    """
    Submit a train + batch-export pipeline as a SLURM job.
    The job calls backend.headless_cli directly on the compute node — no
    running server required.  Token is passed via a secure env var in the
    batch script, never on the command line.
    """

    # Training inputs
    annotations: List[dict]  # [{dzip_url, strokes:[{label, points}]}]
    t_source: Optional[str] = None  # optional override dzip_url for all annotations
    features: dict = {  # type: ignore[assignment]
        "filters": [
            "gaussianSmoothing",
            "laplacianOfGaussian",
            "gaussianGradientMagnitude",
            "hessianOfGaussianEigenvalues",
        ],
        "scales": [0.7, 1.6, 3.5, 5.0],
    }
    level: Optional[int] = None  # training DZI level; None = max (full res)

    # Export inputs
    p_source: str  # data-proxy dir containing source DZIPs
    output_dir: str  # data-proxy dir where prediction DZIPs are written

    # SLURM overrides (all optional; defaults from env vars above)
    partition: Optional[str] = None
    cpus: Optional[int] = None
    mem: Optional[str] = None
    time_limit: Optional[str] = None  # HH:MM:SS


# ── Headless sbatch builder ────────────────────────────────────────────────────


def _sbatch_headless(
    job_id: str,
    req: HeadlessJobRequest,
    token: Optional[str],
) -> tuple[str, str]:
    """
    Generate and submit an sbatch script for the headless pipeline.
    Returns (slurm_job_id, log_path).

    Design:
    - Single node, many CPUs (no GPU — CPU RF inference is fast enough and
      keeps headless jobs off the scarce GPU queue).
    - Token passed via env var in the script — never visible in `ps` or logs.
    - Annotations JSON base64-encoded directly into the script (heredoc
      quoting issues avoided; works even without shared filesystem).
    - srun used so SLURM properly accounts CPU affinity.
    - ThreadPoolExecutor workers = $SLURM_CPUS_PER_TASK so we use exactly
      what we were allocated.
    """
    cpus = req.cpus or SLURM_HL_CPUS
    mem = req.mem or SLURM_HL_MEM
    partition = req.partition or SLURM_HL_PARTITION
    time_limit = req.time_limit or SLURM_HL_TIME
    log_path = f"{WI2_SCRATCH_DIR}/wi2-headless-{job_id}.log"

    # Base64-encode annotations so we can embed them safely in the batch script
    ann_b64 = base64.b64encode(
        json.dumps(req.annotations, separators=(",", ":")).encode()
    ).decode()

    features_json = json.dumps(req.features, separators=(",", ":"))
    level_flag = f"--level {req.level}" if req.level is not None else ""
    t_source_flag = f"--t-source '{req.t_source}'" if req.t_source else ""

    # Token injection: if token provided, set WI2_TOKEN env var in the script.
    # The variable is unexported outside the job and never written to any file.
    token_export = f"export WI2_TOKEN='{token}'" if token else "# no token provided"
    token_flag = "--token-env WI2_TOKEN" if token else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=wi2-hl-{job_id[:8]}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}
#SBATCH --export=NONE

# ── Environment ──────────────────────────────────────────────────────────────
source {WI2_ENV_ACTIVATE}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
{token_export}

echo "[wi2] Job {job_id} starting on $(hostname)"
echo "[wi2] CPUs: $SLURM_CPUS_PER_TASK  MEM: {mem}"
echo "[wi2] p_source:   {req.p_source}"
echo "[wi2] output_dir: {req.output_dir}"
date

# ── Run headless pipeline ────────────────────────────────────────────────────
srun --cpus-per-task=$SLURM_CPUS_PER_TASK \\
    python -m backend.headless_cli \\
        --annotations-b64 '{ann_b64}' \\
        {t_source_flag} \\
        --p-source '{req.p_source}' \\
        --output-dir '{req.output_dir}' \\
        --features '{features_json}' \\
        {level_flag} \\
        {token_flag} \\
        --workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?
echo "[wi2] Pipeline finished with exit code $EXIT_CODE"
date
exit $EXIT_CODE
"""

    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script.encode(),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.decode().strip()}")
    slurm_job_id = result.stdout.decode().strip().split(";")[0]
    return slurm_job_id, log_path


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.post("/headless-jobs")
async def create_headless_job(
    req: HeadlessJobRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Submit a train+export pipeline as a SLURM batch job."""
    user_id = _auth(authorization)

    # Extract raw token to embed in job script
    raw_token: Optional[str] = None
    if authorization:
        parts = authorization.split()
        raw_token = parts[1] if len(parts) == 2 else authorization

    job_id = str(uuid.uuid4())
    try:
        slurm_job_id, log_path = _sbatch_headless(job_id, req, raw_token)
    except Exception as e:
        logger.error("sbatch_headless error: %s", e)
        raise HTTPException(500, f"Failed to submit headless job: {e}")

    _headless_jobs[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "slurm_job_id": slurm_job_id,
        "status": "pending",
        "p_source": req.p_source,
        "output_dir": req.output_dir,
        "log_path": log_path,
        "created_at": time.time(),
    }
    logger.info(
        "Headless job %s submitted for user %s (SLURM %s)",
        job_id,
        user_id,
        slurm_job_id,
    )
    return {
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "status": "pending",
        "log_path": log_path,
    }


@app.get("/headless-jobs")
async def list_headless_jobs(
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    jobs = [j for j in _headless_jobs.values() if j["user_id"] == user_id]
    for j in jobs:
        _refresh_headless_job(j)
    return jobs


@app.get("/headless-jobs/{job_id}")
async def get_headless_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    j = _headless_jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Headless job not found")
    if j["user_id"] != user_id:
        raise HTTPException(403, "Not your job")
    _refresh_headless_job(j)
    return j


@app.delete("/headless-jobs/{job_id}")
async def cancel_headless_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    j = _headless_jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Headless job not found")
    if j["user_id"] != user_id:
        raise HTTPException(403, "Not your job")
    _scancel(j["slurm_job_id"])
    j["status"] = "cancelled"
    return {"cancelled": job_id}


def _refresh_headless_job(j: dict) -> None:
    if j["status"] in ("done", "error", "cancelled"):
        return
    try:
        slurm_state = _squeue_state(j["slurm_job_id"])
        s = slurm_state.upper()
        if s in ("PENDING", "CONFIGURING"):
            j["status"] = "pending"
        elif s == "RUNNING":
            j["status"] = "running"
        elif s in ("COMPLETED", "COMPLETING"):
            j["status"] = "done"
        elif s in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "UNKNOWN"):
            j["status"] = "error" if s != "CANCELLED" else "cancelled"
    except Exception as e:
        logger.warning("Could not refresh headless job %s: %s", j["job_id"], e)


@app.post("/sessions")
async def create_session(authorization: Optional[str] = Header(default=None)):
    user_id = _auth(authorization)
    session_id = str(uuid.uuid4())
    port = _next_port()

    try:
        job_id = _sbatch(session_id, port, user_id)
    except Exception as e:
        logger.error("sbatch error: %s", e)
        raise HTTPException(500, f"Failed to submit job: {e}")

    _sessions[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "job_id": job_id,
        "port": port,
        "status": "pending",
        "url": None,
        "created_at": time.time(),
    }
    logger.info("Session %s created for user %s (job %s)", session_id, user_id, job_id)
    return {"session_id": session_id, "status": "pending", "url": None}


@app.get("/sessions")
async def list_sessions(authorization: Optional[str] = Header(default=None)):
    user_id = _auth(authorization)
    user_sessions = [s for s in _sessions.values() if s["user_id"] == user_id]
    # Refresh status for all
    for s in user_sessions:
        _refresh_session(s)
    return user_sessions


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if s["user_id"] != user_id:
        raise HTTPException(403, "Not your session")
    _refresh_session(s)
    return s


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = _auth(authorization)
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if s["user_id"] != user_id:
        raise HTTPException(403, "Not your session")
    _scancel(s["job_id"])
    del _sessions[session_id]
    return {"deleted": session_id}


# ── Status refresh ─────────────────────────────────────────────────────────────


def _refresh_session(s: dict) -> None:
    if s["status"] in ("done", "error"):
        return
    try:
        slurm_state = _squeue_state(s["job_id"])
        status, url = _map_state(slurm_state, s["port"])
        s["status"] = status
        if url:
            s["url"] = url
    except Exception as e:
        logger.warning("Could not refresh session %s: %s", s["session_id"], e)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.session_allocator:app",
        host="0.0.0.0",
        port=int(os.environ.get("ALLOCATOR_PORT", "8001")),
    )
