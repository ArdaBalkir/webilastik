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

import logging
import os
import re
import subprocess
import time
import uuid
from typing import Dict, Optional

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

# ── In-memory session registry ────────────────────────────────────────────────
_sessions: Dict[str, dict] = (
    {}
)  # session_id → {user_id, job_id, status, url, port, created_at}
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


# ── Routes ─────────────────────────────────────────────────────────────────────


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
