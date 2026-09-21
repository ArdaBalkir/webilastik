"""
Webilastik 2.0 — Annotation / Preview Compute Server

This server runs LOCALLY (or on a small gateway machine).
Its only purpose is to support the annotation UI — it is NOT used for
bulk export.  Bulk export runs on HPC via backend.session_allocator + sbatch.

Endpoints
---------
GET  /health
GET  /dzi-info?dzip_url=...         — read DZI metadata from a remote DZIP
GET  /list-sources?url=...          — list .dzip objects in a data-proxy directory
POST /train                         — fit a Random Forest on brushstroke pixels
GET  /predict/{clf_id}/{level}/{c}_{r}  — prediction tile PNG for live overlay

The following endpoints exist for local single-image testing only and are
NOT part of the HPC pipeline:
POST /export                        — single-image export job (local)
GET  /export/{job_id}
POST /batch-export                  — multi-image export (local, CPU only)
GET  /batch-export/{job_id}
POST /export-zip                    — synchronous DZIP download
POST /headless-run                  — all-in-one train+export (local testing)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import pathlib
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .auth import verify_token, AuthError, get_user_id
from .classifier import Classifier
from .encoding import encode_prediction_png
from .dzi_source import DzipSource
from .features import extract_features

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Webilastik 2.0 Compute Server", version="2.0.0")

_PROD_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "https://app.ilastik.org,http://localhost:5173,http://localhost:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_PROD_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files when dist/ exists (production mode)
_DIST = pathlib.Path(__file__).parent.parent / "dist"
if _DIST.is_dir():
    app.mount("/app", StaticFiles(directory=str(_DIST), html=True), name="ui")

# ── CPU executor pools ───────────────────────────────────────────────────────
# Train/export are heavyweight (sklearn, feature extraction across tiles).
# Keep them on a small pool so they don't starve live prediction requests.
_CPUS = os.cpu_count() or 4
# At most 2 concurrent train/export jobs; each gets up to _CPUS cores via sklearn.
_train_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="train")
# Tile prediction is fast per tile; allow more concurrency here.
_predict_executor = ThreadPoolExecutor(max_workers=_CPUS, thread_name_prefix="predict")

# Hard cap: refuse a 3rd simultaneous train so the queue never grows silently.
_train_semaphore = asyncio.Semaphore(2)


async def _run_train(fn, *args):
    """Run a CPU-heavy training/export task; returns HTTP 503 if already at capacity."""
    if not _train_semaphore._value:  # non-blocking peek
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Server busy — too many concurrent training jobs, please retry shortly",
        )
    async with _train_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_train_executor, fn, *args)


async def _run_predict(fn, *args):
    """Run a lightweight predict/IO task."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_predict_executor, fn, *args)


# Keep _run as an alias for predict (used by misc lightweight helpers).
async def _run(fn, *args):
    return await _run_predict(fn, *args)


# ── In-memory state ───────────────────────────────────────────────────────────
# Each authenticated user owns at most one classifier.  A successful retrain
# installs a fresh classifier ID (which invalidates cached prediction tiles)
# and removes that user's previous classifier immediately.
# classifier_id → {"clf": Classifier, "last_used": float, "user_id": str}
_CLF_TTL = 30 * 60  # 30 minutes
_classifiers: Dict[str, Dict[str, Any]] = {}
_user_classifier_ids: Dict[str, str] = {}
_classifier_lock = threading.Lock()
import time


def _store_clf(user_id: str, clf: "Classifier") -> str:
    """Atomically replace a user's classifier and return its new cache-safe ID."""
    classifier_id = str(uuid.uuid4())
    now = time.time()
    with _classifier_lock:
        previous_id = _user_classifier_ids.get(user_id)
        if previous_id is not None:
            _classifiers.pop(previous_id, None)
        _classifiers[classifier_id] = {
            "clf": clf,
            "last_used": now,
            "user_id": user_id,
        }
        _user_classifier_ids[user_id] = classifier_id

    if previous_id is not None:
        logger.info(
            "[classifier] user %s replaced model %s with %s",
            user_id,
            previous_id,
            classifier_id,
        )
    else:
        logger.info("[classifier] user %s stored model %s", user_id, classifier_id)
    return classifier_id


def _get_clf(classifier_id: str, user_id: str) -> "Classifier":
    """Return the user's current classifier, touch its TTL, or raise 404."""
    with _classifier_lock:
        entry = _classifiers.get(classifier_id)
        if entry is None or entry["user_id"] != user_id:
            raise HTTPException(
                status_code=404,
                detail="Classifier not found (may have been replaced or expired)",
            )
        entry["last_used"] = time.time()
        return entry["clf"]


# (dzip_url, token) → DzipSource (reuse HTTP sessions per token)
_dzip_cache: Dict[tuple, DzipSource] = {}
# export job_id → dict with status/progress
_exports: Dict[str, Dict[str, Any]] = {}


def _get_dzip(url: str, authorization: Optional[str] = None) -> DzipSource:
    # Extract raw token value from "Bearer <token>" header
    token: Optional[str] = None
    if authorization:
        parts = authorization.split()
        token = parts[1] if len(parts) == 2 else authorization
    key = (url, token)
    if key not in _dzip_cache:
        _dzip_cache[key] = DzipSource(url, bearer_token=token)
    return _dzip_cache[key]


# ── EBRAINS data-proxy upload helpers ────────────────────────────────────────

import re as _re

_DP_BASE = "https://data-proxy.ebrains.eu/api/v1"


def _dp_normalize(url: str) -> str:
    """
    Ensure a data-proxy URL is in canonical form:
      https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}/{object}

    Handles user-supplied shortforms like:
      https://data-proxy.ebrains.eu/{bucket}/{object}          (missing /api/v1/buckets)
      https://data-proxy.ebrains.eu/api/{bucket}/{object}      (missing v1/buckets)
    """
    if not "data-proxy.ebrains.eu" in url:
        return url  # not a data-proxy URL, pass through
    # Already canonical
    if "/api/v1/buckets/" in url:
        return url
    # Strip scheme + host, then figure out the path tail
    m = _re.match(r"(https?://data-proxy\.ebrains\.eu)(/.*)", url)
    if not m:
        return url
    path = m.group(2)
    # Drop any /api or /api/v1 prefix that may be present
    path = _re.sub(r"^/api(/v1)?", "", path)
    # Drop a leading /buckets/ if present
    path = _re.sub(r"^/buckets", "", path)
    canonical = f"{_DP_BASE}/buckets{path}"
    logger.debug("dp_normalize: %s → %s", url, canonical)
    return canonical


def _dp_put(url: str, data: bytes, token: str) -> int:
    """
    Upload *data* to a data-proxy object URL using the two-step pre-signed flow:
      1. PUT {dp_url}  (no body, Authorization header) → {"url": presigned_s3_url}
      2. PUT {presigned_s3_url}  (data, no auth)
    Returns the final HTTP status code.
    """
    import requests as _req
    import time as _time

    canonical = _dp_normalize(url)
    logger.info("[dp_put] step-1 request pre-signed URL: %s", canonical)
    r1 = _req.put(
        canonical,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    logger.info("[dp_put] step-1 response: %d  body: %.200s", r1.status_code, r1.text)
    r1.raise_for_status()
    presigned = r1.json()["url"]
    logger.info("[dp_put] step-2 PUT %d bytes to S3 pre-signed URL", len(data))
    r2 = _req.put(presigned, data=data, timeout=120)
    logger.info("[dp_put] step-2 response: %d", r2.status_code)
    r2.raise_for_status()
    return r2.status_code


# ── Shared export helper ───────────────────────────────────────────────────────────


def _build_prediction_dzip(
    dzip_url: str,
    dzi_name: str,
    clf: "Classifier",
    features: "FeatureSpec",
    authorization: Optional[str],
    job_id: Optional[str] = None,
    level: Optional[int] = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Predict every tile at the requested level (default: max_level = full res),
    write PNGs to a temp directory, pack them into a DZIP, and return
    (tmpdir, dzip_path).

    Caller is responsible for shutil.rmtree(tmpdir) after use.
    If job_id is given, updates _exports[job_id]["progress"] as tiles complete.
    """
    import math, zipfile, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dzip_src = _get_dzip(dzip_url, authorization)
    _, meta = dzip_src.find_dzi()

    # Use the requested level; clamp to max_level so we never request tiles
    # that don't exist in the source (which would silently produce empty output).
    level = min(level, meta.max_level) if level is not None else meta.max_level
    scale = 2.0 ** (level - meta.max_level)
    lw = max(1, round(meta.width * scale))
    lh = max(1, round(meta.height * scale))
    ts, ol = meta.tile_size, meta.overlap
    num_cols = math.ceil(lw / ts)
    num_rows = math.ceil(lh / ts)
    total = num_cols * num_rows
    out_name = f"{dzi_name}_predictions"

    logger.info(
        "[build-dzip] %s — %d×%d tiles at level %d (scale %.4f)",
        out_name,
        num_cols,
        num_rows,
        level,
        scale,
    )

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="webilastik_"))
    tiles_dir = tmpdir / f"{out_name}_files" / str(level)
    tiles_dir.mkdir(parents=True)
    logger.info("[build-dzip] tmpdir: %s", tmpdir)

    dzi_xml = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        f'  Format="png" Overlap="{ol}" TileSize="{ts}">\n'
        f'  <Size Width="{lw}" Height="{lh}"/>\n'
        f"</Image>\n"
    )

    errors: list[str] = []
    lock = threading.Lock()
    done: list[int] = [0]  # mutable counter safe to mutate inside threads

    def process_tile(col: int, row: int) -> None:
        try:
            tile_arr = dzip_src.get_tile(dzi_name, level, col, row, meta.format)
            feat = extract_features(tile_arr, features.filters, features.scales)
            h, w = tile_arr.shape[:2]
            proba = clf.predict_proba(feat).reshape(h, w, -1)
            assert clf.classes_ is not None
            png = encode_prediction_png(proba, clf.classes_.tolist())
            (tiles_dir / f"{col}_{row}.png").write_bytes(png)
        except Exception as e:
            with lock:
                errors.append(f"{col},{row}: {e}")
            logger.warning("[build-dzip] tile %d,%d failed: %s", col, row, e)
        finally:
            with lock:
                done[0] += 1
                if job_id:
                    _exports[job_id]["progress"] = (
                        done[0] / total * 0.9
                    )  # 90% = predict
                if done[0] % 50 == 0 or done[0] == total:
                    logger.info("[build-dzip] %d/%d tiles done", done[0], total)

    tile_coords = [(c, r) for r in range(num_rows) for c in range(num_cols)]
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as pool:
        futures = [pool.submit(process_tile, c, r) for c, r in tile_coords]
        for f in as_completed(futures):
            f.result()  # propagate unexpected exceptions

    if errors:
        logger.warning(
            "[build-dzip] %d/%d tile errors: %s",
            len(errors),
            total,
            "; ".join(errors[:5]),
        )

    # Write .dzi manifest and pack DZIP
    (tmpdir / f"{out_name}.dzi").write_text(dzi_xml, encoding="utf-8")
    dzip_path = tmpdir / f"{out_name}.dzip"
    logger.info("[build-dzip] packing %s", dzip_path)
    with zipfile.ZipFile(dzip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tmpdir / f"{out_name}.dzi", f"{out_name}.dzi")
        for png_file in sorted(tiles_dir.iterdir()):
            zf.write(png_file, f"{out_name}_files/{level}/{png_file.name}")
    logger.info("[build-dzip] packed %d bytes", dzip_path.stat().st_size)
    return tmpdir, dzip_path


# ── Auth dependency ───────────────────────────────────────────────────────────


def _auth(authorization: Optional[str]) -> Dict[str, Any]:
    """Verify bearer token; skip in dev if DISABLE_AUTH=1."""
    if os.environ.get("DISABLE_AUTH") == "1":
        return {"sub": "dev-user"}
    try:
        payload = verify_token(authorization)
        # A stable subject is required to enforce one classifier per user.
        get_user_id(payload)
        return payload
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


# ── Models ────────────────────────────────────────────────────────────────────


class StrokeData(BaseModel):
    label: int
    points: List[List[int]]  # [[x, y], ...]


class FeatureSpec(BaseModel):
    filters: List[str]
    scales: List[float]


class TrainRequest(BaseModel):
    dzip_url: str
    dzi_name: str
    level: int
    strokes: List[StrokeData]
    features: FeatureSpec


class TrainResponse(BaseModel):
    classifier_id: str
    num_classes: int


# Multi-image training — matches the annotations JSON format exactly
class AnnotationImage(BaseModel):
    """One annotated image. Matches {dzip_url, level, strokes} annotations JSON entry."""

    dzip_url: str
    level: Optional[int] = None  # DZI level strokes were drawn at; None = max_level
    strokes: List[StrokeData]


class TrainMultiRequest(BaseModel):
    """Train a single classifier from strokes across multiple images."""

    annotations: List[AnnotationImage]
    features: FeatureSpec


class ExportRequest(BaseModel):
    classifier_id: str
    dzip_url: str
    dzi_name: str
    level: int
    features: FeatureSpec
    output_url: Optional[str] = (
        None  # if set, PUT the DZIP there; else return as download
    )


class BatchExportRequest(BaseModel):
    """
    Export every DZIP in *p_source* through the classifier and write results
    to *output_dir*.  Each DZIP is processed sequentially on the server but
    tiles within one image are parallelised across all CPU cores.
    """

    classifier_id: str
    p_source: str  # data-proxy dir URL containing source DZIPs
    output_dir: str  # data-proxy dir URL where predictions will be written
    features: FeatureSpec
    level: Optional[int] = (
        None  # DZI level to export at; None = max_level of each image
    )


class HeadlessRequest(BaseModel):
    """
    All-in-one: train then batch-export without touching the UI.
    Useful for HPC scripts and benchmarking.
    """

    # Training
    t_source: str  # single DZIP URL (or comma-separated list) to train on
    annotations: List[dict]  # [{dzip_url, strokes:[{label, points}]}]
    features: FeatureSpec
    level: int  # training zoom level (use maxLevel for full res)
    # Export
    p_source: str  # data-proxy dir URL to scan for DZIPs to predict
    output_dir: str  # output directory URL


class SaveProjectRequest(BaseModel):
    """
    Upload a project JSON blob to the user's data-proxy bucket so it can be
    shared across sessions / machines without manual file downloads.
    The caller is responsible for supplying a canonical data-proxy object URL:
      https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}/ilastikProjectSaves/{name}.json
    """

    json_content: str  # serialised project JSON
    dest_url: str  # full data-proxy object URL to PUT to


# ── Routes ────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _start_eviction_loop():
    async def _evict():
        while True:
            await asyncio.sleep(60)
            now = time.time()
            with _classifier_lock:
                expired = [
                    k
                    for k, v in _classifiers.items()
                    if now - v["last_used"] > _CLF_TTL
                ]
                for k in expired:
                    entry = _classifiers.pop(k)
                    user_id = entry["user_id"]
                    if _user_classifier_ids.get(user_id) == k:
                        _user_classifier_ids.pop(user_id, None)
            for k in expired:
                logger.info("[evict] classifier %s removed after TTL", k)

    asyncio.create_task(_evict())


@app.get("/health")
async def health():
    with _classifier_lock:
        classifier_count = len(_classifiers)
    return {"status": "ok", "classifiers": classifier_count}


@app.post("/save-project")
async def save_project_to_proxy(
    req: SaveProjectRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Write *json_content* to *dest_url* in the EBRAINS data-proxy using the
    two-step pre-signed S3 upload.  A bearer token is required.
    Destination should follow the convention:
      …/buckets/{bucket}/ilastikProjectSaves/{filename}.json
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401, detail="Authorization bearer token required"
        )

    data = req.json_content.encode("utf-8")
    try:
        await _run(_dp_put, req.dest_url, data, token)
    except Exception as exc:
        logger.warning("save-project upload failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}") from exc

    return {"status": "ok", "url": req.dest_url, "bytes": len(data)}


@app.get("/dzi-info")
async def dzi_info(
    dzip_url: str,
    authorization: Optional[str] = Header(default=None),
):
    _auth(authorization)

    def _load():
        dzip = _get_dzip(dzip_url, authorization)
        name, meta = dzip.find_dzi()
        return name, meta

    name, meta = await _run(_load)
    return {
        "name": name,
        "width": meta.width,
        "height": meta.height,
        "tileSize": meta.tile_size,
        "overlap": meta.overlap,
        "format": meta.format,
        "maxLevel": meta.max_level,
    }


@app.post("/train", response_model=TrainResponse)
async def train(
    req: TrainRequest,
    authorization: Optional[str] = Header(default=None),
):
    user = _auth(authorization)
    user_id = get_user_id(user)
    logger.info("Train request from %s: %d strokes", user_id, len(req.strokes))

    def _train() -> tuple[str, int]:
        dzip = _get_dzip(req.dzip_url, authorization)
        _, meta = dzip.find_dzi()

        scale = 2 ** (req.level - meta.max_level)
        lw = max(1, round(meta.width * scale))
        lh = max(1, round(meta.height * scale))
        ts = meta.tile_size
        ol = meta.overlap

        # Group annotation points by tile
        tile_pts: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for stroke in req.strokes:
            for x, y in stroke.points:
                lx = min(max(0, x), lw - 1)
                ly = min(max(0, y), lh - 1)
                col = lx // ts
                row = ly // ts
                # Pixel inside the tile image (accounting for overlap padding)
                tx = lx - col * ts + (ol if col > 0 else 0)
                ty = ly - row * ts + (ol if row > 0 else 0)
                tile_pts[(col, row)].append((tx, ty, stroke.label))

        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for (col, row), pts in tile_pts.items():
            try:
                tile_arr = dzip.get_tile(req.dzi_name, req.level, col, row, meta.format)
            except (KeyError, Exception) as e:
                logger.warning("Skipping tile %d/%d: %s", col, row, e)
                continue

            h, w = tile_arr.shape[:2]
            feat = extract_features(tile_arr, req.features.filters, req.features.scales)
            # feat shape: (h*w, F)

            for tx, ty, label in pts:
                if 0 <= ty < h and 0 <= tx < w:
                    idx = ty * w + tx
                    X_list.append(feat[idx])
                    y_list.append(label)

        if not X_list:
            raise ValueError("No valid annotated pixels found — check tile coordinates")

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
        logger.info("Training RF on %d samples, %d features", *X.shape)

        clf = Classifier()
        clf.fit(X, y)

        clf_id = _store_clf(user_id, clf)
        return clf_id, clf.n_classes

    clf_id, n_classes = await _run_train(_train)
    return TrainResponse(classifier_id=clf_id, num_classes=n_classes)


@app.post("/train-multi", response_model=TrainResponse)
async def train_multi(
    req: TrainMultiRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Train a classifier from annotations across multiple images."""
    user = _auth(authorization)
    user_id = get_user_id(user)
    total_strokes = sum(len(a.strokes) for a in req.annotations)
    logger.info(
        "TrainMulti from %s: %d images, %d strokes total",
        user_id,
        len(req.annotations),
        total_strokes,
    )

    def _train_multi() -> tuple[str, int]:
        from collections import defaultdict

        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for ann in req.annotations:
            dzip = _get_dzip(ann.dzip_url, authorization)
            dzi_name, meta = dzip.find_dzi()  # resolve name from the DZIP
            level = ann.level if ann.level is not None else meta.max_level
            scale = 2.0 ** (level - meta.max_level)
            lw = max(1, round(meta.width * scale))
            lh = max(1, round(meta.height * scale))
            ts = meta.tile_size
            ol = meta.overlap

            tile_pts: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(
                list
            )
            for stroke in ann.strokes:
                for x, y in stroke.points:
                    lx = min(max(0, x), lw - 1)
                    ly = min(max(0, y), lh - 1)
                    col = lx // ts
                    row = ly // ts
                    tx = lx - col * ts + (ol if col > 0 else 0)
                    ty = ly - row * ts + (ol if row > 0 else 0)
                    tile_pts[(col, row)].append((tx, ty, stroke.label))

            for (col, row), pts in tile_pts.items():
                try:
                    tile_arr = dzip.get_tile(dzi_name, level, col, row, meta.format)
                except Exception as e:
                    logger.warning(
                        "Skipping tile %d/%d in %s: %s", col, row, ann.dzip_url, e
                    )
                    continue
                h, w = tile_arr.shape[:2]
                feat = extract_features(
                    tile_arr, req.features.filters, req.features.scales
                )
                for tx, ty, label in pts:
                    if 0 <= ty < h and 0 <= tx < w:
                        X_list.append(feat[ty * w + tx])
                        y_list.append(label)

        if not X_list:
            raise ValueError("No valid annotated pixels found across all images")

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
        logger.info("TrainMulti RF on %d samples, %d features", *X.shape)
        clf = Classifier()
        clf.fit(X, y)
        clf_id = _store_clf(user_id, clf)
        return clf_id, clf.n_classes

    clf_id, n_classes = await _run_train(_train_multi)
    return TrainResponse(classifier_id=clf_id, num_classes=n_classes)


@app.get("/predict/{classifier_id}/{level}/{tile_spec}")
async def predict_tile(
    classifier_id: str,
    level: int,
    tile_spec: str,
    dzip_url: str,
    dzi_name: str,
    filters: str,
    scales: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    # Accept token via query param (img.src can't send headers)
    effective_auth = authorization or (f"Bearer {token}" if token else None)
    user = _auth(effective_auth)
    clf = _get_clf(classifier_id, get_user_id(user))

    parts = tile_spec.split("_")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="tile_spec must be '{col}_{row}'")
    col, row = int(parts[0]), int(parts[1])
    filter_list = [f.strip() for f in filters.split(",") if f.strip()]
    scale_list = [float(s) for s in scales.split(",") if s.strip()]

    def _predict() -> bytes:
        dzip = _get_dzip(dzip_url, effective_auth)
        _, meta = dzip.find_dzi()

        tile_arr = dzip.get_tile(dzi_name, level, col, row, meta.format)
        h, w = tile_arr.shape[:2]

        feat = extract_features(tile_arr, filter_list, scale_list)  # (H*W, F)
        proba = clf.predict_proba(feat)  # (H*W, n_classes)
        proba = proba.reshape(h, w, -1)
        assert clf.classes_ is not None
        return encode_prediction_png(proba, clf.classes_.tolist())

    try:
        png_bytes = await _run_predict(_predict)
    except KeyError:
        raise HTTPException(status_code=404, detail="Tile not found in archive")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{classifier_id}-{level}-{tile_spec}"',
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/export")
async def start_export(
    req: ExportRequest,
    authorization: Optional[str] = Header(default=None),
):
    user = _auth(authorization)
    clf = _get_clf(req.classifier_id, get_user_id(user))

    job_id = str(uuid.uuid4())
    _exports[job_id] = {"status": "pending", "progress": 0.0}
    logger.info(
        "[export-job %s] queued: output_url=%s dzip=%s level=%d",
        job_id,
        req.output_url,
        req.dzip_url,
        req.level,
    )
    asyncio.create_task(_run_export(job_id, req, clf, authorization))
    return {"job_id": job_id}


@app.get("/export/{job_id}")
async def export_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _auth(authorization)
    job = _exports.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    return job


# ── Source listing ─────────────────────────────────────────────────────────────


@app.get("/list-sources")
async def list_sources(
    url: str,
    ext: str = ".dzip",
    authorization: Optional[str] = Header(default=None),
):
    """
    List DZIP files available at a data-proxy directory URL.
    Returns [{name, object_url, bytes}] for all *.dzip objects found.
    Works with:
      - data-proxy: https://data-proxy.ebrains.eu/api/v1/buckets/{bucket}/{prefix}/
      - plain HTTP directory listing (best-effort, HTML scrape)
    """
    import requests as _req

    _auth(authorization)

    token: Optional[str] = None
    if authorization:
        parts = authorization.split()
        token = parts[1] if len(parts) == 2 else authorization

    def _list() -> list[dict]:
        canonical = _dp_normalize(url.rstrip("/") + "/")
        if "data-proxy.ebrains.eu" in canonical:
            # Parse canonical URL: .../api/v1/buckets/{bucket}/{prefix}
            m = _re.match(
                r"https://data-proxy\.ebrains\.eu/api/v1/buckets/([^/]+)/?(.*)$",
                canonical,
            )
            if not m:
                raise ValueError(f"Cannot parse data-proxy URL: {canonical}")
            bucket = m.group(1)
            prefix = m.group(2).lstrip("/")
            dp_url = f"{_DP_BASE}/buckets/{bucket}"
            params: dict = {"prefix": prefix, "delimiter": "/", "limit": 9999}
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            logger.info("[list-sources] GET %s prefix=%s", dp_url, prefix)
            r = _req.get(dp_url, headers=headers, params=params, timeout=20)
            r.raise_for_status()
            resp = r.json()
            results = []
            for obj in resp.get("objects", []):
                if "name" in obj and obj["name"].endswith(ext):
                    results.append(
                        {
                            "name": obj["name"].split("/")[-1],
                            "object_url": f"{_DP_BASE}/buckets/{bucket}/{obj['name']}",
                            "bytes": obj.get("bytes"),
                        }
                    )
            logger.info("[list-sources] found %d %s files", len(results), ext)
            return results
        raise HTTPException(
            status_code=400, detail="Only data-proxy URLs are supported"
        )

    return await _run(_list)


# ── Batch export ───────────────────────────────────────────────────────────────

_batch_jobs: Dict[str, Dict[str, Any]] = {}


@app.post("/batch-export")
async def start_batch_export(
    req: BatchExportRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Queue a batch export job: list all DZIPs in p_source, predict each one
    with the given classifier, write results to output_dir/{name}.dzip.
    """
    user = _auth(authorization)
    clf = _get_clf(req.classifier_id, get_user_id(user))

    job_id = str(uuid.uuid4())
    _batch_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "total": 0,
        "done": 0,
        "failed": [],
    }
    logger.info(
        "[batch %s] queued: p_source=%s output_dir=%s",
        job_id,
        req.p_source,
        req.output_dir,
    )
    asyncio.create_task(_run_batch(job_id, req, clf, authorization))
    return {"job_id": job_id}


@app.get("/batch-export/{job_id}")
async def batch_export_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _auth(authorization)
    job = _batch_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return job


# ── Headless run ───────────────────────────────────────────────────────────────


@app.post("/headless-run")
async def headless_run(
    req: HeadlessRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Train a classifier on *annotations* from *t_source* images,
    then batch-export all DZIPs in *p_source* to *output_dir*.
    Returns immediately with {train_classifier_id, batch_job_id}.
    """
    user = _auth(authorization)
    user_id = get_user_id(user)
    logger.info(
        "[headless] building train request from %d annotation entries",
        len(req.annotations),
    )

    def _train_headless():
        from collections import defaultdict as _dd

        combined_X: list[np.ndarray] = []
        combined_y: list[int] = []

        for ann in req.annotations:
            dzip_url = ann["dzip_url"]
            strokes_raw = ann["strokes"]
            dzip = _get_dzip(dzip_url, authorization)
            _, meta = dzip.find_dzi()
            # derive dzi_name from URL
            dzi_name = (
                dzip_url.rstrip("/")
                .split("/")[-1]
                .replace(".dzip", "")
                .replace(".zip", "")
            )

            scale = 2 ** (req.level - meta.max_level)
            lw = max(1, round(meta.width * scale))
            lh = max(1, round(meta.height * scale))
            ts, ol = meta.tile_size, meta.overlap

            tile_pts: dict = _dd(list)
            for stroke in strokes_raw:
                for x, y in stroke["points"]:
                    lx, ly = min(max(0, x), lw - 1), min(max(0, y), lh - 1)
                    col, row = lx // ts, ly // ts
                    tx = lx - col * ts + (ol if col > 0 else 0)
                    ty = ly - row * ts + (ol if row > 0 else 0)
                    tile_pts[(col, row)].append((tx, ty, stroke["label"]))

            for (col, row), pts in tile_pts.items():
                try:
                    tile_arr = dzip.get_tile(dzi_name, req.level, col, row, meta.format)
                except Exception as e:
                    logger.warning("[headless] skip tile %d,%d: %s", col, row, e)
                    continue
                h, w = tile_arr.shape[:2]
                feat = extract_features(
                    tile_arr, req.features.filters, req.features.scales
                )
                for tx, ty, label in pts:
                    if 0 <= ty < h and 0 <= tx < w:
                        combined_X.append(feat[ty * w + tx])
                        combined_y.append(label)

        if not combined_X:
            raise RuntimeError("No training pixels extracted — check annotations")

        X = np.vstack(combined_X)
        y = np.array(combined_y, dtype=int)
        clf_obj = Classifier()
        clf_obj.fit(X, y)
        cid = _store_clf(user_id, clf_obj)
        logger.info(
            "[headless] trained classifier %s on %d pixels from %d images",
            cid,
            len(y),
            len(req.annotations),
        )
        return cid, int(np.unique(y).shape[0])

    try:
        cid, num_cls = await _run_train(_train_headless)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}") from e

    # 2 — Batch export
    clf = _get_clf(cid, user_id)
    batch_req = BatchExportRequest(
        classifier_id=cid,
        p_source=req.p_source,
        output_dir=req.output_dir,
        features=req.features,
        level=req.level,
    )
    job_id = str(uuid.uuid4())
    _batch_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "total": 0,
        "done": 0,
        "failed": [],
    }
    asyncio.create_task(_run_batch(job_id, batch_req, clf, authorization))
    logger.info("[headless] batch job queued: %s", job_id)
    return {"classifier_id": cid, "num_classes": num_cls, "batch_job_id": job_id}


@app.post("/export-zip")
async def export_zip(
    req: ExportRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Synchronously process every prediction tile and return the results as a
    downloadable DZIP file (ZIP containing a .dzi manifest + prediction PNGs
    in the standard DZI tile layout).  The output can be opened directly in
    the viewer or any other DZI-capable tool.
    """
    user = _auth(authorization)
    clf = _get_clf(req.classifier_id, get_user_id(user))
    logger.info(
        "[export-zip] start: output_url=%s dzip=%s",
        req.output_url or "(download)",
        req.dzip_url,
    )

    def _do() -> tuple[bytes, str]:
        tmpdir, dzip_path = _build_prediction_dzip(
            req.dzip_url,
            req.dzi_name,
            clf,
            req.features,
            authorization,
            level=req.level,
        )
        try:
            return dzip_path.read_bytes(), dzip_path.name
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.info("[export-zip] cleaned up tmpdir")

    zip_bytes, filename = await _run_train(_do)

    if req.output_url:
        token: Optional[str] = None
        if authorization:
            _parts = authorization.split()
            token = _parts[1] if len(_parts) == 2 else authorization

        target = req.output_url
        if not any(target.endswith(ext) for ext in (".dzip", ".zip")):
            target = target.rstrip("/") + "/" + filename

        def _upload():
            logger.info(
                "[export-zip] uploading %d bytes to: %s", len(zip_bytes), target
            )
            if "data-proxy.ebrains.eu" in target and token:
                return _dp_put(target, zip_bytes, token)
            import requests as _req

            s = _req.Session()
            if token:
                s.headers.update({"Authorization": f"Bearer {token}"})
            r = s.put(target, data=zip_bytes, timeout=120)
            r.raise_for_status()
            return r.status_code

        try:
            status_code = await _run(_upload)
            logger.info(
                "[export-zip] upload done → %s  status=%d  size=%d",
                target,
                status_code,
                len(zip_bytes),
            )
            return JSONResponse(
                {
                    "status": "uploaded",
                    "url": target,
                    "filename": filename,
                    "size": len(zip_bytes),
                }
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upload failed: {e}") from e

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


# ── Export background task ────────────────────────────────────────────────────


async def _run_export(
    job_id: str,
    req: ExportRequest,
    clf: Classifier,
    authorization: Optional[str] = None,
) -> None:
    try:
        _exports[job_id]["status"] = "running"
        logger.info(
            "[export-job %s] start: output_url=%s dzip=%s",
            job_id,
            req.output_url,
            req.dzip_url,
        )

        def _do():
            import requests as _req

            tmpdir, dzip_path = _build_prediction_dzip(
                req.dzip_url,
                req.dzi_name,
                clf,
                req.features,
                authorization,
                job_id=job_id,
                level=req.level,
            )
            try:
                zip_bytes = dzip_path.read_bytes()
                filename = dzip_path.name
                if not req.output_url:
                    logger.warning(
                        "[export-job %s] no output_url — DZIP discarded", job_id
                    )
                    return

                target = req.output_url
                if not any(target.endswith(ext) for ext in (".dzip", ".zip")):
                    target = target.rstrip("/") + "/" + filename

                token: Optional[str] = None
                if authorization:
                    _parts = authorization.split()
                    token = _parts[1] if len(_parts) == 2 else authorization

                logger.info(
                    "[export-job %s] uploading %d bytes → %s",
                    job_id,
                    len(zip_bytes),
                    target,
                )
                _exports[job_id]["progress"] = 0.9

                if "data-proxy.ebrains.eu" in target and token:
                    _dp_put(target, zip_bytes, token)
                else:
                    s = _req.Session()
                    if token:
                        s.headers.update({"Authorization": f"Bearer {token}"})
                    s.put(target, data=zip_bytes, timeout=300).raise_for_status()

                logger.info("[export-job %s] upload done → %s", job_id, target)
                _exports[job_id]["url"] = target
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
                logger.info("[export-job %s] cleaned up tmpdir", job_id)

        await _run_train(_do)
        _exports[job_id]["status"] = "done"
        _exports[job_id]["progress"] = 1.0
    except Exception as e:
        _exports[job_id]["status"] = "error"
        _exports[job_id]["error"] = str(e)
        logger.error("[export-job %s] failed: %s", job_id, e)


# ── Batch export background task ──────────────────────────────────────────────


async def _run_batch(
    job_id: str,
    req: "BatchExportRequest",
    clf: Classifier,
    authorization: Optional[str] = None,
) -> None:
    """
    1. List all .dzip files in req.p_source
    2. For each: _build_prediction_dzip → upload to req.output_dir/{name}
    3. Images are processed sequentially; tiles within each image are parallel.
    """
    import requests as _req

    try:
        _batch_jobs[job_id]["status"] = "running"

        token: Optional[str] = None
        if authorization:
            parts = authorization.split()
            token = parts[1] if len(parts) == 2 else authorization

        # ── Step 1: resolve source list ────────────────────────────────────
        def _list_sources() -> list[dict]:
            canonical = _dp_normalize(req.p_source.rstrip("/") + "/")
            m = _re.match(
                r"https://data-proxy\.ebrains\.eu/api/v1/buckets/([^/]+)/?(.*)$",
                canonical,
            )
            if not m:
                raise RuntimeError(f"Cannot parse p_source: {canonical}")
            bucket, prefix = m.group(1), m.group(2).lstrip("/")
            dp_url = f"{_DP_BASE}/buckets/{bucket}"
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = _req.get(
                dp_url,
                headers=headers,
                params={"prefix": prefix, "delimiter": "/", "limit": 9999},
                timeout=20,
            )
            r.raise_for_status()
            results = []
            for obj in r.json().get("objects", []):
                if "name" in obj and obj["name"].endswith(".dzip"):
                    results.append(
                        {
                            "name": obj["name"].split("/")[-1],
                            "object_url": f"{_DP_BASE}/buckets/{bucket}/{obj['name']}",
                            "bytes": obj.get("bytes"),
                        }
                    )
            return results

        sources = await _run(_list_sources)
        total = len(sources)
        _batch_jobs[job_id]["total"] = total
        logger.info("[batch %s] %d DZIPs to process", job_id, total)

        if total == 0:
            _batch_jobs[job_id]["status"] = "done"
            _batch_jobs[job_id]["progress"] = 1.0
            return

        # ── Step 2: process each image ──────────────────────────────────────
        output_base = req.output_dir.rstrip("/")
        done_count = 0

        for src in sources:
            src_url = src["object_url"]
            src_name = src["name"]  # e.g. "79556738_s306.jpg.dzip"
            out_name = src_name  # keep original name; predictions inside DZIP differ
            dest_url = f"{output_base}/{out_name}"

            logger.info("[batch %s] processing %s → %s", job_id, src_name, dest_url)
            _batch_jobs[job_id]["current"] = src_name

            try:

                def _process(url=src_url, name=src_name):
                    # Derive dzi_name from the DZIP filename
                    dzi_name = name.replace(".dzip", "").replace(".zip", "")
                    tmpdir, dzip_path = _build_prediction_dzip(
                        url,
                        dzi_name,
                        clf,
                        req.features,
                        authorization,
                        job_id=None,  # don't clobber batch progress
                        level=req.level,
                    )
                    try:
                        return dzip_path.read_bytes(), dzip_path.name
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)

                zip_bytes, _ = await _run_train(_process)

                def _upload(data=zip_bytes, target=dest_url):
                    logger.info(
                        "[batch %s] uploading %d bytes → %s", job_id, len(data), target
                    )
                    if "data-proxy.ebrains.eu" in target and token:
                        _dp_put(target, data, token)
                    else:
                        s = _req.Session()
                        if token:
                            s.headers.update({"Authorization": f"Bearer {token}"})
                        s.put(target, data=data, timeout=300).raise_for_status()
                    logger.info("[batch %s] uploaded %s OK", job_id, target)

                await _run(_upload)
                done_count += 1

            except Exception as e:
                logger.error("[batch %s] failed on %s: %s", job_id, src_name, e)
                _batch_jobs[job_id]["failed"].append(
                    {"name": src_name, "error": str(e)}
                )
                done_count += 1  # still advance progress

            _batch_jobs[job_id]["done"] = done_count
            _batch_jobs[job_id]["progress"] = done_count / total

        _batch_jobs[job_id]["status"] = "done"
        _batch_jobs[job_id]["progress"] = 1.0
        logger.info(
            "[batch %s] complete — %d/%d succeeded",
            job_id,
            done_count - len(_batch_jobs[job_id]["failed"]),
            total,
        )

    except Exception as e:
        _batch_jobs[job_id]["status"] = "error"
        _batch_jobs[job_id]["error"] = str(e)
        logger.error("[batch %s] fatal: %s", job_id, e)


# ── PNG encoding ──────────────────────────────────────────────────────────────


# _encode_prediction_png is now encode_prediction_png from backend.encoding


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # single worker; GPU state is process-local
    )
