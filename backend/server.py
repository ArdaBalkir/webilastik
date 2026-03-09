"""
Webilastik 2.0 — Compute Server

Endpoints
---------
GET  /health
GET  /dzi-info?dzip_url=...
POST /train
GET  /predict/{classifier_id}/{level}/{tile_spec}?dzip_url=...&dzi_name=...&filters=...&scales=...
POST /export
GET  /export/{job_id}
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image
from pydantic import BaseModel

from .auth import verify_token, AuthError
from .classifier import Classifier
from .dzi_source import DzipSource
from .features import extract_features

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Webilastik 2.0 Compute Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Process pool for CPU-bound work ──────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


async def _run(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ── In-memory state ───────────────────────────────────────────────────────────
# classifier_id → Classifier
_classifiers: Dict[str, Classifier] = {}
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


# ── Auth dependency ───────────────────────────────────────────────────────────


def _auth(authorization: Optional[str]) -> Dict[str, Any]:
    """Verify bearer token; skip in dev if DISABLE_AUTH=1."""
    if os.environ.get("DISABLE_AUTH") == "1":
        return {"sub": "dev-user"}
    try:
        return verify_token(authorization)
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


class ExportRequest(BaseModel):
    classifier_id: str
    dzip_url: str
    dzi_name: str
    level: int
    features: FeatureSpec
    output_url: Optional[str] = (
        None  # if set, PUT the DZIP there; else return as download
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "classifiers": len(_classifiers)}


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
    logger.info("Train request from %s: %d strokes", user.get("sub"), len(req.strokes))

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

        clf_id = str(uuid.uuid4())
        _classifiers[clf_id] = clf
        return clf_id, clf.n_classes

    clf_id, n_classes = await _run(_train)
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
    _auth(effective_auth)

    clf = _classifiers.get(classifier_id)
    if clf is None:
        raise HTTPException(status_code=404, detail="Classifier not found")

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

        return _encode_prediction_png(proba)

    try:
        png_bytes = await _run(_predict)
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
    clf = _classifiers.get(req.classifier_id)
    if clf is None:
        raise HTTPException(status_code=404, detail="Classifier not found")

    job_id = str(uuid.uuid4())
    _exports[job_id] = {"status": "pending", "progress": 0.0}

    # Fire and forget background task
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
    _auth(authorization)
    clf = _classifiers.get(req.classifier_id)
    if clf is None:
        raise HTTPException(status_code=404, detail="Classifier not found")

    def _build_zip() -> bytes:
        import math
        import zipfile
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dzip = _get_dzip(req.dzip_url, authorization)
        _, meta = dzip.find_dzi()
        scale = 2 ** (req.level - meta.max_level)
        lw = max(1, round(meta.width * scale))
        lh = max(1, round(meta.height * scale))
        ts = meta.tile_size

        num_cols = math.ceil(lw / ts)
        num_rows = math.ceil(lh / ts)

        out_name = f"{req.dzi_name}_predictions"

        # DZI manifest for the prediction output
        dzi_xml = (
            f'<?xml version="1.0" encoding="utf-8"?>\n'
            f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
            f'  Format="png" Overlap="{meta.overlap}" TileSize="{ts}">\n'
            f'  <Size Width="{lw}" Height="{lh}"/>\n'
            f"</Image>\n"
        )

        # Process tiles in parallel, collect (path, png_bytes) results
        results: dict[str, bytes] = {}
        lock = threading.Lock()
        errors: list[str] = []

        def process_tile(col: int, row: int) -> tuple[str, bytes] | None:
            try:
                tile_arr = dzip.get_tile(req.dzi_name, req.level, col, row, meta.format)
                feat = extract_features(
                    tile_arr, req.features.filters, req.features.scales
                )
                h, w = tile_arr.shape[:2]
                proba = clf.predict_proba(feat).reshape(h, w, -1)
                png = _encode_prediction_png(proba)
                path = f"{out_name}_files/{req.level}/{col}_{row}.png"
                return path, png
            except Exception as e:
                with lock:
                    errors.append(f"{col},{row}: {e}")
                return None

        tile_coords = [(col, row) for row in range(num_rows) for col in range(num_cols)]
        max_workers = min(32, os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_tile, col, row): (col, row)
                for col, row in tile_coords
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    path, png = result
                    results[path] = png

        if errors:
            logger.warning(
                "Export had %d tile errors: %s", len(errors), "; ".join(errors[:5])
            )

        # Pack into a ZIP (DZIP)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(f"{out_name}.dzi", dzi_xml.encode("utf-8"))
            for path, png in results.items():
                zf.writestr(path, png)
        return buf.getvalue()

    zip_bytes = await _run(_build_zip)

    # Derive output filename from the source DZIP URL (e.g. fart.dzip → fart.dzip)
    from urllib.parse import urlparse

    src_path = urlparse(req.dzip_url).path
    src_filename = src_path.rstrip("/").split("/")[-1] or f"{req.dzi_name}.dzip"
    if not src_filename.lower().endswith((".dzip", ".zip")):
        src_filename += ".dzip"
    filename = src_filename

    # If an upload URL was given, PUT the DZIP there and return JSON status
    if req.output_url:
        import requests as _req

        token: Optional[str] = None
        if authorization:
            _parts = authorization.split()
            token = _parts[1] if len(_parts) == 2 else authorization

        # Determine final target URL: append filename if URL ends with /
        target = req.output_url
        if not any(target.endswith(ext) for ext in (".dzip", ".zip")):
            target = target.rstrip("/") + "/" + filename

        def _upload():
            s = _req.Session()
            if token:
                s.headers.update({"Authorization": f"Bearer {token}"})
            r = s.put(target, data=zip_bytes, timeout=120)
            r.raise_for_status()
            return r.status_code

        try:
            status_code = await _run(_upload)
            logger.info("Uploaded %s → %s (%d)", filename, target, status_code)
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

    # No upload URL — stream as browser download
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

        def _do_export():
            import math
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            import requests as _req

            dzip = _get_dzip(req.dzip_url, authorization)
            _, meta = dzip.find_dzi()
            scale = 2 ** (req.level - meta.max_level)
            lw = max(1, round(meta.width * scale))
            lh = max(1, round(meta.height * scale))
            ts = meta.tile_size

            num_cols = math.ceil(lw / ts)
            num_rows = math.ceil(lh / ts)
            total = num_cols * num_rows

            # Build auth header value once
            token: Optional[str] = None
            if authorization:
                _parts = authorization.split()
                token = _parts[1] if len(_parts) == 2 else authorization

            # Thread-local sessions so each worker thread has its own connection pool
            _local = threading.local()

            def _get_session() -> _req.Session:
                if not hasattr(_local, "session"):
                    s = _req.Session()
                    if token:
                        s.headers.update({"Authorization": f"Bearer {token}"})
                    _local.session = s
                return _local.session

            # Each worker: download tile → features → predict → encode → upload
            # Retry up to 3 times on transient network errors (RemoteDisconnected etc.)
            failed_tiles: list[str] = []

            def _put_with_retry(
                session, url: str, data: bytes, retries: int = 3
            ) -> None:
                import time as _time

                for attempt in range(retries):
                    try:
                        r = session.put(url, data=data, timeout=60)
                        r.raise_for_status()
                        return
                    except Exception as exc:
                        if attempt == retries - 1:
                            raise
                        _time.sleep(2**attempt)

            def process_tile(col: int, row: int) -> None:
                try:
                    tile_arr = dzip.get_tile(
                        req.dzi_name, req.level, col, row, meta.format
                    )
                    feat = extract_features(
                        tile_arr, req.features.filters, req.features.scales
                    )
                    h, w = tile_arr.shape[:2]
                    proba = clf.predict_proba(feat).reshape(h, w, -1)
                    png = _encode_prediction_png(proba)
                    path = f"{req.dzi_name}_predictions/{req.level}/{col}_{row}.png"
                    target = req.output_url.rstrip("/") + "/" + path
                    _put_with_retry(_get_session(), target, png)
                except Exception as e:
                    with lock:
                        failed_tiles.append(f"{col},{row}: {e}")
                    logger.warning("Export tile %d,%d failed: %s", col, row, e)

            tile_coords = [
                (col, row) for row in range(num_rows) for col in range(num_cols)
            ]

            # Use up to 32 workers on HPC nodes — I/O bound (download + upload) benefits
            # greatly from concurrency even on CPU-only machines
            max_workers = min(32, os.cpu_count() or 4)
            done_count = 0
            lock = threading.Lock()

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(process_tile, col, row): (col, row)
                    for col, row in tile_coords
                }
                for fut in as_completed(futures):
                    fut.result()  # process_tile no longer swallows
                    with lock:
                        done_count += 1
                        _exports[job_id]["progress"] = done_count / total

            # Upload the .dzi manifest so the result is a complete DZI dataset
            scale = 2 ** (req.level - meta.max_level)
            lw2 = max(1, round(meta.width * scale))
            lh2 = max(1, round(meta.height * scale))
            dzi_xml = (
                f'<?xml version="1.0" encoding="utf-8"?>\n'
                f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
                f'  Format="png" Overlap="{meta.overlap}" TileSize="{meta.tile_size}">\n'
                f'  <Size Width="{lw2}" Height="{lh2}"/>\n'
                f"</Image>\n"
            )
            dzi_path = f"{req.dzi_name}_predictions.dzi"
            dzi_target = req.output_url.rstrip("/") + "/" + dzi_path
            try:
                r = _get_session().put(dzi_target, data=dzi_xml.encode(), timeout=30)
                r.raise_for_status()
            except Exception as e:
                logger.warning("Failed to upload .dzi manifest: %s", e)

            if failed_tiles:
                raise RuntimeError(
                    f"{len(failed_tiles)}/{total} tiles failed: "
                    + "; ".join(failed_tiles[:5])
                )

        await _run(_do_export)
        _exports[job_id]["status"] = "done"
        _exports[job_id]["progress"] = 1.0
    except Exception as e:
        _exports[job_id]["status"] = "error"
        _exports[job_id]["error"] = str(e)
        logger.error("Export %s failed: %s", job_id, e)


# ── PNG encoding ──────────────────────────────────────────────────────────────


def _encode_prediction_png(proba: np.ndarray) -> bytes:
    """
    Encode an (H, W, n_classes) float32 probability array as PNG.
    Up to 4 classes are stored in R, G, B, A channels (uint8, value = prob*255).
    For 2-class problems, just R and G are used.
    """
    h, w, n = proba.shape
    n_ch = min(n, 4)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(n_ch):
        rgba[:, :, i] = (proba[:, :, i] * 255).clip(0, 255).astype(np.uint8)

    mode = "RGBA" if n >= 3 else ("RGB" if n == 2 else "L")
    if mode == "L":
        arr = rgba[:, :, 0]
    elif mode == "RGB":
        arr = rgba[:, :, :3]
    else:
        arr = rgba

    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False)
    return buf.getvalue()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # single worker; GPU state is process-local
    )
