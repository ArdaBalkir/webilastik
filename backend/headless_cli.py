"""
Webilastik 2.0 — Headless CLI
==============================
Full pipeline (train → batch predict → upload) without a running server.
Designed to be called from an sbatch job or a local terminal.

Usage — local
-------------
    python -m backend.headless_cli \\
        --annotations annotations.json \\
        --p-source  "https://data-proxy.ebrains.eu/api/v1/buckets/bucket/images/" \\
        --output-dir "https://data-proxy.ebrains.eu/api/v1/buckets/bucket/segs/" \\
        --token-file ~/.wi2_token \\
        --workers 16

Usage — from SLURM job (token via env var, workers from $SLURM_CPUS_PER_TASK)
-------------------------------------------------------------------------------
    WI2_TOKEN=$TOKEN srun --cpus-per-task=$SLURM_CPUS_PER_TASK \\
        python -m backend.headless_cli \\
            --annotations /scratch/annotations.json \\
            --p-source  "https://…/images/" \\
            --output-dir "https://…/segmentations/" \\
            --token-env WI2_TOKEN \\
            --workers $SLURM_CPUS_PER_TASK

Annotations JSON format
-----------------------
    [
      {
        "dzip_url": "https://data-proxy…/image1.dzip",
        "strokes": [
          {"label": 1, "points": [[10, 20], [11, 21]]},
          {"label": 2, "points": [[50, 60]]}
        ]
      }
    ]

You can also supply --t-source flag to specify explicitly which DZIP the
annotations belong to (when there is only one training image and you want
to skip putting dzip_url inside the JSON).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .classifier import Classifier
from .dzi_source import DzipSource, LocalDzipSource
from .encoding import encode_prediction_png
from .features import extract_features

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("wi2.headless")

# ── Data-proxy helpers (mirrors server.py) ────────────────────────────────────

_DP_BASE = "https://data-proxy.ebrains.eu/api/v1"
_RE = re.compile(r"")  # initialised below


def _make_session(workers: int, token: Optional[str]) -> Any:
    """
    Build a requests.Session whose connection pool is sized to the number of
    concurrent workers so urllib3 never discards connections under load.
    Rule of thumb: pool_connections = ceil(workers / 4) host buckets,
    pool_maxsize = workers + a few spare slots.
    """
    import requests
    from requests.adapters import HTTPAdapter

    pool_size = workers + 4  # spare slots so burst doesn't discard
    adapter = HTTPAdapter(
        pool_connections=max(4, workers // 4),
        pool_maxsize=pool_size,
        max_retries=3,
    )
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def _dp_normalize(url: str) -> str:
    if "data-proxy.ebrains.eu" not in url:
        return url
    if "/api/v1/buckets/" in url:
        return url
    m = re.match(r"(https?://data-proxy\.ebrains\.eu)(/.*)", url)
    if not m:
        return url
    path = m.group(2)
    path = re.sub(r"^/api(/v1)?", "", path)
    path = re.sub(r"^/buckets", "", path)
    return f"{_DP_BASE}/buckets{path}"


def _dp_put(url: str, data: bytes, token: str) -> None:
    import requests

    canonical = _dp_normalize(url)
    logger.info("  [upload] step-1 pre-sign: %s", canonical)
    r1 = requests.put(
        canonical,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r1.raise_for_status()
    presigned = r1.json()["url"]
    logger.info("  [upload] step-2 PUT %s bytes", f"{len(data):,}")
    r2 = requests.put(presigned, data=data, timeout=300)
    r2.raise_for_status()
    logger.info("  [upload] done (%d)", r2.status_code)


def _list_dzips(dir_url: str, token: Optional[str]) -> list[dict]:
    import requests

    canonical = _dp_normalize(dir_url.rstrip("/") + "/")
    m = re.match(
        r"https://data-proxy\.ebrains\.eu/api/v1/buckets/([^/]+)/?(.*)$",
        canonical,
    )
    if not m:
        raise ValueError(
            f"Cannot parse p_source URL: {canonical}\n"
            "Expected: https://data-proxy.ebrains.eu/api/v1/buckets/<bucket>/<prefix>/"
        )
    bucket, prefix = m.group(1), m.group(2).lstrip("/")
    endpoint = f"{_DP_BASE}/buckets/{bucket}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(
        endpoint,
        headers=headers,
        params={"prefix": prefix, "delimiter": "/"},
        timeout=30,
    )
    r.raise_for_status()
    results = []
    for obj in r.json().get("objects", []):
        if "name" in obj and obj["name"].endswith(".dzip"):
            results.append(
                {
                    "name": obj["name"].split("/")[-1],
                    "object_url": f"{_DP_BASE}/buckets/{bucket}/{obj['name']}",
                }
            )
    return results


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    annotations: list[dict],
    features: dict,
    level_hint: Optional[int],
    token: Optional[str],
    workers: int,
) -> Classifier:
    """
    Build a Classifier from a list of annotated DZIPs.
    annotations = [{"dzip_url": ..., "strokes": [{label, points}]}]
    """
    combined_X: list[np.ndarray] = []
    combined_y: list[int] = []
    filters: list[str] = features["filters"]
    scales: list[float] = features["scales"]

    for ann_idx, ann in enumerate(annotations, 1):
        dzip_url = ann["dzip_url"]
        strokes_raw = ann["strokes"]
        dzi_name = (
            dzip_url.rstrip("/").split("/")[-1].replace(".dzip", "").replace(".zip", "")
        )
        logger.info(
            "Training image %d/%d: %s (%d strokes)",
            ann_idx,
            len(annotations),
            dzi_name,
            len(strokes_raw),
        )

        src = DzipSource(
            dzip_url, session=_make_session(workers, token), bearer_token=token
        )
        _, meta = src.find_dzi()
        # Per-annotation level takes priority, then the global level_hint, then max
        ann_level = ann.get("level")
        level = (
            ann_level
            if ann_level is not None
            else (level_hint if level_hint is not None else meta.max_level)
        )

        scale = 2.0 ** (level - meta.max_level)
        lw = max(1, round(meta.width * scale))
        lh = max(1, round(meta.height * scale))
        ts, ol = meta.tile_size, meta.overlap

        tile_pts: dict = defaultdict(list)
        for stroke in strokes_raw:
            for x, y in stroke["points"]:
                lx, ly = min(max(0, x), lw - 1), min(max(0, y), lh - 1)
                col, row = lx // ts, ly // ts
                tx = lx - col * ts + (ol if col > 0 else 0)
                ty = ly - row * ts + (ol if row > 0 else 0)
                tile_pts[(col, row)].append((tx, ty, stroke["label"]))

        for (col, row), pts in tile_pts.items():
            try:
                tile_arr = src.get_tile(dzi_name, level, col, row, meta.format)
            except Exception as e:
                logger.warning("  skip tile %d,%d: %s", col, row, e)
                continue
            h, w2 = tile_arr.shape[:2]
            feat = extract_features(tile_arr, filters, scales)
            for tx, ty, label in pts:
                if 0 <= ty < h and 0 <= tx < w2:
                    combined_X.append(feat[ty * w2 + tx])
                    combined_y.append(label)

    if not combined_X:
        raise RuntimeError("No training pixels extracted — check annotations and URLs")

    X = np.vstack(combined_X)
    y = np.array(combined_y, dtype=np.int32)
    classes = np.unique(y)
    logger.info(
        "Training RF on %s pixels, %d classes %s, %d features",
        f"{len(y):,}",
        len(classes),
        classes.tolist(),
        X.shape[1],
    )

    clf = Classifier()
    clf.fit(X, y)
    logger.info("Training complete")
    return clf


# ── Prefetch ──────────────────────────────────────────────────────────────────


def prefetch_sources(
    sources: list[dict],
    dest_dir: pathlib.Path,
    token: Optional[str],
    workers: int,
) -> dict[str, pathlib.Path]:
    """
    Download all source DZIPs in parallel to dest_dir.
    Returns {src_name: local_path}.
    """
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pathlib.Path] = {}
    lock = __import__("threading").Lock()

    def _download(src: dict) -> None:
        name = src["name"]
        url = src["object_url"]
        dest = dest_dir / name
        if dest.exists():
            logger.info("  [prefetch] %s already cached, skipping", name)
            with lock:
                results[name] = dest
            return
        logger.info("  [prefetch] downloading %s ...", name)
        sess = _make_session(4, token)
        # data-proxy: get pre-signed S3 URL first
        if token and "data-proxy.ebrains.eu" in url:
            r = sess.get(url + "?redirect=false", timeout=30)
            r.raise_for_status()
            try:
                data = r.json()
                pre = (
                    data.get("url")
                    or data.get("URL")
                    or next(iter(data.values()), None)
                )
                if isinstance(pre, str) and pre.startswith("http"):
                    url = pre
                    sess = requests.Session()  # no auth for S3
            except Exception:
                pass
        with sess.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
        logger.info("  [prefetch] %s done (%s bytes)", name, f"{dest.stat().st_size:,}")
        with lock:
            results[name] = dest

    dl_workers = min(len(sources), workers, 16)  # cap at 16 parallel downloads
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        futs = [pool.submit(_download, src) for src in sources]
        for f in as_completed(futs):
            f.result()
    return results


# ── Multiprocessing worker (module-level so it is picklable) ────────────────────
# Each worker process receives a copy of the tile cache + classifier via the
# initializer, then processes tiles with its own GIL — true parallelism.

_mp: dict = {}  # per-process state set by initializer


def _mp_init_worker(
    clf: "Classifier",
    tile_cache: "dict[tuple[int,int], bytes]",
    filters: "list[str]",
    scales: "list[float]",
    fmt: str,
    tiles_dir_str: str,
) -> None:
    _mp["clf"] = clf
    _mp["cache"] = tile_cache
    _mp["filters"] = filters
    _mp["scales"] = scales
    _mp["fmt"] = fmt
    _mp["out"] = pathlib.Path(tiles_dir_str)


def _mp_predict_tile(col_row: "tuple[int, int]") -> "tuple[int, int, str | None]":
    """Run in a worker process. Returns (col, row, error_or_None)."""
    col, row = col_row
    raw = _mp["cache"].get((col, row))
    if raw is None:
        return col, row, "missing"
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        feat = extract_features(arr, _mp["filters"], _mp["scales"])
        h, w = arr.shape[:2]
        clf = _mp["clf"]
        proba = clf.predict_proba(feat).reshape(h, w, -1)
        png = encode_prediction_png(proba, clf.classes_.tolist())
        (_mp["out"] / f"{col}_{row}.png").write_bytes(png)
        return col, row, None
    except Exception as exc:
        return col, row, str(exc)


# ── Prediction / DZIP building ────────────────────────────────────────────────


def build_prediction_dzip(
    dzip_url: str,
    dzi_name: str,
    clf: Classifier,
    features: dict,
    token: Optional[str],
    workers: int,
    local_path: Optional[pathlib.Path] = None,
    level: Optional[int] = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Predict all tiles at the requested level (default: max_level = full res),
    write to tmpdir, pack as DZIP.
    If local_path is provided, reads tiles from it (no network) — use after prefetch.
    Returns (tmpdir, dzip_path).  Caller must shutil.rmtree(tmpdir).
    """
    import threading

    filters: list[str] = features["filters"]
    scales: list[float] = features["scales"]

    if local_path is not None:
        src: "DzipSource | LocalDzipSource" = LocalDzipSource(local_path)
    else:
        src = DzipSource(
            dzip_url, session=_make_session(workers, token), bearer_token=token
        )
    _, meta = src.find_dzi()
    # Clamp to max_level: requesting a level that doesn't exist in the source
    # zip silently yields an empty tile cache and a manifest-only output.
    level = min(level, meta.max_level) if level is not None else meta.max_level
    scale = 2.0 ** (level - meta.max_level)
    lw = max(1, round(meta.width * scale))
    lh = max(1, round(meta.height * scale))
    ts, ol = meta.tile_size, meta.overlap
    num_cols = math.ceil(lw / ts)
    num_rows = math.ceil(lh / ts)
    total = num_cols * num_rows

    out_name = f"{dzi_name}_predictions"
    dzi_xml = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        f'  Format="png" Overlap="{ol}" TileSize="{ts}">\n'
        f'  <Size Width="{lw}" Height="{lh}"/>\n'
        f"</Image>\n"
    )

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="wi2_"))
    tiles_dir = tmpdir / f"{out_name}_files" / str(level)
    tiles_dir.mkdir(parents=True)

    tile_coords = [(c, r) for r in range(num_rows) for c in range(num_cols)]

    # When local: preload all tiles from zip into memory in one sequential pass
    # so workers never do file I/O (128 threads hammering the same zip = slower
    # than 6 threads on a laptop due to zipfile + GPFS contention).
    tile_cache: dict[tuple[int, int], bytes] | None = None
    if local_path is not None:
        logger.info("  preloading %d tiles into memory ...", total)
        import zipfile as _zf

        tile_cache = {}
        import re as _re

        _tile_re = _re.compile(r"^.+_files/" + str(level) + r"/(\d+)_(\d+)\.\w+$")
        with _zf.ZipFile(local_path) as zf:
            # Scan the zip's actual entries at the target level rather than
            # constructing paths from dzi_name + format, which may not match
            # what was actually stored (different filename conventions, .jpg vs
            # .jpeg, etc.).
            for entry in zf.namelist():
                m = _tile_re.match(entry)
                if m:
                    tile_cache[(int(m.group(1)), int(m.group(2)))] = zf.read(entry)
        logger.info(
            "  preloaded %d tiles (%.1f MB) from zip at level %d",
            len(tile_cache),
            sum(len(v) for v in tile_cache.values()) / 1e6,
            level,
        )

    errors: list[str] = []
    t_start = time.time()

    if tile_cache is not None:
        # Local mode — ProcessPoolExecutor: each process has its own GIL so
        # numpy/PIL work truly parallelises. Tile cache is copied once per
        # worker process via the initializer (one-time pickling cost).
        actual_workers = min(workers, total)
        logger.info("  predicting %d tiles with %d processes", total, actual_workers)
        done = 0
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_mp_init_worker,
            initargs=(clf, tile_cache, filters, scales, meta.format, str(tiles_dir)),
        ) as pool:
            futs = {
                pool.submit(_mp_predict_tile, (c, r)): (c, r) for c, r in tile_coords
            }
            for f in as_completed(futs):
                col, row, err = f.result()
                done += 1
                if err and err != "missing":
                    errors.append(f"{col},{row}: {err}")
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "  tiles %d/%d  %.1f/s  ETA %.0fs", done, total, rate, eta
                    )
    else:
        # Remote mode — ThreadPoolExecutor: I/O-bound (data-proxy), GIL is fine.
        actual_workers = min(workers, total, 128)
        logger.info("  predicting %d tiles with %d workers", total, actual_workers)
        lock = threading.Lock()
        done_count: list[int] = [0]

        def process_tile(col: int, row: int) -> None:
            try:
                tile_arr = src.get_tile(dzi_name, level, col, row, meta.format)
                feat = extract_features(tile_arr, filters, scales)
                h, w = tile_arr.shape[:2]
                proba = clf.predict_proba(feat).reshape(h, w, -1)
                assert clf.classes_ is not None
                png = encode_prediction_png(proba, clf.classes_.tolist())
                (tiles_dir / f"{col}_{row}.png").write_bytes(png)
            except Exception as e:
                with lock:
                    errors.append(f"{col},{row}: {e}")
            finally:
                with lock:
                    done_count[0] += 1
                    n = done_count[0]
                    if n % 100 == 0 or n == total:
                        elapsed = time.time() - t_start
                        rate = n / elapsed if elapsed > 0 else 0
                        eta = (total - n) / rate if rate > 0 else 0
                        logger.info(
                            "  tiles %d/%d  %.1f/s  ETA %.0fs", n, total, rate, eta
                        )

        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = [pool.submit(process_tile, c, r) for c, r in tile_coords]
            for f in as_completed(futures):
                f.result()

    if errors:
        logger.warning("  %d tile errors: %s", len(errors), "; ".join(errors[:5]))

    dzip_path = tmpdir / f"{out_name}.dzip"
    (tmpdir / f"{out_name}.dzi").write_text(dzi_xml, encoding="utf-8")
    logger.info("  packing %s", dzip_path.name)
    with zipfile.ZipFile(dzip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tmpdir / f"{out_name}.dzi", f"{out_name}.dzi")
        for png_file in sorted(tiles_dir.iterdir()):
            zf.write(png_file, f"{out_name}_files/{level}/{png_file.name}")

    logger.info("  packed %s bytes", f"{dzip_path.stat().st_size:,}")
    return tmpdir, dzip_path


# ── Upload ────────────────────────────────────────────────────────────────────


def upload_dzip(dzip_path: pathlib.Path, dest_url: str, token: Optional[str]) -> None:
    import requests

    data = dzip_path.read_bytes()
    if token and "data-proxy.ebrains.eu" in dest_url:
        _dp_put(dest_url, data, token)
    else:
        s = requests.Session()
        if token:
            s.headers.update({"Authorization": f"Bearer {token}"})
        s.put(dest_url, data=data, timeout=300).raise_for_status()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(
    annotations: list[dict],
    features: dict,
    level_hint: Optional[int],
    p_source: str,
    output_dir: str,
    token: Optional[str],
    workers: int,
    prefetch_dir: Optional[pathlib.Path] = None,
) -> int:
    """Returns exit code."""
    t0 = time.time()

    # ── 1. Train ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — TRAINING")
    logger.info("=" * 60)
    clf = train(annotations, features, level_hint, token, workers)

    # Determine export level: prefer per-annotation level, fall back to level_hint
    export_level: Optional[int] = level_hint
    for ann in annotations:
        if ann.get("level") is not None:
            export_level = ann["level"]
            break
    if export_level is not None:
        logger.info("Export level: %d", export_level)
    else:
        logger.info("Export level: max_level of each image (full res)")

    # ── 2. List sources ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — LISTING SOURCES: %s", p_source)
    logger.info("=" * 60)
    sources = _list_dzips(p_source, token)
    total = len(sources)
    if total == 0:
        logger.error("No .dzip files found in %s", p_source)
        return 1
    logger.info("Found %d images to process", total)

    # ── 2.5 Prefetch — always download DZIPs locally before prediction ─────────
    # Reading tiles tile-by-tile from data-proxy during prediction is throttled
    # to ~6-7 tiles/s per token.  Downloading the whole DZIP first lets the
    # predict step run fully CPU-bound from local disk (much faster).
    # If the caller supplied an explicit prefetch_dir (e.g. /scratch), reuse it
    # across jobs and skip re-downloads.  Otherwise we use a throwaway tmpdir.
    _own_prefetch_dir = prefetch_dir is None
    if _own_prefetch_dir:
        prefetch_dir = pathlib.Path(tempfile.mkdtemp(prefix="wi2_dl_"))

    logger.info("=" * 60)
    logger.info("PHASE 2.5 — PREFETCH to %s", prefetch_dir)
    logger.info("=" * 60)
    local_cache = prefetch_sources(sources, prefetch_dir, token, workers)
    logger.info("Prefetch complete: %d files", len(local_cache))

    # ── 3. Predict + upload each image ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — BATCH EXPORT  (%d workers per image, local disk)", workers)
    logger.info("=" * 60)
    output_base = output_dir.rstrip("/")
    failed: list[str] = []

    for idx, src in enumerate(sources, 1):
        src_url = src["object_url"]
        src_name = src["name"]
        dzi_name = src_name.replace(".dzip", "").replace(".zip", "")
        dest_url = f"{output_base}/{src_name}"
        local_path = local_cache.get(src_name)

        logger.info(
            "[%d/%d] %s%s", idx, total, src_name, " (local)" if local_path else ""
        )
        tmpdir = None
        try:
            tmpdir, dzip_path = build_prediction_dzip(
                src_url,
                dzi_name,
                clf,
                features,
                token,
                workers,
                local_path=local_path,
                level=export_level,
            )
            logger.info("  uploading → %s", dest_url)
            upload_dzip(dzip_path, dest_url, token)
            logger.info("  ✓ done")
            # Free the downloaded source DZIP immediately to reclaim workdir space
            if local_path and local_path.exists():
                local_path.unlink(missing_ok=True)
                logger.info("  cleaned %s from workdir", local_path.name)
        except Exception as e:
            logger.error("  ✗ FAILED: %s", e)
            failed.append(src_name)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    ok = total - len(failed)
    logger.info("=" * 60)
    logger.info("DONE  %d/%d succeeded  elapsed %.0fs", ok, total, elapsed)
    if failed:
        logger.error("Failed images: %s", ", ".join(failed))
    logger.info("=" * 60)

    if _own_prefetch_dir and prefetch_dir is not None:
        shutil.rmtree(prefetch_dir, ignore_errors=True)

    return 0 if not failed else 1


# ── Shared argument helpers ────────────────────────────────────────────────────

_DEFAULT_FEATURES = (
    '{"filters":["gaussianSmoothing","laplacianOfGaussian",'
    '"gaussianGradientMagnitude","hessianOfGaussianEigenvalues"],'
    '"scales":[0.7,1.6,3.5,5.0]}'
)
_DEFAULT_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 8)))


def _add_token_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--token", metavar="TOKEN", help="Bearer token (avoid on shared systems)"
    )
    g.add_argument("--token-file", metavar="FILE", help="File containing bearer token")
    g.add_argument(
        "--token-env",
        metavar="VAR",
        help="Env var holding bearer token (safe for batch scripts)",
    )


def _add_features_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--features",
        metavar="JSON",
        default=_DEFAULT_FEATURES,
        help="Feature config as JSON string",
    )
    g.add_argument("--features-file", metavar="FILE", help="Feature config JSON file")


def _add_annotations_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--annotations", metavar="FILE", help="Annotations JSON file (- to read stdin)"
    )
    g.add_argument(
        "--annotations-b64",
        metavar="B64",
        help="Base64-encoded annotations JSON (for sbatch embedding)",
    )


def _resolve_token(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "token", None):
        return args.token.strip()
    if getattr(args, "token_file", None):
        return pathlib.Path(args.token_file).read_text().strip()
    if getattr(args, "token_env", None):
        return os.environ.get(args.token_env, "").strip() or None
    return None


def _resolve_features(args: argparse.Namespace) -> dict:
    if getattr(args, "features_file", None):
        with open(args.features_file, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(args.features)


def _resolve_annotations(args: argparse.Namespace) -> list[dict]:
    if getattr(args, "annotations_b64", None):
        return json.loads(base64.b64decode(args.annotations_b64).decode())
    src = args.annotations
    if src == "-":
        return json.load(sys.stdin)
    with open(src, encoding="utf-8") as f:
        return json.load(f)


# ── Subcommand: list ───────────────────────────────────────────────────────────
#   python -m backend.headless_cli list --url URL [--token-env VAR]
#
#   Discover all .dzip files in a data-proxy directory.
#   Good first test: verifies bucket access and URL parsing.


# when public: python -m backend.headless_cli list --url https://data-proxy.ebrains.eu/api/v1/buckets/rwb-arda2014/demo_project/CP_Pvalb/zipped_images/
# w tken --token
def _cmd_list(args: argparse.Namespace) -> int:
    try:
        token = _resolve_token(args)
    except Exception as e:
        logger.error("proceeding with no token: %s", e)
        return 1

    logger.info("Listing DZIPs in: %s", args.url)
    sources = _list_dzips(args.url, token)
    if not sources:
        logger.warning("No .dzip files found.")
        return 1
    print(f"Found {len(sources)} DZIP(s):")
    for s in sources:
        print(f"  {s['name']:40s}  {s['object_url']}")
    return 0


# ── Subcommand: train ──────────────────────────────────────────────────────────
#   python -m backend.headless_cli train --annotations FILE --output-model FILE
#
#   Train a classifier and save it to disk (pickle).
#   Lets you verify feature extraction and RF training independently.


def _cmd_train(args: argparse.Namespace) -> int:
    import pickle

    annotations = _resolve_annotations(args)
    features = _resolve_features(args)
    token = _resolve_token(args)

    if args.t_source:
        for ann in annotations:
            ann.setdefault("dzip_url", args.t_source)

    logger.info(
        "TRAIN: %d annotation entries, features: %s @ %s",
        len(annotations),
        features["filters"],
        features["scales"],
    )
    clf = train(annotations, features, args.level, token, args.workers)

    out = pathlib.Path(args.output_model)
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved to %s", out)
    return 0


# ── Subcommand: predict ────────────────────────────────────────────────────────
#   python -m backend.headless_cli predict \
#       --model model.pkl --p-source URL --output-dir URL
#
#   Load a pre-trained model and batch-export all images in p-source.
#   Lets you re-run or retry the export without re-training.


def _cmd_predict(args: argparse.Namespace) -> int:
    import pickle

    token = _resolve_token(args)
    features = _resolve_features(args)

    with open(args.model, "rb") as f:
        clf: Classifier = pickle.load(f)
    logger.info("Loaded model from %s (%d classes)", args.model, clf.n_classes)

    sources = _list_dzips(args.p_source, token)
    if not sources:
        logger.error("No .dzip files found in %s", args.p_source)
        return 1

    output_base = args.output_dir.rstrip("/")
    failed: list[str] = []
    for idx, src in enumerate(sources, 1):
        src_url = src["object_url"]
        src_name = src["name"]
        dzi_name = src_name.replace(".dzip", "").replace(".zip", "")
        dest_url = f"{output_base}/{src_name}"
        logger.info("[%d/%d] %s", idx, len(sources), src_name)
        tmpdir = None
        try:
            tmpdir, dzip_path = build_prediction_dzip(
                src_url,
                dzi_name,
                clf,
                features,
                token,
                args.workers,
                level=getattr(args, "level", None),
            )
            upload_dzip(dzip_path, dest_url, token)
            logger.info("  ✓ done")
        except Exception as e:
            logger.error("  ✗ FAILED: %s", e)
            failed.append(src_name)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info("DONE %d/%d succeeded", len(sources) - len(failed), len(sources))
    return 0 if not failed else 1


# ── Subcommand: run ────────────────────────────────────────────────────────────
#   python -m backend.headless_cli run --annotations FILE \
#       --p-source URL --output-dir URL
#
#   Full pipeline: train + predict + upload.  This is what sbatch calls.


def _cmd_run(args: argparse.Namespace) -> int:
    annotations = _resolve_annotations(args)
    features = _resolve_features(args)
    token = _resolve_token(args)

    if getattr(args, "t_source", None):
        for ann in annotations:
            ann.setdefault("dzip_url", args.t_source)

    logger.info("Workers:    %d", args.workers)
    logger.info("Features:   %s @ scales %s", features["filters"], features["scales"])
    logger.info("p_source:   %s", args.p_source)
    logger.info("output_dir: %s", args.output_dir)
    logger.info(
        "Auth:       %s", "token present" if token else "NO TOKEN (public buckets only)"
    )

    return run(
        annotations=annotations,
        features=features,
        level_hint=getattr(args, "level", None),
        p_source=args.p_source,
        output_dir=args.output_dir,
        token=token,
        workers=args.workers,
        prefetch_dir=(
            pathlib.Path(args.prefetch_dir)
            if getattr(args, "prefetch_dir", None)
            else None
        ),
    )


# ── Root parser ────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    root = argparse.ArgumentParser(
        prog="python -m backend.headless_cli",
        description=(
            "Webilastik 2.0 headless pipeline.\n\n"
            "Subcommands:\n"
            "  list     — discover .dzip files in a data-proxy directory\n"
            "  train    — train a classifier from annotations, save model\n"
            "  predict  — load a saved model, predict all images in a dir\n"
            "  run      — full pipeline: train + predict + upload (for sbatch)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command")

    # ── list ──────────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List .dzip files in a data-proxy directory")
    p_list.add_argument(
        "--url", required=True, metavar="URL", help="Data-proxy directory URL"
    )
    _add_token_args(p_list)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train classifier, save model to disk")
    _add_annotations_args(p_train)
    p_train.add_argument(
        "--t-source", metavar="URL", help="Override dzip_url for all annotations"
    )
    p_train.add_argument(
        "--output-model",
        required=True,
        metavar="FILE",
        help="Output path for the pickled model",
    )
    _add_features_args(p_train)
    p_train.add_argument(
        "--level",
        type=int,
        default=None,
        help="DZI training level (default: max = full res)",
    )
    _add_token_args(p_train)
    p_train.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predict from saved model, upload results")
    p_pred.add_argument(
        "--model",
        required=True,
        metavar="FILE",
        help="Path to pickled model (from 'train')",
    )
    p_pred.add_argument(
        "--p-source",
        required=True,
        metavar="URL",
        help="Data-proxy directory of source DZIPs",
    )
    p_pred.add_argument(
        "--output-dir",
        required=True,
        metavar="URL",
        help="Data-proxy directory for output DZIPs",
    )
    _add_features_args(p_pred)
    _add_token_args(p_pred)
    p_pred.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    p_pred.add_argument(
        "--level",
        type=int,
        default=None,
        help="DZI level to export at (default: max = full res)",
    )

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Full pipeline: train + predict + upload")
    _add_annotations_args(p_run)
    p_run.add_argument(
        "--t-source", metavar="URL", help="Override dzip_url for all annotations"
    )
    p_run.add_argument(
        "--p-source",
        required=True,
        metavar="URL",
        help="Data-proxy directory of source DZIPs to predict",
    )
    p_run.add_argument(
        "--output-dir",
        required=True,
        metavar="URL",
        help="Data-proxy directory for output DZIPs",
    )
    _add_features_args(p_run)
    p_run.add_argument("--level", type=int, default=None)
    _add_token_args(p_run)
    p_run.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    p_run.add_argument(
        "--prefetch-dir",
        metavar="DIR",
        default=None,
        help="Download all source DZIPs here first, then predict from local disk "
        "(eliminates network bottleneck during prediction — use to benchmark CPU throughput)",
    )

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = root.parse_args(argv)
    if args.command is None:
        root.print_help()
        sys.exit(1)

    dispatch = {
        "list": _cmd_list,
        "train": _cmd_train,
        "predict": _cmd_predict,
        "run": _cmd_run,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
