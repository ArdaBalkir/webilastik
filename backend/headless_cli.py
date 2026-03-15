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
from .dzi_source import DzipSource
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


# ── PNG encoding (mirrors server.py) ─────────────────────────────────────────


def _encode_prediction_png(proba: np.ndarray) -> bytes:
    h, w, n = proba.shape
    n_ch = min(n, 4)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(n_ch):
        rgba[:, :, i] = (proba[:, :, i] * 255).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if n >= 3 else ("RGB" if n == 2 else "L")
    arr = rgba[:, :, :n_ch] if mode != "L" else rgba[:, :, 0]
    img = Image.fromarray(arr, mode=mode)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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

        src = DzipSource(dzip_url, bearer_token=token)
        _, meta = src.find_dzi()
        level = level_hint if level_hint is not None else meta.max_level

        scale = 2 ** (level - meta.max_level)
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


# ── Prediction / DZIP building ────────────────────────────────────────────────


def build_prediction_dzip(
    dzip_url: str,
    dzi_name: str,
    clf: Classifier,
    features: dict,
    token: Optional[str],
    workers: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Predict all tiles at full resolution, write to tmpdir, pack as DZIP.
    Returns (tmpdir, dzip_path).  Caller must shutil.rmtree(tmpdir).
    """
    import threading

    filters: list[str] = features["filters"]
    scales: list[float] = features["scales"]

    src = DzipSource(dzip_url, bearer_token=token)
    _, meta = src.find_dzi()
    level = meta.max_level
    lw, lh, ts, ol = meta.width, meta.height, meta.tile_size, meta.overlap
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

    errors: list[str] = []
    lock = threading.Lock()
    done_count: list[int] = [0]
    t_start = time.time()

    def process_tile(col: int, row: int) -> None:
        try:
            tile_arr = src.get_tile(dzi_name, level, col, row, meta.format)
            feat = extract_features(tile_arr, filters, scales)
            h, w = tile_arr.shape[:2]
            proba = clf.predict_proba(feat).reshape(h, w, -1)
            png = _encode_prediction_png(proba)
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
                    logger.info("  tiles %d/%d  %.1f/s  ETA %.0fs", n, total, rate, eta)

    tile_coords = [(c, r) for r in range(num_rows) for c in range(num_cols)]
    # Use min(workers, total, 64) — beyond ~64 threads I/O contention outweighs gains
    actual_workers = min(workers, total, 64)
    logger.info("  predicting %d tiles with %d workers", total, actual_workers)

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = [pool.submit(process_tile, c, r) for c, r in tile_coords]
        for f in as_completed(futures):
            f.result()

    if errors:
        logger.warning("  %d tile errors: %s", len(errors), "; ".join(errors[:5]))

    (tmpdir / f"{out_name}.dzi").write_text(dzi_xml, encoding="utf-8")
    dzip_path = tmpdir / f"{out_name}.dzip"
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
) -> int:
    """Returns exit code."""
    t0 = time.time()

    # ── 1. Train ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — TRAINING")
    logger.info("=" * 60)
    clf = train(annotations, features, level_hint, token, workers)

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

    # ── 3. Predict + upload each image ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — BATCH EXPORT  (%d workers per image)", workers)
    logger.info("=" * 60)
    output_base = output_dir.rstrip("/")
    failed: list[str] = []

    for idx, src in enumerate(sources, 1):
        src_url = src["object_url"]
        src_name = src["name"]
        dzi_name = src_name.replace(".dzip", "").replace(".zip", "")
        dest_url = f"{output_base}/{src_name}"

        logger.info("[%d/%d] %s", idx, total, src_name)
        tmpdir = None
        try:
            tmpdir, dzip_path = build_prediction_dzip(
                src_url, dzi_name, clf, features, token, workers
            )
            logger.info("  uploading → %s", dest_url)
            upload_dzip(dzip_path, dest_url, token)
            logger.info("  ✓ done")
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
    return 0 if not failed else 1


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m backend.headless_cli",
        description="Webilastik 2.0 — train + batch export without a running server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Annotations
    ann = p.add_mutually_exclusive_group(required=True)
    ann.add_argument(
        "--annotations",
        metavar="FILE",
        help="Path to annotations JSON file (or - to read from stdin)",
    )
    ann.add_argument(
        "--annotations-b64",
        metavar="B64",
        help="Base64-encoded annotations JSON (used by sbatch scripts)",
    )

    # Sources
    p.add_argument(
        "--t-source",
        metavar="URL",
        help="Optional: single DZIP URL for training (overrides dzip_url in JSON)",
    )
    p.add_argument(
        "--p-source",
        required=True,
        metavar="URL",
        help="Data-proxy directory URL containing source DZIPs to predict",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="URL",
        help="Data-proxy directory URL where prediction DZIPs will be written",
    )

    # Features
    feat = p.add_mutually_exclusive_group()
    feat.add_argument(
        "--features",
        metavar="JSON",
        default='{"filters":["gaussianSmoothing","laplacianOfGaussian","gaussianGradientMagnitude","hessianOfGaussianEigenvalues"],"scales":[0.7,1.6,3.5,5.0]}',
        help="Feature config JSON string",
    )
    feat.add_argument(
        "--features-file", metavar="FILE", help="Path to features JSON file"
    )

    # Training level
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="DZI level to train at (default: max level = full res)",
    )

    # Auth
    tok = p.add_mutually_exclusive_group()
    tok.add_argument(
        "--token",
        metavar="TOKEN",
        help="Bearer token (avoid on shared systems — prefer --token-file)",
    )
    tok.add_argument(
        "--token-file", metavar="FILE", help="Path to file containing bearer token"
    )
    tok.add_argument(
        "--token-env",
        metavar="ENV_VAR",
        help="Name of environment variable holding bearer token",
    )

    # Parallelism
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 8))),
        help="Number of parallel tile workers (default: $SLURM_CPUS_PER_TASK or cpu_count)",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    # ── Load annotations ──────────────────────────────────────────────────────
    if args.annotations_b64:
        raw = base64.b64decode(args.annotations_b64).decode("utf-8")
        annotations: list[dict] = json.loads(raw)
    elif args.annotations == "-":
        annotations = json.load(sys.stdin)
    else:
        with open(args.annotations, encoding="utf-8") as f:
            annotations = json.load(f)

    # If --t-source overrides all dzip_urls in the annotations
    if args.t_source:
        for ann in annotations:
            ann.setdefault("dzip_url", args.t_source)

    # ── Load features ─────────────────────────────────────────────────────────
    if args.features_file:
        with open(args.features_file, encoding="utf-8") as f:
            features: dict = json.load(f)
    else:
        features = json.loads(args.features)

    # ── Load token ────────────────────────────────────────────────────────────
    token: Optional[str] = None
    if args.token:
        token = args.token.strip()
    elif args.token_file:
        token = pathlib.Path(args.token_file).read_text().strip()
    elif args.token_env:
        token = os.environ.get(args.token_env, "").strip() or None

    logger.info("Workers:    %d", args.workers)
    logger.info("Features:   %s @ scales %s", features["filters"], features["scales"])
    logger.info("p_source:   %s", args.p_source)
    logger.info("output_dir: %s", args.output_dir)
    logger.info("Auth:       %s", "token present" if token else "NO TOKEN")

    rc = run(
        annotations=annotations,
        features=features,
        level_hint=args.level,
        p_source=args.p_source,
        output_dir=args.output_dir,
        token=token,
        workers=args.workers,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
