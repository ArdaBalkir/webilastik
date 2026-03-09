#!/usr/bin/env python3
"""
Webilastik 2.0 — standalone HPC export script
=================================================
Trains a random-forest classifier from a saved project JSON file and exports
all prediction tiles to an EBRAINS data-proxy bucket (or any S3-compatible
PUT target).

Usage
-----
    python -m backend.export_cli \\
        --project    project.json \\
        --dzip-url   https://data-proxy.ebrains.eu/api/v1/buckets/my-bucket/data.dzip \\
        --output-url https://data-proxy.ebrains.eu/api/v1/buckets/my-bucket/results \\
        --token      $EBRAINS_TOKEN \\
        --workers    32

SLURM example
-------------
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 \\
        python -m backend.export_cli \\
        --project project.json \\
        --dzip-url ... --output-url ... --token $EBRAINS_TOKEN --workers 128
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import requests


def _load_project(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_dzip(url: str, bearer_token: Optional[str] = None):
    # Import from sibling package
    from backend.dzi_source import DzipSource

    return DzipSource(url, bearer_token=bearer_token)


def _extract_features(tile_arr, filters, scales):
    from backend.features import extract_features

    return extract_features(tile_arr, filters, scales)


def _train(dzip, project: dict, level: int, bearer_token: Optional[str]):
    from backend.classifier import Classifier

    _, meta = dzip.find_dzi()
    dzi_name = project["dziName"]
    strokes = project["strokes"]
    filters = project["featureConfig"]["filters"]
    scales = project["featureConfig"]["scales"]

    samples, labels = [], []
    for stroke in strokes:
        slevel = stroke["level"]
        for x, y in stroke["points"]:
            # Convert stroke coords to level coords
            factor = 2 ** (level - slevel)
            lx = round(x * factor)
            ly = round(y * factor)

            tile_col = lx // meta.tile_size
            tile_row = ly // meta.tile_size
            try:
                tile = dzip.get_tile(dzi_name, level, tile_col, tile_row, meta.format)
            except Exception:
                continue

            px = lx - tile_col * meta.tile_size
            py = ly - tile_row * meta.tile_size
            if px >= tile.shape[1] or py >= tile.shape[0]:
                continue

            feat = _extract_features(tile, filters, scales)
            feat_h, feat_w = tile.shape[:2]
            feat_map = feat.reshape(feat_h, feat_w, -1)
            samples.append(feat_map[py, px])
            labels.append(stroke["labelId"])

    if not samples:
        raise ValueError("No training samples could be extracted from strokes")

    clf = Classifier()
    clf.fit(np.array(samples), np.array(labels))
    print(f"  Trained on {len(samples)} samples, {len(set(labels))} classes")
    return clf, meta, dzi_name


def _encode_png(proba: np.ndarray) -> bytes:
    from backend.server import _encode_prediction_png

    return _encode_prediction_png(proba)


def main():
    parser = argparse.ArgumentParser(description="Webilastik 2.0 HPC export")
    parser.add_argument("--project", required=True, help="Path to project.json")
    parser.add_argument("--dzip-url", required=True, help="Source DZIP URL")
    parser.add_argument(
        "--output-url", required=True, help="Destination URL (directory on data-proxy)"
    )
    parser.add_argument("--token", default=None, help="EBRAINS bearer token")
    parser.add_argument(
        "--level-offset",
        type=int,
        default=0,
        help="0=full res, 1=half res, 2=quarter res",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, __import__("os").cpu_count() or 4),
        help="Number of parallel tile workers",
    )
    args = parser.parse_args()

    token: Optional[str] = args.token
    # Strip "Bearer " prefix if user pasted the full header value
    if token and token.lower().startswith("bearer "):
        token = token[7:]

    print(f"Loading project: {args.project}")
    project = _load_project(args.project)

    print(f"Connecting to DZIP: {args.dzip_url}")
    dzip = _get_dzip(args.dzip_url, bearer_token=token)

    _, meta = dzip.find_dzi()
    level = max(0, meta.max_level - args.level_offset)
    print(
        f"Working at DZI level {level} of {meta.max_level} "
        f"({int(meta.width * 2**(level - meta.max_level))} × "
        f"{int(meta.height * 2**(level - meta.max_level))} px)"
    )

    print("Training classifier from strokes…")
    clf, meta, dzi_name = _train(dzip, project, level, token)

    scale = 2 ** (level - meta.max_level)
    lw = max(1, round(meta.width * scale))
    lh = max(1, round(meta.height * scale))
    ts = meta.tile_size
    num_cols = math.ceil(lw / ts)
    num_rows = math.ceil(lh / ts)
    total = num_cols * num_rows
    filters = project["featureConfig"]["filters"]
    scales_list = project["featureConfig"]["scales"]

    output_dir = args.output_url.rstrip("/")
    print(f"Exporting {total} tiles → {output_dir}/  ({args.workers} workers)")

    # Upload .dzi manifest first
    dzi_xml = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        f'  Format="png" Overlap="{meta.overlap}" TileSize="{ts}">\n'
        f'  <Size Width="{lw}" Height="{lh}"/>\n'
        f"</Image>\n"
    )
    _local = threading.local()

    def _session() -> requests.Session:
        if not hasattr(_local, "s"):
            s = requests.Session()
            if token:
                s.headers.update({"Authorization": f"Bearer {token}"})
            _local.s = s
        return _local.s

    manifest_url = f"{output_dir}/{dzi_name}_predictions.dzi"
    _session().put(manifest_url, data=dzi_xml.encode(), timeout=30).raise_for_status()
    print(f"  Uploaded manifest → {manifest_url}")

    done = 0
    failed = 0
    lock = threading.Lock()
    t0 = time.time()

    def _put_with_retry(url: str, data: bytes, retries: int = 3) -> None:
        for attempt in range(retries):
            try:
                r = _session().put(url, data=data, timeout=60)
                r.raise_for_status()
                return
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)

    def process_tile(col: int, row: int) -> None:
        nonlocal done, failed
        try:
            tile_arr = dzip.get_tile(dzi_name, level, col, row, meta.format)
            feat = _extract_features(tile_arr, filters, scales_list)
            h, w = tile_arr.shape[:2]
            proba = clf.predict_proba(feat).reshape(h, w, -1)
            png = _encode_png(proba)
            path = f"{dzi_name}_predictions/{level}/{col}_{row}.png"
            _put_with_retry(f"{output_dir}/{path}", png)
        except Exception as e:
            with lock:
                failed += 1
            print(f"  WARN tile {col},{row}: {e}", file=sys.stderr)
        finally:
            with lock:
                done += 1

    tile_coords = [(col, row) for row in range(num_rows) for col in range(num_cols)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_tile, c, r): (c, r) for c, r in tile_coords}
        last_print = 0.0
        for _fut in as_completed(futures):
            _fut.result()
            now = time.time()
            if now - last_print >= 5:
                pct = done / total * 100
                elapsed = now - t0
                eta = (elapsed / done * (total - done)) if done else 0
                print(
                    f"  {done}/{total} tiles  {pct:.1f}%  "
                    f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                    f"failed {failed}"
                )
                last_print = now

    elapsed = time.time() - t0
    print(
        f"\nDone: {done - failed}/{total} tiles in {elapsed:.1f}s " f"({failed} failed)"
    )
    if failed:
        print(
            f"WARNING: {failed} tiles failed — check stderr for details",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
