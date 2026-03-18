"""
DzipSource — reads DZI tiles from a remote DZIP file via HTTP range requests.
Mirrors the browser-side dzip_helper.ts logic in Python.
"""

from __future__ import annotations

import io
import math
import struct
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image


@dataclass
class _ZipEntry:
    name: str
    method: int
    comp_size: int
    uncomp_size: int
    offset: int  # local file header offset in the ZIP


@dataclass
class DziMeta:
    width: int
    height: int
    tile_size: int
    overlap: int
    format: str  # "jpeg" or "png"
    max_level: int  # ceil(log2(max(width, height)))


class DzipSource:
    """
    Opens a remote DZIP (DZI-in-ZIP) via HTTP range requests and exposes
    individual tile arrays without downloading the whole archive.

    The central directory is read once on construction; individual tiles are
    fetched on demand.  Thread-safe for concurrent tile reads.
    """

    def __init__(
        self,
        url: str,
        session: Optional[requests.Session] = None,
        bearer_token: Optional[str] = None,
    ):
        self._bearer_token = bearer_token
        self._proxy_url = url  # original data-proxy URL (or plain URL)
        self._is_data_proxy = bool(bearer_token and "data-proxy.ebrains.eu" in url)
        self._auth_session = session or requests.Session()
        if bearer_token:
            self._auth_session.headers.update(
                {"Authorization": f"Bearer {bearer_token}"}
            )
        self.url = url  # kept for reference; actual requests use _get_url()
        self._entries: Dict[str, _ZipEntry] = {}
        self._dzi_meta: Optional[DziMeta] = None
        self._dzi_name: Optional[str] = None
        self._load_central_directory()

    def _get_url(self) -> Tuple[str, "requests.Session"]:
        """
        Return (url, session) to use for a single range request.
        For EBRAINS data-proxy: call ?redirect=false each time to get a fresh
        pre-signed S3 URL (they expire in ~10 seconds).
        """
        if not self._is_data_proxy:
            return self._proxy_url, self._auth_session

        from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

        p = urlparse(self._proxy_url)
        qs = parse_qs(p.query, keep_blank_values=True)
        qs["redirect"] = ["false"]
        proxy_url = urlunparse(
            p._replace(query=urlencode({k: v[0] for k, v in qs.items()}))
        )
        r = self._auth_session.get(proxy_url, timeout=30)
        r.raise_for_status()
        try:
            data = r.json()
            pre_signed = (
                data.get("url") or data.get("URL") or next(iter(data.values()), None)
            )
            if isinstance(pre_signed, str) and pre_signed.startswith("http"):
                return pre_signed, requests.Session()  # S3 — no auth needed
        except Exception:
            pass
        return self._proxy_url, self._auth_session

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _range(self, start: int, end: int) -> bytes:
        """Inclusive range GET — gets a fresh pre-signed URL each call for data-proxy."""
        url, session = self._get_url()
        r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        r.raise_for_status()
        return r.content

    def _tail(self, n: int) -> bytes:
        """Last n bytes."""
        url, session = self._get_url()
        r = session.get(url, headers={"Range": f"bytes=-{n}"}, timeout=30)
        r.raise_for_status()
        return r.content

    # ── ZIP central directory parsing ─────────────────────────────────────────

    def _load_central_directory(self) -> None:
        footer = self._tail(22)
        sig = struct.unpack_from("<I", footer, 0)[0]
        if sig != 0x06054B50:
            raise ValueError(f"Not a ZIP file or has a comment: {self.url!r}")

        dir_size: int = struct.unpack_from("<I", footer, 12)[0]
        dir_offset: int = struct.unpack_from("<I", footer, 16)[0]

        if dir_offset == 0xFFFF_FFFF:
            # ZIP64 — find the EOCD64 locator 20 bytes before EOCD
            locator = self._tail(42)
            if struct.unpack_from("<I", locator, 0)[0] != 0x07064B50:
                raise ValueError("ZIP64 EOCD locator signature not found")
            zip64_offset: int = struct.unpack_from("<Q", locator, 8)[0]
            eocd64 = self._range(zip64_offset, zip64_offset + 55)
            if struct.unpack_from("<I", eocd64, 0)[0] != 0x06064B50:
                raise ValueError("ZIP64 EOCD signature not found")
            dir_offset = struct.unpack_from("<Q", eocd64, 48)[0]
            dir_size = struct.unpack_from("<Q", eocd64, 40)[0]

        central = self._range(dir_offset, dir_offset + dir_size - 1)
        pos = 0
        while pos + 46 <= len(central):
            sig = struct.unpack_from("<I", central, pos)[0]
            if sig != 0x02014B50:
                break

            method: int = struct.unpack_from("<H", central, pos + 10)[0]
            comp_size: int = struct.unpack_from("<I", central, pos + 20)[0]
            uncomp_size: int = struct.unpack_from("<I", central, pos + 24)[0]
            name_len: int = struct.unpack_from("<H", central, pos + 28)[0]
            extra_len: int = struct.unpack_from("<H", central, pos + 30)[0]
            comment_len: int = struct.unpack_from("<H", central, pos + 32)[0]
            offset: int = struct.unpack_from("<I", central, pos + 42)[0]
            pos += 46

            name = central[pos : pos + name_len].decode("utf-8", errors="replace")
            extra = central[pos + name_len : pos + name_len + extra_len]

            # ZIP64 extended info in Extra field
            if offset == 0xFFFF_FFFF:
                ep = 0
                while ep + 4 <= len(extra):
                    hid = struct.unpack_from("<H", extra, ep)[0]
                    dsz = struct.unpack_from("<H", extra, ep + 2)[0]
                    if hid == 0x0001 and dsz >= 8:
                        offset = struct.unpack_from("<Q", extra, ep + 4)[0]
                        break
                    ep += 4 + dsz

            self._entries[name] = _ZipEntry(
                name=name,
                method=method,
                comp_size=comp_size,
                uncomp_size=uncomp_size,
                offset=offset,
            )
            pos += name_len + extra_len + comment_len

    # ── Entry reading ─────────────────────────────────────────────────────────

    def get_bytes(self, name: str) -> bytes:
        """Decompress and return the raw bytes for the named entry."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Entry {name!r} not in DZIP {self.url!r}")

        # Read local file header to find exact data offset
        local = self._range(entry.offset, entry.offset + 29)
        name_len = struct.unpack_from("<H", local, 26)[0]
        extra_len = struct.unpack_from("<H", local, 28)[0]
        data_offset = entry.offset + 30 + name_len + extra_len

        raw = self._range(data_offset, data_offset + entry.comp_size - 1)

        if entry.method == 0:
            return raw
        if entry.method == 8:
            return zlib.decompress(raw, -15)  # raw deflate
        raise ValueError(f"Unsupported ZIP method {entry.method} for {name!r}")

    # ── DZI helpers ───────────────────────────────────────────────────────────

    def find_dzi(self) -> Tuple[str, DziMeta]:
        """Parse the .dzi file inside the archive and return (name, meta)."""
        if self._dzi_meta is not None:
            return self._dzi_name, self._dzi_meta  # type: ignore[return-value]

        dzi_key = next(
            (k for k in self._entries if k.endswith(".dzi") and "/" not in k),
            None,
        ) or next((k for k in self._entries if k.endswith(".dzi")), None)

        if dzi_key is None:
            raise ValueError(f"No .dzi file found in archive {self.url!r}")

        xml_str = self.get_bytes(dzi_key).decode("utf-8")
        meta = _parse_dzi_xml(xml_str)
        name = dzi_key.removesuffix(".dzi")
        self._dzi_name = name
        self._dzi_meta = meta
        return name, meta

    def get_tile(
        self, dzi_name: str, level: int, col: int, row: int, fmt: str
    ) -> np.ndarray:
        """
        Return an (H, W, 3) uint8 RGB numpy array for the given DZI tile.
        Raises KeyError if the tile doesn't exist in the archive.
        """
        path = f"{dzi_name}_files/{level}/{col}_{row}.{fmt}"
        data = self.get_bytes(path)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def tile_path(self, dzi_name: str, level: int, col: int, row: int, fmt: str) -> str:
        return f"{dzi_name}_files/{level}/{col}_{row}.{fmt}"

    @property
    def entry_names(self):
        return list(self._entries.keys())


class LocalDzipSource:
    """
    Reads a DZIP that has already been downloaded to a local file.
    Same interface as DzipSource (find_dzi / get_tile / get_bytes) but
    uses Python's zipfile module — no HTTP, no auth, pure disk I/O.
    Thread-safe: ZipFile opened per-call to avoid locking.
    """

    def __init__(self, path: "pathlib.Path"):
        import pathlib

        self.path = pathlib.Path(path)
        self._dzi_meta: Optional[DziMeta] = None
        self._dzi_name: Optional[str] = None

    def get_bytes(self, name: str) -> bytes:
        import zipfile as _zf

        with _zf.ZipFile(self.path) as zf:
            return zf.read(name)

    def find_dzi(self) -> Tuple[str, DziMeta]:
        if self._dzi_meta is not None:
            return self._dzi_name, self._dzi_meta  # type: ignore[return-value]
        import zipfile as _zf

        with _zf.ZipFile(self.path) as zf:
            names = zf.namelist()
        dzi_key = next(
            (k for k in names if k.endswith(".dzi") and "/" not in k),
            None,
        ) or next((k for k in names if k.endswith(".dzi")), None)
        if dzi_key is None:
            raise ValueError(f"No .dzi file found in {self.path}")
        xml_str = self.get_bytes(dzi_key).decode("utf-8")
        meta = _parse_dzi_xml(xml_str)
        name = dzi_key.removesuffix(".dzi")
        self._dzi_name = name
        self._dzi_meta = meta
        return name, meta

    def get_tile(
        self, dzi_name: str, level: int, col: int, row: int, fmt: str
    ) -> np.ndarray:
        path = f"{dzi_name}_files/{level}/{col}_{row}.{fmt}"
        data = self.get_bytes(path)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


# ── DZI XML parsing ───────────────────────────────────────────────────────────


def _parse_dzi_xml(xml_str: str) -> DziMeta:
    root = ET.fromstring(xml_str)
    # Strip namespace if present
    tag = root.tag
    ns_prefix = tag.split("}")[0] + "}" if "}" in tag else ""

    size_el = root.find(f"{ns_prefix}Size")
    if size_el is None:
        raise ValueError("DZI XML missing <Size> element")

    width = int(size_el.attrib["Width"])
    height = int(size_el.attrib["Height"])
    tile_size = int(root.attrib["TileSize"])
    overlap = int(root.attrib.get("Overlap", "0"))
    fmt = root.attrib.get("Format", "jpeg").lower()
    max_level = math.ceil(math.log2(max(width, height)))

    return DziMeta(
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        format=fmt,
        max_level=max_level,
    )
