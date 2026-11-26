"""Regression: data-proxy pre-signed URLs point at a storage host that is not on
WI2_ALLOWED_HTTP_HOSTS. Training, saving and uploads must still work."""
import asyncio
import io
import json
import os
import socket
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import requests
from PIL import Image

from backend import security, server

PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
PROXY_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/b/img.dzip"
STORAGE_URL = "https://object.cscs.ch/v1/AUTH_x/b/img.dzip?temp_url_sig=s"


def _make_dzip() -> bytes:
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    jpg = io.BytesIO()
    Image.fromarray(tile).save(jpg, format="JPEG")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("img.dzi", '<?xml version="1.0"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
                    'Format="jpg" Overlap="0" TileSize="64"><Size Width="64" Height="64"/></Image>')
        zf.writestr("img_files/6/0_0.jpg", jpg.getvalue())
    return buf.getvalue()


class FakeStorage:
    def __init__(self):
        self.blob = _make_dzip()
        self.calls = []

    def __call__(self, session, method, url, **kw):
        self.calls.append((type(session).__name__, method, url, dict(kw.get("headers") or {})))
        r = requests.Response()
        r.url = url
        if url.startswith(PROXY_URL) and "redirect=false" in url:
            r.status_code = 200
            r._content = json.dumps({"url": STORAGE_URL}).encode()
        elif url == STORAGE_URL and method == "GET":
            rng = kw["headers"]["Range"].removeprefix("bytes=")
            a, b = rng.split("-")
            data = self.blob[-int(b):] if a == "" else self.blob[int(a): int(b) + 1]
            r.status_code = 206
            r._content = data
        elif url == STORAGE_URL and method == "PUT":
            r.status_code = 201
            r._content = b""
        elif url.startswith(PROXY_URL) and method == "PUT":
            r.status_code = 200
            r._content = json.dumps({"url": STORAGE_URL}).encode()
        else:
            r.status_code = 404
            r._content = b""
        return r


class PresignedStorage(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"WI2_ALLOWED_HTTP_HOSTS": ""})
        self.dns = patch("socket.getaddrinfo", return_value=PUBLIC_DNS)
        self.env.start(); self.dns.start()
        self.fake = FakeStorage()
        self.net = patch.object(requests.Session, "request", autospec=True, side_effect=self.fake)
        self.net.start()
        server._dzip_cache.clear()

    def tearDown(self):
        self.net.stop(); self.dns.stop(); self.env.stop()
        server._dzip_cache.clear()

    def test_train_multi_reads_tiles_from_presigned_storage(self):
        req = server.TrainMultiRequest(
            annotations=[{"dzip_url": PROXY_URL, "strokes": [
                {"label": 1, "points": [[1, 1], [2, 2], [3, 3]]},
                {"label": 2, "points": [[50, 50], [60, 60]]}]}],
            features={"filters": ["gaussianSmoothing"], "scales": [1.0]})
        with patch.object(server, "_auth", return_value={"sub": "u"}):
            resp = asyncio.run(server.train_multi(req, "Bearer tok"))
        server._classifiers.clear() if hasattr(server, "_classifiers") else None
        self.assertEqual(resp.num_classes, 2)
        storage = [c for c in self.fake.calls if c[2] == STORAGE_URL]
        self.assertTrue(storage)
        for _, _, _, headers in storage:  # bearer token never leaves the proxy
            self.assertIsNone(headers.get("Authorization"))

    def test_upload_puts_to_presigned_storage(self):
        self.assertEqual(server._dp_put(PROXY_URL, b"data", "tok"), 201)

    def test_user_supplied_storage_host_still_refused(self):
        with self.assertRaises(ValueError):
            security.validate_url(STORAGE_URL)
        with self.assertRaises(ValueError):
            security.validate_url(STORAGE_URL, proxy_only=True, issued_by_proxy=True)

    def test_presigned_session_keeps_transport_checks(self):
        for bad in ("http://object.cscs.ch/x", "https://object.cscs.ch:8443/x", "https://u:p@object.cscs.ch/x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                security.PresignedSession().get(bad)
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))]), \
                self.assertRaises(ValueError):
            security.PresignedSession().get("https://object.cscs.ch/x")


if __name__ == "__main__":
    unittest.main()
