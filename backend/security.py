"""Shared outbound HTTP policy and filesystem-safe identity helpers."""
import hashlib
import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit

import requests

DATA_PROXY_HOST = "data-proxy.ebrains.eu"


def is_data_proxy(url):
    return urlsplit(url).hostname == DATA_PROXY_HOST


def validate_url(url, proxy_only=False, issued_by_proxy=False):
    """
    issued_by_proxy: the URL is a pre-signed link the data-proxy itself returned
    (its storage backend lives on another host). That host needs no allowlist
    entry because the proxy, not the user, chose it; all other checks apply.
    """
    parsed = urlsplit(url)
    host = parsed.hostname
    allowed = {DATA_PROXY_HOST}
    allowed.update(h.strip().lower() for h in os.getenv("WI2_ALLOWED_HTTP_HOSTS", "").split(",") if h.strip())
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Outbound URLs must use HTTPS without credentials or custom ports")
    if proxy_only and host != DATA_PROXY_HOST:
        raise ValueError("Outbound hostname is not approved")
    if host not in allowed and not issued_by_proxy:
        raise ValueError("Outbound hostname is not approved")
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Outbound hostname must resolve to public addresses")
    return url


class Session(requests.Session):
    _issued_by_proxy = False

    def __init__(self):
        super().__init__()
        # Ignore ambient proxy/netrc settings, which can alter routing/auth.
        self.trust_env = False

    def request(self, method, url, **kwargs):
        validate_url(url, issued_by_proxy=self._issued_by_proxy)
        # Never follow redirects to an unchecked destination.
        kwargs["allow_redirects"] = False
        headers = dict(self.headers)
        headers.update(kwargs.pop("headers", {}) or {})
        if not is_data_proxy(url):
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = None
        kwargs["headers"] = headers
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise ValueError("Outbound redirects are disabled; use the canonical URL")
        return response


class PresignedSession(Session):
    """For pre-signed storage URLs the data-proxy returned. Sends no auth (the
    host isn't the proxy), follows no redirects, still needs HTTPS to a public
    address."""

    _issued_by_proxy = True


def put_presigned(url, **kwargs):
    with PresignedSession() as session:
        return session.put(url, **kwargs)


def get(url, **kwargs):
    with Session() as session:
        return session.get(url, **kwargs)


def put(url, **kwargs):
    with Session() as session:
        return session.put(url, **kwargs)


def workspace_name(payload):
    # Username comes exclusively from verified claims. Subject disambiguates
    # renamed users and names that normalize to the same filesystem component.
    subject = str(payload["sub"])
    username = str(payload.get("preferred_username") or subject)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", username)[:64] or "user"
    return safe + "-" + hashlib.sha256(subject.encode()).hexdigest()[:12]


def safe_output_name(name):
    # Source entry names may contain directories; output files never do.
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", basename)[:120].strip(".")
    return (safe or "image") + "_predictions"
