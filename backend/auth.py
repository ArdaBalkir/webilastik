"""
EBRAINS OIDC bearer-token verification.

Fetches the JWKS from EBRAINS IAM, caches it for `CACHE_TTL` seconds,
and validates incoming JWT access tokens.  Returns the decoded payload on
success; raises AuthError on any failure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import jwt
import requests

logger = logging.getLogger(__name__)

EBRAINS_IAM_JWKS_URL = (
    "https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect/certs"
)
EXPECTED_ISSUER = "https://iam.ebrains.eu/auth/realms/hbp"
CACHE_TTL = 3600  # seconds


class AuthError(Exception):
    """Raised when bearer token validation fails."""


class _JwksCache:
    def __init__(self) -> None:
        self._keys: Optional[jwt.PyJWKClient] = None
        self._fetched_at: float = 0.0

    def get(self) -> jwt.PyJWKClient:
        if self._keys is None or (time.monotonic() - self._fetched_at) > CACHE_TTL:
            self._keys = jwt.PyJWKClient(EBRAINS_IAM_JWKS_URL)
            self._fetched_at = time.monotonic()
            logger.debug("JWKS refreshed from EBRAINS IAM")
        return self._keys


_cache = _JwksCache()


def verify_token(authorization_header: Optional[str]) -> Dict[str, Any]:
    """
    Validate an 'Authorization: Bearer <token>' header.

    Returns the decoded JWT payload dict on success.
    Raises AuthError on any validation failure.
    """
    if not authorization_header:
        raise AuthError("Missing Authorization header")

    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be 'Bearer <token>'")

    raw_token = parts[1]
    try:
        jwks_client = _cache.get()
        signing_key = jwks_client.get_signing_key_from_jwt(raw_token)
        payload = jwt.decode(
            raw_token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=EXPECTED_ISSUER,
            options={"verify_aud": False},  # audience varies by client
        )
        return payload
    except jwt.ExpiredSignatureError as e:
        raise AuthError("Token expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}") from e
    except Exception as e:
        raise AuthError(f"Token verification failed: {e}") from e


def get_user_id(payload: Dict[str, Any]) -> str:
    """Extract the EBRAINS user ID (subject) from a verified JWT payload."""
    sub = payload.get("sub")
    if not sub:
        raise AuthError("JWT payload missing 'sub' claim")
    return str(sub)
