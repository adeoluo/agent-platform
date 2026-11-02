"""API keys: issuance, hashed storage, verification, and revocation.

A key has the form ``ap_<key_id>_<secret>``. Only ``key_id`` (a lookup handle) and an
HMAC-SHA256 digest of the secret are stored; the plaintext secret is shown once at
creation and never persisted. Verification looks the row up by ``key_id`` and compares
digests in constant time, so a database leak does not reveal usable keys and the check
does not leak timing information about how much of a guess was correct.

The tenant a request acts for is taken from the key row, never from the request body.
That is the root of tenant isolation for every other component.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass

from agent_platform.db import Database

_TENANT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PREFIX = "ap"


class AuthError(Exception):
    """Raised when a presented key is missing, malformed, unknown, or revoked."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Principal:
    """The authenticated identity behind a request."""

    tenant: str
    key_id: str


@dataclass(frozen=True)
class IssuedKey:
    """A newly created key. ``secret`` is the full key and is only available here."""

    key_id: str
    tenant: str
    secret: str


@dataclass(frozen=True)
class KeyInfo:
    """Non-secret metadata about a stored key."""

    key_id: str
    tenant: str
    label: str
    created_at: float
    revoked: bool


def validate_tenant(tenant: str) -> str:
    """Return ``tenant`` if it is a valid identifier, else raise ``ValueError``."""
    if not _TENANT.match(tenant):
        raise ValueError(
            "tenant must be 1 to 64 characters of lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return tenant


class KeyStore:
    """Creates and verifies hashed API keys stored in SQLite."""

    def __init__(self, db: Database, *, pepper: str) -> None:
        if not pepper:
            raise ValueError("pepper must be non-empty")
        self._db = db
        self._pepper = pepper.encode("utf-8")

    def _digest(self, secret: str) -> str:
        return hmac.new(self._pepper, secret.encode("utf-8"), hashlib.sha256).hexdigest()

    def create(self, tenant: str, *, label: str = "") -> IssuedKey:
        """Issue a new key for ``tenant`` and store only its digest."""
        validate_tenant(tenant)
        key_id = secrets.token_hex(6)
        secret = secrets.token_urlsafe(32)
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, tenant, key_hash, label, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key_id, tenant, self._digest(secret), label, time.time()),
            )
        return IssuedKey(key_id=key_id, tenant=tenant, secret=f"{_PREFIX}_{key_id}_{secret}")

    def verify(self, presented: str | None) -> Principal:
        """Return the principal for ``presented`` or raise :class:`AuthError`."""
        if not presented:
            raise AuthError("missing_key")
        parts = presented.strip().split("_", 2)
        if len(parts) != 3 or parts[0] != _PREFIX or not parts[1] or not parts[2]:
            raise AuthError("malformed_key")
        _, key_id, secret = parts
        row = (
            self._db.conn()
            .execute(
                "SELECT tenant, key_hash, revoked_at FROM api_keys WHERE key_id = ?", (key_id,)
            )
            .fetchone()
        )
        # Compute the digest even for unknown ids so both paths cost the same.
        candidate = self._digest(secret)
        stored = row["key_hash"] if row is not None else "0" * len(candidate)
        if not hmac.compare_digest(candidate, stored) or row is None:
            raise AuthError("invalid_key")
        if row["revoked_at"] is not None:
            raise AuthError("revoked_key")
        return Principal(tenant=str(row["tenant"]), key_id=key_id)

    def revoke(self, key_id: str) -> bool:
        """Revoke ``key_id``; return whether an active key was revoked."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
        return cursor.rowcount == 1

    def list(self, tenant: str | None = None) -> list[KeyInfo]:
        """List stored keys (never secrets), optionally for a single tenant."""
        query = "SELECT key_id, tenant, label, created_at, revoked_at FROM api_keys"
        params: tuple[str, ...] = ()
        if tenant is not None:
            query += " WHERE tenant = ?"
            params = (tenant,)
        rows = self._db.conn().execute(query + " ORDER BY created_at", params).fetchall()
        return [
            KeyInfo(
                key_id=str(row["key_id"]),
                tenant=str(row["tenant"]),
                label=str(row["label"]),
                created_at=float(row["created_at"]),
                revoked=row["revoked_at"] is not None,
            )
            for row in rows
        ]
