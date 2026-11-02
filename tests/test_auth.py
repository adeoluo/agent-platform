from __future__ import annotations

import pytest

from agent_platform.app import Platform
from agent_platform.auth import AuthError, KeyStore, validate_tenant
from agent_platform.db import Database


def test_created_key_verifies_to_its_tenant(platform: Platform) -> None:
    issued = platform.keys.create("acme", label="ci")
    principal = platform.keys.verify(issued.secret)
    assert principal.tenant == "acme"
    assert principal.key_id == issued.key_id


def test_only_a_hash_is_stored(platform: Platform) -> None:
    issued = platform.keys.create("acme")
    rows = platform.db.conn().execute("SELECT * FROM api_keys").fetchall()
    stored = " ".join(str(value) for row in rows for value in tuple(row))
    secret_part = issued.secret.split("_", 2)[2]
    assert secret_part not in stored
    assert len(rows[0]["key_hash"]) == 64


@pytest.mark.parametrize(
    ("presented", "reason"),
    [
        (None, "missing_key"),
        ("", "missing_key"),
        ("not-a-key", "malformed_key"),
        ("xx_abc_def", "malformed_key"),
        ("ap__secret", "malformed_key"),
        ("ap_000000000000_wrong", "invalid_key"),
    ],
)
def test_bad_keys_are_rejected(platform: Platform, presented: str | None, reason: str) -> None:
    with pytest.raises(AuthError) as caught:
        platform.keys.verify(presented)
    assert caught.value.reason == reason


def test_wrong_secret_for_a_real_key_id_is_invalid(platform: Platform) -> None:
    issued = platform.keys.create("acme")
    forged = f"ap_{issued.key_id}_{'x' * 43}"
    with pytest.raises(AuthError, match="invalid_key"):
        platform.keys.verify(forged)


def test_revoked_key_is_rejected_and_listed(platform: Platform) -> None:
    issued = platform.keys.create("acme")
    assert platform.keys.revoke(issued.key_id)
    assert not platform.keys.revoke(issued.key_id)
    with pytest.raises(AuthError, match="revoked_key"):
        platform.keys.verify(issued.secret)
    [info] = platform.keys.list("acme")
    assert info.revoked


def test_a_different_pepper_cannot_verify(platform: Platform) -> None:
    issued = platform.keys.create("acme")
    other = KeyStore(platform.db, pepper="another-pepper")
    with pytest.raises(AuthError, match="invalid_key"):
        other.verify(issued.secret)


def test_key_listing_filters_by_tenant(platform: Platform) -> None:
    platform.keys.create("acme")
    platform.keys.create("globex")
    assert {k.tenant for k in platform.keys.list()} == {"acme", "globex"}
    assert [k.tenant for k in platform.keys.list("globex")] == ["globex"]


@pytest.mark.parametrize("tenant", ["", "Acme", "-x", "a b", "x" * 65])
def test_invalid_tenant_names(tenant: str) -> None:
    with pytest.raises(ValueError):
        validate_tenant(tenant)


def test_empty_pepper_is_rejected(platform: Platform) -> None:
    db: Database = platform.db
    with pytest.raises(ValueError):
        KeyStore(db, pepper="")
