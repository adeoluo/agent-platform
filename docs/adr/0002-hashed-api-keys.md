# ADR 0002: Hashed API keys with a server-side pepper

## Status

Accepted

## Context

Clients authenticate with long-lived API keys. A database copy (backup, leaked file, or a
read-only SQL injection elsewhere) must not yield usable keys, and verification should not
leak timing information.

## Decision

* Keys have the form `ap_<key_id>_<secret>`. `key_id` is 12 random hex characters used only
  for lookup. `secret` is 32 random bytes, URL-safe base64 encoded.
* Only `HMAC-SHA256(pepper, secret)` is stored. The pepper comes from
  `AGENT_PLATFORM_KEY_PEPPER` and is not stored in the database.
* Verification looks up the row by `key_id`, computes the digest for every request (also
  when the id is unknown), and compares with `hmac.compare_digest`.
* Keys can be revoked; revoked keys fail with a distinct reason that is counted in metrics.
* The plaintext key is printed once by `agent-platform keys create` and never again.

## Consequences

* A slow password hash (scrypt, Argon2) is unnecessary because secrets have 256 bits of
  entropy; brute force against the digest is not practical. HMAC keeps verification cheap.
* Losing the pepper invalidates every key. Rotating it requires reissuing keys.
* The default pepper is for development only and is documented as such.
* There is no key expiry or scope model yet.
