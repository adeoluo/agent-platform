# ADR 0001: SQLite job leases with fencing tokens

## Status

Accepted

## Context

Jobs run for seconds to minutes across several steps. Workers can die at any point
(SIGKILL, OOM, host loss). The platform needs to resume a job from its last completed step,
never record a step twice, and never let two workers own a job at the same time. The
project has a zero third-party runtime dependency rule, so a broker such as Redis or a
server database is out of scope.

## Decision

Store jobs in SQLite (WAL mode) and coordinate workers with leases:

* A claim is one `BEGIN IMMEDIATE` transaction that selects the oldest `queued` job, or a
  `running` job whose `lease_expires` is in the past, and sets `lease_owner`,
  `lease_expires`, and `lease_token = lease_token + 1`.
* Each worker renews its lease from a heartbeat thread every third of the lease duration.
* Every later write (attempt log, checkpoint, completion) is conditional on
  `lease_owner = ? AND lease_token = ?` inside the same transaction. The token is a fencing
  token: a stale owner cannot write after a takeover even if its clock or scheduling
  misbehaves.
* A checkpoint writes the ledger row and the blackboard snapshot in one transaction.
* A job claimed more than `max_attempts` times is failed instead of retried forever.

## Consequences

* Correctness is checked by a cross-process test that SIGKILLs a worker and asserts on the
  ledger, the attempt log, and the claim history.
* Recovery latency is roughly the lease duration plus process start time. The recovery
  benchmark measures it.
* The design is single node. Moving to PostgreSQL would keep the same protocol
  (`SELECT ... FOR UPDATE SKIP LOCKED` plus the same token checks).
* Execution is at least once. Step idempotency is the caller's responsibility.
