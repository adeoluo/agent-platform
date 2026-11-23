"""Load benchmark: many tenants and concurrent clients against the real HTTP API.

The server runs as a separate process (``agent-platform serve``). Each tenant gets its own
API key and its own copy of the corpus. Client threads then send ``POST /v1/ask`` requests,
choosing tenant and question with a fixed seed. Two passes are measured:

* ``repeated``: questions repeat, so most model calls are served by each tenant's gateway
  cache (retrieval and the agent loop still run on every request).
* ``unique``: every question is made unique, so every model call misses the cache.

Usage: uv run python benchmarks/load.py [--tenants 20] [--clients 16] [--requests 2000]
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter

from _common import fresh_dir, start_server

from agent_platform.auth import KeyStore
from agent_platform.client import Client
from agent_platform.config import DEFAULT_PEPPER
from agent_platform.corpus import ACME_DOCUMENTS, EVAL_CASES
from agent_platform.db import Database
from agent_platform.evaluation import percentile


def run_pass(
    url: str, keys: list[str], *, clients: int, requests: int, unique: bool, seed: int
) -> dict[str, object]:
    rng = random.Random(seed)
    plan = [
        (rng.randrange(len(keys)), rng.randrange(len(EVAL_CASES)), index)
        for index in range(requests)
    ]
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    lock = threading.Lock()
    cursor = iter(plan)

    def client_loop() -> None:
        sessions = [Client(url, key, timeout=60) for key in keys]
        local_lat: list[float] = []
        local_status: Counter[int] = Counter()
        while True:
            with lock:
                item = next(cursor, None)
            if item is None:
                break
            tenant, case, index = item
            question = EVAL_CASES[case].question
            if unique:
                question = f"{question} (request {index})"
            start = time.perf_counter()
            status, _ = sessions[tenant].request("POST", "/v1/ask", {"question": question})
            local_lat.append((time.perf_counter() - start) * 1000.0)
            local_status[status] += 1
        with lock:
            latencies.extend(local_lat)
            statuses.update(local_status)

    threads = [threading.Thread(target=client_loop) for _ in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    return {
        "mode": "unique" if unique else "repeated",
        "requests": requests,
        "clients": clients,
        "tenants": len(keys),
        "seconds": round(elapsed, 2),
        "throughput_rps": round(requests / elapsed, 1),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "statuses": dict(statuses),
    }


def server_gauges(url: str, key: str) -> dict[str, int]:
    """Open SQLite connections and file descriptors of the server process, from /metrics."""
    names = {
        "agent_platform_db_connections": "db_connections",
        "agent_platform_open_file_descriptors": "open_fds",
    }
    out = {}
    for line in Client(url, key).metrics().splitlines():
        name, _, value = line.partition(" ")
        if name in names:
            out[names[name]] = int(float(value))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenants", type=int, default=20)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workdir = fresh_dir("load")
    db_path = workdir / "load.db"
    db = Database(db_path)
    store = KeyStore(db, pepper=DEFAULT_PEPPER)
    keys = [store.create(f"tenant-{i:03d}").secret for i in range(args.tenants)]
    db.close()

    env = {"AGENT_PLATFORM_RATE_CAPACITY": "1000000", "AGENT_PLATFORM_RATE_REFILL": "1000000"}
    server, url = start_server(db_path, env)
    try:
        for key in keys:
            client = Client(url, key)
            client.ingest(ACME_DOCUMENTS)
            # Warm-up: the first query per tenant builds its index; exclude that one-off cost.
            client.ask("warm up the index")
        before = server_gauges(url, keys[0])
        results = [
            run_pass(url, keys, clients=args.clients, requests=args.requests, unique=u, seed=7)
            for u in (False, True)
        ]
        after = server_gauges(url, keys[0])
    finally:
        server.terminate()
        server.wait(timeout=10)
    resources = {"before": before, "after": after}
    if args.json:
        print(json.dumps({"passes": results, "server_resources": resources}, indent=2))
        return
    print(
        f"{'mode':<9} {'requests':>8} {'clients':>7} {'tenants':>7} {'req/s':>8} "
        f"{'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}  statuses"
    )
    for r in results:
        print(
            f"{r['mode']:<9} {r['requests']:>8} {r['clients']:>7} {r['tenants']:>7} "
            f"{r['throughput_rps']:>8} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8}  "
            f"{r['statuses']}"
        )
    total = sum(int(str(r["requests"])) for r in results)
    print(
        f"server after {total} requests: {after.get('db_connections')} SQLite connections "
        f"({before.get('db_connections')} before), {after.get('open_fds')} open file "
        f"descriptors ({before.get('open_fds')} before)"
    )


if __name__ == "__main__":
    main()
