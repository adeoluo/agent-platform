"""In-process request metrics for the HTTP service.

Production services need to answer "how many requests, how many errors, how fast"
without a separate dependency. This module records per-route request counts, error
counts, and a latency histogram, and renders them in the Prometheus text exposition
format so any standard scraper can read ``GET /metrics``. The recorder is mutable
and lives on the (otherwise frozen) service, so recording a request never mutates
the service's own configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Cumulative histogram upper bounds in milliseconds.
_BUCKETS_MS: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0)


@dataclass
class RouteStats:
    """Counters and a latency histogram for one route."""

    requests: int = 0
    errors: int = 0
    duration_ms_sum: float = 0.0
    bucket_counts: list[int] = field(default_factory=lambda: [0] * len(_BUCKETS_MS))


@dataclass
class ServiceMetrics:
    """Per-route request metrics, rendered in the Prometheus text format."""

    routes: dict[str, RouteStats] = field(default_factory=dict)

    def observe(self, route: str, duration_ms: float, *, error: bool) -> None:
        """Record one handled request on ``route``."""
        stats = self.routes.setdefault(route, RouteStats())
        stats.requests += 1
        if error:
            stats.errors += 1
        stats.duration_ms_sum += duration_ms
        for index, upper in enumerate(_BUCKETS_MS):
            if duration_ms <= upper:
                stats.bucket_counts[index] += 1

    @property
    def total_requests(self) -> int:
        return sum(stats.requests for stats in self.routes.values())

    @property
    def total_errors(self) -> int:
        return sum(stats.errors for stats in self.routes.values())

    def render_prometheus(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""
        lines: list[str] = []
        lines.append("# HELP service_requests_total Requests handled, by route.")
        lines.append("# TYPE service_requests_total counter")
        for route, stats in sorted(self.routes.items()):
            lines.append(f'service_requests_total{{route="{route}"}} {stats.requests}')
        lines.append("# HELP service_errors_total Error responses, by route.")
        lines.append("# TYPE service_errors_total counter")
        for route, stats in sorted(self.routes.items()):
            lines.append(f'service_errors_total{{route="{route}"}} {stats.errors}')
        lines.append("# HELP service_request_duration_ms_sum Summed request duration, by route.")
        lines.append("# TYPE service_request_duration_ms_sum counter")
        for route, stats in sorted(self.routes.items()):
            lines.append(
                f'service_request_duration_ms_sum{{route="{route}"}} {stats.duration_ms_sum:.3f}'
            )
        lines.append("# HELP service_request_duration_ms Request latency histogram, by route.")
        lines.append("# TYPE service_request_duration_ms histogram")
        for route, stats in sorted(self.routes.items()):
            for index, upper in enumerate(_BUCKETS_MS):
                count = stats.bucket_counts[index]
                lines.append(
                    f'service_request_duration_ms_bucket{{route="{route}",le="{upper}"}} {count}'
                )
            lines.append(
                f'service_request_duration_ms_bucket{{route="{route}",le="+Inf"}} {stats.requests}'
            )
        return "\n".join(lines) + "\n"
