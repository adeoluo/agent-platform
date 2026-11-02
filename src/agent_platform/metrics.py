"""A small thread-safe metrics registry rendered in the Prometheus text format."""

from __future__ import annotations

import bisect
import threading
from collections.abc import Iterable

Labels = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _labels(labels: dict[str, str]) -> Labels:
    return tuple(sorted(labels.items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Iterable[tuple[str, str]]) -> str:
    items = [f'{key}="{_escape(value)}"' for key, value in labels]
    return "{" + ",".join(items) + "}" if items else ""


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.total += value
        self.count += 1


class Metrics:
    """Counters, gauges, and histograms with labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, tuple[str, str]] = {}
        self._counters: dict[str, dict[Labels, float]] = {}
        self._gauges: dict[str, dict[Labels, float]] = {}
        self._histograms: dict[str, dict[Labels, _Histogram]] = {}

    def _declare(self, name: str, kind: str, help_text: str) -> None:
        self._help.setdefault(name, (kind, help_text))

    def inc(self, name: str, help_text: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._declare(name, "counter", help_text)
            series = self._counters.setdefault(name, {})
            key = _labels(labels)
            series[key] = series.get(key, 0.0) + value

    def set_counter(self, name: str, help_text: str, value: float, **labels: str) -> None:
        """Set a counter mirrored from another component's monotonic total."""
        with self._lock:
            self._declare(name, "counter", help_text)
            self._counters.setdefault(name, {})[_labels(labels)] = value

    def set_gauge(self, name: str, help_text: str, value: float, **labels: str) -> None:
        with self._lock:
            self._declare(name, "gauge", help_text)
            self._gauges.setdefault(name, {})[_labels(labels)] = value

    def observe(
        self,
        name: str,
        help_text: str,
        value: float,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        **labels: str,
    ) -> None:
        with self._lock:
            self._declare(name, "histogram", help_text)
            series = self._histograms.setdefault(name, {})
            key = _labels(labels)
            histogram = series.get(key)
            if histogram is None:
                histogram = series[key] = _Histogram(buckets)
            histogram.observe(value)

    def counter_value(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_labels(labels), 0.0)

    def render(self) -> str:
        """Render every series in the Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._help):
                kind, help_text = self._help[name]
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                if kind == "counter":
                    for labels, value in sorted(self._counters.get(name, {}).items()):
                        lines.append(f"{name}{_render_labels(labels)} {_number(value)}")
                elif kind == "gauge":
                    for labels, value in sorted(self._gauges.get(name, {}).items()):
                        lines.append(f"{name}{_render_labels(labels)} {_number(value)}")
                else:
                    for labels, hist in sorted(self._histograms.get(name, {}).items()):
                        cumulative = 0
                        for bound, count in zip(hist.buckets, hist.counts, strict=False):
                            cumulative += count
                            bucket_labels = (*labels, ("le", _number(bound)))
                            lines.append(
                                f"{name}_bucket{_render_labels(bucket_labels)} {cumulative}"
                            )
                        inf_labels = (*labels, ("le", "+Inf"))
                        lines.append(f"{name}_bucket{_render_labels(inf_labels)} {hist.count}")
                        lines.append(f"{name}_sum{_render_labels(labels)} {_number(hist.total)}")
                        lines.append(f"{name}_count{_render_labels(labels)} {hist.count}")
        return "\n".join(lines) + "\n"


def _number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)
