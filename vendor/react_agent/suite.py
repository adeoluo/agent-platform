"""A small built-in benchmark suite for the agent.

A handful of tasks spanning the tool types (arithmetic, lookup, unit conversion), each
with a judge for the final answer. It exists to produce a concrete, reproducible agent
success-rate number with the offline rule-policy model and the full toolset.
"""

from __future__ import annotations

from react_agent.benchmark import BenchmarkTask, contains, every

SUITE: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        name="arithmetic-basic",
        question="What is 21 * 2?",
        judge=contains("42"),
        tags=("calculator",),
    ),
    BenchmarkTask(
        name="arithmetic-order",
        question="What is 2 * (3 + 4)?",
        judge=contains("14"),
        tags=("calculator",),
    ),
    BenchmarkTask(
        name="capital-france",
        question="What is the capital of France?",
        judge=contains("Paris"),
        tags=("knowledge_base",),
    ),
    BenchmarkTask(
        name="capital-japan",
        question="What is the capital of Japan?",
        judge=contains("Tokyo"),
        tags=("knowledge_base",),
    ),
    BenchmarkTask(
        name="convert-length",
        question="Convert 1000 m to km?",
        judge=contains("1"),
        tags=("convert_units",),
    ),
    BenchmarkTask(
        name="convert-temperature",
        question="Convert 100 C to F?",
        judge=contains("212"),
        tags=("convert_units",),
    ),
    BenchmarkTask(
        name="multi-capital-and-arithmetic",
        question="What is the capital of France, and what is 21 * 2?",
        judge=every("Paris", "42"),
        tags=("knowledge_base", "calculator", "multi-step"),
    ),
    BenchmarkTask(
        name="multi-two-sums",
        question="What is 10 + 5, and what is 6 * 7?",
        judge=every("15", "42"),
        tags=("calculator", "multi-step"),
    ),
)
