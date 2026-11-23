"""Constrained Generation Library - schema-constrained decoding from scratch."""

from __future__ import annotations

from constrained_generation_library.constrainer import (
    EOS,
    Constraint,
    GenerationResult,
    constrained_generate,
    unconstrained_generate,
)
from constrained_generation_library.data import biased_model, default_vocab, random_model
from constrained_generation_library.evaluation import ValidityReport, validity_report
from constrained_generation_library.regex import Regex, RegexParseError
from constrained_generation_library.schema import Property, compile_schema, schema_to_pattern

__version__ = "0.1.0"

__all__ = [
    "EOS",
    "Constraint",
    "GenerationResult",
    "Property",
    "Regex",
    "RegexParseError",
    "ValidityReport",
    "__version__",
    "biased_model",
    "compile_schema",
    "constrained_generate",
    "default_vocab",
    "random_model",
    "schema_to_pattern",
    "unconstrained_generate",
    "validity_report",
]
