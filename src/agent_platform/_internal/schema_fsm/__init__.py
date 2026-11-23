"""Schema-constrained finite-automaton decoding subsystem."""

from __future__ import annotations

from agent_platform._internal.schema_fsm.constrainer import (
    EOS,
    Constraint,
    GenerationResult,
    constrained_generate,
    unconstrained_generate,
)
from agent_platform._internal.schema_fsm.data import biased_model, default_vocab, random_model
from agent_platform._internal.schema_fsm.regex import Regex, RegexParseError
from agent_platform._internal.schema_fsm.schema import Property, compile_schema, schema_to_pattern

__all__ = [
    "EOS",
    "Constraint",
    "GenerationResult",
    "Property",
    "Regex",
    "RegexParseError",
    "biased_model",
    "compile_schema",
    "constrained_generate",
    "default_vocab",
    "random_model",
    "schema_to_pattern",
    "unconstrained_generate",
]
