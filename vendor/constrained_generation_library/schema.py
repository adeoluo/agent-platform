"""Compile a small JSON Schema into a regular expression for compact JSON.

The supported schema is an object with an ordered list of typed properties. Each
property type maps to a regular expression fragment for its JSON serialization,
and the object pattern requires the keys in order with no insignificant
whitespace. Compiling the schema to a regular expression is what lets the token
constrainer guarantee that generated output is valid: the automaton rejects any
prefix that could not extend to a legal document.

Supported types: ``string``, ``integer``, ``number``, ``boolean``, and ``enum``
(a fixed set of string values).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from constrained_generation_library.regex import Regex


@dataclass(frozen=True)
class Property:
    """One object property: a name and a type (plus enum values if applicable)."""

    name: str
    type: str
    values: tuple[str, ...] = ()
    max_length: int = 16


_BOOLEAN = r"(true|false)"


def _bounded_repeat(fragment: str, count: int) -> str:
    """Return ``fragment`` repeated up to ``count`` times (each optional)."""

    return f"({fragment})?" * count


def _escape_literal(text: str) -> str:
    out = []
    for ch in text:
        if ch in r".[]()|*+?\\^$":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _json_string_literal(text: str) -> str:
    """Return a regex fragment matching the exact JSON serialization of ``text``.

    Property names and enum values appear in the output as JSON strings, so they
    must be matched against their escaped serialization, not the raw text. For
    example a property name containing a quote serializes to ``"a\\"b"`` and the
    pattern has to match those literal characters. ``json.dumps`` produces the
    escaping (quotes, backslashes, control characters) and ``_escape_literal``
    then protects any regex metacharacters it introduced. Raw non-ASCII is kept
    (``ensure_ascii=False``) to match the string-value convention.
    """

    return _escape_literal(json.dumps(text, ensure_ascii=False))


def _value_pattern(prop: Property) -> str:
    if prop.type == "string":
        return '"' + _bounded_repeat('[^"]', prop.max_length) + '"'
    if prop.type == "integer":
        return "-?(0|[1-9]" + _bounded_repeat("\\d", max(prop.max_length - 1, 0)) + ")"
    if prop.type == "number":
        integer = "-?(0|[1-9]" + _bounded_repeat("\\d", max(prop.max_length - 1, 0)) + ")"
        return integer + "(\\.\\d" + _bounded_repeat("\\d", 3) + ")?"
    if prop.type == "boolean":
        return _BOOLEAN
    if prop.type == "enum":
        if not prop.values:
            raise ValueError(f"enum property {prop.name!r} needs values")
        alts = "|".join(_json_string_literal(v) for v in prop.values)
        return f"({alts})"
    raise ValueError(f"unsupported type: {prop.type!r}")


def schema_to_pattern(properties: list[Property]) -> str:
    """Return a regular expression matching compact JSON for ``properties``."""

    if not properties:
        return r"\{\}"
    parts = []
    for prop in properties:
        parts.append(_json_string_literal(prop.name) + ":" + _value_pattern(prop))
    return r"\{" + ",".join(parts) + r"\}"


def compile_schema(properties: list[Property]) -> Regex:
    """Compile ``properties`` into a :class:`Regex` over compact JSON."""

    return Regex(schema_to_pattern(properties))
