"""Structured answers via schema-constrained decoding.

A client can ask for a JSON object with a declared schema (an ordered list of typed
properties). The schema is compiled to a token-level automaton by
``constrained_generation_library`` and the answer is decoded under that automaton: the
decoder prefers the characters of a draft built from the agent's result, but can only emit
tokens the automaton allows, so the output is valid JSON for the schema by construction even
when the draft is not (too long, wrong characters, unknown enum value).
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from typing import Any

from constrained_generation_library import (
    EOS,
    Property,
    compile_schema,
    constrained_generate,
    default_vocab,
)

_TYPES = frozenset({"string", "integer", "number", "boolean", "enum"})
MAX_PROPERTIES = 8
MAX_STRING_LENGTH = 160
_EXTRA_CHARS = "!#$%&'()*+/;<=>?@[]^`{|}~"


@dataclass(frozen=True)
class StructuredOutput:
    """A decoded object plus decoding diagnostics."""

    value: dict[str, Any]
    text: str
    tokens: int
    mean_allowed_fraction: float


def parse_schema(raw: object) -> list[Property]:
    """Validate a ``response_schema`` JSON object into constrained-generation properties."""
    if not isinstance(raw, dict) or not isinstance(raw.get("properties"), list):
        raise ValueError("'response_schema' must be an object with a 'properties' array")
    items = raw["properties"]
    if not items or len(items) > MAX_PROPERTIES:
        raise ValueError(f"'properties' must have 1 to {MAX_PROPERTIES} entries")
    properties: list[Property] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each property must be an object")
        name = item.get("name")
        kind = item.get("type")
        if not isinstance(name, str) or not name.isidentifier() or name in seen:
            raise ValueError("property 'name' must be a unique identifier")
        if kind not in _TYPES:
            raise ValueError(f"property {name!r} has unsupported type {kind!r}")
        seen.add(name)
        values = item.get("values", [])
        if kind == "enum" and (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, str) and v.isidentifier() for v in values)
        ):
            raise ValueError(f"enum property {name!r} needs identifier-like string 'values'")
        default_length = 120 if kind == "string" else 6
        max_length = item.get("max_length", default_length)
        if not isinstance(max_length, int) or not 1 <= max_length <= MAX_STRING_LENGTH:
            raise ValueError(f"property {name!r} max_length must be 1..{MAX_STRING_LENGTH}")
        properties.append(
            Property(
                name=name,
                type=str(kind),
                values=tuple(values) if kind == "enum" else (),
                max_length=max_length,
            )
        )
    return properties


def _vocab() -> list[str]:
    vocab = default_vocab()
    present = set(vocab)
    for ch in string.ascii_letters + string.digits + _EXTRA_CHARS:
        if ch not in present:
            vocab.insert(len(vocab) - 1, ch)
            present.add(ch)
    return vocab


_VOCAB = _vocab()
_ALLOWED_CHARS = frozenset(token for token in _VOCAB if len(token) == 1) - {'"', "\\"}


def _clean_string(value: str, max_length: int) -> str:
    cleaned = "".join(ch if ch in _ALLOWED_CHARS else " " for ch in value)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length].rstrip()


def _draft_value(prop: Property, fields: dict[str, Any]) -> Any:
    raw = fields.get(prop.name)
    if prop.type == "string":
        return _clean_string(str(raw) if raw is not None else "", prop.max_length)
    if prop.type == "boolean":
        return bool(raw)
    if prop.type == "integer":
        whole = int(raw) if isinstance(raw, int | float) else 0
        return whole if len(str(abs(whole))) <= prop.max_length else 0
    if prop.type == "number":
        number = float(raw) if isinstance(raw, int | float) else 0.0
        if not abs(number) < 10**prop.max_length:
            return 0.0
        return round(number, 3)
    # enum: use the field if it names a valid value, else the first value.
    return raw if raw in prop.values else prop.values[0]


def draft_json(properties: list[Property], fields: dict[str, Any]) -> str:
    """Build the compact JSON the decoder will try to reproduce."""
    draft = {prop.name: _draft_value(prop, fields) for prop in properties}
    return json.dumps(draft, separators=(",", ":"), ensure_ascii=True)


def generate_structured(properties: list[Property], fields: dict[str, Any]) -> StructuredOutput:
    """Decode a schema-valid JSON object that follows ``fields`` wherever it can."""
    regex = compile_schema(properties)
    target = draft_json(properties, fields)
    vocab = _VOCAB

    def score(token_id: int, prefix: str) -> float:
        token = vocab[token_id]
        if token == EOS:
            return 3.0 if prefix == target else -1.0
        if target.startswith(prefix + token):
            return 2.0 + len(token)
        return -float(len(token))

    budget = len(target) + sum(prop.max_length + len(prop.name) + 8 for prop in properties)
    result = constrained_generate(regex, vocab, score, max_tokens=budget)
    try:
        value = json.loads(result.text)
    except json.JSONDecodeError as error:  # pragma: no cover - the draft is always valid
        raise ValueError(f"decoding did not reach an accepting state: {error}") from None
    return StructuredOutput(
        value=value,
        text=result.text,
        tokens=result.steps,
        mean_allowed_fraction=round(result.mean_allowed_fraction, 4),
    )
