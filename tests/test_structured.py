from __future__ import annotations

import json

import pytest

from agent_platform.structured import draft_json, generate_structured, parse_schema


def test_parse_schema_accepts_all_types() -> None:
    props = parse_schema(
        {
            "properties": [
                {"name": "a", "type": "string"},
                {"name": "b", "type": "integer"},
                {"name": "c", "type": "number"},
                {"name": "d", "type": "boolean"},
                {"name": "e", "type": "enum", "values": ["x", "y"]},
            ]
        }
    )
    assert [p.type for p in props] == ["string", "integer", "number", "boolean", "enum"]
    assert props[0].max_length == 120
    assert props[4].values == ("x", "y")


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {},
        {"properties": []},
        {"properties": [1]},
        {"properties": [{"name": "bad name", "type": "string"}]},
        {"properties": [{"name": "a", "type": "object"}]},
        {"properties": [{"name": "a", "type": "string"}, {"name": "a", "type": "string"}]},
        {"properties": [{"name": "e", "type": "enum", "values": []}]},
        {"properties": [{"name": "e", "type": "enum", "values": ["has space"]}]},
        {"properties": [{"name": "a", "type": "string", "max_length": 0}]},
        {"properties": [{"name": "a", "type": "string", "max_length": 500}]},
        {"properties": [{"name": f"p{i}", "type": "boolean"} for i in range(9)]},
    ],
)
def test_parse_schema_rejects_invalid(schema: object) -> None:
    with pytest.raises(ValueError):
        parse_schema(schema)


def test_generation_is_valid_even_when_the_draft_does_not_fit() -> None:
    props = parse_schema(
        {
            "properties": [
                {"name": "answer", "type": "string", "max_length": 12},
                {"name": "count", "type": "integer", "max_length": 2},
                {"name": "score", "type": "number", "max_length": 2},
                {"name": "ok", "type": "boolean"},
                {"name": "label", "type": "enum", "values": ["good", "bad"]},
            ]
        }
    )
    fields = {
        "answer": 'A "quoted"\nanswer\\ that is far too long',
        "count": 12345,
        "score": 1e9,
        "ok": 1,
        "label": "unknown",
    }
    out = generate_structured(props, fields)
    assert json.loads(out.text) == out.value
    assert out.value == {
        "answer": "A quoted ans",
        "count": 0,
        "score": 0.0,
        "ok": True,
        "label": "good",
    }
    assert 0.0 < out.mean_allowed_fraction < 1.0
    assert out.tokens > 0


def test_draft_uses_defaults_for_missing_fields() -> None:
    props = parse_schema(
        {
            "properties": [
                {"name": "answer", "type": "string"},
                {"name": "n", "type": "integer"},
                {"name": "x", "type": "number"},
            ]
        }
    )
    assert draft_json(props, {"x": 2.71828}) == '{"answer":"","n":0,"x":2.718}'
