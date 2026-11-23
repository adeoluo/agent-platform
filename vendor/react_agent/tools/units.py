"""Unit conversion tool.

Covers the common conversions an agent needs for everyday quantitative questions -
length, mass, and temperature - through a small, explicit table of linear factors (plus
special handling for temperature, which is affine, not linear). The conversions are exact
and inspectable rather than relying on an external units library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from react_agent.core.tool import ToolError

# Factors convert a unit to a canonical base (metres, grams).
_LENGTH = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
    "yd": 0.9144,
}
_MASS = {
    "g": 1.0,
    "kg": 1000.0,
    "mg": 0.001,
    "lb": 453.59237,
    "oz": 28.349523125,
}
_INPUT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)\s*$")


@dataclass(frozen=True)
class UnitConversionTool:
    """Convert a quantity between units of length, mass, or temperature."""

    name: str = "convert_units"
    description: str = (
        "Convert a quantity between units, for example '10 km to mi' or '100 C to F'. "
        "Supports length (m, km, cm, mm, mi, ft, in, yd), mass (g, kg, mg, lb, oz), "
        "and temperature (C, F, K)."
    )

    def __call__(self, tool_input: str) -> str:
        match = _INPUT.match(tool_input)
        if match is None:
            raise ToolError("convert_units expects '<value> <from> to <to>'")
        value = float(match.group(1))
        source = match.group(2)
        target = match.group(3)
        result = self._convert(value, source, target)
        return _format(result)

    def _convert(self, value: float, source: str, target: str) -> float:
        if source.upper() in {"C", "F", "K"} or target.upper() in {"C", "F", "K"}:
            return _convert_temperature(value, source.upper(), target.upper())
        for table in (_LENGTH, _MASS):
            if source in table and target in table:
                return value * table[source] / table[target]
        raise ToolError(f"cannot convert {source!r} to {target!r}")


def _convert_temperature(value: float, source: str, target: str) -> float:
    if source not in {"C", "F", "K"} or target not in {"C", "F", "K"}:
        raise ToolError(f"cannot convert {source!r} to {target!r}")
    # Normalize to Celsius first.
    if source == "F":
        celsius = (value - 32) * 5 / 9
    elif source == "K":
        celsius = value - 273.15
    else:
        celsius = value
    if target == "F":
        return celsius * 9 / 5 + 32
    if target == "K":
        return celsius + 273.15
    return celsius


def _format(value: float) -> str:
    rounded = round(value, 4)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)
