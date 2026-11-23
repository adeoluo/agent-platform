"""A safe arithmetic calculator tool.

Expressions are parsed into an abstract syntax tree and evaluated against a fixed
allow-list of binary and unary operators. Python's :func:`eval` is never used, so a
model (or an upstream prompt injection) cannot execute arbitrary code through this tool.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from react_agent.core.tool import ToolError

Number = int | float

_BINARY: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


class CalculatorTool:
    """Evaluate an arithmetic expression such as ``2 * (3 + 4)``."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression, for example '2 * (3 + 4)'. "
        "Supports + - * / % ** and parentheses."
    )

    def __call__(self, tool_input: str) -> str:
        expression = tool_input.strip()
        if not expression:
            raise ToolError("calculator received an empty expression")
        try:
            tree = ast.parse(expression, mode="eval")
            value = _evaluate(tree.body)
        except ToolError:
            raise
        except (SyntaxError, ValueError, TypeError) as error:
            raise ToolError(f"invalid expression {expression!r}: {error}") from None
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("numeric overflow") from None
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ToolError("result is not a finite number")
            if value == int(value):
                return str(int(value))
            return str(value)
        return str(value)


def _evaluate(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        # Keep the native type: integers stay exact (Python ints are unbounded),
        # so large products are not silently rounded through float.
        return node.value
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ToolError("unsupported unary operator")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    raise ToolError("unsupported expression")
