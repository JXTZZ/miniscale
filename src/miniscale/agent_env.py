from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import operator
import re


@dataclass(frozen=True, slots=True)
class CalculatorTask:
    question: str
    expression: str
    answer: str | tuple[str, ...]
    system_prompt: str | None = None


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def safe_calculate(expression: str) -> int | float:
    """Evaluate arithmetic without exposing Python eval or arbitrary calls."""

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1_000_000):
                raise ValueError("power expression is too large")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("only numeric arithmetic is allowed")

    if len(expression) > 100:
        raise ValueError("expression is too long")
    return evaluate(ast.parse(expression, mode="eval").body)


def parse_tool_call(text: str) -> str | None:
    match = _TOOL_CALL.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if payload.get("name", "calculator") not in {"calculator", "calculate_math"}:
        return None
    arguments = payload.get("arguments", payload)
    expression = arguments.get("expression") if isinstance(arguments, dict) else None
    return expression if isinstance(expression, str) else None


class CalculatorEnv:
    tool_prompt = (
        "Tool calculator accepts JSON expression. To use it reply "
        '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>.'
    )

    def __init__(self, task: CalculatorTask) -> None:
        self.task = task
        self.valid_calls = 0
        self.invalid_calls = 0

    def execute(self, assistant_text: str) -> str | None:
        expression = parse_tool_call(assistant_text)
        if expression is None:
            if "<tool_call>" in assistant_text:
                self.invalid_calls += 1
            return None
        try:
            value = safe_calculate(expression)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            self.invalid_calls += 1
            return "error: invalid arithmetic expression"
        self.valid_calls += 1
        return str(value)

    def reward(self, final_answer: str) -> float:
        numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", final_answer)
        expected = (self.task.answer,) if isinstance(self.task.answer, str) else self.task.answer

        def normalize(value: str) -> float | str:
            try:
                return float(value)
            except ValueError:
                return value.lstrip("+")

        normalized_numbers = [normalize(number) for number in numbers]
        matched = sum(normalize(item) in normalized_numbers for item in expected)
        correctness = matched / len(expected) if expected else 0.0
        tool_bonus = 0.2 * min(self.valid_calls / max(len(expected), 1), 1.0)
        return correctness + tool_bonus - 0.1 * self.invalid_calls
