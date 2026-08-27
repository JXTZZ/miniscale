from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
import operator
import re

from .rewards import MathReward, score_math_answer


@dataclass(frozen=True, slots=True)
class CalculatorTask:
    question: str
    expression: str
    answer: str | tuple[str, ...]
    system_prompt: str | None = None
    tools: object | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolExecution:
    observation: str | None
    valid: bool
    error: str | None = None


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
CALCULATOR_NAMES = frozenset({"calculator", "calculate_math"})


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
    result = evaluate(ast.parse(expression, mode="eval").body)
    if type(result) not in {int, float} or not math.isfinite(result):
        raise ValueError("calculation result must be a finite real number")
    if abs(result) > 1_000_000_000_000:
        raise ValueError("calculation result is too large")
    return result


def parse_tool_call_payload(text: str) -> ToolCall | None:
    matches = _TOOL_CALL.findall(text)
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name", "calculator")
    arguments = payload.get("arguments", payload)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def parse_tool_call(text: str) -> str | None:
    call = parse_tool_call_payload(text)
    if call is None or call.name not in CALCULATOR_NAMES:
        return None
    expression = call.arguments.get("expression")
    return expression if isinstance(expression, str) else None


def filter_calculator_tools(tools: object | None) -> list[dict[str, object]] | None:
    """Return only schemas executable by CalculatorEnv.

    MiniMind math rows also advertise weather. Exposing that schema while the
    environment cannot execute it teaches an impossible action, so it is
    deliberately removed at the data boundary.
    """

    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as error:
            raise ValueError("tools must contain valid JSON") from error
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    supported: list[dict[str, object]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        name = function.get("name") if isinstance(function, dict) else None
        if name in CALCULATOR_NAMES:
            supported.append(tool)
    return supported or None


class CalculatorEnv:
    tool_prompt = (
        "Tool calculator accepts JSON expression. To use it reply "
        '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>.'
    )

    def __init__(self, task: CalculatorTask) -> None:
        self.task = task
        self.valid_calls = 0
        self.invalid_calls = 0
        self.last_error: str | None = None

    def step(self, assistant_text: str) -> ToolExecution:
        call = parse_tool_call_payload(assistant_text)
        if call is None:
            if "<tool_call>" in assistant_text:
                self.invalid_calls += 1
                self.last_error = "invalid_tool_call"
                return ToolExecution("error: invalid tool call", False, self.last_error)
            return ToolExecution(None, False, None)
        if call.name not in CALCULATOR_NAMES:
            self.invalid_calls += 1
            self.last_error = "unsupported_tool"
            return ToolExecution(f"error: unsupported tool {call.name}", False, self.last_error)
        expression = call.arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            self.invalid_calls += 1
            self.last_error = "invalid_arguments"
            return ToolExecution("error: expression must be a non-empty string", False, self.last_error)
        try:
            value = safe_calculate(expression)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            self.invalid_calls += 1
            self.last_error = "execution_error"
            return ToolExecution("error: invalid arithmetic expression", False, self.last_error)
        self.valid_calls += 1
        self.last_error = None
        return ToolExecution(str(value), True)

    def execute(self, assistant_text: str) -> str | None:
        return self.step(assistant_text).observation

    def reward_components(self, final_answer: str) -> dict[str, float]:
        base: MathReward = score_math_answer(final_answer, self.task.answer)
        tool_bonus = 0.2 * base.correctness if self.valid_calls else 0.0
        invalid_penalty = 0.1 * self.invalid_calls
        return {
            **base.metrics(),
            "tool_bonus": tool_bonus,
            "invalid_tool_penalty": invalid_penalty,
            "total": base.total + tool_bonus - invalid_penalty,
        }

    def reward(self, final_answer: str) -> float:
        return self.reward_components(final_answer)["total"]
