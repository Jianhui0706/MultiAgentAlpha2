from __future__ import annotations

import re
from dataclasses import dataclass

from trading_agents import factor_runtime
from trading_agents.agents.tools.factor_expression_ast import BinaryOpNode
from trading_agents.agents.tools.factor_expression_ast import ConditionalNode
from trading_agents.agents.tools.factor_expression_ast import FunctionNode
from trading_agents.agents.tools.factor_expression_ast import Node
from trading_agents.agents.tools.factor_expression_ast import UnaryOpNode
from trading_agents.agents.tools.factor_expression_ast import VarNode
from trading_agents.agents.tools.factor_expression_ast import NumberNode
from trading_agents.agents.tools.factor_expression_ast import parse_expression


_WINDOW_ALIAS_FUNCTIONS = {
    "MAX": "TS_MAX",
    "MIN": "TS_MIN",
    "MEDIAN": "TS_MEDIAN",
    "MEAN": "TS_MEAN",
    "STD": "TS_STD",
}

_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    "RANK": (1, 1),
    "ZSCORE": (1, 1),
    "MEAN": (1, 1),
    "STD": (1, 1),
    "MAX": (1, 1),
    "MIN": (1, 1),
    "MEDIAN": (1, 1),
    "SKEW": (1, 1),
    "KURT": (1, 1),
    "ABS": (1, 1),
    "LOG": (1, 1),
    "SQRT": (1, 1),
    "SIGN": (1, 1),
    "EXP": (1, 1),
    "DELTA": (2, 2),
    "DELAY": (2, 2),
    "TS_MEAN": (2, 2),
    "TS_SUM": (2, 2),
    "TS_RANK": (2, 2),
    "TS_ZSCORE": (2, 2),
    "TS_MEDIAN": (2, 2),
    "TS_PCTCHANGE": (2, 2),
    "TS_MIN": (2, 2),
    "TS_MAX": (2, 2),
    "TS_STD": (2, 2),
    "TS_VAR": (2, 2),
    "TS_CORR": (3, 3),
    "TS_COV": (3, 3),
    "TS_RATIO": (2, 2),
    "PERCENTILE": (3, 3),
    "TS_QUANTILE": (3, 3),
    "TS_ARGMAX": (2, 2),
    "TS_ARGMIN": (2, 2),
    "TS_MAD": (2, 2),
    "HIGHDAY": (2, 2),
    "LOWDAY": (2, 2),
    "SUMAC": (2, 2),
    "SMA": (2, 3),
    "WMA": (2, 2),
    "EMA": (2, 2),
    "DECAYLINEAR": (2, 2),
    "COUNT": (2, 2),
    "SUMIF": (3, 3),
    "FILTER": (2, 2),
    "SEQUENCE": (1, 1),
    "REGBETA": (3, 3),
    "REGRESI": (3, 3),
    "PROD": (2, 2),
    "INV": (1, 1),
    "FLOOR": (1, 1),
    "SCALE": (1, 1),
    "POW": (2, 2),
    "RSI": (2, 2),
    "MACD": (2, 3),
    "BB_UPPER": (2, 3),
    "BB_LOWER": (2, 3),
    "BB_MIDDLE": (2, 2),
}


@dataclass(frozen=True)
class FactorExpressionValidationResult:
    passed: bool
    unsupported_features: tuple[str, ...]
    unsupported_functions: tuple[str, ...]
    invalid_function_arities: tuple[str, ...] = ()

    def to_feedback(self) -> str:
        messages: list[str] = []
        if self.unsupported_features:
            messages.append(f"unsupported features: {list(self.unsupported_features)}")
        if self.unsupported_functions:
            messages.append(f"unsupported functions: {list(self.unsupported_functions)}")
        if self.invalid_function_arities:
            messages.append(f"invalid function arity: {list(self.invalid_function_arities)}")
        return "; ".join(messages) if messages else "expression symbols validated"


def extract_expression_features(expression: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)}


def extract_expression_functions(expression: str) -> set[str]:
    return {match.group(1) for match in re.finditer(r"\b([A-Z][A-Z0-9_]*)\s*\(", expression)}


def get_supported_runtime_functions() -> set[str]:
    return {str(item).strip() for item in getattr(factor_runtime, "__all__", []) if str(item).strip()}


def _format_node(node: Node) -> str:
    if isinstance(node, FunctionNode):
        name = node.name.name if isinstance(node.name, VarNode) else str(node.name)
        return f"{name}({', '.join(_format_node(arg) for arg in node.args)})"
    if isinstance(node, BinaryOpNode):
        return f"({_format_node(node.left)} {node.op} {_format_node(node.right)})"
    if isinstance(node, ConditionalNode):
        return f"({_format_node(node.condition)} ? {_format_node(node.true_expr)} : {_format_node(node.false_expr)})"
    if isinstance(node, UnaryOpNode):
        return f"({node.op}{_format_node(node.operand)})"
    if isinstance(node, VarNode):
        return node.name
    if isinstance(node, NumberNode):
        value = node.value
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(node)


def _rewrite_window_aliases(node: Node) -> tuple[Node, bool]:
    if isinstance(node, FunctionNode):
        rewritten_pairs = [_rewrite_window_aliases(arg) for arg in node.args]
        rewritten_args = [item[0] for item in rewritten_pairs]
        changed = any(item[1] for item in rewritten_pairs)
        raw_name = node.name.name if isinstance(node.name, VarNode) else str(node.name)
        name = raw_name.upper()
        if name in _WINDOW_ALIAS_FUNCTIONS and len(rewritten_args) == 2:
            return FunctionNode(name=_WINDOW_ALIAS_FUNCTIONS[name], args=rewritten_args), True
        return FunctionNode(name=raw_name, args=rewritten_args), changed
    if isinstance(node, BinaryOpNode):
        left, left_changed = _rewrite_window_aliases(node.left)
        right, right_changed = _rewrite_window_aliases(node.right)
        return BinaryOpNode(op=node.op, left=left, right=right), left_changed or right_changed
    if isinstance(node, ConditionalNode):
        condition, condition_changed = _rewrite_window_aliases(node.condition)
        true_expr, true_changed = _rewrite_window_aliases(node.true_expr)
        false_expr, false_changed = _rewrite_window_aliases(node.false_expr)
        return (
            ConditionalNode(condition=condition, true_expr=true_expr, false_expr=false_expr),
            condition_changed or true_changed or false_changed,
        )
    if isinstance(node, UnaryOpNode):
        operand, changed = _rewrite_window_aliases(node.operand)
        return UnaryOpNode(op=node.op, operand=operand), changed
    return node, False


def normalize_window_function_aliases(expression: str) -> str:
    """Convert common model mistakes like MAX(A, n) into TS_MAX(A, n)."""
    text = str(expression or "").strip()
    if not text:
        return text
    try:
        parsed = parse_expression(text)
    except Exception:
        return text
    rewritten, changed = _rewrite_window_aliases(parsed)
    if not changed:
        return text
    return _format_node(rewritten)


def _collect_function_arity_errors(node: Node, errors: list[str]) -> None:
    if isinstance(node, FunctionNode):
        raw_name = node.name.name if isinstance(node.name, VarNode) else str(node.name)
        name = raw_name.upper()
        expected = _FUNCTION_ARITY.get(name)
        if expected is not None:
            min_args, max_args = expected
            actual = len(node.args)
            if actual < min_args or actual > max_args:
                expected_text = str(min_args) if min_args == max_args else f"{min_args}-{max_args}"
                errors.append(f"{name} expected {expected_text} args, got {actual}")
        for arg in node.args:
            _collect_function_arity_errors(arg, errors)
        return
    if isinstance(node, BinaryOpNode):
        _collect_function_arity_errors(node.left, errors)
        _collect_function_arity_errors(node.right, errors)
        return
    if isinstance(node, ConditionalNode):
        _collect_function_arity_errors(node.condition, errors)
        _collect_function_arity_errors(node.true_expr, errors)
        _collect_function_arity_errors(node.false_expr, errors)
        return
    if isinstance(node, UnaryOpNode):
        _collect_function_arity_errors(node.operand, errors)


def validate_expression_function_arities(expression: str) -> tuple[bool, tuple[str, ...]]:
    try:
        parsed = parse_expression(str(expression or "").strip())
    except Exception:
        return True, ()
    errors: list[str] = []
    _collect_function_arity_errors(parsed, errors)
    return not errors, tuple(errors)


def validate_expression_symbols(
    expression: str,
    *,
    allowed_features: set[str],
    allowed_functions: set[str] | None = None,
) -> FactorExpressionValidationResult:
    normalized_allowed_functions = allowed_functions or get_supported_runtime_functions()
    used_features = extract_expression_features(expression)
    used_functions = extract_expression_functions(expression)
    unsupported_features = tuple(sorted(feature for feature in used_features if feature not in allowed_features))
    unsupported_functions = tuple(sorted(fn for fn in used_functions if fn not in normalized_allowed_functions))
    arity_passed, arity_errors = validate_expression_function_arities(expression)
    return FactorExpressionValidationResult(
        passed=not unsupported_features and not unsupported_functions and arity_passed,
        unsupported_features=unsupported_features,
        unsupported_functions=unsupported_functions,
        invalid_function_arities=arity_errors,
    )
