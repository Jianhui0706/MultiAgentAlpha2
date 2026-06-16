from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from trading_agents.agents.tools.factor_expression_ast import BinaryOpNode
from trading_agents.agents.tools.factor_expression_ast import ConditionalNode
from trading_agents.agents.tools.factor_expression_ast import FunctionNode
from trading_agents.agents.tools.factor_expression_ast import Node
from trading_agents.agents.tools.factor_expression_ast import UnaryOpNode
from trading_agents.agents.tools.factor_expression_ast import VarNode
from trading_agents.agents.tools.factor_expression_ast import parse_expression


@dataclass(frozen=True)
class ExpressionComplexityBudget:
    max_expression_length: int | None = None
    max_function_count: int | None = None
    max_nesting_depth: int | None = None
    max_input_feature_count: int | None = None
    max_operator_count: int | None = None
    max_filter_count: int | None = None
    max_complexity_score: float | None = None


@dataclass(frozen=True)
class ExpressionComplexityResult:
    expression: str
    expression_length: int
    function_count: int
    nesting_depth: int
    input_feature_count: int
    input_features: tuple[str, ...]
    operator_count: int
    filter_count: int
    complexity_score: float
    passed: bool
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "expression_length": self.expression_length,
            "function_count": self.function_count,
            "nesting_depth": self.nesting_depth,
            "input_feature_count": self.input_feature_count,
            "input_features": list(self.input_features),
            "operator_count": self.operator_count,
            "filter_count": self.filter_count,
            "complexity_score": self.complexity_score,
            "passed": self.passed,
            "violations": list(self.violations),
        }


def _walk(node: Node) -> tuple[int, int, int, set[str], int]:
    """
    Returns:
        function_count,
        max_function_nesting_depth,
        operator_count,
        input_features,
        filter_count
    """
    if isinstance(node, VarNode):
        features = {node.name} if node.name.startswith("$") else set()
        return 0, 0, 0, features, 0

    if isinstance(node, FunctionNode):
        child_stats = [_walk(arg) for arg in node.args]
        function_count = 1 + sum(item[0] for item in child_stats)
        child_depth = max((item[1] for item in child_stats), default=0)
        operator_count = sum(item[2] for item in child_stats)
        features: set[str] = set()
        for item in child_stats:
            features.update(item[3])
        filter_count = sum(item[4] for item in child_stats)
        raw_name = node.name.name if isinstance(node.name, VarNode) else str(node.name)
        name_upper = raw_name.upper()
        if name_upper in {"IF", "FILTER"}:
            filter_count += 1
        return function_count, child_depth + 1, operator_count, features, filter_count

    if isinstance(node, UnaryOpNode):
        function_count, depth, operator_count, features, filter_count = _walk(node.operand)
        return function_count, depth, operator_count + 1, features, filter_count

    if isinstance(node, BinaryOpNode):
        left = _walk(node.left)
        right = _walk(node.right)
        features = set(left[3])
        features.update(right[3])
        filter_count = left[4] + right[4]
        if node.op in {">", "<", ">=", "<=", "==", "!=", "&", "&&", "|", "||"}:
            filter_count += 1
        return (
            left[0] + right[0],
            max(left[1], right[1]),
            left[2] + right[2] + 1,
            features,
            filter_count,
        )

    if isinstance(node, ConditionalNode):
        cond = _walk(node.condition)
        true_expr = _walk(node.true_expr)
        false_expr = _walk(node.false_expr)
        features = set(cond[3])
        features.update(true_expr[3])
        features.update(false_expr[3])
        return (
            cond[0] + true_expr[0] + false_expr[0],
            max(cond[1], true_expr[1], false_expr[1]),
            cond[2] + true_expr[2] + false_expr[2] + 1,
            features,
            cond[4] + true_expr[4] + false_expr[4] + 1,
        )

    return 0, 0, 0, set(), 0


def calculate_expression_complexity(expr: str) -> ExpressionComplexityResult:
    normalized_expr = expr.strip()
    if not normalized_expr:
        raise ValueError("Expression is empty.")

    ast = parse_expression(normalized_expr)
    function_count, nesting_depth, operator_count, input_features, filter_count = _walk(ast)
    input_feature_count = len(input_features)
    expression_length = len(normalized_expr)

    complexity_score = round(
        expression_length / 40.0
        + function_count * 1.5
        + nesting_depth * 2.0
        + input_feature_count * 1.25
        + operator_count * 0.5
        + filter_count * 1.5,
        4,
    )

    return ExpressionComplexityResult(
        expression=normalized_expr,
        expression_length=expression_length,
        function_count=function_count,
        nesting_depth=nesting_depth,
        input_feature_count=input_feature_count,
        input_features=tuple(sorted(input_features)),
        operator_count=operator_count,
        filter_count=filter_count,
        complexity_score=complexity_score,
        passed=True,
        violations=(),
    )


def evaluate_expression_complexity(
    expr: str,
    *,
    budget: ExpressionComplexityBudget | None = None,
) -> ExpressionComplexityResult:
    result = calculate_expression_complexity(expr)
    if budget is None:
        return result

    violations: list[str] = []

    def check_limit(name: str, actual: int | float, limit: int | float | None) -> None:
        if limit is None:
            return
        if not isfinite(float(limit)):
            return
        if actual > limit:
            violations.append(f"{name} exceeded: {actual} > {limit}")

    check_limit("expression_length", result.expression_length, budget.max_expression_length)
    check_limit("function_count", result.function_count, budget.max_function_count)
    check_limit("nesting_depth", result.nesting_depth, budget.max_nesting_depth)
    check_limit("input_feature_count", result.input_feature_count, budget.max_input_feature_count)
    check_limit("operator_count", result.operator_count, budget.max_operator_count)
    check_limit("filter_count", result.filter_count, budget.max_filter_count)
    check_limit("complexity_score", result.complexity_score, budget.max_complexity_score)

    return ExpressionComplexityResult(
        expression=result.expression,
        expression_length=result.expression_length,
        function_count=result.function_count,
        nesting_depth=result.nesting_depth,
        input_feature_count=result.input_feature_count,
        input_features=result.input_features,
        operator_count=result.operator_count,
        filter_count=result.filter_count,
        complexity_score=result.complexity_score,
        passed=not violations,
        violations=tuple(violations),
    )
