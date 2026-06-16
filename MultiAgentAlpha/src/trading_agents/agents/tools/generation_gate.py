from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_agents.agents.tools.factor_expression_ast import count_base_features
from trading_agents.agents.tools.factor_expression_ast import count_free_args


@dataclass(frozen=True)
class GenerationGateConfig:
    profile: str = "quantaalpha"
    max_symbol_length: int = 250
    max_base_features: int = 6
    max_free_args_ratio: float = 0.5


@dataclass(frozen=True)
class GenerationGateResult:
    expression: str
    passed: bool
    metrics: dict[str, Any]
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "passed": bool(self.passed),
            "metrics": dict(self.metrics),
            "violations": list(self.violations),
        }


class GenerationGate:
    """Paper-aligned hard gate for controllable factor generation."""

    def __init__(self, config: GenerationGateConfig | None = None) -> None:
        self.config = config or GenerationGateConfig()

    @classmethod
    def from_profile(cls, profile: str) -> "GenerationGate":
        normalized = str(profile or "quantaalpha").strip().lower()
        if normalized in {"long-study", "long_study", "longstudy"}:
            return cls(GenerationGateConfig(profile="long-study", max_symbol_length=200, max_base_features=4))
        return cls(GenerationGateConfig(profile=normalized or "quantaalpha"))

    def evaluate(self, expression: str) -> GenerationGateResult:
        text = str(expression or "").strip()
        violations: list[str] = []
        symbol_length = len(text)
        try:
            free_arg_count = count_free_args(text)
            base_feature_count = count_base_features(text)
        except Exception as exc:
            return GenerationGateResult(
                expression=text,
                passed=False,
                metrics={"parse_error": str(exc), "symbol_length": symbol_length},
                violations=(f"parse_error: {exc}",),
            )
        free_args_ratio = free_arg_count / max(1, symbol_length)
        if symbol_length > self.config.max_symbol_length:
            violations.append(f"symbol_length exceeded: {symbol_length} > {self.config.max_symbol_length}")
        if base_feature_count > self.config.max_base_features:
            violations.append(f"base_features exceeded: {base_feature_count} > {self.config.max_base_features}")
        if free_args_ratio >= self.config.max_free_args_ratio:
            violations.append(f"free_args_ratio exceeded: {free_args_ratio:.6f} >= {self.config.max_free_args_ratio:g}")
        return GenerationGateResult(
            expression=text,
            passed=not violations,
            metrics={
                "profile": self.config.profile,
                "symbol_length": symbol_length,
                "base_feature_count": base_feature_count,
                "free_arg_count": free_arg_count,
                "free_args_ratio": round(float(free_args_ratio), 6),
                "max_symbol_length": self.config.max_symbol_length,
                "max_base_features": self.config.max_base_features,
                "max_free_args_ratio": self.config.max_free_args_ratio,
            },
            violations=tuple(violations),
        )
