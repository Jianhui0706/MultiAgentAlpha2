from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.tools import get_supported_runtime_functions
from trading_agents.agents.tools.factor_expression_validator import normalize_window_function_aliases
from trading_agents.agents.tools.factor_expression_validator import validate_expression_symbols


def test_validation_regression_unsupported_feature_still_fails() -> None:
    result = validate_expression_symbols(
        "RANK($volume_change_4h)",
        allowed_features={"$close", "$volume_change_1h"},
        allowed_functions=get_supported_runtime_functions(),
    )

    assert not result.passed
    assert "$volume_change_4h" in result.unsupported_features


def test_validation_regression_unsupported_function_still_fails() -> None:
    result = validate_expression_symbols(
        "FOO($close)",
        allowed_features={"$close"},
        allowed_functions=get_supported_runtime_functions(),
    )

    assert not result.passed
    assert "FOO" in result.unsupported_functions


def test_validation_regression_invalid_arity_still_fails_and_alias_rewrite_works() -> None:
    normalized = normalize_window_function_aliases("MAX($close, 5)")
    assert normalized == "TS_MAX($close, 5)"

    result = validate_expression_symbols(
        "TS_MAX($close)",
        allowed_features={"$close"},
        allowed_functions=get_supported_runtime_functions(),
    )

    assert not result.passed
    assert any("TS_MAX expected 2 args" in item for item in result.invalid_function_arities)
