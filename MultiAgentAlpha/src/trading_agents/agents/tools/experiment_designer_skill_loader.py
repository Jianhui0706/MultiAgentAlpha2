from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from trading_agents.agents.tools.factor_expression_validator import extract_expression_functions
from trading_agents.agents.tools.factor_expression_validator import get_supported_runtime_functions


_BASE_RULE_FILES = ("SKILL.md", "core_rules.md", "output_contract.md")
_HIGH_FREQUENCY_FUNCTIONS = (
    "RANK",
    "ZSCORE",
    "TS_PCTCHANGE",
    "TS_MEAN",
    "TS_STD",
    "TS_MAX",
    "TS_MIN",
    "DELAY",
)
_RELATED_FUNCTION_RULES: dict[str, tuple[str, ...]] = {
    "MAX": ("TS_MAX",),
    "MIN": ("TS_MIN",),
    "MEAN": ("TS_MEAN",),
    "STD": ("TS_STD",),
}
_MODE_LIMITS = {
    "generation": 8,
    "retry": 4,
}


@dataclass(frozen=True)
class ExperimentDesignerSkillContext:
    text: str
    metadata: dict[str, object]


class ExperimentDesignerSkillLoader:
    def __init__(self, base_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.base_dir = base_dir or (root / "skills" / "experiment_designer")
        self.function_rules_dir = self.base_dir / "function_rules"

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def _available_function_rule_names(self) -> list[str]:
        if not self.function_rules_dir.exists():
            return []
        names = [
            path.stem.upper()
            for path in self.function_rules_dir.glob("*.md")
            if path.is_file() and path.stem.strip()
        ]
        return sorted(dict.fromkeys(names))

    def _normalize_feature_name(self, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        raw = text.lstrip("$")
        if not raw:
            return ""
        return f"${raw}"

    def _normalize_feedback_items(self, feedback_items: Iterable[object] | None) -> list[str]:
        items: list[str] = []
        for item in list(feedback_items or []):
            text = str(item or "").strip()
            if text:
                items.append(text)
        return items

    def load_base_rule_blocks(self) -> dict[str, str]:
        blocks: dict[str, str] = {}
        for name in _BASE_RULE_FILES:
            path = self.base_dir / name
            blocks[name] = self._read_text(path)
        return blocks

    def build_feature_schema_block(self, domain_dataset_summary: dict[str, object]) -> str:
        raw_columns = domain_dataset_summary.get("feature_columns", [])
        if not isinstance(raw_columns, list):
            raw_columns = []
        normalized = [
            self._normalize_feature_name(item)
            for item in raw_columns
        ]
        features = sorted(dict.fromkeys(item for item in normalized if item))
        if not features:
            return "Dataset feature whitelist: not provided."
        return (
            f"Dataset feature whitelist ({len(features)} total):\n"
            + ", ".join(features)
        )

    def build_supported_functions_block(self) -> str:
        functions = sorted(get_supported_runtime_functions())
        if not functions:
            return "Supported runtime functions: not available."
        return (
            f"Supported runtime functions ({len(functions)} total):\n"
            + ", ".join(functions)
        )

    def extract_function_names(
        self,
        expressions: Iterable[object] | None,
        feedback_items: Iterable[object] | None,
    ) -> list[str]:
        found: list[str] = []
        available_rule_names = self._available_function_rule_names()
        available_set = set(available_rule_names)
        supported_functions = {name.upper() for name in get_supported_runtime_functions()}
        known_names = sorted(available_set | supported_functions | set(_RELATED_FUNCTION_RULES))

        for expression in list(expressions or []):
            text = str(expression or "").strip()
            if not text:
                continue
            for fn in sorted(extract_expression_functions(text)):
                upper = str(fn).upper()
                if upper in known_names:
                    found.append(upper)

        feedback_text = "\n".join(self._normalize_feedback_items(feedback_items)).upper()
        if feedback_text:
            for name in known_names:
                if re.search(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])", feedback_text):
                    found.append(name)

        return list(dict.fromkeys(found))

    def select_function_rule_files(
        self,
        mode: str,
        expressions: Iterable[object] | None = None,
        feedback_items: Iterable[object] | None = None,
    ) -> list[Path]:
        normalized_mode = str(mode or "generation").strip().lower()
        if normalized_mode not in _MODE_LIMITS:
            normalized_mode = "generation"

        available = {name: self.function_rules_dir / f"{name}.md" for name in self._available_function_rule_names()}
        selected_names: list[str] = []

        if normalized_mode == "generation":
            selected_names.extend(name for name in _HIGH_FREQUENCY_FUNCTIONS if name in available)
        else:
            explicit_names = self.extract_function_names(expressions, feedback_items)
            for name in explicit_names:
                if name in available:
                    selected_names.append(name)
                for related in _RELATED_FUNCTION_RULES.get(name, ()):
                    if related in available:
                        selected_names.append(related)

        deduplicated = list(dict.fromkeys(selected_names))
        limit = _MODE_LIMITS[normalized_mode]
        return [available[name] for name in deduplicated[:limit]]

    def build_skill_context(
        self,
        *,
        mode: str,
        domain_dataset_summary: dict[str, object] | None = None,
        expressions: Iterable[object] | None = None,
        feedback_items: Iterable[object] | None = None,
    ) -> ExperimentDesignerSkillContext:
        base_blocks = self.load_base_rule_blocks()
        function_rule_paths = self.select_function_rule_files(
            mode=mode,
            expressions=expressions,
            feedback_items=feedback_items,
        )
        function_rule_blocks = [
            f"## Function Rule: {path.stem}\n{self._read_text(path)}"
            for path in function_rule_paths
        ]

        feature_block = self.build_feature_schema_block(domain_dataset_summary or {})
        supported_functions_block = self.build_supported_functions_block()
        text_parts = [
            f"## {name}\n{content}" for name, content in base_blocks.items()
        ]
        text_parts.append(f"## feature_schema\n{feature_block}")
        text_parts.append(f"## supported_functions\n{supported_functions_block}")
        text_parts.extend(function_rule_blocks)
        combined_text = "\n\n".join(part.strip() for part in text_parts if part.strip())

        raw_columns = (domain_dataset_summary or {}).get("feature_columns", [])
        feature_columns = raw_columns if isinstance(raw_columns, list) else []
        normalized_features = [
            self._normalize_feature_name(item)
            for item in feature_columns
        ]
        metadata = {
            "mode": str(mode or "generation").strip().lower() or "generation",
            "base_rule_files": [str(self.base_dir / name) for name in _BASE_RULE_FILES],
            "function_rule_files": [str(path) for path in function_rule_paths],
            "function_rule_names": [path.stem for path in function_rule_paths],
            "feature_schema_count": len({item for item in normalized_features if item}),
            "supported_function_count": len(get_supported_runtime_functions()),
        }
        return ExperimentDesignerSkillContext(text=combined_text, metadata=metadata)
