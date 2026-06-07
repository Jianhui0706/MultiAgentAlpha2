from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.tools import ExperimentDesignerSkillLoader


def test_skill_loader_builds_base_context_and_feature_block() -> None:
    loader = ExperimentDesignerSkillLoader()

    blocks = loader.load_base_rule_blocks()
    assert set(blocks) == {"SKILL.md", "core_rules.md", "output_contract.md"}
    assert "Workflow" in blocks["SKILL.md"]
    assert "strict JSON" in blocks["output_contract.md"]

    feature_block = loader.build_feature_schema_block({"feature_columns": ["close", "$volume"]})
    assert "$close" in feature_block
    assert "$volume" in feature_block

    context = loader.build_skill_context(
        mode="generation",
        domain_dataset_summary={"feature_columns": ["close", "volume"]},
    )
    assert context.text
    assert context.metadata["feature_schema_count"] == 2
    assert "RANK" in context.metadata["function_rule_names"]


def test_skill_loader_selects_targeted_retry_rules_with_alias_mapping() -> None:
    loader = ExperimentDesignerSkillLoader()

    names = loader.extract_function_names(
        expressions=["MAX($close, 5)"],
        feedback_items=["MAX expected 1 args, got 2"],
    )
    assert "MAX" in names

    selected = loader.select_function_rule_files(
        mode="retry",
        expressions=["MAX($close, 5)"],
        feedback_items=["MAX expected 1 args, got 2"],
    )
    selected_names = [path.stem for path in selected]
    assert selected_names == ["MAX", "TS_MAX"]


def test_skill_loader_base_rules_explicitly_block_common_illegal_outputs() -> None:
    loader = ExperimentDesignerSkillLoader()
    blocks = loader.load_base_rule_blocks()

    combined = "\n".join(blocks.values())
    assert "$volume_change_4h" in combined
    assert "$volume_change_12h" in combined
    assert "TS_PERCENTILE" in combined
