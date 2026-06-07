from __future__ import annotations

from trading_agents.agents.experiment_designer_agent import ExperimentDesignerAgent
from trading_agents.agents.tools import GenerationGate


def test_evolution_novelty_gate_rejects_identical_parent_expression() -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    expression = "RANK(TS_MEAN($close, 10))"

    feedback, metrics = agent._evolution_novelty_violation(
        phase="mutation",
        mutation_type="expression_mutation",
        expression=expression,
        expression_variables={"$close"},
        parent_expressions=[expression],
        evolution_plan_summaries=[],
    )

    assert "identical" in feedback
    assert metrics["expression_similarity"] == 1.0


def test_parameter_mutation_must_change_numeric_parameter_not_structure() -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    parent = "RANK(TS_MEAN($close, 10))"

    unchanged_feedback, _ = agent._evolution_novelty_violation(
        phase="mutation",
        mutation_type="parameter_mutation",
        expression="RANK(TS_MEAN($close, 10))",
        expression_variables={"$close"},
        parent_expressions=[parent],
        evolution_plan_summaries=[],
    )
    changed_feedback, _ = agent._evolution_novelty_violation(
        phase="mutation",
        mutation_type="parameter_mutation",
        expression="RANK(TS_MEAN($close, 20))",
        expression_variables={"$close"},
        parent_expressions=[parent],
        evolution_plan_summaries=[],
    )

    assert unchanged_feedback
    assert changed_feedback == ""


def test_crossover_derived_expression_needs_two_parent_anchors() -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")

    feedback, metrics = agent._evolution_novelty_violation(
        phase="evolved",
        mutation_type="generation_branch",
        expression="RANK(TS_MEAN($close, 20))",
        expression_variables={"$close"},
        parent_expressions=[
            "RANK(TS_MEAN($close, 10))",
            "RANK(TS_STD($volume, 10))",
        ],
        evolution_plan_summaries=[{"phase": "crossover"}],
    )

    assert "at least two parents" in feedback
    assert metrics["parent_hit_count"] == 1


def test_generation_gate_enforces_profile_complexity_budget() -> None:
    quanta_gate = GenerationGate.from_profile("quantaalpha")
    long_gate = GenerationGate.from_profile("long-study")
    expression = "RANK(" + " + ".join([f"$feature_{idx}" for idx in range(5)]) + ")"

    assert quanta_gate.evaluate(expression).passed
    long_result = long_gate.evaluate(expression)

    assert not long_result.passed
    assert any("base_features" in item for item in long_result.violations)
