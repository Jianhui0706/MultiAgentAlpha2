#!/usr/bin/env python3
"""
Alpha Factor Mining Report Generator

This script generates a Chinese summary report from the log files of the alpha
factor mining workflow. It extracts research-report interpretation, hypothesis,
experiment design, factor expressions, metrics, and feedback from each loop,
then generates a Markdown file and optionally converts it to PDF.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine
from trading_agents.factor_runtime import calculate_factor_expression
from trading_agents.infra.factor_signal import build_factor_signal_diagnostics, build_loop_signal_diagnostics
from trading_agents.infra.run_store import RunStore

REPORT_TOP_SHARPE_FACTOR_COUNT = 6
REPORT_LAYER_PLOT_GROUPS = 5


def find_latest_log_dir() -> Path | None:
    """Find the latest log directory."""
    log_base = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop"
    if not log_base.exists():
        return None
    
    log_dirs = sorted([d for d in log_base.iterdir() if d.is_dir()], reverse=True)
    return log_dirs[0] if log_dirs else None


def load_loop_file(log_dir: Path, loop_index: int) -> dict[str, Any] | None:
    """Load a loop JSON file."""
    store_path = RunStore.default_path(PROJECT_ROOT)
    if store_path.exists():
        try:
            stored = RunStore(store_path).load_loop_compatible(run_id=log_dir.name, loop_index=loop_index)
            if stored is not None:
                return stored
        except Exception:
            pass
    loop_file = log_dir / f"loop_{loop_index:03d}.json"
    if not loop_file.exists():
        return None
    
    with loop_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_hypothesis(loop_data: dict[str, Any]) -> dict[str, Any]:
    """Extract hypothesis information from loop data."""
    final_context = loop_data.get("final_shared_context", {})
    
    hypothesis_info = {
        "hypothesis": final_context.get("hypothesis", "N/A"),
        "reasoning": final_context.get("hypothesis_reasoning", "N/A"),
        "structured": final_context.get("hypothesis_structured", {}),
    }
    
    return hypothesis_info


def extract_experiment_design(loop_data: dict[str, Any]) -> dict[str, Any]:
    """Extract experiment design information from loop data."""
    final_context = loop_data.get("final_shared_context", {})
    experiment_spec = final_context.get("experiment_spec", {})
    qlib_experiment = final_context.get("qlib_factor_experiment", {})
    
    experiment_info = {
        "experiment_id": qlib_experiment.get("experiment_id", "N/A"),
        "target_hypothesis": experiment_spec.get("target_hypothesis", "N/A"),
        "factor_names": experiment_spec.get("factor_names", []),
        "task_plan": qlib_experiment.get("task_plan", []),
        "research_stage": (
            experiment_spec.get("research_stage")
            or final_context.get("research_stage")
            or (final_context.get("feedback", {}) or {}).get("next_research_stage")
            or "N/A"
        ),
        "complexity_budget": experiment_spec.get("complexity_budget", {}),
        "complexity_diagnostics": experiment_spec.get("complexity_diagnostics", []),
        "counterfactual_factor_count": experiment_spec.get("counterfactual_factor_count", 0),
        "counterfactual_factor_target_count": experiment_spec.get("counterfactual_factor_target_count", 0),
        "factor_group_counts": experiment_spec.get("factor_group_counts", {}),
        "factor_group_targets": experiment_spec.get("factor_group_targets", {}),
        "factor_group_shortfall_diagnostics": experiment_spec.get("factor_group_shortfall_diagnostics", {}),
    }
    
    return experiment_info


def extract_backtest_metadata(loop_data: dict[str, Any]) -> dict[str, Any]:
    """Extract backtest metadata useful for report-side visualization."""
    final_context = loop_data.get("final_shared_context", {})
    qlib_experiment = final_context.get("qlib_factor_experiment", {})
    backtest_report = qlib_experiment.get("backtest_report", {})
    if not isinstance(backtest_report, dict):
        backtest_report = final_context.get("backtest_report", {})
    if not isinstance(backtest_report, dict):
        backtest_report = {}
    return {
        "panel_data_path": str(backtest_report.get("panel_data_path", "")).strip(),
        "target_column": str(backtest_report.get("target_column", "")).strip(),
        "sharpe_sampling_step": backtest_report.get("sharpe_sampling_step"),
        "sharpe_annualization_factor": backtest_report.get("sharpe_annualization_factor"),
        "panel_time_step": str(backtest_report.get("panel_time_step", "")).strip(),
    }


def extract_factor_info(loop_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract factor information from loop data."""
    final_context = loop_data.get("final_shared_context", {})
    qlib_experiment = final_context.get("qlib_factor_experiment", {})
    factors = qlib_experiment.get("factors", [])
    metrics = qlib_experiment.get("metrics", {})
    backtest_report = qlib_experiment.get("backtest_report", {})
    backtest_errors = backtest_report.get("errors", {}) if isinstance(backtest_report, dict) else {}
    
    # First check per_factor_metrics directly in final shared context
    per_factor_metrics = qlib_experiment.get("per_factor_metrics", {})
    
    if not per_factor_metrics:
        # Fallback: check records, especially backtest_runner_agent
        for record in loop_data.get("records", []):
            output_snapshot = record.get("output_snapshot", {})
            
            # Check shared_updates for per_factor_metrics
            shared_updates = output_snapshot.get("shared_updates", {})
            if shared_updates.get("per_factor_metrics"):
                per_factor_metrics = shared_updates["per_factor_metrics"]
                break
            
            # Check private_context directly
            private_context = output_snapshot.get("private_context", {})
            if "latest_per_factor_metrics" in private_context:
                per_factor_metrics = private_context["latest_per_factor_metrics"]
                break
            
            # Check private_context for latest_experiment
            latest_exp = private_context.get("latest_experiment", {})
            if latest_exp.get("latest_per_factor_metrics"):
                per_factor_metrics = latest_exp["latest_per_factor_metrics"]
                break
    
    factor_list = []
    hypothesis_structured = (
        final_context.get("hypothesis_structured", {})
        if isinstance(final_context.get("hypothesis_structured", {}), dict)
        else {}
    )
    complexity_map = {
        str(item.get("factor_name", "")): item
        for item in list((final_context.get("experiment_spec", {}) or {}).get("complexity_diagnostics", []))
        if isinstance(item, dict)
    }
    implementation_map = {
        str(item.get("factor_name", "")): item
        for item in list((qlib_experiment.get("factor_implementation", {}) or {}).get("implementations", []))
        if isinstance(item, dict)
    }
    for factor in factors:
        factor_name = factor.get("factor_name", "Unknown")
        if factor_name in per_factor_metrics:
            factor_metrics = per_factor_metrics.get(factor_name, {})
        elif not per_factor_metrics:
            factor_metrics = metrics
        else:
            factor_metrics = {}
        implementation = implementation_map.get(factor_name, {})
        failure_detail = {}
        if isinstance(backtest_errors, dict) and isinstance(backtest_errors.get(factor_name, {}), dict):
            failure_detail = dict(backtest_errors.get(factor_name, {}))
        factor_expected_direction = _expected_factor_sign_for_factor(
            hypothesis_structured,
            factor_name=factor_name,
            factor_expression=factor.get("expression", ""),
            factor_description=factor.get("description", ""),
        )
        factor_info = {
            "name": factor_name,
            "description": factor.get("description", "N/A"),
            "expression": factor.get("expression", "N/A"),
            "formulation": factor.get("formulation", "N/A"),
            "metrics": factor_metrics,
            "complexity": complexity_map.get(factor_name, {}),
            "implementation_status": implementation.get("status", ""),
            "source_pickle_path": str(implementation.get("source_pickle_path", "")).strip(),
            "signal_cache_key": str(implementation.get("signal_cache_key", "")).strip(),
            "theory_alignment": factor.get("theory_alignment", "mainline"),
            "counterfactual_reason": factor.get("counterfactual_reason", ""),
            "counterfactual_thesis": factor.get("counterfactual_thesis", ""),
            "counterfactual_mechanism": factor.get("counterfactual_mechanism", ""),
            "counterfactual_critique": factor.get("counterfactual_critique", ""),
            "counterfactual_expected_direction": factor.get("counterfactual_expected_direction", ""),
            "backtest_failure": failure_detail,
            "validation_warnings": list(factor.get("validation_warnings", []))
            if isinstance(factor.get("validation_warnings", []), list)
            else [],
            "retained_for_backtest": bool(factor.get("retained_for_backtest", True)),
            "anchor_diagnostics": {
                "matched_anchor_features": list(factor.get("matched_anchor_features", []))
                if isinstance(factor.get("matched_anchor_features", []), list)
                else [],
                "matched_anchor_pattern": bool(factor.get("matched_anchor_pattern", False)),
                "expected_anchor_features": list(factor.get("expected_anchor_features", []))
                if isinstance(factor.get("expected_anchor_features", []), list)
                else [],
                "expected_anchor_patterns": list(factor.get("expected_anchor_patterns", []))
                if isinstance(factor.get("expected_anchor_patterns", []), list)
                else [],
                "anchor_fallback_reason": factor.get("anchor_fallback_reason", ""),
            },
            "direction_diagnostics": _build_direction_diagnostics(factor_expected_direction, factor_metrics),
            "signal_diagnostics": build_factor_signal_diagnostics(factor_metrics),
        }
        factor_list.append(factor_info)
    
    return factor_list


def extract_feedback(loop_data: dict[str, Any]) -> dict[str, Any]:
    """Extract feedback information from loop data."""
    final_context = loop_data.get("final_shared_context", {})
    qlib_experiment = final_context.get("qlib_factor_experiment", {})
    feedback = final_context.get("feedback", {}) or qlib_experiment.get("feedback", {})
    
    feedback_info = {
        "decision": feedback.get("decision", "N/A"),
        "summary": feedback.get("summary", "N/A"),
        "observations": feedback.get("observations", "N/A"),
        "reason": feedback.get("reason", "N/A"),
        "new_hypothesis": feedback.get("new_hypothesis", ""),
        "hypothesis_supported": feedback.get("hypothesis_supported", "N/A"),
        "replace_best_result": feedback.get("replace_best_result", "N/A"),
        "semantic_diagnosis": feedback.get("semantic_diagnosis", "N/A"),
        "semantic_diagnosis_reason": feedback.get("semantic_diagnosis_reason", "N/A"),
        "next_research_stage": feedback.get("next_research_stage", "N/A"),
    }
    
    return feedback_info


def extract_evolution_context(loop_data: dict[str, Any]) -> dict[str, Any]:
    """Extract genetic iteration metadata from loop final shared context."""
    final_context = loop_data.get("final_shared_context", {})
    qlib_experiment = final_context.get("qlib_factor_experiment", {})
    phase = str(final_context.get("phase", qlib_experiment.get("phase", "")))
    round_idx = int(final_context.get("round_idx", qlib_experiment.get("round_idx", 0)) or 0)
    paper_iteration = _paper_iteration_from_round(round_idx)
    return {
        "phase": phase,
        "round_idx": round_idx,
        "paper_iteration": paper_iteration,
        "paper_phase": _paper_phase_label(phase, round_idx),
        "direction_id": int(final_context.get("direction_id", 0) or 0),
        "direction": str(final_context.get("direction", "")),
        "trajectory_id": str(final_context.get("trajectory_id", qlib_experiment.get("trajectory_id", ""))),
        "parent_ids": [str(item) for item in list(final_context.get("parent_ids", qlib_experiment.get("parent_ids", [])))],
        "mutation_type": str(final_context.get("mutation_type", "")),
        "evolution_planner_mode": str(final_context.get("evolution_planner_mode", "")),
        "evolution_seed": str(final_context.get("evolution_seed", "")),
        "evolution_guidance": str(final_context.get("evolution_guidance", "")),
        "evolution_plan_summaries": [
            dict(item)
            for item in list(final_context.get("evolution_plan_summaries", []))
            if isinstance(item, dict)
        ],
    }


def _paper_iteration_from_round(round_idx: Any) -> int:
    try:
        value = int(round_idx)
    except Exception:
        return 0
    if value <= 0:
        return 0
    return (value + 1) // 2


def _paper_phase_label(phase: Any, round_idx: Any) -> str:
    phase_text = str(phase or "").strip().lower()
    if phase_text == "mutation":
        return "Mutation phase"
    if phase_text == "crossover":
        return "Crossover phase"
    if phase_text == "original" or _paper_iteration_from_round(round_idx) == 0:
        return "Initialization"
    return phase_text or "N/A"


def _format_paper_iteration_label(item: dict[str, Any]) -> str:
    iteration = int(item.get("paper_iteration", _paper_iteration_from_round(item.get("round_idx", 0))) or 0)
    phase = _paper_phase_label(item.get("phase", ""), item.get("round_idx", 0))
    if iteration <= 0 or phase == "Initialization":
        return "Initialization"
    return f"Iteration {iteration} / {phase}"


def _planned_task_key(item: dict[str, Any]) -> str:
    trajectory_id = str(item.get("trajectory_id", "")).strip()
    if trajectory_id:
        return trajectory_id
    return json.dumps(
        {
            "phase": item.get("phase", ""),
            "direction": item.get("direction", ""),
            "parent_ids": item.get("parent_ids", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _add_planned_task(bucket: dict[str, Any], item: dict[str, Any]) -> None:
    phase = str(item.get("phase", "")).strip()
    if phase not in {"mutation", "crossover"}:
        return
    seen = bucket.setdefault("_planned_task_keys", set())
    key = f"{phase}:{_planned_task_key(item)}"
    if key in seen:
        return
    seen.add(key)
    bucket[f"{phase}_tasks"].append(
        {
            "loop_index": "",
            "trajectory_id": str(item.get("trajectory_id", "")),
            "round_idx": int((list(bucket.get("round_indices", [])) or [0])[0]),
            "paper_iteration": int(bucket.get("paper_iteration", 0) or 0),
            "paper_phase": phase,
            "direction_id": int(item.get("direction_id", 0) or 0),
            "direction": str(item.get("direction", "")),
            "parent_ids": [str(parent_id) for parent_id in list(item.get("parent_ids", []))],
            "mutation_type": str(item.get("mutation_type", "")),
            "evolution_planner_mode": str(item.get("planner_mode", item.get("evolution_planner_mode", ""))),
            "evolution_seed": "",
            "evolution_guidance": "",
            "rationale": str(item.get("rationale", "")),
            "fusion_logic": str(item.get("fusion_logic", "")),
            "evolution_plan_summaries": [],
        }
    )


def summarize_evolution_from_loops(all_loops: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact planning summary from loop files."""
    initial_directions: list[dict[str, Any]] = []
    plans_by_round: dict[int, dict[str, Any]] = {}
    for loop in all_loops:
        evolution = dict(loop.get("evolution", {}))
        phase = str(evolution.get("phase", ""))
        if not phase:
            continue
        row = {
            "loop_index": loop["index"],
            "trajectory_id": evolution.get("trajectory_id", ""),
            "round_idx": evolution.get("round_idx", 0),
            "paper_iteration": evolution.get("paper_iteration", _paper_iteration_from_round(evolution.get("round_idx", 0))),
            "paper_phase": evolution.get("paper_phase", _paper_phase_label(phase, evolution.get("round_idx", 0))),
            "direction_id": evolution.get("direction_id", 0),
            "direction": evolution.get("direction", ""),
            "parent_ids": evolution.get("parent_ids", []),
            "mutation_type": evolution.get("mutation_type", ""),
            "evolution_planner_mode": evolution.get("evolution_planner_mode", ""),
            "evolution_seed": evolution.get("evolution_seed", ""),
            "evolution_guidance": evolution.get("evolution_guidance", ""),
            "evolution_plan_summaries": list(evolution.get("evolution_plan_summaries", []) or []),
        }
        if phase == "original":
            row["direction_decomposer_mode"] = row["evolution_planner_mode"]
            initial_directions.append(row)
            continue
        paper_iteration = int(row.get("paper_iteration", 0) or 0)
        bucket = plans_by_round.setdefault(
            paper_iteration,
            {
                "paper_iteration": paper_iteration,
                "round_indices": set(),
                "mutation_tasks": [],
                "crossover_tasks": [],
                "evolved_tasks": [],
                "_planned_task_keys": set(),
            },
        )
        bucket["round_indices"].add(int(row["round_idx"] or 0))
        if phase == "evolved" or (row["mutation_type"] == "generation_branch" and row["evolution_plan_summaries"]):
            bucket["evolved_tasks"].append(row)
            for plan_item in row["evolution_plan_summaries"]:
                if isinstance(plan_item, dict):
                    _add_planned_task(bucket, plan_item)
        elif phase == "mutation":
            bucket["mutation_tasks"].append(row)
        elif phase == "crossover":
            bucket["crossover_tasks"].append(row)
    evolution_agent_plans = [
        {
            **{
                key: (sorted(value) if key == "round_indices" and isinstance(value, set) else value)
                for key, value in bucket.items()
                if key != "_planned_task_keys"
            },
            "planned_mutation_count": len(bucket["mutation_tasks"]),
            "planned_crossover_count": len(bucket["crossover_tasks"]),
            "executed_evolved_count": len(bucket["evolved_tasks"]),
        }
        for _paper_iteration, bucket in sorted(plans_by_round.items())
    ]
    return {
        "initial_direction_count": len(initial_directions),
        "initial_directions": initial_directions,
        "planned_mutation_count": sum(plan["planned_mutation_count"] for plan in evolution_agent_plans),
        "planned_crossover_count": sum(plan["planned_crossover_count"] for plan in evolution_agent_plans),
        "executed_evolved_count": sum(plan["executed_evolved_count"] for plan in evolution_agent_plans),
        "evolution_agent_plans": evolution_agent_plans,
    }


def extract_loop_runtime_status(loop_data: dict[str, Any]) -> dict[str, Any]:
    final_context = loop_data.get("final_shared_context", {}) or {}
    records = loop_data.get("records", []) or []
    failed_record = next((record for record in reversed(records) if record.get("error")), None)

    stale_reused = False
    final_exp = (final_context.get("qlib_factor_experiment") or {})
    round_idx = final_exp.get("round_idx")
    if failed_record is not None and isinstance(round_idx, int):
        try:
            loop_index = int(Path(loop_data.get("__loop_file__", "loop_000.json")).stem.split("_")[-1])
            stale_reused = round_idx != loop_index
        except Exception:
            stale_reused = False

    return {
        "round_status": final_context.get("round_status", "completed" if failed_record is None else "failed_incomplete"),
        "failed_agent_name": final_context.get("failed_agent_name") or (failed_record.get("agent_name") if failed_record else ""),
        "failed_error": final_context.get("failed_error") or (str(failed_record.get("error")) if failed_record else ""),
        "stale_results_removed": bool(final_context.get("stale_results_removed", False)),
        "stale_result_keys_removed": list(final_context.get("stale_result_keys_removed", [])),
        "stale_results_reused_for_reference": stale_reused,
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics for display."""
    if not metrics:
        return "N/A"
    
    lines = []
    key_order = ["IC", "ICIR", "Rank IC", "Rank ICIR", "coverage", "turnover", "sharpe", "drawdown", "long_short_positive_ratio", "long_short_mean_return", "layer_monotonicity"]
    label_map = {
        "coverage": "覆盖率",
        "turnover": "换手率",
        "sharpe": "夏普",
        "drawdown": "回撤",
        "long_short_positive_ratio": "多空正收益占比",
        "long_short_mean_return": "多空平均收益",
        "layer_monotonicity": "分层单调性",
        "layer_1_mean_return": "第1层平均收益",
        "layer_2_mean_return": "第2层平均收益",
        "layer_3_mean_return": "第3层平均收益",
        "layer_4_mean_return": "第4层平均收益",
        "layer_5_mean_return": "第5层平均收益",
        "IC_abs": "IC绝对值",
        "IC_sign": "IC方向",
        "Rank IC_abs": "Rank IC绝对值",
        "Rank IC_sign": "Rank IC方向",
        "ICIR_abs": "ICIR绝对值",
        "sharpe_abs": "夏普绝对值",
    }
    
    for key in key_order:
        if key in metrics:
            value = metrics[key]
            label = label_map.get(key, key)
            if isinstance(value, float):
                lines.append(f"- **{label}**: {value:.6f}")
            else:
                lines.append(f"- **{label}**: {value}")
    
    for key, value in metrics.items():
        if key not in key_order:
            label = label_map.get(key, key)
            if isinstance(value, float):
                lines.append(f"- **{label}**: {value:.6f}")
            else:
                lines.append(f"- **{label}**: {value}")
    
    return "\n".join(lines) if lines else "N/A"


def _safe_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text if text else "N/A"


def _extract_research_report_payload(log_dir: Path) -> dict[str, Any]:
    sample_report = load_loop_file(log_dir, 1)
    if sample_report is None:
        return {}
    payload = (
        sample_report.get("final_shared_context", {})
        .get("research_report", {})
    )
    return payload if isinstance(payload, dict) else {}


def _extract_named_section(text: str, heading: str, *, max_lines: int = 4) -> list[str]:
    lines = str(text or "").splitlines()
    selected: list[str] = []
    active = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if active and selected:
                break
            continue
        if stripped.rstrip(":") == heading.rstrip(":"):
            active = True
            continue
        if active and not stripped.startswith("-") and stripped.endswith(":"):
            break
        if active:
            selected.append(stripped)
            if len(selected) >= max_lines:
                break
    return selected


def _format_research_report_source_summary(research_report: dict[str, Any]) -> str:
    source_reports = research_report.get("source_reports", [])
    if isinstance(source_reports, list) and source_reports:
        names = [
            str(item.get("file_name", "") or item.get("file_path", "")).strip()
            for item in source_reports
            if isinstance(item, dict) and str(item.get("file_name", "") or item.get("file_path", "")).strip()
        ]
        return ", ".join(names) if names else "combined_reports"
    source = research_report.get("source", {})
    if isinstance(source, dict):
        return str(source.get("file_name", "")).strip()
    return ""


def build_multi_report_summary_cn(research_report: dict[str, Any]) -> list[str]:
    source_reports = research_report.get("source_reports", [])
    if not isinstance(source_reports, list) or len(source_reports) <= 1:
        return []
    synthesis = research_report.get("multi_report_synthesis", {})
    if not isinstance(synthesis, dict):
        synthesis = {}
    report_text = str(research_report.get("report_text", ""))
    conflicts = _extract_named_section(report_text, "冲突或需要保留的限定观点", max_lines=5)
    if not conflicts:
        conflicts = _extract_named_section(report_text, "Conflicting or caveated views", max_lines=5)
    transferable = _extract_named_section(report_text, "可迁移到 crypto 的逻辑", max_lines=3)
    if not transferable:
        transferable = _extract_named_section(report_text, "Transferable logic to crypto", max_lines=3)
    proxies = list(synthesis.get("common_proxy_candidates", [])) if isinstance(synthesis.get("common_proxy_candidates", []), list) else []
    lines = [
        "### 0.1 多研报合成摘要",
        "",
        f"- **输入研报数量**: {len(source_reports)}",
        f"- **合成格式**: {_safe_text(synthesis.get('format') or research_report.get('multi_report_synthesis_status'))}",
    ]
    names = [
        str(item.get("file_name", "") or item.get("file_path", "")).strip()
        for item in source_reports
        if isinstance(item, dict) and str(item.get("file_name", "") or item.get("file_path", "")).strip()
    ]
    if names:
        lines.append(f"- **研报列表**: {', '.join(names)}")
    if proxies:
        lines.append(f"- **共同代理候选**: {', '.join(str(item) for item in proxies[:8])}")
    if conflicts:
        lines.append(f"- **冲突/保留意见**: {' '.join(conflicts)}")
    if transferable:
        lines.append(f"- **可迁移逻辑**: {' '.join(transferable)}")
    return lines


def _clean_family_name(item: str) -> str:
    text = str(item or "").strip()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return text or "未命名代理"


def _format_family_cn(item: str) -> str:
    mapping = {
        "price_momentum_divergence": "价格动量背离",
        "momentum_non_confirmation": "动量不确认",
        "returns_deceleration": "收益减速",
        "spread_stress": "价差压力",
        "volume_shrinking_confirmation": "量能收缩确认",
        "vwap_deviation": "VWAP 偏离",
        "momentum_exhaustion_divergence": "动量衰竭背离",
        "multi_horizon_divergence": "多周期背离",
        "spread_stress_indicator": "价差压力指示",
        "volume_regime_filter": "成交量状态过滤",
        "price_acceleration": "价格加速度",
        "volume_non_confirmation_score": "量能不确认强度",
        "cross_sectional_rank": "横截面排序",
        "momentum_divergence": "动量背离",
    }
    raw = _clean_family_name(item)
    return mapping.get(raw, raw.replace("_", " "))


def _format_non_transferable_cn(item: str) -> str:
    text = str(item or "").strip()
    replacements = {
        "K-line containing relationship processing and stroke identification methodology": "K 线包含关系与笔划识别方法",
        "A-share specific trading session patterns": "A 股特有的分时与交易时段结构",
        "Industry index classification and sector rotation dynamics": "行业指数分类与板块轮动框架",
        "Confirmation-delay waiting period approach (wait for next peak/trough)": "等待下一确认高低点的延迟确认法",
        "K-line '包含关系' processing methodology (specific to Chinese candlestick patterns)": "K 线“包含关系”处理方法",
        "'顶分型/底分型' pivot identification rules (domain-specific)": "顶分型/底分型识别规则",
        "'向上一笔/向下一笔' stroke definitions (heavily dependent on session boundaries)": "向上一笔/向下一笔的笔划定义",
        "The 65% win rate statistic (A-share specific microstructure)": "65% 胜率这一 A 股特定统计结论",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _format_task_plan_cn(task_plan: list[str], factor_names: str = "无因子") -> str:
    if not task_plan:
        return factor_names
    text = " ".join(str(x).strip() for x in task_plan if str(x).strip()).lower()
    snippets: list[str] = []
    if "divergence" in text or "price-volume" in text:
        snippets.append("复现量价背离主信号")
    if "non-confirmation" in text:
        snippets.append("补充动量不确认代理")
    if "deceleration" in text:
        snippets.append("构造收益减速代理")
    if "normalization" in text or "comparability" in text or "z-score" in text or "rank" in text:
        snippets.append("统一做横截面标准化")
    if "window" in text or "threshold" in text:
        snippets.append("调整窗口与阈值参数")
    if "volatility" in text:
        snippets.append("加入波动率归一化")
    if "spread" in text:
        snippets.append("增加价差压力确认")
    if "filter" in text or "regime" in text:
        snippets.append("增加状态过滤条件")
    if not snippets:
        return factor_names
    deduped: list[str] = []
    for s in snippets:
        if s not in deduped:
            deduped.append(s)
    return "；".join(deduped[:4])


def _format_stage_cn(stage: Any) -> str:
    mapping = {
        "base_reconstruction": "基础复现",
        "parameter_tuning": "参数调优",
        "resonance_filtering": "共振过滤",
        "thesis_revision": "假设修正",
        "free_exploration": "自由探索",
        "N/A": "未标注",
        None: "未标注",
    }
    return mapping.get(stage, str(stage))


def _format_expected_direction_cn(direction: Any) -> str:
    mapping = {
        "positive_forward_return": "未来24小时正收益",
        "negative_forward_return": "未来24小时负收益",
        "positive": "正相关",
        "negative": "负相关",
        "zero": "零",
        "unknown": "未知",
        "depends_on_encoding": "取决于因子编码",
        "mixed_or_conditional": "条件性方向",
        "N/A": "未标注",
        None: "未标注",
    }
    return mapping.get(direction, str(direction))


def _format_diag_cn(diag: Any) -> str:
    mapping = {
        "thesis_invalid": "核心假设失效",
        "proxy_weak": "代理变量偏弱",
        "parameter_not_tuned": "参数尚未调优到位",
        "implementation_unstable": "实现或回测不稳定",
        "direction_mismatch": "信号方向不匹配",
        "N/A": "未标注",
        None: "未标注",
    }
    return mapping.get(diag, str(diag))


def _localize_hypothesis_cn(text: Any) -> str:
    """Translate common hypothesis wording into Chinese for the final report summary."""
    raw = _safe_text(text)
    lowered = raw.lower()
    if "price momentum significantly outpaces volume momentum" in lowered:
        return "当近期价格动量显著强于成交量动量并形成背离时，价格上涨可能缺乏真实成交确认，代表短期动量衰竭，因而预期未来收益偏负。"
    if "multi-period price momentum deceleration" in lowered and "quote-volume deceleration" in lowered:
        return "多周期价格动量减速经过已实现波动率归一化后，再结合报价成交量减速确认，可以刻画量价信念背离下的动量衰竭，并预期加密资产未来收益偏负。"
    if "high cross-sectional price momentum rank" in lowered and "below-median volume participation rank" in lowered:
        return "横截面价格动量排名较高、但成交参与度排名低于中位数的资产，可能代表缺乏成交支撑的拥挤价格推动，预期未来收益偏负；相反，低价格动量但高成交参与度可能代表更有信念的反转机会，预期未来收益偏正。"

    replacements = [
        ("crypto markets", "加密资产市场"),
        ("assets exhibiting", "表现出"),
        ("recent price momentum", "近期价格动量"),
        ("price momentum", "价格动量"),
        ("volume momentum", "成交量动量"),
        ("quote-volume", "报价成交量"),
        ("volume participation", "成交参与度"),
        ("cross-sectional", "横截面"),
        ("rank", "排名"),
        ("divergence", "背离"),
        ("weak conviction", "信念不足"),
        ("short-term exhaustion", "短期衰竭"),
        ("negative forward returns", "未来收益偏负"),
        ("positive forward returns", "未来收益偏正"),
        ("forward returns", "未来收益"),
        ("realized volatility", "已实现波动率"),
        ("normalized by", "经过归一化"),
        ("combined with", "结合"),
        ("predicts", "预示"),
        ("predicting", "预示"),
        ("while", "而"),
    ]
    translated = raw
    for src, dst in replacements:
        translated = translated.replace(src, dst).replace(src.title(), dst)
    return translated


def _localize_revision_reason_cn(text: Any) -> str:
    raw = _safe_text(text)
    if raw == "N/A":
        return raw
    if raw.startswith("Counterfactual group is moderately stronger than mainline"):
        metric = ""
        marker = "best |Rank IC|="
        if marker in raw:
            metric = raw.split(marker, 1)[1].split(")", 1)[0].strip()
        metric_part = f"（最佳 |Rank IC|={metric}）" if metric else ""
        return f"理论外对照组中等程度强于理论内主线{metric_part}；在翻转核心假设前，应优先扩大并重构代理变量。"
    if raw.startswith("Counterfactual group is strong"):
        return "理论外对照组在组级表现上显著强于理论内主线，提示当前假设可能需要进行 thesis-level 修正。"
    replacements = {
        "Counterfactual group": "理论外对照组",
        "mainline": "理论内主线",
        "broaden proxy reconstruction before flipping the thesis": "在翻转假设前先扩大并重构代理变量",
        "persistent noise": "持续噪声",
        "hypothesis replacement": "替换假设",
    }
    translated = raw
    for src, dst in replacements.items():
        translated = translated.replace(src, dst)
    return translated


def _format_challenge_key_cn(value: Any) -> str:
    return {
        "none": "无挑战",
        "weak": "弱挑战",
        "moderate": "中等挑战",
        "strong": "强挑战",
    }.get(str(value), str(value))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_expected_factor_return_sign(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    aliases = {
        "negative_forward_return": "negative",
        "positive_forward_return": "positive",
        "negative": "negative",
        "positive": "positive",
        "depends_on_encoding": "depends_on_encoding",
        "mixed_or_conditional": "depends_on_encoding",
        "conditional": "depends_on_encoding",
    }
    if text in aliases:
        return aliases[text]
    has_negative = "negative" in text
    has_positive = "positive" in text
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    if has_negative and has_positive:
        return "depends_on_encoding"
    return text


def _canonical_expected_factor_return_sign(_value: Any = None) -> dict[str, str]:
    return {
        "top_divergence_strength": "negative",
        "bottom_divergence_strength": "positive",
        "mixed_divergence_score": "depends_on_encoding",
    }


def _expected_factor_sign_from_structured(structured: dict[str, Any]) -> str:
    raw = structured.get("expected_factor_return_sign")
    if isinstance(raw, dict):
        canonical = _canonical_expected_factor_return_sign(raw)
        for key in ("default", "primary", "expected"):
            normalized = _normalize_expected_factor_return_sign(raw.get(key))
            if normalized in {"negative", "positive", "depends_on_encoding"}:
                return normalized
        if any(key in raw for key in canonical):
            return "depends_on_encoding"
    normalized = _normalize_expected_factor_return_sign(raw)
    if normalized in {"negative", "positive", "depends_on_encoding"}:
        return normalized
    return _normalize_expected_factor_return_sign(structured.get("expected_direction"))


def _expected_factor_sign_for_factor(
    structured: dict[str, Any],
    *,
    factor_name: str = "",
    factor_expression: str = "",
    factor_description: str = "",
) -> str:
    raw = structured.get("expected_factor_return_sign")
    if not isinstance(raw, dict):
        return _expected_factor_sign_from_structured(structured)

    canonical = _canonical_expected_factor_return_sign(raw)
    context = " ".join([factor_name, factor_expression, factor_description]).lower()
    category_keywords = (
        (
            "top_divergence_strength",
            ("top", "顶部", "顶背离", "bearish", "overbought", "exhaustion", "deceleration"),
        ),
        (
            "bottom_divergence_strength",
            ("bottom", "底部", "底背离", "bullish", "oversold", "rebound"),
        ),
        (
            "mixed_divergence_score",
            ("mixed", "composite", "combo", "combined", "ratio", "spread"),
        ),
    )
    for category, keywords in category_keywords:
        if any(keyword in context for keyword in keywords):
            normalized = _normalize_expected_factor_return_sign(canonical.get(category))
            if normalized in {"negative", "positive", "depends_on_encoding"}:
                return normalized
    return _expected_factor_sign_from_structured(structured)


def _rank_ic_sign(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def _build_direction_diagnostics(expected_direction: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    expected = _normalize_expected_factor_return_sign(expected_direction)
    actual = _rank_ic_sign(metrics.get("Rank IC"))
    if actual not in {"positive", "negative"}:
        aligned = None
    elif expected == "positive":
        aligned = actual == "positive"
    elif expected == "negative":
        aligned = actual == "negative"
    else:
        aligned = None
    recommended_usage = "as_is"
    if aligned is False and actual in {"positive", "negative"} and expected in {"positive", "negative"}:
        recommended_usage = "negate_factor"
    return {
        "expected_direction": expected,
        "actual_rank_ic_sign": actual,
        "direction_aligned": aligned,
        "recommended_usage": recommended_usage,
    }


def _format_direction_alignment_cn(value: Any) -> str:
    if value is True:
        return "符合"
    if value is False:
        return "不符合"
    return "无法判断"


def _format_metric_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return f"{value:.4f}"
    return str(value)


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _mermaid_safe(text: str) -> str:
    cleaned = str(text).replace('"', "'").replace("\n", " ")
    cleaned = cleaned.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return _truncate(cleaned, 56)


def build_evolution_mermaid(all_loops: list[dict[str, Any]], *, has_research_report: bool = True) -> str:
    if not all_loops:
        return ""

    first_structured = all_loops[0]["hypothesis"].get("structured", {})
    proxy = first_structured.get("proxy_reconstruction_plan", {}) or {}
    core = [_format_family_cn(x) for x in proxy.get("core_proxy_family", [])[:3]]
    supporting = [_format_family_cn(x) for x in proxy.get("supporting_proxy_family", [])[:2]]
    direction = _format_expected_direction_cn(first_structured.get("expected_direction"))
    default_thesis = "围绕研报主线构建横截面信号" if has_research_report else "围绕市场背景构建横截面信号"
    thesis = _mermaid_safe(" / ".join(core) if core else default_thesis)
    mechanism = _mermaid_safe(f"通过{' / '.join(supporting) if supporting else '量价关系'}识别趋势衰竭与背离")
    crypto_rewrite = _mermaid_safe(f"在加密横截面中验证{direction}")

    lines = [
        "```mermaid",
        "flowchart TD",
        '    A["PDF 研报输入"]' if has_research_report else '    A["市场背景与数据摘要输入"]',
        f'    B["提取核心论点: {thesis}"]' if has_research_report else f'    B["形成初始研究方向: {thesis}"]',
        f'    C["机制提炼: {mechanism}"]',
        f'    D["Crypto 迁移: {crypto_rewrite}"]',
        '    E["代理重建方案"]',
        "    A --> B --> C --> D --> E",
    ]
    for loop in all_loops:
        exp = loop["experiment_design"]
        factors = loop["factors"]
        stage = _format_stage_cn(loop.get("current_research_stage", exp.get("research_stage", "N/A")))
        task_plan = exp.get("task_plan", [])
        factor_names = ", ".join(f["name"] for f in factors[:3]) if factors else "无因子"
        lines.append(
            f'    E --> L{loop["index"]}["第 {loop["index"]} 轮 | {stage} | {_mermaid_safe(_format_task_plan_cn(task_plan, factor_names))}"]'
        )
    lines.append("```")
    return "\n".join(lines)


def infer_current_research_stage(all_loops: list[dict[str, Any]], idx: int) -> str:
    """Infer the current loop stage from prior-loop transition state."""
    loop = all_loops[idx]
    explicit_stage = loop["experiment_design"].get("research_stage")
    if explicit_stage and explicit_stage != "N/A":
        return explicit_stage

    if idx > 0:
        prev_next = all_loops[idx - 1]["feedback"].get("next_research_stage")
        if prev_next and prev_next != "N/A":
            return str(prev_next)

    structured = loop["hypothesis"].get("structured", {}) or {}
    if structured.get("thesis") or structured.get("proxy_reconstruction_plan"):
        return "base_reconstruction"
    return "free_exploration"


def summarize_best_factor(loop: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = None
    for factor in loop["factors"]:
        metrics = factor.get("metrics", {})
        rank_ic = _to_float(metrics.get("Rank IC"))
        if rank_ic is None:
            continue
        expected = (factor.get("direction_diagnostics", {}) or {}).get("expected_direction")
        if expected == "negative":
            score = -rank_ic
        elif expected == "depends_on_encoding":
            score = abs(rank_ic)
        else:
            score = rank_ic
        if best_score is None or score > best_score:
            best_score = score
            best = factor
    return best


def collect_top_sharpe_factors(all_loops: list[dict[str, Any]], top_k: int = REPORT_TOP_SHARPE_FACTOR_COUNT) -> list[dict[str, Any]]:
    """Collect top-k factors ranked by net final direction-aware Sharpe."""
    candidates: list[dict[str, Any]] = []
    for loop in all_loops:
        runtime_status = loop.get("runtime_status", {}) or {}
        if runtime_status.get("stale_results_reused_for_reference"):
            continue
        for factor in loop.get("factors", []):
            metrics = factor.get("metrics", {}) or {}
            sharpe = _to_float(metrics.get("net_sharpe"))
            if sharpe is None:
                sharpe = _to_float(metrics.get("sharpe"))
            if sharpe is None:
                continue
            gross_sharpe = _to_float(metrics.get("gross_sharpe"))
            candidates.append(
                {
                    "loop_index": loop.get("index"),
                    "factor": factor,
                    "sharpe": sharpe,
                    "gross_sharpe": gross_sharpe,
                    "rank_ic_abs": abs(_to_float(metrics.get("Rank IC")) or 0.0),
                }
            )
    candidates.sort(key=lambda item: (item["sharpe"], item["rank_ic_abs"]), reverse=True)
    return candidates[:top_k]


def summarize_alignment_groups(loop: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {
        "mainline": {"count": 0, "best": None},
        "counterfactual": {"count": 0, "best": None},
    }
    for factor in loop["factors"]:
        alignment = str(factor.get("theory_alignment", "mainline") or "mainline")
        if alignment not in groups:
            groups[alignment] = {"count": 0, "best": None}
        groups[alignment]["count"] = int(groups[alignment]["count"]) + 1
        metrics = factor.get("metrics", {})
        rank_ic = _to_float(metrics.get("Rank IC"))
        if rank_ic is None:
            continue
        best = groups[alignment].get("best")
        expected = (factor.get("direction_diagnostics", {}) or {}).get("expected_direction")
        score = -rank_ic if expected == "negative" else abs(rank_ic) if expected == "depends_on_encoding" else rank_ic
        if isinstance(best, dict):
            best_rank_ic = _to_float((best or {}).get("metrics", {}).get("Rank IC"))
            best_expected = (best.get("direction_diagnostics", {}) or {}).get("expected_direction")
            best_score = (
                -best_rank_ic
                if best_expected == "negative" and best_rank_ic is not None
                else abs(best_rank_ic)
                if best_expected == "depends_on_encoding" and best_rank_ic is not None
                else best_rank_ic
            )
        else:
            best_score = None
        if best is None or best_score is None or score > best_score:
            groups[alignment]["best"] = factor
    return groups


def format_alignment_cn(value: Any) -> str:
    return "理论外对照" if str(value or "mainline") == "counterfactual" else "理论内主线"


def _format_signal_strength_cn(value: Any) -> str:
    mapping = {
        "noise": "噪声",
        "weak": "弱信号",
        "moderate": "中等",
        "strong": "强",
        "stronger": "更强",
        "unknown": "未知",
    }
    return mapping.get(str(value or "unknown"), str(value or "未知"))


def _format_signal_action_cn(value: Any) -> str:
    mapping = {
        "replace_hypothesis_if_persistent": "若长期持续则换假设",
        "revise_theory_or_proxy": "修改理论/代理",
        "tune_parameters": "更换参数",
        "strengthen_hypothesis": "强化假设",
        "insufficient_data": "数据不足",
    }
    return mapping.get(str(value or "insufficient_data"), str(value or "数据不足"))


def _format_bool_cn(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "无法判断"


def _short_cn_text(value: Any, limit: int = 70) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "N/A"
    return text if len(text) <= limit else f"{text[:limit]}…"


def _md_table_cell(value: Any, limit: int | None = None) -> str:
    text = _short_cn_text(value, limit) if limit is not None else str(value if value is not None else "N/A")
    return text.replace("|", "\\|").replace("\n", " ")


def build_hypothesis_evolution_rows(all_loops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_signal_diagnostics: list[dict[str, Any]] = []
    for loop in all_loops:
        factors_for_signal = [
            {
                "factor_name": factor.get("name"),
                "theory_alignment": factor.get("theory_alignment", "mainline"),
            }
            for factor in loop.get("factors", [])
        ]
        per_factor_metrics = {
            str(factor.get("name")): dict(factor.get("metrics", {}))
            for factor in loop.get("factors", [])
            if isinstance(factor.get("metrics", {}), dict)
        }
        signal_diagnostics = build_loop_signal_diagnostics(
            factors=factors_for_signal,
            per_factor_metrics=per_factor_metrics,
            previous_loop_diagnostics=previous_signal_diagnostics,
        )
        previous_signal_diagnostics.append(signal_diagnostics)
        loop["signal_diagnostics"] = signal_diagnostics
        best = summarize_best_factor(loop)
        structured = loop["hypothesis"].get("structured", {}) or {}
        hypothesis_text = structured.get("hypothesis") or loop["hypothesis"].get("hypothesis")
        rows.append(
            {
                "loop": loop.get("index"),
                "stage": loop.get("current_research_stage", loop["experiment_design"].get("research_stage", "N/A")),
                "hypothesis": hypothesis_text,
                "best_factor": (best or {}).get("name", "N/A") if isinstance(best, dict) else "N/A",
                "best_rank_ic": ((best or {}).get("metrics", {}) or {}).get("Rank IC") if isinstance(best, dict) else None,
                "evaluated_factor_count": signal_diagnostics.get("evaluated_factor_count"),
                "weak_signal_count": signal_diagnostics.get("weak_signal_count"),
                "majority_weak_signal": signal_diagnostics.get("majority_weak_signal"),
                "direction_stable": signal_diagnostics.get("direction_stable"),
                "recommended_action": signal_diagnostics.get("recommended_action"),
                "revision_reason": signal_diagnostics.get("revision_reason"),
            }
        )
    return rows


def build_hypothesis_evolution_summary_cn(all_loops: list[dict[str, Any]]) -> list[str]:
    """Build a narrative Chinese summary of how the hypothesis evolved."""
    if not all_loops:
        return ["暂无可用于总结的 loop 结果。"]

    lines: list[str] = []
    lines.append("### 假设演化总结")
    lines.append("")

    for loop in all_loops:
        loop_idx = loop.get("index", "N/A")
        structured = loop.get("hypothesis", {}).get("structured", {}) or {}
        hypothesis_text = structured.get("hypothesis") or loop.get("hypothesis", {}).get("hypothesis") or "未记录"
        stage = _format_stage_cn(loop.get("current_research_stage", loop.get("experiment_design", {}).get("research_stage", "N/A")))
        best = summarize_best_factor(loop)
        best_name = (best or {}).get("name", "N/A") if isinstance(best, dict) else "N/A"
        best_rank_ic = ((best or {}).get("metrics", {}) or {}).get("Rank IC") if isinstance(best, dict) else None
        signal = loop.get("signal_diagnostics", {}) or {}
        group_perf = signal.get("group_performance", {}) or {}
        main = group_perf.get("mainline", {}) or {}
        counter = group_perf.get("counterfactual", {}) or {}
        challenge = str(signal.get("counterfactual_challenge_level", "none") or "none")
        action = str(signal.get("recommended_action", "continue") or "continue")
        action_cn = {
            "continue": "继续当前方向",
            "parameter_tuning": "继续参数调优",
            "proxy_reconstruction": "重构代理变量",
            "hypothesis_replacement": "替换假设",
            "hypothesis_revision": "修正假设",
        }.get(action, action)
        challenge_cn = {
            "none": "无明显反事实挑战",
            "weak": "反事实仅弱挑战",
            "moderate": "反事实形成中等挑战",
            "strong": "反事实形成强挑战",
        }.get(challenge, challenge)

        lines.append(f"**第 {loop_idx} 轮（{stage}）**：{_safe_text(hypothesis_text)}")
        lines.append(
            f"本轮最优因子为 `{best_name}`，Rank IC={_format_metric_cell(best_rank_ic)}。"
            f"理论内组最佳 |Rank IC|={_format_metric_cell(main.get('best_abs_rank_ic'))}，"
            f"理论外对照组最佳 |Rank IC|={_format_metric_cell(counter.get('best_abs_rank_ic'))}，"
            f"判断为“{challenge_cn}”，建议动作是“{action_cn}”。"
        )
        revision_reason = signal.get("revision_reason")
        if revision_reason:
            lines.append(f"原因：{_safe_text(revision_reason)}")
        lines.append("")

    first_hyp = (
        all_loops[0].get("hypothesis", {}).get("structured", {}).get("hypothesis")
        or all_loops[0].get("hypothesis", {}).get("hypothesis")
        or ""
    )
    last_hyp = (
        all_loops[-1].get("hypothesis", {}).get("structured", {}).get("hypothesis")
        or all_loops[-1].get("hypothesis", {}).get("hypothesis")
        or ""
    )
    challenge_counts = Counter(
        str((loop.get("signal_diagnostics", {}) or {}).get("counterfactual_challenge_level", "none") or "none")
        for loop in all_loops
    )
    lines.append("**总体结论**：")
    lines.append(
        f"本次实验从“{_safe_text(first_hyp)}”出发，最终演化到“{_safe_text(last_hyp)}”。"
        "如果理论外对照只是 weak/moderate，应优先理解为 proxy 或方向解释受到挑战；"
        "只有连续组级胜出并达到 strong 时，才应视为需要 thesis-level 修正。"
    )
    lines.append(
        "反事实挑战统计："
        + "，".join(f"{key}={value}" for key, value in sorted(challenge_counts.items()))
        + "。"
    )
    return lines


def _build_report_backtest_engine() -> QlibFactorBacktestEngine:
    config_path = PROJECT_ROOT / "src" / "trading_agents" / "configs" / "qlib" / "backtest_engine.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"invalid backtest engine config: {config_path}")
    return QlibFactorBacktestEngine(config, config_path)


def _resolve_signal_cache_path(raw_path: str) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    fallback = PROJECT_ROOT / "factor_library" / "cache" / path.name
    if fallback.exists():
        return fallback
    if not path.is_absolute():
        resolved = (PROJECT_ROOT / path).resolve()
        if resolved.exists():
            return resolved
    return None


def _safe_filename_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return text.strip("._") or "factor"


def _load_signal_series_for_factor(
    engine: QlibFactorBacktestEngine,
    factor: dict[str, Any],
    panel: pd.DataFrame,
) -> pd.Series:
    cache_path = _resolve_signal_cache_path(str(factor.get("source_pickle_path", "")))
    if cache_path is not None:
        loaded = engine._load_signal_from_cache({"source_pickle_path": str(cache_path)}, panel)
        if loaded is not None:
            return loaded
    signal = calculate_factor_expression(str(factor.get("expression", "")).strip(), panel, str(factor.get("name", "factor")))
    if signal is None:
        raise ValueError(f"failed to calculate signal for factor {factor.get('name')}")
    return signal


def _build_layer_plot_payload(
    engine: QlibFactorBacktestEngine,
    factor: dict[str, Any],
    panel: pd.DataFrame,
    *,
    target_column: str,
    n_groups: int = REPORT_LAYER_PLOT_GROUPS,
) -> dict[str, Any]:
    label_series = pd.to_numeric(panel[target_column], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    label_series = pd.to_numeric(label_series, errors="coerce").rename("label")
    signal_series = pd.to_numeric(_load_signal_series_for_factor(engine, factor, panel), errors="coerce").rename("signal")
    factor_frame = engine._build_layered_factor_frame(signal_series, label_series, n_layers=n_groups)
    if factor_frame.empty:
        raise ValueError(f"empty layered frame for factor {factor.get('name')}")

    layer_returns_by_dt = (
        factor_frame.groupby([factor_frame.index.get_level_values("datetime"), "layer"])["label"]
        .mean()
        .unstack("layer")
        .sort_index()
    )
    if layer_returns_by_dt.empty:
        raise ValueError(f"empty layer returns for factor {factor.get('name')}")

    benchmark = label_series.groupby(level="datetime").mean().dropna().sort_index()
    sampling_step = max(1, int(engine._current_sharpe_sampling_step))
    if sampling_step > 1:
        layer_returns_by_dt = layer_returns_by_dt.iloc[::sampling_step]
        benchmark = benchmark.iloc[::sampling_step]

    layer_cost_by_dt = pd.DataFrame(0.0, index=layer_returns_by_dt.index, columns=layer_returns_by_dt.columns)
    if float(getattr(engine, "one_way_fee_rate", 0.0) or 0.0) > 0:
        previous_weights: dict[Any, pd.Series] = {}
        cost_rows: list[dict[Any, float]] = []
        cost_index: list[Any] = []
        for dt, frame in factor_frame.groupby(level="datetime", sort=True):
            row_cost: dict[Any, float] = {}
            symbols = frame.index.get_level_values("symbol")
            layers = pd.Series(frame["layer"].to_numpy(), index=symbols)
            for layer in layer_returns_by_dt.columns:
                layer_symbols = list(layers[layers == layer].index)
                if layer_symbols:
                    weights = pd.Series(
                        1.0 / max(1, len(layer_symbols)),
                        index=pd.Index(layer_symbols, name="symbol"),
                        dtype=float,
                    )
                else:
                    weights = pd.Series(dtype=float)
                previous = previous_weights.get(layer, pd.Series(dtype=float))
                aligned_index = previous.index.union(weights.index)
                turnover = float(
                    (
                        weights.reindex(aligned_index, fill_value=0.0)
                        - previous.reindex(aligned_index, fill_value=0.0)
                    )
                    .abs()
                    .sum()
                )
                row_cost[layer] = turnover * float(engine.one_way_fee_rate)
                previous_weights[layer] = weights
            cost_rows.append(row_cost)
            cost_index.append(dt)
        if cost_rows:
            layer_cost_by_dt = (
                pd.DataFrame(cost_rows, index=pd.Index(cost_index, name="datetime"))
                .reindex(index=layer_returns_by_dt.index, columns=layer_returns_by_dt.columns)
                .fillna(0.0)
            )

    rank_ic = _to_float((factor.get("metrics", {}) or {}).get("Rank IC")) or 0.0
    display_columns = list(layer_returns_by_dt.columns)
    # Align the visual order with the final Sharpe direction:
    # positive Rank IC -> highest-signal group first; negative/zero -> lowest-signal group first.
    if rank_ic > 0:
        display_columns = list(sorted(display_columns, reverse=True))
    else:
        display_columns = list(sorted(display_columns))
    ordered_returns = layer_returns_by_dt.reindex(columns=display_columns)
    ordered_cost = layer_cost_by_dt.reindex(columns=display_columns).fillna(0.0)
    group_curves: dict[str, pd.Series] = {}
    for idx, column in enumerate(display_columns, start=1):
        group_returns = (ordered_returns[column] - ordered_cost[column]).fillna(0.0)
        group_curves[f"Group {idx}"] = (1.0 + group_returns).cumprod()

    benchmark_curve = (1.0 + benchmark.fillna(0.0)).cumprod()
    orientation = "high_minus_low" if rank_ic > 0 else "low_minus_high"
    return {
        "group_curves": group_curves,
        "benchmark_curve": benchmark_curve,
        "benchmark_label": "Listed Universe Mean",
        "orientation": orientation,
        "rank_ic": rank_ic,
        "fee_model": "net_of_one_way_fee",
        "one_way_fee_rate": float(getattr(engine, "one_way_fee_rate", 0.0) or 0.0),
    }


def _format_layer_plot_title(
    *,
    factor_name: str,
    loop_index: int,
    sharpe: float,
    rank_ic: float,
    orientation: str,
    one_way_fee_rate: float,
) -> str:
    fee_bps = one_way_fee_rate * 10000.0
    return f"{factor_name} | Loop {loop_index} | Net Sharpe={sharpe:.3f} | Rank IC={rank_ic:.4f} | {orientation} | fee={fee_bps:.1f}bps/side"


def _format_expression_title_line(expression: Any, *, max_width: int = 160) -> str:
    expr = " ".join(str(expression or "N/A").split())
    if len(expr) <= max_width:
        return f"Expression: {expr}"
    return f"Expression: {textwrap.shorten(expr, width=max_width, placeholder='...')}"


def _draw_layer_plot_on_axis(
    ax: Any,
    *,
    factor_name: str,
    factor_expression: str,
    loop_index: int,
    sharpe: float,
    plot_payload: dict[str, Any],
    title_fontsize: int,
    legend_fontsize: int,
    show_legend: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    log_floor = 1e-6

    def _positive_curve_values(curve: pd.Series) -> pd.Series:
        values = pd.to_numeric(curve, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
        return values.astype(float).clip(lower=log_floor)

    colors = plt.cm.tab10.colors
    for idx, (label, curve) in enumerate(plot_payload["group_curves"].items()):
        safe_curve = _positive_curve_values(curve)
        ax.plot(safe_curve.index, safe_curve.values, label=label, linewidth=1.45, color=colors[idx % len(colors)])
    benchmark_curve = plot_payload["benchmark_curve"]
    safe_benchmark_curve = _positive_curve_values(benchmark_curve)
    ax.plot(
        safe_benchmark_curve.index,
        safe_benchmark_curve.values,
        label=str(plot_payload.get("benchmark_label", "Listed Universe Mean")),
        linewidth=1.85,
        color="black",
        linestyle="--",
    )
    title_line_1 = _format_layer_plot_title(
        factor_name=factor_name,
        loop_index=loop_index,
        sharpe=sharpe,
        rank_ic=float(plot_payload["rank_ic"]),
        orientation=str(plot_payload["orientation"]),
        one_way_fee_rate=float(plot_payload.get("one_way_fee_rate", 0.0) or 0.0),
    )
    title_line_2 = _format_expression_title_line(factor_expression)
    ax.set_title(f"{title_line_1}\n{title_line_2}", fontsize=title_fontsize, pad=10)
    ax.set_xlabel("Datetime", fontsize=9)
    ax.set_ylabel("Net Cumulative Return (log scale)", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_fontsize, ncol=1, frameon=True)


def _render_layer_plot(
    *,
    factor_name: str,
    factor_expression: str,
    loop_index: int,
    sharpe: float,
    plot_payload: dict[str, Any],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.6))
    _draw_layer_plot_on_axis(
        ax,
        factor_name=factor_name,
        factor_expression=factor_expression,
        loop_index=loop_index,
        sharpe=sharpe,
        plot_payload=plot_payload,
        title_fontsize=11,
        legend_fontsize=8,
        show_legend=True,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_combined_layer_plot(
    *,
    rendered_plots: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rendered_plots:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 13), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, rendered in enumerate(rendered_plots[: n_rows * n_cols]):
        ax = axes.ravel()[idx]
        ax.set_visible(True)
        factor = rendered["factor"]
        _draw_layer_plot_on_axis(
            ax,
            factor_name=str(factor.get("name", "factor")),
            factor_expression=str(factor.get("expression", "N/A")),
            loop_index=int(rendered["loop_index"] or 0),
            sharpe=float(rendered["sharpe"]),
            plot_payload=rendered["plot_payload"],
            title_fontsize=7,
            legend_fontsize=6,
            show_legend=idx == 0,
        )

    fig.suptitle("Top 6 Net Sharpe Factors - Layered Backtest", fontsize=16, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_top_sharpe_layer_plots(
    *,
    log_dir: Path,
    all_loops: list[dict[str, Any]],
    output_file: Path,
) -> list[dict[str, Any]]:
    """Generate stratified backtest plots for the top final-Sharpe factors."""
    del log_dir
    top_factors = collect_top_sharpe_factors(all_loops)
    if not top_factors:
        return []

    panel_path = ""
    target_column = ""
    for loop in reversed(all_loops):
        backtest_meta = loop.get("backtest_report", {}) or {}
        panel_path = str(backtest_meta.get("panel_data_path", "")).strip() or panel_path
        target_column = str(backtest_meta.get("target_column", "")).strip() or target_column
        if panel_path and target_column:
            break
    if not panel_path:
        return []

    panel_file = Path(panel_path)
    if not panel_file.exists():
        panel_file = (PROJECT_ROOT / panel_path).resolve() if not Path(panel_path).is_absolute() else panel_file
    if not panel_file.exists():
        raise FileNotFoundError(f"panel data not found: {panel_path}")

    engine = _build_report_backtest_engine()
    prepared = engine._prepare_panel(str(panel_file))
    if prepared is None:
        raise ValueError(f"unable to prepare panel data from {panel_file}")
    panel, detected_target = prepared
    target_column = target_column or detected_target
    engine.target_column = target_column
    engine._configure_sharpe_sampling(panel)

    assets_dir = output_file.parent / "charts" / "layered_returns"
    chart_entries: list[dict[str, Any]] = []
    rendered_plots: list[dict[str, Any]] = []
    for rank, item in enumerate(top_factors, start=1):
        factor = dict(item["factor"])
        try:
            plot_payload = _build_layer_plot_payload(engine, factor, panel, target_column=target_column)
            image_name = f"top_net_sharpe_{rank:02d}_loop_{int(item['loop_index'] or 0):03d}_{_safe_filename_component(factor['name'])}.png"
            image_path = assets_dir / image_name
            _render_layer_plot(
                factor_name=str(factor.get("name", "factor")),
                factor_expression=str(factor.get("expression", "N/A")),
                loop_index=int(item["loop_index"] or 0),
                sharpe=float(item["sharpe"]),
                plot_payload=plot_payload,
                output_path=image_path,
            )
            rendered_plots.append(
                {
                    "loop_index": item["loop_index"],
                    "factor": factor,
                    "sharpe": item["sharpe"],
                    "plot_payload": plot_payload,
                }
            )
            chart_entries.append(
                {
                    "rank": rank,
                    "loop_index": item["loop_index"],
                    "factor_name": factor.get("name", "N/A"),
                    "expression": factor.get("expression", "N/A"),
                    "sharpe": item["sharpe"],
                    "gross_sharpe": item["gross_sharpe"],
                    "rank_ic": plot_payload["rank_ic"],
                    "orientation": plot_payload["orientation"],
                    "group_count": len(plot_payload["group_curves"]),
                    "target_column": target_column,
                    "image_relative_path": os.path.relpath(image_path, output_file.parent),
                }
            )
        except Exception as exc:
            print(f"Warning: failed to generate layer plot for {factor.get('name', 'N/A')}: {exc}")
    if rendered_plots:
        combined_image_path = assets_dir / "top_net_sharpe_2x3_combined.png"
        try:
            _render_combined_layer_plot(rendered_plots=rendered_plots, output_path=combined_image_path)
            combined_relative_path = os.path.relpath(combined_image_path, output_file.parent)
            for entry in chart_entries:
                entry["combined_image_relative_path"] = combined_relative_path
        except Exception as exc:
            print(f"Warning: failed to generate combined top-sharpe layer plot: {exc}")
    return chart_entries


def build_report_summary_cn(structured: dict[str, Any]) -> list[str]:
    adaptation = structured.get("crypto_adaptation_summary", {}) or {}
    proxy = structured.get("proxy_reconstruction_plan", {}) or {}
    core = [_format_family_cn(x) for x in proxy.get("core_proxy_family", [])]
    supporting = [_format_family_cn(x) for x in proxy.get("supporting_proxy_family", [])]
    input_features = proxy.get("input_features", [])
    non_transferable = [_format_non_transferable_cn(x) for x in (adaptation.get("non_transferable_parts", []) or [])]

    lines = []
    lines.append(
        f"- **研究目标**: 以 {_format_expected_direction_cn(structured.get('expected_direction'))} 为验证方向，围绕「{' / '.join(core[:3]) if core else '当前研报逻辑'}」构建横截面因子。"
    )
    if supporting:
        lines.append(f"- **辅助确认**: {' / '.join(supporting[:3])}")
    if input_features:
        lines.append(f"- **实际输入特征**: {', '.join(input_features[:10])}")
    if non_transferable:
        lines.append(f"- **已剔除的原市场特有机制**: {', '.join(non_transferable[:4])}")
    return lines


def build_loop_hypothesis_cn(loop: dict[str, Any]) -> list[str]:
    structured = loop["hypothesis"].get("structured", {}) or {}
    proxy = structured.get("proxy_reconstruction_plan", {}) or {}
    adaptation = structured.get("crypto_adaptation_summary", {}) or {}
    core = [_format_family_cn(x) for x in proxy.get("core_proxy_family", [])]
    supporting = [_format_family_cn(x) for x in proxy.get("supporting_proxy_family", [])]
    lines = []
    lines.append(
        f"**本轮核心假设**: 围绕「{' / '.join(core[:3]) if core else '当前代理族'}」构建信号，目标是 {_format_expected_direction_cn(structured.get('expected_direction'))}。"
    )
    expected_factor_sign = _expected_factor_sign_from_structured(structured)
    if expected_factor_sign != "unknown":
        lines.append(f"- **因子值与收益预期方向**: {_format_expected_direction_cn(expected_factor_sign)}")
    if supporting:
        lines.append(f"- **辅助确认**: {' / '.join(supporting[:3])}")
    if proxy.get("input_features"):
        lines.append(f"- **输入特征**: {', '.join(proxy.get('input_features', [])[:10])}")
    if adaptation.get("non_transferable_parts"):
        converted = [_format_non_transferable_cn(x) for x in adaptation.get("non_transferable_parts", [])[:4]]
        lines.append(f"- **已避免直接照搬**: {', '.join(converted)}")
    return lines


def build_stage_focus_cn(stage: str, factor_count: int) -> str:
    stage_cn = _format_stage_cn(stage)
    if stage == "base_reconstruction":
        return f"本轮处于{stage_cn}阶段，重点是先把研报逻辑用 {factor_count} 个朴素因子落地验证。"
    if stage == "parameter_tuning":
        return f"本轮处于{stage_cn}阶段，重点是围绕上一轮有效代理做窗口、归一化和轻量组合调整。"
    if stage == "resonance_filtering":
        return f"本轮处于{stage_cn}阶段，重点是增加确认条件与过滤器，判断哪些市场状态下信号更稳定。"
    if stage == "thesis_revision":
        return f"本轮处于{stage_cn}阶段，重点是保留有效因子线索，对假设主线做保守修正。"
    return f"本轮处于{stage_cn}阶段。"


def build_complexity_budget_cn(budget: dict[str, Any]) -> list[str]:
    if not budget:
        return ["- **复杂度预算**: 本轮未显式设置阶段预算。"]
    return [
        f"- **复杂度预算**: 长度 <= {budget.get('max_expression_length', 'N/A')}，函数数 <= {budget.get('max_function_count', 'N/A')}，嵌套层数 <= {budget.get('max_nesting_depth', 'N/A')}，原始特征数 <= {budget.get('max_input_feature_count', 'N/A')}，运算符数 <= {budget.get('max_operator_count', 'N/A')}，过滤器数 <= {budget.get('max_filter_count', 'N/A')}，复杂度得分 <= {budget.get('max_complexity_score', 'N/A')}。"
    ]


def build_factor_complexity_cn(factor: dict[str, Any]) -> list[str]:
    complexity = factor.get("complexity", {}) or {}
    if not complexity:
        return ["- **复杂度诊断**: 本因子未记录复杂度结果。"]
    passed = complexity.get("passed")
    status = "通过" if passed else "未通过"
    lines = [
        f"- **复杂度诊断**: {status}",
        f"- **复杂度得分**: {_format_metric_cell(complexity.get('complexity_score'))}",
        f"- **复杂度拆解**: 长度={complexity.get('expression_length', 'N/A')}，函数数={complexity.get('function_count', 'N/A')}，嵌套层数={complexity.get('nesting_depth', 'N/A')}，原始特征数={complexity.get('input_feature_count', 'N/A')}，运算符数={complexity.get('operator_count', 'N/A')}，过滤器数={complexity.get('filter_count', 'N/A')}",
    ]
    violations = complexity.get("violations", [])
    if violations:
        lines.append(f"- **复杂度违规**: {'；'.join(str(item) for item in violations)}")
    else:
        lines.append("- **复杂度违规**: 无")
    return lines


def build_feedback_cn(loop: dict[str, Any]) -> list[str]:
    feedback = loop["feedback"]
    best = summarize_best_factor(loop)
    diag = feedback.get("semantic_diagnosis")
    lines = []
    lines.append(f"**语义诊断**: {_format_diag_cn(diag)}")
    lines.append(
        f"**假设支持情况**: {'支持' if feedback.get('hypothesis_supported') is True else '不支持' if feedback.get('hypothesis_supported') is False else '未明确'}"
    )
    lines.append(f"**下一阶段建议**: {_format_stage_cn(feedback.get('next_research_stage'))}")
    if best:
        m = best.get("metrics", {})
        lines.append(
            f"**最佳单因子**: `{best['name']}`，IC={_format_metric_cell(m.get('IC'))}，Rank IC={_format_metric_cell(m.get('Rank IC'))}，夏普={_format_metric_cell(m.get('sharpe'))}，回撤={_format_metric_cell(m.get('drawdown'))}。"
        )

    diag_reason_map = {
        "parameter_not_tuned": "当前主线没有被彻底推翻，但参数、窗口或归一化方式仍有优化空间。",
        "proxy_weak": "当前代理变量对研报逻辑的表达还不够强，需要换更贴近机制的可观测代理。",
        "thesis_invalid": "当前结果更像是在否定核心假设本身，而不是单纯参数没调好。",
        "implementation_unstable": "当前轮结果受到实现或回测稳定性影响，暂不宜据此下过强结论。",
        "direction_mismatch": "当前因子抓到了某种结构，但结果方向和原始假设并不一致。",
    }
    lines.append(f"**中文结论**: {diag_reason_map.get(diag, '当前轮需要结合指标与实验设计继续判断。')}")
    return lines


def generate_markdown_report(log_dir: Path, output_file: Path) -> None:
    """Generate a Markdown report from log files."""
    loop_index = 1
    all_loops = []
    
    while True:
        loop_data = load_loop_file(log_dir, loop_index)
        if loop_data is None:
            break
        
        loop_info = {
            "index": loop_index,
            "evolution": extract_evolution_context(loop_data),
            "hypothesis": extract_hypothesis(loop_data),
            "experiment_design": extract_experiment_design(loop_data),
            "backtest_report": extract_backtest_metadata(loop_data),
            "factors": extract_factor_info(loop_data),
            "feedback": extract_feedback(loop_data),
            "runtime_status": extract_loop_runtime_status({**loop_data, "__loop_file__": f"loop_{loop_index:03d}.json"}),
        }
        all_loops.append(loop_info)
        loop_index += 1
    
    if not all_loops:
        print(f"No loop files found in {log_dir}")
        return

    for idx, loop in enumerate(all_loops):
        loop["current_research_stage"] = infer_current_research_stage(all_loops, idx)
    hypothesis_evolution_rows = build_hypothesis_evolution_rows(all_loops)
    try:
        top_sharpe_charts = generate_top_sharpe_layer_plots(
            log_dir=log_dir,
            all_loops=all_loops,
            output_file=output_file,
        )
    except Exception as exc:
        print(f"Warning: failed to generate top-sharpe layer plots: {exc}")
        top_sharpe_charts = []
    
    md_content = generate_markdown_content(log_dir, all_loops, hypothesis_evolution_rows, top_sharpe_charts)
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Markdown 报告已保存：{output_file}")


def generate_markdown_content(
    log_dir: Path,
    all_loops: list[dict[str, Any]],
    hypothesis_evolution_rows: list[dict[str, Any]] | None = None,
    top_sharpe_charts: list[dict[str, Any]] | None = None,
) -> str:
    """Generate Markdown content from loop data."""
    lines = []

    latest_context = all_loops[-1]["hypothesis"].get("structured", {}) if all_loops else {}
    research_report = _extract_research_report_payload(log_dir)
    report_source = _format_research_report_source_summary(research_report)
    has_research_report = bool(str(research_report.get("report_text", "")).strip())

    lines.append("# 因子挖掘研究报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**日志目录**: `{log_dir.name}`")
    lines.append(f"**总轮数**: {len(all_loops)}")
    lines.append(f"**研报来源**: {_safe_text(report_source) if has_research_report else '非研报模式，本次运行未提供 PDF 研报'}")
    lines.append("")
    evolution_summary = summarize_evolution_from_loops(all_loops)
    if evolution_summary["initial_direction_count"] or evolution_summary["evolution_agent_plans"]:
        lines.append("## Evolution Planning")
        lines.append("")
        lines.append(f"- **Initial direction count**: {evolution_summary['initial_direction_count']}")
        lines.append(f"- **Mutation branches**: {evolution_summary['planned_mutation_count']}")
        lines.append(f"- **Crossover branches**: {evolution_summary['planned_crossover_count']}")
        lines.append(f"- **Legacy evolved branches**: {evolution_summary.get('executed_evolved_count', 0)}")
        lines.append("")
        if evolution_summary["initial_directions"]:
            lines.append("### Initial Directions")
            lines.append("")
            for item in evolution_summary["initial_directions"]:
                lines.append(
                    f"- `{item.get('trajectory_id', '')}` / direction {item.get('direction_id', '')}: "
                    f"{item.get('direction', '')} "
                    f"(mode={item.get('direction_decomposer_mode', '')})"
                )
            lines.append("")
        if evolution_summary["evolution_agent_plans"]:
            lines.append("### Evolution Agent Plans")
            lines.append("")
            for plan in evolution_summary["evolution_agent_plans"]:
                lines.append(
                    f"- **Iteration {plan.get('paper_iteration', '')}** "
                    f"(internal phases {plan.get('round_indices', [])}): "
                    f"{plan['planned_mutation_count']} mutation branch, "
                    f"{plan['planned_crossover_count']} crossover branch, "
                    f"{plan.get('executed_evolved_count', 0)} legacy evolved branch"
                )
                for task in plan["mutation_tasks"]:
                    lines.append(
                        f"  - Mutation `{task.get('trajectory_id', '')}` from {task.get('parent_ids', [])}: "
                        f"{task.get('direction', '')} (mode={task.get('evolution_planner_mode', '')})"
                    )
                for task in plan["crossover_tasks"]:
                    lines.append(
                        f"  - Crossover `{task.get('trajectory_id', '')}` from {task.get('parent_ids', [])}: "
                        f"{task.get('direction', '')} (mode={task.get('evolution_planner_mode', '')})"
                    )
                for task in plan.get("evolved_tasks", []):
                    source_phases = sorted(
                        {
                            str(item.get("phase", "")).strip()
                            for item in list(task.get("evolution_plan_summaries", []) or [])
                            if isinstance(item, dict) and str(item.get("phase", "")).strip()
                        }
                    )
                    source_text = f" from plan phases {source_phases}" if source_phases else ""
                    lines.append(
                        f"  - Executed evolved branch `{task.get('trajectory_id', '')}`{source_text}: "
                        f"{task.get('direction', '')} (mode={task.get('evolution_planner_mode', '')})"
                    )
            lines.append("")
    factor_pool_path = log_dir / "factor_pool.json"
    if factor_pool_path.exists():
        try:
            factor_pool = json.loads(factor_pool_path.read_text(encoding="utf-8"))
        except Exception:
            factor_pool = {}
        if isinstance(factor_pool, dict) and factor_pool:
            lines.append("## Factor Pool")
            lines.append("")
            config = factor_pool.get("config", {}) if isinstance(factor_pool.get("config", {}), dict) else {}
            lines.append(f"- **Status**: {factor_pool.get('status', '')}")
            lines.append(f"- **Candidates**: {factor_pool.get('candidate_count', 0)}")
            lines.append(f"- **Selected**: {factor_pool.get('selected_count', 0)}")
            lines.append(f"- **Rejected**: {factor_pool.get('rejected_count', 0)}")
            lines.append(f"- **Correlation threshold**: {config.get('corr_threshold', '')}")
            combined = factor_pool.get("combined_backtest", {})
            if isinstance(combined, dict) and combined:
                metrics = combined.get("metrics", {}) if isinstance(combined.get("metrics", {}), dict) else {}
                metric_text = ", ".join(
                    f"{key}={_format_metric_cell(metrics.get(key))}"
                    for key in ["IC", "Rank IC", "Rank ICIR", "sharpe", "drawdown"]
                    if key in metrics
                )
                lines.append(f"- **Combined evaluation**: {combined.get('status', '')}" + (f" ({metric_text})" if metric_text else ""))
            selected = [
                item
                for item in list(factor_pool.get("selected_factors", []))
                if isinstance(item, dict)
            ][:10]
            if selected:
                lines.append("")
                lines.append("### Selected Pool Factors")
                lines.append("")
                for item in selected:
                    lines.append(
                        f"- `{item.get('factor_name', item.get('factor_id', ''))}`: "
                        f"{item.get('metric', 'Rank IC')}={_format_metric_cell(item.get('metric_value'))}, "
                        f"max_abs_corr={_format_metric_cell(item.get('max_abs_corr'))}"
                    )
            lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 0. 总览")
    lines.append("")
    if has_research_report:
        lines.append("本报告以 PDF 研报为起点，展示：")
    else:
        lines.append("本报告以市场背景、数据摘要与模型自动生成的初始假设为起点，展示：")
    lines.append("")
    if has_research_report:
        lines.append("- 研报中抽取出的核心论点与作用机制")
        lines.append("- 如何迁移到加密资产横截面场景")
    else:
        lines.append("- 初始研究方向与假设如何形成")
        lines.append("- 如何结合市场背景构建加密资产横截面因子")
    lines.append("- 每一轮假设如何演化")
    lines.append("- 每一层实验设计与因子表达式")
    lines.append("- 各轮核心指标对比（包含 IC / Rank IC / 夏普 / 回撤 等）")
    lines.append("- 报告末尾保留“全部因子表现明细（最终总表）”，逐个展示每个因子的信号强度、方向评价和回测指标")
    lines.append("")

    multi_report_summary = build_multi_report_summary_cn(research_report)
    if multi_report_summary:
        lines.extend(multi_report_summary)
        lines.append("")

    if has_research_report and latest_context:
        lines.append("### 0.2 研报抽取摘要")
        lines.append("")
        lines.extend(build_report_summary_cn(latest_context))
        lines.append("")

    lines.append("### 0.3 研究流程图")
    lines.append("")
    lines.append(build_evolution_mermaid(all_loops, has_research_report=has_research_report))
    lines.append("")

    lines.append("### 0.4 各轮核心指标总表")
    lines.append("")
    lines.append("| 轮次 | 阶段 | 最优因子 | IC | Rank IC | ICIR | Rank ICIR | 夏普 | 回撤 | 覆盖率 | 换手率 |")
    lines.append("|------|------|----------|----|---------|------|-----------|--------|----------|----------|----------|")
    for loop in all_loops:
        best = summarize_best_factor(loop)
        stage = loop.get("current_research_stage", loop["experiment_design"].get("research_stage", "N/A"))
        stage = _format_stage_cn(stage)
        runtime_status = loop.get("runtime_status", {})
        if runtime_status.get("round_status") == "failed_incomplete":
            stage = f"{stage}（未完成）"
        if best is None:
            lines.append(f"| {loop['index']} | {stage} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        m = best.get("metrics", {})
        lines.append(
            f"| {loop['index']} | {stage} | {best['name'][:26]} | {_format_metric_cell(m.get('IC'))} | {_format_metric_cell(m.get('Rank IC'))} | {_format_metric_cell(m.get('ICIR'))} | {_format_metric_cell(m.get('Rank ICIR'))} | {_format_metric_cell(m.get('sharpe'))} | {_format_metric_cell(m.get('drawdown'))} | {_format_metric_cell(m.get('coverage'))} | {_format_metric_cell(m.get('turnover'))} |"
        )
    lines.append("")

    layer_sampling_target = ""
    if top_sharpe_charts:
        layer_sampling_target = str((top_sharpe_charts[0] or {}).get("target_column", "") or "").strip()
    if not layer_sampling_target:
        for loop in reversed(all_loops):
            backtest_meta = loop.get("backtest_report", {}) or {}
            layer_sampling_target = str(backtest_meta.get("target_column", "") or "").strip()
            if layer_sampling_target:
                break
    layer_sampling_target = layer_sampling_target or "fwd_returns_*"

    if top_sharpe_charts:
        lines.append("### 0.5 Top 6 净夏普分层回测图")
        lines.append("")
        lines.append(
            f"以下图表按扣除手续费后的 `net_sharpe`（旧日志回退到 `sharpe`）选出全报告净夏普最高的 {len(top_sharpe_charts)} 个因子。"
            f"图中使用 {REPORT_LAYER_PLOT_GROUPS} 分组可视化，并沿用 `{layer_sampling_target}`、按预测窗口非重叠采样的口径。"
        )
        lines.append("图中累计收益曲线为扣除单边万4交易费后的净收益；`gross_sharpe` 保留为未扣费对照口径。")
        lines.append("图里的 benchmark 使用所有已上市币种在该时点的等权平均收益累计曲线。")
        lines.append("`Group 1` 始终表示与最终 Sharpe 方向一致的理想多头分层，最后一组表示对应空头分层。")
        lines.append("")
        combined_image = str(top_sharpe_charts[0].get("combined_image_relative_path", "")).strip()
        if combined_image:
            lines.append(f"![Top 6 net Sharpe layered backtests]({combined_image})")
            lines.append("")
        for chart in top_sharpe_charts:
            gross_sharpe = chart.get("gross_sharpe")
            gross_text = (
                f"，Gross Sharpe={gross_sharpe:.6f}"
                if isinstance(gross_sharpe, (int, float))
                else ""
            )
            lines.append(
                f"**Top {chart['rank']}**: `{chart['factor_name']}`"
                f"（Loop {chart['loop_index']}，Net Sharpe={chart['sharpe']:.6f}{gross_text}，Rank IC={chart['rank_ic']:.6f}，方向=`{chart['orientation']}`）"
            )
            lines.append(f"`{chart.get('expression', 'N/A')}`")
            image_path = str(chart.get("image_relative_path", "")).strip()
            if image_path:
                lines.append("")
                lines.append(f"![Top {chart['rank']} layered return chart]({image_path})")
            lines.append("")

    lines.append("### 0.6 理论内因子 vs 理论外对照")
    lines.append("")

    lines.append("### 0.7 假设演化与信号强弱")
    lines.append("")
    lines.append("| 轮次 | 阶段 | 本轮假设（中文） | 最优因子 | 最优 Rank IC | 已回测因子 | 弱信号数 | 多数弱信号 | 方向稳定 | 建议动作 | 原因 |")
    lines.append("|------|------|------------------|----------|--------------|------------|----------|------------|----------|----------|------|")
    for row in hypothesis_evolution_rows or []:
        action_cn = {
            "continue": "继续",
            "parameter_tuning": "参数调优",
            "proxy_reconstruction": "代理重构",
            "hypothesis_replacement": "替换假设",
            "hypothesis_revision": "修正假设",
        }.get(str(row.get("recommended_action", "")), str(row.get("recommended_action", "")) or "继续")
        lines.append(
            f"| {row.get('loop')} | {_format_stage_cn(row.get('stage'))} | {_md_table_cell(row.get('hypothesis'), 90)} | {_md_table_cell(str(row.get('best_factor', 'N/A'))[:24])} | {_format_metric_cell(row.get('best_rank_ic'))} | {row.get('evaluated_factor_count', 0)} | {row.get('weak_signal_count', 0)} | {_format_bool_cn(row.get('majority_weak_signal'))} | {_format_bool_cn(row.get('direction_stable'))} | {action_cn} | {_md_table_cell(row.get('revision_reason'), 80)} |"
        )
    lines.append("")
    lines.append("| 轮次 | 理论内因子数 | 理论内最佳 | 理论内 Rank IC | 理论外因子数 | 理论外最佳 | 理论外 Rank IC | 观察 |")
    lines.append("|------|--------------|------------|----------------|--------------|------------|----------------|------|")
    for loop in all_loops:
        groups = summarize_alignment_groups(loop)
        main = groups.get("mainline", {})
        counter = groups.get("counterfactual", {})
        main_best = main.get("best") if isinstance(main, dict) else None
        counter_best = counter.get("best") if isinstance(counter, dict) else None
        main_rank = _to_float((main_best or {}).get("metrics", {}).get("Rank IC")) if isinstance(main_best, dict) else None
        counter_rank = _to_float((counter_best or {}).get("metrics", {}).get("Rank IC")) if isinstance(counter_best, dict) else None
        main_abs = abs(main_rank) if main_rank is not None else None
        counter_abs = abs(counter_rank) if counter_rank is not None else None
        if counter_abs is not None and (main_abs is None or counter_abs > main_abs):
            observation = "理论外对照绝对 Rank IC 更强，需检查假设方向"
        elif counter_rank is not None:
            observation = "理论外对照可作为反证参考"
        else:
            observation = "暂无理论外对照"
        lines.append(
            f"| {loop['index']} | {main.get('count', 0)} | {(main_best or {}).get('name', 'N/A')[:24] if isinstance(main_best, dict) else 'N/A'} | {_format_metric_cell(main_rank)} | {counter.get('count', 0)} | {(counter_best or {}).get('name', 'N/A')[:24] if isinstance(counter_best, dict) else 'N/A'} | {_format_metric_cell(counter_rank)} | {observation} |"
        )
    lines.append("")

    for loop in all_loops:
        loop_index = loop["index"]
        hypothesis = loop["hypothesis"]
        experiment = loop["experiment_design"]
        factors = loop["factors"]
        feedback = loop["feedback"]
        evolution = loop.get("evolution", {})
        runtime_status = loop.get("runtime_status", {})

        lines.append(f"## 运行序号 {loop_index}")
        lines.append("")

        if evolution.get("phase"):
            lines.append("### Evolution Metadata")
            lines.append("")
            lines.append(f"- **Paper iteration**: `{evolution.get('paper_iteration', '')}`")
            lines.append(f"- **Paper phase**: `{evolution.get('paper_phase', '')}`")
            lines.append(f"- **Phase**: `{evolution.get('phase', '')}`")
            lines.append(f"- **Internal phase index**: `{evolution.get('round_idx', '')}`")
            lines.append(f"- **Direction ID**: `{evolution.get('direction_id', '')}`")
            lines.append(f"- **Trajectory ID**: `{evolution.get('trajectory_id', '')}`")
            lines.append(f"- **Parent IDs**: `{', '.join(evolution.get('parent_ids', [])) or 'None'}`")
            if evolution.get("evolution_planner_mode"):
                lines.append(f"- **Planner mode**: `{evolution.get('evolution_planner_mode', '')}`")
            if evolution.get("direction"):
                lines.append(f"- **Direction**: {evolution.get('direction', '')}")
            if evolution.get("evolution_seed"):
                lines.append("")
                lines.append("**Evolution Seed**:")
                lines.append("```")
                lines.append(str(evolution.get("evolution_seed", "")))
                lines.append("```")
            lines.append("")

        if runtime_status.get("round_status") == "failed_incomplete":
            lines.append("> 本轮实验未完成。")
            if runtime_status.get("failed_agent_name"):
                lines.append(f"> 失败位置：`{runtime_status.get('failed_agent_name')}`")
            if runtime_status.get("failed_error"):
                lines.append(f"> 错误信息：`{runtime_status.get('failed_error')}`")
            if runtime_status.get("stale_results_reused_for_reference"):
                lines.append("> 以下指标沿用上一轮结果，仅供排障参考，不应视为本轮新实验结果。")
            lines.append("")

        lines.append("### 1. 本轮假设")
        lines.append("")
        lines.extend(build_loop_hypothesis_cn(loop))
        lines.append("")

        lines.append("### 2. 本层实验设计")
        lines.append("")
        lines.append(f"- **Experiment ID**: `{experiment['experiment_id']}`")
        current_stage = loop.get('current_research_stage', experiment.get('research_stage', 'N/A'))
        lines.append(f"- **当前阶段**: {_format_stage_cn(current_stage)}")
        lines.append(f"- **阶段目标**: {build_stage_focus_cn(current_stage, len(factors))}")
        lines.append(f"- **本轮生成因子数**: {len(factors)}")
        counterfactual_count = int(experiment.get("counterfactual_factor_count", 0) or 0)
        counterfactual_target = int(experiment.get("counterfactual_factor_target_count", 0) or 0)
        if counterfactual_target or counterfactual_count:
            lines.append(
                f"- **理论外对抗因子**: 目标 {counterfactual_target} 个，实际保留 {counterfactual_count} 个，用于检验相反经济解释是否更强。"
            )
        shortfall = experiment.get("factor_group_shortfall_diagnostics", {}) or {}
        if isinstance(shortfall, dict) and shortfall.get("status") == "shortfall":
            lines.append(f"- **因子分组不足警告**: {shortfall.get('message')}")
        lines.extend(build_complexity_budget_cn(experiment.get("complexity_budget", {})))
        lines.append("")

        lines.append("### 3. 因子与表达式")
        lines.append("")

        groups = summarize_alignment_groups(loop)
        if groups.get("counterfactual", {}).get("count", 0):
            lines.append("**本轮包含理论外对照因子**: 这些因子合法且参与回测，但偏离当前 thesis 锚点，用于判断是否存在更强的替代机制。")
            lines.append("")

        if runtime_status.get("round_status") == "failed_incomplete" and runtime_status.get("stale_results_reused_for_reference"):
            lines.append("**注意**: 本节因子与指标来自上一轮残留结果，仅用于帮助定位失败时的上下文，不代表本轮已成功生成并回测。")
            lines.append("")

        for i, factor in enumerate(factors, 1):
            lines.append(f"#### 因子 {i}: {factor['name']}")
            lines.append("")
            lines.append(f"**理论归属**: {format_alignment_cn(factor.get('theory_alignment'))}")
            if factor.get("counterfactual_reason"):
                lines.append(f"**对照原因**: {factor.get('counterfactual_reason')}")
            if factor.get("counterfactual_thesis"):
                lines.append(f"**反事实批判假设**: {factor.get('counterfactual_thesis')}")
            if factor.get("counterfactual_mechanism"):
                lines.append(f"**反事实机制**: {factor.get('counterfactual_mechanism')}")
            if factor.get("counterfactual_expected_direction"):
                lines.append(f"**反事实预期方向**: {factor.get('counterfactual_expected_direction')}")
            if factor.get("validation_warnings"):
                lines.append(f"**校验警告**: {', '.join(str(item) for item in factor.get('validation_warnings', []))}")
            anchor = factor.get("anchor_diagnostics", {}) or {}
            expected_features = anchor.get("expected_anchor_features", []) or []
            matched_features = anchor.get("matched_anchor_features", []) or []
            expected_patterns = anchor.get("expected_anchor_patterns", []) or []
            if expected_features or expected_patterns:
                missing_features = sorted(set(str(item) for item in expected_features) - set(str(item) for item in matched_features))
                lines.append(
                    f"**Anchor 匹配**: 匹配特征={', '.join(matched_features) if matched_features else '无'}；"
                    f"缺失特征={', '.join(missing_features) if missing_features else '无'}；"
                    f"表达式模式匹配={'是' if anchor.get('matched_anchor_pattern') else '否'}"
                )
                if expected_patterns:
                    lines.append(f"**预期 Anchor 模式**: {', '.join(str(item) for item in expected_patterns[:5])}")
                if anchor.get("anchor_fallback_reason"):
                    lines.append(f"**Anchor fallback 原因**: {anchor.get('anchor_fallback_reason')}")
            if factor.get("implementation_status"):
                lines.append(f"**实现状态**: {factor.get('implementation_status')}")
            failure = factor.get("backtest_failure", {}) or {}
            if failure:
                reason_parts = [
                    str(failure.get(key, "")).strip()
                    for key in ("message", "value_feedback", "final_feedback", "code_feedback", "execution_feedback")
                    if str(failure.get(key, "")).strip()
                ]
                lines.append(f"**失败原因**: {reason_parts[0] if reason_parts else failure}")
            direction_diag = factor.get("direction_diagnostics", {}) or {}
            if direction_diag:
                lines.append(
                    f"**方向评价**: 预期={_format_expected_direction_cn(direction_diag.get('expected_direction'))}；"
                    f"实际 Rank IC 符号={_format_expected_direction_cn(direction_diag.get('actual_rank_ic_sign'))}；"
                    f"是否符合={_format_direction_alignment_cn(direction_diag.get('direction_aligned'))}；"
                    f"建议用法=`{direction_diag.get('recommended_usage', 'as_is')}`"
                )
            signal_diag = factor.get("signal_diagnostics", {}) or {}
            if signal_diag:
                lines.append(
                    f"**信号强度**: {_format_signal_strength_cn(signal_diag.get('signal_strength'))}；"
                    f"|Rank IC|={_format_metric_cell(signal_diag.get('abs_rank_ic'))}；"
                    f"弱信号={_format_bool_cn(signal_diag.get('weak_signal_flag'))}；"
                    f"建议={_format_signal_action_cn(signal_diag.get('signal_action'))}"
                )
            lines.append("")
            lines.append("**说明**: 本因子为当前阶段自动生成的候选表达式，用于检验该代理在横截面上的预测能力。")
            lines.append("")
            lines.append("**表达式**:")
            lines.append("```text")
            lines.append(factor['expression'])
            lines.append("```")
            lines.append("")
            lines.append("**指标**:")
            lines.append("")
            lines.append(format_metrics(factor['metrics']))
            lines.append("")
            lines.append("**复杂度诊断**:")
            lines.append("")
            lines.extend(build_factor_complexity_cn(factor))
            lines.append("")

        lines.append("### 4. 反馈与结论")
        lines.append("")
        if runtime_status.get("round_status") == "failed_incomplete" and runtime_status.get("stale_results_reused_for_reference"):
            lines.append("**状态说明**: 本轮没有形成新的完整反馈，以下内容为残留上下文，仅供排障参考。")
            lines.append("")
        signal_summary = loop.get("signal_diagnostics", {}) or {}
        if signal_summary.get("majority_weak_signal") is True:
            lines.append("**弱信号提醒**: 本轮因子 Rank IC 绝对值普遍低于 0.025，建议修改 proxy 或重构假设。")
            if signal_summary.get("revision_reason"):
                lines.append(f"**修订原因**: {signal_summary.get('revision_reason')}")
            lines.append("")
        lines.extend(build_feedback_cn(loop))
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## 总结")
    lines.append("")
    lines.extend(build_hypothesis_evolution_summary_cn(all_loops))
    lines.append("")
    lines.append("### 全部因子表现明细（最终总表）")
    lines.append("")
    lines.append("该表逐个列出所有 loop 生成的因子，用于横向比较信号强弱、方向是否符合预期、是否建议取反，以及核心回测指标。")
    lines.append("")
    lines.append("| 轮次 | 理论归属 | 因子名 | 信号强度 | 信号建议 | Abs Rank IC | 弱信号 | 预期方向 | Rank IC符号 | 方向符合 | 建议用法 | IC | Rank IC | ICIR | Rank ICIR | 夏普 | 回撤 | 覆盖率 | 换手率 |")
    lines.append("|------|----------|--------|----------|----------|-----------|--------|----------|------------|----------|----------|----|---------|------|-----------|--------|----------|----------|----------|")
    
    for loop in all_loops:
        for factor in loop["factors"]:
            metrics = factor["metrics"]
            direction_diag = factor.get("direction_diagnostics", {}) or {}
            signal_diag = factor.get("signal_diagnostics", {}) or {}
            lines.append(
                f"| {loop['index']} | {format_alignment_cn(factor.get('theory_alignment'))} | {factor['name'][:30]} | {_format_signal_strength_cn(signal_diag.get('signal_strength'))} | {_format_signal_action_cn(signal_diag.get('signal_action'))} | {_format_metric_cell(signal_diag.get('abs_rank_ic'))} | {_format_bool_cn(signal_diag.get('weak_signal_flag'))} | {_format_expected_direction_cn(direction_diag.get('expected_direction'))} | {_format_expected_direction_cn(direction_diag.get('actual_rank_ic_sign'))} | {_format_direction_alignment_cn(direction_diag.get('direction_aligned'))} | `{direction_diag.get('recommended_usage', 'as_is')}` | {_format_metric_cell(metrics.get('IC'))} | {_format_metric_cell(metrics.get('Rank IC'))} | {_format_metric_cell(metrics.get('ICIR'))} | {_format_metric_cell(metrics.get('Rank ICIR'))} | {_format_metric_cell(metrics.get('sharpe'))} | {_format_metric_cell(metrics.get('drawdown'))} | {_format_metric_cell(metrics.get('coverage'))} | {_format_metric_cell(metrics.get('turnover'))} |"
            )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 Alpha Factor Mining Report Generator 自动生成（中文版）*")
    
    return "\n".join(lines)


def convert_to_pdf(md_file: Path, pdf_file: Path) -> bool:
    """Convert Markdown file to PDF.
    
    Try multiple methods:
    1. pandoc (best quality)
    2. weasyprint (good quality, pure Python)
    3. reportlab (fallback, simple text)
    """
    method_used = None
    
    if _try_pandoc_pdf(md_file, pdf_file):
        method_used = "pandoc"
    elif _try_weasyprint_pdf(md_file, pdf_file):
        method_used = "weasyprint"
    elif _try_reportlab_pdf(md_file, pdf_file):
        method_used = "reportlab"
    else:
        print("Warning: Could not generate PDF. Install one of:")
        print("  - pandoc: brew install pandoc")
        print("  - weasyprint: pip install weasyprint && brew install pango gdk-pixbuf libffi")
        print("  - reportlab: pip install reportlab")
        return False
    
    if pdf_file.exists():
        print(f"PDF report saved to: {pdf_file} (using {method_used})")
        return True
    return False


def _try_pandoc_pdf(md_file: Path, pdf_file: Path) -> bool:
    """Try to convert using pandoc."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False
    
    try:
        result = subprocess.run(
            [
                "pandoc",
                str(md_file),
                "-o", str(pdf_file),
                "--pdf-engine=xelatex",
                "-V", "mainfont=Arial Unicode MS",
                "-V", "geometry:margin=1in",
                "--highlight-style=tango",
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_weasyprint_pdf(md_file: Path, pdf_file: Path) -> bool:
    """Try to convert using weasyprint."""
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        return False
    
    try:
        with md_file.open("r", encoding="utf-8") as f:
            md_content = f.read()
        
        html_content = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "codehilite"]
        )
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #7f8c8d;
                    margin-top: 20px;
                }}
                h4 {{
                    color: #95a5a6;
                    margin-top: 15px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Monaco', 'Menlo', monospace;
                    font-size: 10pt;
                }}
                pre {{
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-x: auto;
                    font-size: 9pt;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    font-size: 10pt;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #eee;
                    margin: 30px 0;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                    padding: 10px 20px;
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        HTML(string=full_html).write_pdf(pdf_file)
        return True
    except Exception as e:
        return False


def _try_reportlab_pdf(md_file: Path, pdf_file: Path) -> bool:
    """Try to convert using reportlab (simple text-based PDF)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        return False
    
    try:
        with md_file.open("r", encoding="utf-8") as f:
            md_content = f.read()
        
        doc = SimpleDocTemplate(str(pdf_file), pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Code',
                                  fontName='Courier',
                                  fontSize=8,
                                  leading=10,
                                  leftIndent=20))
        
        story = []
        
        for line in md_content.split('\n'):
            if line.startswith('# '):
                text = line[2:]
                story.append(Paragraph(text, styles['Heading1']))
                story.append(Spacer(1, 12))
            elif line.startswith('## '):
                text = line[3:]
                story.append(Paragraph(text, styles['Heading2']))
                story.append(Spacer(1, 10))
            elif line.startswith('### '):
                text = line[4:]
                story.append(Paragraph(text, styles['Heading3']))
                story.append(Spacer(1, 8))
            elif line.startswith('#### '):
                text = line[5:]
                story.append(Paragraph(text, styles['Heading4']))
                story.append(Spacer(1, 6))
            elif line.startswith('```'):
                continue
            elif line.startswith('|'):
                story.append(Paragraph(line.replace('|', ' | '), styles['Code']))
            elif line.startswith('- '):
                text = '• ' + line[2:]
                story.append(Paragraph(text, styles['Normal']))
            elif line.strip():
                text = line.replace('**', '<b>').replace('**', '</b>')
                text = text.replace('`', '<font face="Courier">').replace('`', '</font>')
                story.append(Paragraph(text, styles['Normal']))
            else:
                story.append(Spacer(1, 6))
        
        doc.build(story)
        return True
    except Exception:
        return False


def _try_markdown_pdf(md_file: Path, pdf_file: Path) -> bool:
    """Try to convert using markdown-pdf."""
    try:
        from md2pdf.core import md2pdf
    except ImportError:
        return False
    
    try:
        md2pdf(str(pdf_file), md_content=md_file.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def generate_report(log_dir: Path | None = None, no_pdf: bool = False) -> tuple[Path | None, Path | None]:
    """Generate report from log directory.
    
    Args:
        log_dir: Path to log directory. If None, uses the latest log directory.
        no_pdf: If True, skip PDF generation.
    
    Returns:
        Tuple of (markdown_file_path, pdf_file_path) or (None, None) if failed.
    """
    if log_dir is None:
        log_dir = find_latest_log_dir()
    
    if log_dir is None:
        print("未找到运行日志目录。")
        return None, None
    
    print(f"正在处理运行目录：{log_dir}")
    
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = reports_dir / f"factor_mining_report_{log_dir.name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_file = report_dir / "factor_mining_report.md"
    pdf_file = report_dir / "factor_mining_report.pdf"
    
    generate_markdown_report(log_dir, md_file)
    
    pdf_file_result = None
    if md_file.exists() and not no_pdf:
        convert_to_pdf(md_file, pdf_file)
        if pdf_file.exists():
            pdf_file_result = pdf_file
    
    return md_file, pdf_file_result


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Alpha Factor Mining Report")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Path to log directory (default: latest)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation",
    )
    
    args = parser.parse_args()
    
    md_file, pdf_file = generate_report(args.log_dir, args.no_pdf)
    
    if md_file:
        print(f"\n报告生成完成。")
        print(f"Markdown：{md_file}")
        if pdf_file and pdf_file.exists():
            print(f"PDF：{pdf_file}")


if __name__ == "__main__":
    main()
