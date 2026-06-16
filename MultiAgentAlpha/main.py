from __future__ import annotations

import argparse
import atexit
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.domains import CryptoCrossSectionDomainAdapter
from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.infra.factor_pool_manager import FactorPoolConfig, FactorPoolManager
from trading_agents.infra.llm import GatewayModelClient, LLMGateway, RetryPolicy
from trading_agents.infra.llm.providers import AzureOpenAIProvider, OpenAICompatibleProvider, ZhipuNativeProvider
from trading_agents.infra.research_report_loader import build_research_report_payload
from trading_agents.infra.run_store import RunStore
from trading_agents.workflows import AlphaFactorMiningWorkflow, EvolutionConfig
from scripts.generate_report import generate_report


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        value = value[1:-1]
    return key, value


def _load_dotenv_files() -> None:
    for env_file in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
        if not env_file.exists() or not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _print_debug_event(tag: str, payload: dict[str, Any]) -> None:
    event = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
    }
    print(json.dumps(_json_safe(event), ensure_ascii=False), file=sys.stderr, flush=True)


def _format_console_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).strftime("%H:%M:%S")
    except ValueError:
        return text[:8] if len(text) >= 8 else text


def _progress_prefix(payload: dict[str, Any], *, time_key: str | None = None) -> str:
    if time_key:
        value = _format_console_time(payload.get(time_key))
        if value:
            return f"[{value}]"
    for key in ("timestamp", "started_at", "finished_at", "failed_at"):
        value = _format_console_time(payload.get(key))
        if value:
            return f"[{value}]"
    return f"[{datetime.now().strftime('%H:%M:%S')}]"


_AGENT_DISPLAY_NAMES = {
    "hypothesis_agent": "假说生成",
    "experiment_designer_agent": "实验设计",
    "factor_coder_agent": "因子编码",
    "backtest_runner_agent": "回测执行",
    "feedback_summarizer_agent": "反馈总结",
    "direction_decomposer_agent": "初始化假设",
    "evolution_agent": "进化规划",
}

_PHASE_DISPLAY_NAMES = {
    "original": "初始化/原始探索",
    "mutation": "Mutation phase",
    "crossover": "Crossover phase",
    "evolved": "进化分支",
}

_AGENT_ACTIONS = {
    "hypothesis_agent": "生成/修正因子假说",
    "experiment_designer_agent": "设计候选因子",
    "factor_coder_agent": "生成并校验因子表达式",
    "backtest_runner_agent": "回测并筛选因子",
    "feedback_summarizer_agent": "总结反馈和下一步",
    "direction_decomposer_agent": "生成初始假设",
    "evolution_agent": "规划变异/交叉任务",
}

_STAGE_DISPLAY_NAMES = {
    "evolution_round_planned": "进化轮次规划",
    "evolution_round_skipped": "进化轮次跳过",
    "evolution_task_planned": "进化任务规划",
    "evolution_task_started": "进化任务开始",
    "evolution_task_finished": "进化任务完成",
    "direction_decomposer_started": "初始假设生成开始",
    "direction_decomposer_finished": "初始假设生成完成",
    "direction_decomposer_failed": "初始假设生成失败",
    "evolved_seed_decomposed": "下一代假设生成完成",
    "evolved_direction_decomposer_started": "下一代假设生成开始",
    "evolved_direction_decomposer_finished": "下一代假设生成完成",
    "evolved_direction_decomposer_failed": "下一代假设生成失败",
    "evolution_agent_started": "进化规划开始",
    "evolution_agent_finished": "进化规划完成",
    "evolution_agent_failed": "进化规划失败",
    "factor_pool_finished": "因子池构建完成",
    "factor_pool_failed": "因子池构建失败",
    "market_payload_started": "市场信息读取开始",
    "market_payload_finished": "市场信息读取完成",
    "research_report_payload_started": "研报读取开始",
    "research_report_payload_finished": "研报读取完成",
    "initialization_payload_ready": "初始化输入准备完成",
    "holdout_validation_finished": "留样验证完成",
    "holdout_validation_failed": "留样验证失败",
    "report_generation_failed": "报告生成失败",
    "factor_cache_cleanup_finished": "因子缓存清理完成",
    "session_finished": "会话完成",
}

_STATUS_DISPLAY_NAMES = {
    "completed": "完成",
    "failed": "失败",
    "partial": "部分完成",
    "finished": "完成",
}


_AGENT_DISPLAY_NAMES.update(
    {
        "direction_decomposer_agent": "初始假说",
        "hypothesis_agent": "假说生成",
        "experiment_designer_agent": "实验设计",
        "factor_coder_agent": "因子实现",
        "backtest_runner_agent": "因子评估",
        "feedback_summarizer_agent": "反馈诊断",
        "evolution_agent": "进化规划",
    }
)

_PHASE_DISPLAY_NAMES.update(
    {
        "original": "初始探索",
        "mutation": "变异轮",
        "crossover": "交叉轮",
        "evolved": "进化轮",
    }
)

_AGENT_ACTIONS.update(
    {
        "direction_decomposer_agent": "生成初始研究假说",
        "hypothesis_agent": "生成可检验假说",
        "experiment_designer_agent": "设计候选因子",
        "factor_coder_agent": "实现并校验因子表达式",
        "backtest_runner_agent": "计算因子指标并筛选",
        "feedback_summarizer_agent": "诊断失败原因和下一步",
        "evolution_agent": "规划变异/交叉任务",
    }
)

_STAGE_DISPLAY_NAMES.update(
    {
        "market_payload_started": "数据准备开始",
        "market_payload_finished": "数据摘要完成",
        "research_report_payload_started": "研报输入开始",
        "research_report_payload_finished": "研报输入完成",
        "initialization_payload_ready": "初始化输入完成",
        "direction_decomposer_started": "初始假说开始",
        "direction_decomposer_finished": "初始假说完成",
        "direction_decomposer_failed": "初始假说失败",
        "evolved_seed_decomposed": "子代假说完成",
        "evolved_direction_decomposer_started": "子代假说开始",
        "evolved_direction_decomposer_finished": "子代假说完成",
        "evolved_direction_decomposer_failed": "子代假说失败",
        "evolution_agent_started": "进化规划开始",
        "evolution_agent_finished": "进化规划完成",
        "evolution_agent_failed": "进化规划失败",
        "evolution_round_planned": "轮次规划完成",
        "evolution_round_skipped": "轮次跳过",
        "evolution_round_all_failed": "本轮全失败",
        "evolution_task_planned": "轨迹任务已规划",
        "evolution_task_started": "轨迹任务开始",
        "evolution_task_finished": "轨迹任务完成",
        "round_finished": "轨迹完成",
        "holdout_validation_finished": "留样验证完成",
        "holdout_validation_failed": "留样验证失败",
        "factor_pool_finished": "因子入库完成",
        "factor_pool_failed": "因子入库失败",
        "factor_cache_cleanup_finished": "缓存清理完成",
        "report_generation_failed": "报告生成失败",
        "session_finished": "运行完成",
    }
)

_STATUS_DISPLAY_NAMES.update(
    {
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已终止",
        "partial": "部分完成",
        "finished": "已完成",
    }
)


def _display_with_raw(chinese: str, raw: str) -> str:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return chinese
    return f"{chinese}（{raw_text}）"


def _display_agent(agent_name: str) -> str:
    raw = str(agent_name or "").strip()
    return _display_with_raw(_AGENT_DISPLAY_NAMES.get(raw, "智能体"), raw)


def _display_agent_short(agent_name: Any) -> str:
    raw = str(agent_name or "").strip()
    if not raw:
        return "假设生成"
    return _AGENT_DISPLAY_NAMES.get(raw, raw)


def _display_phase(phase: str) -> str:
    raw = str(phase or "").strip()
    if not raw:
        return ""
    return _display_with_raw(_PHASE_DISPLAY_NAMES.get(raw, "阶段"), raw)


def _paper_iteration_from_round(round_idx: Any) -> int:
    try:
        value = int(round_idx)
    except Exception:
        return 0
    if value <= 0:
        return 0
    return (value + 1) // 2


def _format_paper_iteration_context(round_idx: Any, phase: str) -> str:
    phase_text = str(phase or "").strip().lower()
    iteration = _paper_iteration_from_round(round_idx)
    if iteration <= 0 or phase_text == "original":
        return "初始化"
    return f"第{iteration}次进化"


def _paper_phase_label(phase: Any, round_idx: Any) -> str:
    phase_text = str(phase or "").strip().lower()
    if phase_text == "mutation":
        return "mutation"
    if phase_text == "crossover":
        return "crossover"
    if phase_text == "original" or _paper_iteration_from_round(round_idx) == 0:
        return "initialization"
    return phase_text or ""


def _display_stage(stage: str) -> str:
    raw = str(stage or "").strip()
    return _display_with_raw(_STAGE_DISPLAY_NAMES.get(raw, "进度事件"), raw)


def _display_status(status: Any) -> str:
    raw = str(status or "").strip()
    if not raw:
        return ""
    return _display_with_raw(_STATUS_DISPLAY_NAMES.get(raw, "状态"), raw)


def _format_round_context(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    round_idx = payload.get("round_idx")
    loop_index = payload.get("loop_index")
    raw_phase = str(payload.get("phase", "")).strip()
    phase = _display_phase(raw_phase)
    trajectory_id = str(payload.get("trajectory_id", "")).strip()
    if round_idx not in (None, ""):
        parts.append(_format_paper_iteration_context(round_idx, raw_phase))
    if loop_index not in (None, ""):
        parts.append(f"loop {loop_index}")
    if phase:
        parts.append(phase)
    if trajectory_id:
        parts.append(f"轨迹 {trajectory_id}")
    return " · ".join(parts)


def _format_factor_display(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    return f"因子（{raw}）"


def _format_factor_list(names: list[str], *, limit: int = 3) -> str:
    shown = [_format_factor_display(name) for name in names[:limit] if str(name).strip()]
    if not shown:
        return ""
    suffix = "等" if len(names) > limit else ""
    return "、".join(shown) + suffix


def _print_console_line(payload: dict[str, Any], label: str, message: str, *, time_key: str | None = None) -> None:
    prefix = _progress_prefix(payload, time_key=time_key)
    context = _format_round_context(payload)
    pieces = [prefix, label]
    if context:
        pieces.append(context)
    if message:
        pieces.append(message)
    print(" | ".join(pieces), file=sys.stderr, flush=True)


def _print_console_progress(payload: dict[str, Any]) -> None:
    stage = str(payload.get("stage", "")).strip()
    loop_index = payload.get("loop_index")
    agent_name = str(payload.get("agent_name", "")).strip()
    elapsed = payload.get("elapsed_seconds")
    error_count = payload.get("error_count")
    summary = payload.get("progress_summary", {})
    summary_text = str(summary.get("text", "")).strip() if isinstance(summary, dict) else ""

    if stage == "agent_started":
        action = _AGENT_ACTIONS.get(agent_name, "运行当前步骤")
        _print_console_line(
            payload,
            _display_agent(agent_name),
            f"开始：{action}",
            time_key="started_at",
        )
        return
    if stage == "agent_finished":
        elapsed_text = f" 用时={elapsed}s" if elapsed not in (None, "") else ""
        detail_text = f"；{summary_text}" if summary_text else ""
        _print_console_line(
            payload,
            _display_agent(agent_name),
            f"完成{elapsed_text}{detail_text}",
            time_key="finished_at",
        )
        return
    if stage == "agent_failed":
        elapsed_text = f" 用时={elapsed}s" if elapsed not in (None, "") else ""
        _print_console_line(
            payload,
            _display_agent(agent_name),
            f"失败{elapsed_text}：{payload.get('error', '')}",
            time_key="failed_at",
        )
        return
    if stage == "round_finished":
        _print_console_line(payload, "轮次完成（round_finished）", f"错误数 {error_count}")
        return
    if stage in {
        "evolution_round_planned",
        "evolution_round_all_failed",
        "evolution_round_skipped",
        "evolution_task_planned",
        "evolution_task_started",
        "evolution_task_finished",
        "direction_decomposer_started",
        "direction_decomposer_finished",
        "direction_decomposer_failed",
        "evolved_seed_decomposed",
        "evolved_direction_decomposer_started",
        "evolved_direction_decomposer_finished",
        "evolved_direction_decomposer_failed",
        "evolution_agent_started",
        "evolution_agent_finished",
        "evolution_agent_failed",
        "market_payload_started",
        "market_payload_finished",
        "research_report_payload_started",
        "research_report_payload_finished",
        "initialization_payload_ready",
    }:
        message = _build_evolution_console_message(stage, payload)
        _print_console_line(payload, _display_stage(stage), message)
        return
    if stage in {
        "holdout_validation_finished",
        "report_generation_failed",
        "holdout_validation_failed",
        "factor_pool_finished",
        "factor_pool_failed",
        "factor_cache_cleanup_finished",
        "session_finished",
    }:
        _print_console_line(payload, _display_stage(stage), _build_system_console_message(stage, payload))


def _compact_event_text(value: object, *, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _research_report_progress_fields(research_report_payload: object) -> dict[str, Any]:
    if not isinstance(research_report_payload, dict):
        return {}
    source = research_report_payload.get("source", {})
    if not isinstance(source, dict):
        source = {}
    source_reports = [
        dict(item)
        for item in list(research_report_payload.get("source_reports", []))
        if isinstance(item, dict)
    ]
    report_summary: dict[str, Any] = {}
    if source_reports:
        thesis = " ".join(_compact_event_text(item.get("thesis_summary", ""), max_chars=220) for item in source_reports[:3])
        mechanism = " ".join(_compact_event_text(item.get("mechanism_summary", ""), max_chars=180) for item in source_reports[:3])
        proxies: list[str] = []
        synthesis = research_report_payload.get("multi_report_synthesis", {})
        if isinstance(synthesis, dict):
            proxies.extend(str(item) for item in list(synthesis.get("common_proxy_candidates", [])) if str(item).strip())
        for item in source_reports[:3]:
            proxies.extend(str(proxy) for proxy in list(item.get("proxy_candidates", [])) if str(proxy).strip())
        report_summary = {
            "thesis_summary": _compact_event_text(thesis, max_chars=520),
            "mechanism_summary": _compact_event_text(mechanism, max_chars=420),
            "proxy_candidates": list(dict.fromkeys(proxies))[:8],
        }
    else:
        report_text = str(
            research_report_payload.get("cleaned_report_text")
            or research_report_payload.get("report_text")
            or ""
        )
        try:
            from trading_agents.infra.research_report_loader import _summarize_clean_report

            report_summary = _summarize_clean_report(
                {"cleaned_report_text": report_text},
                [str(item) for item in list(research_report_payload.get("dataset_features", []))],
            )
        except Exception:
            report_summary = {"compact_excerpt": _compact_event_text(report_text, max_chars=700)}

    initial_seed = research_report_payload.get("report_initial_seed", {})
    if not isinstance(initial_seed, dict):
        initial_seed = {}
    source_names = [str(item.get("file_name", "")).strip() for item in source_reports if str(item.get("file_name", "")).strip()]
    if not source_names and source.get("file_name"):
        source_names = [str(source.get("file_name", "")).strip()]
    report_summary_text = " ".join(
        part
        for part in [
            str(report_summary.get("thesis_summary", "")).strip(),
            str(report_summary.get("mechanism_summary", "")).strip(),
            str(report_summary.get("compact_excerpt", "")).strip(),
        ]
        if part
    )
    return {
        "source_report_count": len(source_reports) if source_reports else (1 if source else 0),
        "source_report_names": source_names[:10],
        "report_summary": {
            "thesis_summary": _compact_event_text(report_summary.get("thesis_summary", ""), max_chars=520),
            "mechanism_summary": _compact_event_text(report_summary.get("mechanism_summary", ""), max_chars=420),
            "proxy_candidates": [
                str(item)
                for item in list(report_summary.get("proxy_candidates", []))
                if str(item).strip()
            ][:8],
            "compact_excerpt": _compact_event_text(report_summary.get("compact_excerpt", ""), max_chars=700),
        },
        "report_summary_text": _compact_event_text(report_summary_text, max_chars=900),
        "report_suggested_hypotheses": _report_suggested_hypotheses(report_summary, initial_seed),
        "report_initial_seed": {
            "seed_type": str(initial_seed.get("seed_type", "")).strip(),
            "seed_text": _compact_event_text(initial_seed.get("seed_text", ""), max_chars=520),
            "hypothesis_hint": _compact_event_text(initial_seed.get("hypothesis_hint", ""), max_chars=520),
            "expression": _compact_event_text(initial_seed.get("expression", ""), max_chars=260),
        },
    }


def _report_suggested_hypotheses(report_summary: dict[str, Any], initial_seed: dict[str, Any]) -> list[dict[str, str]]:
    thesis = _compact_event_text(report_summary.get("thesis_summary", ""), max_chars=240)
    mechanism = _compact_event_text(report_summary.get("mechanism_summary", ""), max_chars=220)
    seed_text = _compact_event_text(initial_seed.get("seed_text", initial_seed.get("hypothesis_hint", "")), max_chars=220)
    proxies = [
        str(item).strip()
        for item in list(report_summary.get("proxy_candidates", []))
        if str(item).strip()
    ][:5]
    suggestions: list[dict[str, str]] = []
    anchor = thesis or seed_text or mechanism
    if anchor:
        suggestions.append(
            {
                "title": "研报主机制迁移",
                "hypothesis": (
                    "将研报主机制迁移为 crypto 截面排序代理，检验该代理的分位排序是否预测下一期收益："
                    f"{anchor}"
                ),
            }
        )
    for proxy in proxies[:4]:
        suggestions.append(
            {
                "title": f"{proxy} 代理检验",
                "hypothesis": (
                    f"用可用 crypto panel 字段构造稳定的 {proxy} 截面代理，"
                    "在控制换手后检验 abs(IC)、Rank IC 和净 Sharpe。"
                ),
            }
        )
    return suggestions[:5]


def _build_evolution_console_message(stage: str, payload: dict[str, Any]) -> str:
    if stage == "evolution_round_planned":
        evolved = int(payload.get("evolved_task_count", 0) or 0)
        evolved_text = f"，进化 {evolved}" if evolved else ""
        return (
            f"计划轨迹任务 {payload.get('task_count', 0)} 个"
            f"（原始 {payload.get('original_task_count', 0)}，变异 {payload.get('mutation_task_count', 0)}，交叉 {payload.get('crossover_task_count', 0)}{evolved_text}）"
        )
    if stage == "evolution_round_all_failed":
        return (
            f"本轮 {payload.get('task_count', 0)} 条轨迹全部失败，"
            f"累计失败轮次 {payload.get('no_progress_rounds', 0)}/"
            f"{payload.get('early_stop_patience', 0)}"
        )
    if stage == "evolution_round_skipped":
        return "没有可执行任务，等待下一轮或触发早停"
    if stage == "evolution_task_planned":
        direction_id = payload.get("direction_id")
        mutation_type = str(payload.get("mutation_type", "")).strip()
        parent_ids = payload.get("parent_ids", [])
        parent_text = f"，父代 {len(parent_ids)} 个" if isinstance(parent_ids, list) and parent_ids else ""
        mutation_text = f"，变异类型 {mutation_type}" if mutation_type else ""
        return f"规划轨迹任务 {direction_id}{mutation_text}{parent_text}"
    if stage == "evolution_task_started":
        start_agent = str(payload.get("partial_replay_start_agent", "")).strip()
        if start_agent:
            return f"开始执行因子挖掘链路，从{_display_agent_short(start_agent)}开始"
        return "开始执行因子挖掘链路，从假设生成开始"
    if stage == "evolution_task_finished":
        return f"完成因子挖掘链路，执行步骤 {payload.get('agent_count', 0)} 个，错误 {payload.get('error_count', 0)} 个"
    if stage in {"direction_decomposer_started", "evolved_direction_decomposer_started"}:
        return f"目标生成 {payload.get('target_direction_count', 0)} 条假设"
    if stage in {"direction_decomposer_finished", "evolved_direction_decomposer_finished"}:
        mode = str(payload.get("direction_decomposer_mode", "")).strip()
        mode_text = f"，模式 {mode}" if mode else ""
        count = payload.get("initial_hypothesis_count", payload.get("direction_count", 0))
        return f"得到 {count} 条假设{mode_text}"
    if stage in {"direction_decomposer_failed", "evolved_direction_decomposer_failed", "evolution_agent_failed"}:
        return f"错误：{payload.get('error', '')}"
    if stage == "evolved_seed_decomposed":
        parent_ids = payload.get("parent_ids", [])
        parent_text = f"，父代 {len(parent_ids)} 个" if isinstance(parent_ids, list) else ""
        return f"生成下一代假设 {payload.get('direction_count', 0)} 条{parent_text}"
    if stage == "evolution_agent_started":
        return f"目标：变异 {payload.get('mutation_target_count', 0)} 个，交叉 {payload.get('crossover_target_count', 0)} 个"
    if stage == "evolution_agent_finished":
        return f"产出：变异 {payload.get('mutation_task_count', 0)} 个，交叉 {payload.get('crossover_task_count', 0)} 个"
    if stage == "market_payload_started":
        return f"读取本地 panel 数据：{payload.get('panel_data_path', '')}"
    if stage == "market_payload_finished":
        features = payload.get("feature_count", 0)
        specs = payload.get("parametric_specification_count", 0)
        regime = "有" if payload.get("has_market_regime_summary") else "无"
        return f"完成市场/数据摘要，特征 {features} 个，参数设定 {specs} 组，市场状态摘要：{regime}"
    if stage == "research_report_payload_started":
        count = payload.get("report_count", 0)
        return f"读取并清洗研报 {count} 份"
    if stage == "research_report_payload_finished":
        count = payload.get("source_report_count", payload.get("report_count", 0))
        status = str(payload.get("multi_report_synthesis_status", "")).strip()
        status_text = f"，合成状态 {status}" if status else ""
        return f"完成研报 payload，来源 {count} 份{status_text}"
    if stage == "initialization_payload_ready":
        sources = []
        if payload.get("has_research_report"):
            sources.append("研报")
        if payload.get("has_market_regime_summary"):
            sources.append("市场状态")
        if payload.get("has_dataset_summary"):
            sources.append("数据摘要")
        if payload.get("parametric_specification_count", 0):
            sources.append("参数设定")
        source_text = "、".join(sources) if sources else "基础输入"
        return f"A0 输入已准备：{source_text}；目标初始假设 {payload.get('target_initial_hypotheses', 0)} 条"
    return ""


def _build_system_console_message(stage: str, payload: dict[str, Any]) -> str:
    if stage == "holdout_validation_finished":
        return f"状态 {_display_status(payload.get('status', ''))}，因子 {payload.get('factor_count', 0)} 个，输出 {payload.get('output_file', '')}"
    if stage == "holdout_validation_failed":
        return f"错误：{payload.get('error', '')}，输出 {payload.get('output_file', '')}"
    if stage == "report_generation_failed":
        return f"错误：{payload.get('error', '')}"
    if stage == "factor_pool_finished":
        return (
            f"状态 {_display_status(payload.get('status', ''))}，"
            f"入池 {payload.get('selected_count', 0)} 个，输出 {payload.get('output_file', '')}"
        )
    if stage == "factor_pool_failed":
        return f"错误：{payload.get('error', '')}"
    if stage == "factor_cache_cleanup_finished":
        cleanup = payload.get("cleanup", {})
        if isinstance(cleanup, dict):
            return f"状态 {_display_status(cleanup.get('status', ''))}，删除 {cleanup.get('deleted_files', 0)} 个文件"
        return ""
    if stage == "session_finished":
        return f"状态 {_display_status(payload.get('status', ''))}，loop {payload.get('loop_count', 0)} 个，错误 {payload.get('total_errors', 0)} 个"
    return ""



def _print_final_run_summary(output: dict[str, Any]) -> None:
    """Print a compact human-facing summary; detailed data stays in SQLite/JSON files."""
    overview = output.get("overview", {})
    artifacts = output.get("artifacts", {})
    if not isinstance(overview, dict):
        overview = {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    lines = [
        "",
        "==================== 运行完成 ====================",
        f"运行 ID: {output.get('run_id', '')}",
        f"状态: {_display_status(output.get('status', ''))}",
        (
            f"轨迹数(小轮次): {overview.get('trajectory_count', overview.get('loop_count', 0))}，"
            f"进化阶段(中轮次): {overview.get('ran_phase_count', 0)} 个（"
            f"完成 {overview.get('completed_evolution_phase_count', 0)}，"
            f"全败 {overview.get('failed_evolution_phase_count', 0)}），"
            f"进化迭代(大轮次): {overview.get('completed_iteration_count', 0)}/{overview.get('target_iteration_count', 0)}，"
            f"失败轨迹: {overview.get('failed_trajectory_count', overview.get('failed_round_count', 0))}，"
            f"错误数: {overview.get('total_errors', 0)}"
        ),
        f"运行目录: {artifacts.get('log_dir', '')}",
        f"SQLite 主存储: {artifacts.get('run_store', '')}",
        f"摘要文件: {artifacts.get('summary', '')}",
    ]
    factor_summary = output.get("factor_summary", {})
    if isinstance(factor_summary, dict):
        lines.append(
            f"因子摘要: 共 {factor_summary.get('factor_count', 0)} 条，去重后 {factor_summary.get('unique_factor_count', 0)} 个"
        )
    report_path = artifacts.get("report_markdown")
    if report_path:
        lines.append(f"报告文件: {report_path}")
    cache_cleanup = overview.get("factor_cache_cleanup", {})
    if isinstance(cache_cleanup, dict):
        lines.append(f"缓存清理: {cache_cleanup.get('status', 'unknown')}")
    lines.append("完整细节已写入 SQLite 和 summary.json，终端不再打印大段 JSON。")
    lines.append("==================================================")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _build_debug_logger(enabled: bool, tag: str) -> Callable[[dict[str, Any]], None] | None:
    if not enabled:
        return None

    def _logger(payload: dict[str, Any]) -> None:
        _print_debug_event(tag, payload)

    return _logger


def _extract_keys(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return sorted(str(key) for key in payload.keys())


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，缺失或非法时返回默认值。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_int_alias(primary: str, fallback: str, default: int) -> int:
    """读取新版环境变量；缺失时兼容旧变量名。"""
    if os.getenv(primary, "").strip():
        return _env_int(primary, default)
    return _env_int(fallback, default)


def _env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，支持常见真值写法。"""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量，缺失或非法时返回默认值。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_optional(name: str) -> float | None:
    """读取可选浮点环境变量，缺失或非法时返回 None。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _summarize_best_trajectories(session: Any, top_n: int = 5) -> list[dict[str, Any]]:
    """生成命令行最终输出中的最佳轨迹摘要。"""
    trajectory_pool = getattr(session, "trajectory_pool", None)
    if trajectory_pool is None:
        return []
    return [
        {
            "trajectory_id": item.trajectory_id,
            "phase": item.phase.value if hasattr(item.phase, "value") else str(item.phase),
            "round_idx": item.round_idx,
            "direction_id": item.direction_id,
            "parent_ids": list(item.parent_ids),
            "reward_score": item.get_reward_score(),
            "metrics": _compact_metrics(dict(item.backtest_metrics)),
        }
        for item in trajectory_pool.get_best_trajectories(top_n=top_n)
    ]


def _extract_evolution_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """从进度事件或嵌套共享快照中提取遗传迭代元数据。"""
    metadata_keys = (
        "phase",
        "round_idx",
        "direction_id",
        "trajectory_id",
        "parent_ids",
        "mutation_type",
        "evolution_planner_mode",
        "direction_decomposer_mode",
    )
    metadata: dict[str, Any] = {}

    def _merge(source: Any) -> None:
        if not isinstance(source, dict):
            return
        for key in metadata_keys:
            if key in metadata and metadata[key] not in (None, "", []):
                continue
            if key in source:
                metadata[key] = source[key]

    _merge(payload)
    for snapshot_key in ("input_snapshot", "output_snapshot"):
        snapshot = payload.get(snapshot_key, {})
        if not isinstance(snapshot, dict):
            continue
        _merge(snapshot.get("shared_context", {}))
        _merge(snapshot.get("shared_updates", {}))
    return metadata


def _enrich_progress_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把嵌套的遗传迭代元数据补到进度事件顶层。"""
    enriched = dict(payload)
    for key, value in _extract_evolution_metadata(payload).items():
        enriched.setdefault(key, value)
    return enriched


def _summarize_rounds(session: Any) -> list[dict[str, Any]]:
    """为摘要文件汇总已执行轮次和血缘字段。"""
    rows: list[dict[str, Any]] = []
    traces = list(getattr(session, "loop_traces", []))
    for idx, item in enumerate(getattr(session, "rounds", [])):
        trace = traces[idx] if idx < len(traces) else None
        shared = getattr(trace, "final_shared_context", {}) if trace is not None else {}
        loop_summary, _final_context, _records = _build_loop_summary(item.loop_index, trace) if trace is not None else ({}, {}, [])
        rows.append(
            {
                "loop_index": item.loop_index,
                "attempt_index": item.attempt_index,
                "phase": str(shared.get("phase", item.phase)),
                "round_idx": int(shared.get("round_idx", item.loop_index) or 0),
                "paper_iteration": _paper_iteration_from_round(shared.get("round_idx", item.loop_index)),
                "paper_phase": _paper_phase_label(shared.get("phase", item.phase), shared.get("round_idx", item.loop_index)),
                "direction_id": int(shared.get("direction_id", 0) or 0),
                "trajectory_id": str(shared.get("trajectory_id", item.trajectory_id)),
                "parent_ids": [str(parent_id) for parent_id in list(shared.get("parent_ids", item.parent_ids))],
                "mutation_type": str(shared.get("mutation_type", "")),
                "evolution_planner_mode": str(shared.get("evolution_planner_mode", "")),
                "error_count": item.error_count,
                "agent_count": item.agent_count,
                "hypothesis": str(loop_summary.get("hypothesis", shared.get("hypothesis", ""))),
                "factors": list(loop_summary.get("factors", []) or []),
                "metrics": _first_dict(loop_summary.get("metrics")),
                "per_factor_metrics": _first_dict(loop_summary.get("per_factor_metrics")),
                "feedback_summary": str(loop_summary.get("feedback_summary", "")),
            }
        )
    return rows


def _summarize_factors_across_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a readable factor index for summary.json without storing raw traces."""
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}

    for round_row in rounds:
        loop_index = round_row.get("loop_index")
        phase = round_row.get("phase", "")
        trajectory_id = round_row.get("trajectory_id", "")
        parent_ids = list(round_row.get("parent_ids", []) or [])
        for factor in list(round_row.get("factors", []) or []):
            if not isinstance(factor, dict):
                continue
            name = str(factor.get("factor_name", factor.get("name", ""))).strip()
            if not name:
                continue
            metrics = _first_dict(factor.get("metrics"))
            row = {
                "loop_index": loop_index,
                "phase": phase,
                "trajectory_id": trajectory_id,
                "parent_ids": parent_ids,
                "factor_name": name,
                "expression": str(factor.get("expression", "")),
                "description": str(factor.get("description", "")),
                "implementation_status": str(factor.get("implementation_status", "")),
                "signal_cache_key": str(factor.get("signal_cache_key", "")),
                "metrics": _compact_metrics(metrics),
            }
            rows.append(row)

            existing = by_name.get(name)
            current_sharpe = metrics.get("sharpe")
            existing_sharpe = _first_dict(existing.get("metrics") if existing else {}).get("sharpe") if existing else None
            if existing is None or (
                isinstance(current_sharpe, (int, float))
                and (not isinstance(existing_sharpe, (int, float)) or float(current_sharpe) > float(existing_sharpe))
            ):
                by_name[name] = row

    return {
        "factor_count": len(rows),
        "unique_factor_count": len(by_name),
        "factors": rows,
        "best_by_name": list(by_name.values()),
    }


def _export_run_factor_summary_csv(
    *,
    run_id: str,
    rounds: list[dict[str, Any]],
    holdout_validation: dict[str, Any] | None,
    output_path: Path,
) -> Path:
    """Write a per-run readable factor table for quick inspection after each run."""
    fieldnames = [
        "round_idx",
        "paper_iteration",
        "paper_phase",
        "phase",
        "factor_name",
        "factor_expression",
        "return",
        "IC",
        "Rank IC",
        "sharpe",
    ]
    rows: list[dict[str, Any]] = []
    for round_row in rounds:
        per_factor_metrics = _first_dict(round_row.get("per_factor_metrics"))
        for factor in list(round_row.get("factors", []) or []):
            if not isinstance(factor, dict):
                continue
            factor_name = str(factor.get("factor_name", factor.get("name", ""))).strip()
            if not factor_name:
                continue
            expression = str(factor.get("expression", "")).strip()
            metrics = _first_dict(factor.get("metrics"))
            if not metrics and isinstance(per_factor_metrics, dict):
                metrics = _first_dict(per_factor_metrics.get(factor_name, {}))
            row: dict[str, Any] = {
                "round_idx": round_row.get("round_idx", ""),
                "paper_iteration": round_row.get("paper_iteration", ""),
                "paper_phase": round_row.get("paper_phase", ""),
                "phase": round_row.get("phase", ""),
                "factor_name": factor_name,
                "factor_expression": expression,
                "return": metrics.get("long_short_mean_return", metrics.get("annualized_return", "")),
                "IC": metrics.get("IC", ""),
                "Rank IC": metrics.get("Rank IC", ""),
                "sharpe": metrics.get("sharpe", ""),
            }
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _build_lineage_tree(session: Any) -> list[dict[str, Any]]:
    """为遗传迭代报告生成嵌套的父子轨迹树。"""
    trajectory_pool = getattr(session, "trajectory_pool", None)
    if trajectory_pool is None:
        return []
    nodes: dict[str, dict[str, Any]] = {}
    for trajectory in trajectory_pool.get_all():
        nodes[trajectory.trajectory_id] = {
            "trajectory_id": trajectory.trajectory_id,
            "phase": trajectory.phase.value if hasattr(trajectory.phase, "value") else str(trajectory.phase),
            "round_idx": trajectory.round_idx,
            "direction_id": trajectory.direction_id,
            "parent_ids": list(trajectory.parent_ids),
            "mutation_type": str(trajectory.extra_info.get("mutation_type", "")),
            "evolution_planner_mode": str(trajectory.extra_info.get("evolution_planner_mode", "")),
            "reward_score": trajectory.get_reward_score(),
            "children": [],
        }
    child_ids: set[str] = set()
    for trajectory in trajectory_pool.get_all():
        child = nodes.get(trajectory.trajectory_id)
        if child is None:
            continue
        for parent_id in trajectory.parent_ids:
            parent = nodes.get(str(parent_id))
            if parent is None:
                continue
            parent["children"].append(child)
            child_ids.add(trajectory.trajectory_id)
    roots = [node for trajectory_id, node in nodes.items() if trajectory_id not in child_ids]
    return roots or list(nodes.values())


def _summarize_segment_lineage(session: Any) -> list[dict[str, Any]]:
    """Compact trajectory segment lineage for QuantaAlpha-style reports."""
    trajectory_pool = getattr(session, "trajectory_pool", None)
    if trajectory_pool is None:
        return []
    rows: list[dict[str, Any]] = []
    for trajectory in trajectory_pool.get_all():
        extra = dict(getattr(trajectory, "extra_info", {}) or {})
        raw_segments = extra.get("segments", {})
        segments: dict[str, Any] = {}
        if isinstance(raw_segments, dict):
            for name in ("hypothesis_generation", "factor_expression", "implementation", "evaluation_feedback"):
                segment = raw_segments.get(name, {})
                if not isinstance(segment, dict):
                    continue
                anchors = segment.get("reusable_anchors", [])
                segments[name] = {
                    "summary": str(segment.get("summary", "")),
                    "failure_label": str(segment.get("failure_label", "")),
                    "reusable_anchors": list(anchors) if isinstance(anchors, list) else [],
                }
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "phase": trajectory.phase.value if hasattr(trajectory.phase, "value") else str(trajectory.phase),
                "round_idx": int(trajectory.round_idx),
                "paper_iteration": _paper_iteration_from_round(trajectory.round_idx),
                "paper_phase": _paper_phase_label(
                    trajectory.phase.value if hasattr(trajectory.phase, "value") else str(trajectory.phase),
                    trajectory.round_idx,
                ),
                "parent_ids": list(trajectory.parent_ids),
                "target_segment": str(extra.get("target_segment", "")),
                "partial_replay_start_agent": str(extra.get("partial_replay_start_agent", "")),
                "source_segments": [
                    dict(item)
                    for item in list(extra.get("source_segments", []))
                    if isinstance(item, dict)
                ],
                "segments": segments,
            }
        )
    return rows


def _summarize_evolution_planning(session: Any) -> dict[str, Any]:
    """汇总初始化假设与进化规划，供 summary/report 直接展示。"""
    trajectory_pool = getattr(session, "trajectory_pool", None)
    trajectories = [] if trajectory_pool is None else trajectory_pool.get_all()
    initial_hypotheses: list[dict[str, Any]] = []
    evolution_agent_plans_by_round: dict[int, dict[str, Any]] = {}

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
                "trajectory_id": str(item.get("trajectory_id", "")),
                "round_idx": int(bucket["round_idx"]),
                "direction_id": int(item.get("direction_id", 0) or 0),
                "direction": str(item.get("direction", "")),
                "parent_ids": [str(parent_id) for parent_id in list(item.get("parent_ids", []))],
                "mutation_type": str(item.get("mutation_type", "")),
                "evolution_planner_mode": str(item.get("planner_mode", item.get("evolution_planner_mode", ""))),
                "rationale": str(item.get("rationale", "")),
                "fusion_logic": str(item.get("fusion_logic", "")),
                "target_segment": str(item.get("target_segment", "")),
                "source_segments": [dict(segment) for segment in list(item.get("source_segments", [])) if isinstance(segment, dict)],
                "fusion_plan": str(item.get("fusion_plan", "")),
            }
        )

    for trajectory in trajectories:
        phase = trajectory.phase.value if hasattr(trajectory.phase, "value") else str(trajectory.phase)
        row = {
            "trajectory_id": trajectory.trajectory_id,
            "round_idx": int(trajectory.round_idx),
            "direction_id": int(trajectory.direction_id),
            "direction": str(trajectory.extra_info.get("direction", "")),
            "parent_ids": list(trajectory.parent_ids),
            "mutation_type": str(trajectory.extra_info.get("mutation_type", "")),
            "evolution_planner_mode": str(trajectory.extra_info.get("evolution_planner_mode", "")),
            "rationale": str(trajectory.extra_info.get("rationale", "")),
            "fusion_logic": str(trajectory.extra_info.get("fusion_logic", "")),
            "evolution_seed": str(trajectory.extra_info.get("evolution_seed", "")),
            "target_segment": str(trajectory.extra_info.get("target_segment", "")),
            "fault_diagnosis": str(trajectory.extra_info.get("fault_diagnosis", "")),
            "source_segments": [
                dict(item)
                for item in list(trajectory.extra_info.get("source_segments", []))
                if isinstance(item, dict)
            ],
            "fusion_plan": str(trajectory.extra_info.get("fusion_plan", "")),
            "evolution_plan_summaries": [
                dict(item)
                for item in list(trajectory.extra_info.get("evolution_plan_summaries", []))
                if isinstance(item, dict)
            ],
        }
        if phase == "original":
            row["direction_decomposer_mode"] = row["evolution_planner_mode"]
            initial_hypotheses.append(row)
            continue
        bucket = evolution_agent_plans_by_round.setdefault(
            int(trajectory.round_idx),
            {
                "round_idx": int(trajectory.round_idx),
                "mutation_tasks": [],
                "crossover_tasks": [],
                "evolved_tasks": [],
                "_planned_task_keys": set(),
            },
        )
        if phase == "evolved" or (row["mutation_type"] == "generation_branch" and row["evolution_plan_summaries"]):
            bucket["evolved_tasks"].append(row)
            for plan_item in row["evolution_plan_summaries"]:
                _add_planned_task(bucket, plan_item)
        elif phase == "mutation":
            bucket["mutation_tasks"].append(row)
        elif phase == "crossover":
            bucket["crossover_tasks"].append(row)

    evolution_agent_plans = [
        {
            **{key: value for key, value in bucket.items() if key != "_planned_task_keys"},
            "planned_mutation_count": len(bucket["mutation_tasks"]),
            "planned_crossover_count": len(bucket["crossover_tasks"]),
            "executed_evolved_count": len(bucket["evolved_tasks"]),
        }
        for _round_idx, bucket in sorted(evolution_agent_plans_by_round.items())
    ]
    return {
        "direction_decomposer_mode": _dominant_mode(item.get("direction_decomposer_mode", "") for item in initial_hypotheses),
        "initial_hypothesis_count": len(initial_hypotheses),
        "initial_hypotheses": initial_hypotheses,
        # Backward-compatible summary keys kept for older reports/scripts.
        "initial_direction_count": len(initial_hypotheses),
        "initial_directions": initial_hypotheses,
        "evolution_planner_mode": _dominant_mode(
            task.get("evolution_planner_mode", "")
            for plan in evolution_agent_plans
            for task in [*plan["mutation_tasks"], *plan["crossover_tasks"], *plan.get("evolved_tasks", [])]
        ),
        "planned_mutation_count": sum(int(plan["planned_mutation_count"]) for plan in evolution_agent_plans),
        "planned_crossover_count": sum(int(plan["planned_crossover_count"]) for plan in evolution_agent_plans),
        "executed_evolved_count": sum(int(plan["executed_evolved_count"]) for plan in evolution_agent_plans),
        "evolution_agent_plans": evolution_agent_plans,
    }


def _dominant_mode(values: Any) -> str:
    modes = [str(value).strip() for value in values if str(value).strip()]
    if not modes:
        return ""
    return "llm" if "llm" in modes else modes[0]


def _append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(payload), ensure_ascii=False))
        file.write("\n")


def _append_raw_llm_io(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def _append_jsonl_gz(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file_path, "at", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def _write_json_gz(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file_path, "wt", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, separators=(",", ":"))


def _artifact_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict_keys(value: Any) -> list[str]:
    return sorted(str(key) for key in value.keys()) if isinstance(value, dict) else []


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _finite_metric_text(metrics: dict[str, Any], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={float(value):.4f}")
        elif value not in (None, ""):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _reward_score_from_metrics(metrics: dict[str, Any], factor_entry: dict[str, Any] | None = None) -> float | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    try:
        return float(FactorLibraryManager.compute_reward_score(metrics, factor_entry=factor_entry))
    except Exception:
        return None


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["sharpe", "Rank IC", "Rank ICIR", "IC", "ICIR", "coverage", "turnover"]
    return {key: metrics[key] for key in keys if key in metrics}


def _compact_holdout_validation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    keys = ["status", "metric", "threshold", "comparison", "factor_count", "errors"]
    compact = {key: payload[key] for key in keys if key in payload}
    if "train_window" in payload:
        compact["train_window"] = payload["train_window"]
    if "validation_window" in payload:
        compact["validation_window"] = payload["validation_window"]
    elif "test_window" in payload:
        compact["validation_window"] = payload["test_window"]
    return compact


def _write_detail_json(file_path: Path, payload: dict[str, Any] | list[Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _trajectory_pool_path() -> Path:
    return Path(
        os.getenv(
            "TRAJECTORY_POOL_PATH",
            str(PROJECT_ROOT / "artifacts" / "trajectory_pool.jsonl"),
        )
    )


def _build_progress_summary_from_updates(shared_updates: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(shared_updates, dict):
        return {}
    agent_summary: dict[str, Any] = {}
    if isinstance(shared_updates.get("hypothesis_structured"), dict) or shared_updates.get("hypothesis"):
        hypothesis = str(shared_updates.get("hypothesis", "")).strip()
        agent_summary["text"] = f"假设已生成：{hypothesis[:80]}" if hypothesis else "假设已生成"
        return agent_summary
    experiment_spec = _first_dict(shared_updates.get("experiment_spec"))
    if experiment_spec:
        factors = list(experiment_spec.get("factors", []) or [])
        names = [
            str(item.get("factor_name", item.get("name", ""))).strip()
            for item in factors
            if isinstance(item, dict)
        ]
        factor_text = _format_factor_list(names)
        agent_summary["text"] = f"实验设计完成，候选因子 {len(factors)} 个" + (f"：{factor_text}" if factor_text else "")
        return agent_summary
    calculation_report = _first_dict(shared_updates.get("calculation_report"))
    factor_implementation = _first_dict(shared_updates.get("factor_implementation"))
    if calculation_report or factor_implementation:
        total = calculation_report.get("total_factors", factor_implementation.get("implementation_count", 0))
        generated = calculation_report.get("generated_factors", 0)
        failed = calculation_report.get("failed_factors", [])
        failed_count = len(failed) if isinstance(failed, list) else 0
        cache_hits = calculation_report.get("signal_cache_hits", 0)
        agent_summary["text"] = f"因子代码生成完成，成功 {generated}/{total}，失败 {failed_count}，缓存命中 {cache_hits}"
        return agent_summary
    backtest_report = _first_dict(shared_updates.get("backtest_report"))
    metrics = _first_dict(shared_updates.get("metrics"))
    if backtest_report or metrics:
        evaluated = backtest_report.get("evaluated_factors", 0)
        retained = backtest_report.get("train_retained_factors", 0)
        filtered = backtest_report.get("train_filtered_factors", 0)
        selected = str(backtest_report.get("selected_factor_name", "")).strip()
        per_factor_metrics = _first_dict(shared_updates.get("per_factor_metrics"))
        selected_metrics = (
            _first_dict(per_factor_metrics.get(selected))
            if selected and isinstance(per_factor_metrics, dict)
            else {}
        )
        display_metrics = selected_metrics or metrics
        reward_score = _reward_score_from_metrics(display_metrics)
        metric_text = _finite_metric_text(display_metrics, ["sharpe", "Rank IC", "Rank ICIR", "IC", "ICIR"])
        pieces = [f"回测完成，评估 {evaluated} 个，保留 {retained} 个，过滤 {filtered} 个"]
        # 评估 0 个时把失败原因带出来，便于排查（否则只看到 0 个无从下手）。
        if not evaluated:
            cause = str(backtest_report.get("error", "")).strip()
            errors = backtest_report.get("errors", {})
            if isinstance(errors, dict) and errors:
                first = next(iter(errors.values()), {})
                if isinstance(first, dict):
                    fname = str(first.get("factor_name", "")).strip()
                    msg = str(first.get("message", first.get("status", ""))).strip()
                    etype = str(first.get("exception_type", "")).strip()
                    detail = " ".join(p for p in [etype, msg] if p).strip()
                    if detail:
                        cause = f"{detail}" + (f"（如 {fname}）" if fname else "")
            summary_code = str(backtest_report.get("summary", "")).strip()
            if cause:
                pieces.append(f"失败原因：{cause}")
            elif summary_code:
                pieces.append(f"状态：{summary_code}")
        if selected:
            pieces.append(f"最佳{_format_factor_display(selected)}")
        if reward_score is not None:
            pieces.append(f"Reward={reward_score:.4f}")
        if metric_text:
            pieces.append(metric_text)
        agent_summary["text"] = "；".join(pieces)
        if selected:
            agent_summary["best_factor_name"] = selected
        if reward_score is not None:
            agent_summary["reward_score"] = reward_score
        if display_metrics:
            agent_summary["metrics"] = _compact_metrics(display_metrics)
        return agent_summary
    feedback = _first_dict(shared_updates.get("feedback"))
    if feedback:
        action_labels = {
            "continue": "继续当前方向",
            "proxy_reconstruction": "重构代理变量",
            "parameter_tuning": "参数调优",
            "hypothesis_revision": "修订假说",
            "hypothesis_replacement": "替换假说",
            "hypothesis_reset": "重置假说",
        }
        pieces: list[str] = []
        supported = feedback.get("hypothesis_supported")
        if isinstance(supported, bool):
            pieces.append("假说成立" if supported else "假说不成立")
        action = str(feedback.get("recommended_action", "")).strip()
        if action:
            pieces.append(f"建议{action_labels.get(action, action)}")
        # 最佳主线因子（名字可能很长，截断展示）
        diagnostics = _first_dict(feedback.get("signal_diagnostics"))
        mainline = _first_dict(_first_dict(diagnostics.get("group_performance")).get("mainline"))
        best_factor = str(mainline.get("best_factor", "")).strip()
        if best_factor:
            short = best_factor if len(best_factor) <= 32 else f"{best_factor[:31]}…"
            pieces.append(f"最佳因子 {short}")
        # 结构化字段都缺时，退回 LLM 文本摘要（截断）
        if not pieces:
            summary = str(feedback.get("summary", "")).strip()
            if summary:
                pieces.append(summary[:80] + ("…" if len(summary) > 80 else ""))
        agent_summary["text"] = " · ".join(pieces)
        return agent_summary
    return {}


def _extract_compact_final_context(final_context: dict[str, Any]) -> dict[str, Any]:
    qlib_experiment = _first_dict(final_context.get("qlib_factor_experiment"))
    experiment_spec = _first_dict(final_context.get("experiment_spec"), qlib_experiment.get("experiment_spec"))
    feedback = _first_dict(final_context.get("feedback"), qlib_experiment.get("feedback"))
    backtest_report = _first_dict(final_context.get("backtest_report"), qlib_experiment.get("backtest_report"))
    calculation_report = _first_dict(final_context.get("calculation_report"), qlib_experiment.get("calculation_report"))
    factor_implementation = _first_dict(final_context.get("factor_implementation"), qlib_experiment.get("factor_implementation"))
    metrics = _first_dict(final_context.get("metrics"), qlib_experiment.get("metrics"))
    per_factor_metrics = _first_dict(final_context.get("per_factor_metrics"), qlib_experiment.get("per_factor_metrics"))
    factors = list(qlib_experiment.get("factors", experiment_spec.get("factors", [])) or [])
    compact_experiment = {
        "experiment_id": str(qlib_experiment.get("experiment_id", final_context.get("experiment_id", ""))),
        "target_hypothesis": str(qlib_experiment.get("target_hypothesis", experiment_spec.get("target_hypothesis", ""))),
        "factors": factors,
        "factor_names": list(qlib_experiment.get("factor_names", experiment_spec.get("factor_names", [])) or []),
        "task_plan": list(qlib_experiment.get("task_plan", final_context.get("task_plan", [])) or []),
        "factor_implementation": factor_implementation,
        "calculation_report": calculation_report,
        "backtest_report": backtest_report,
        "metrics": metrics,
        "per_factor_metrics": per_factor_metrics,
        "feedback": feedback,
        "phase": str(qlib_experiment.get("phase", final_context.get("phase", ""))),
        "round_idx": int(qlib_experiment.get("round_idx", final_context.get("round_idx", 0)) or 0),
        "trajectory_id": str(qlib_experiment.get("trajectory_id", final_context.get("trajectory_id", ""))),
        "parent_ids": list(qlib_experiment.get("parent_ids", final_context.get("parent_ids", [])) or []),
    }
    return {
        "hypothesis": final_context.get("hypothesis", compact_experiment["target_hypothesis"]),
        "hypothesis_reasoning": final_context.get("hypothesis_reasoning", ""),
        "hypothesis_structured": _first_dict(final_context.get("hypothesis_structured")),
        "research_report": _first_dict(final_context.get("research_report")),
        "experiment_spec": experiment_spec,
        "qlib_factor_experiment": compact_experiment,
        "factor_implementation": factor_implementation,
        "calculation_report": calculation_report,
        "backtest_report": backtest_report,
        "metrics": metrics,
        "per_factor_metrics": per_factor_metrics,
        "feedback": feedback,
        "phase": str(final_context.get("phase", compact_experiment["phase"])),
        "round_idx": int(final_context.get("round_idx", compact_experiment["round_idx"]) or 0),
        "direction_id": int(final_context.get("direction_id", 0) or 0),
        "trajectory_id": str(final_context.get("trajectory_id", compact_experiment["trajectory_id"])),
        "parent_ids": list(final_context.get("parent_ids", compact_experiment["parent_ids"]) or []),
        "mutation_type": str(final_context.get("mutation_type", "")),
        "evolution_planner_mode": str(final_context.get("evolution_planner_mode", "")),
        "direction_decomposer_mode": str(final_context.get("direction_decomposer_mode", "")),
        "evolution_guidance": str(final_context.get("evolution_guidance", "")),
        "evolution_seed": str(final_context.get("evolution_seed", "")),
        "evolution_plan_summaries": [
            dict(item)
            for item in list(final_context.get("evolution_plan_summaries", []))
            if isinstance(item, dict)
        ],
        "round_status": str(final_context.get("round_status", "completed")),
        "failed_agent_name": str(final_context.get("failed_agent_name", "")),
        "failed_error": str(final_context.get("failed_error", "")),
        "stale_results_removed": bool(final_context.get("stale_results_removed", False)),
        "stale_result_keys_removed": list(final_context.get("stale_result_keys_removed", []) or []),
    }


def _factor_rows_from_context(final_context: dict[str, Any]) -> list[dict[str, Any]]:
    qlib_experiment = _first_dict(final_context.get("qlib_factor_experiment"))
    factors = list(qlib_experiment.get("factors", []) or [])
    per_factor_metrics = _first_dict(qlib_experiment.get("per_factor_metrics"), final_context.get("per_factor_metrics"))
    implementations = list(_first_dict(qlib_experiment.get("factor_implementation"), final_context.get("factor_implementation")).get("implementations", []) or [])
    implementation_map = {
        str(item.get("factor_name", "")).strip(): item
        for item in implementations
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        name = str(factor.get("factor_name", factor.get("name", ""))).strip()
        implementation = implementation_map.get(name, {})
        rows.append(
            {
                **factor,
                "factor_name": name,
                "expression": str(implementation.get("expression", factor.get("expression", ""))),
                "implementation_status": str(implementation.get("status", "")),
                "signal_cache_key": str(implementation.get("signal_cache_key", "")),
                "source_pickle_path": str(implementation.get("source_pickle_path", "")),
                "metrics": _first_dict(per_factor_metrics.get(name, {})) if isinstance(per_factor_metrics, dict) else {},
            }
        )
    return rows


def _build_loop_summary(loop_index: int, trace: Any) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(getattr(trace, "records", []), start=1):
        input_snapshot = getattr(record, "input_snapshot", {})
        output_snapshot = getattr(record, "output_snapshot", {})
        records.append(
            {
                "agent_index": idx,
                "agent_name": getattr(record, "agent_name", ""),
                "started_at": getattr(record, "started_at", None),
                "ended_at": getattr(record, "ended_at", None),
                "elapsed_seconds": getattr(record, "elapsed_seconds", 0.0),
                "input_keys": {
                    "private_context": _dict_keys(_first_dict(input_snapshot).get("private_context", {})),
                    "shared_context": _dict_keys(_first_dict(input_snapshot).get("shared_context", {})),
                },
                "output_keys": {
                    "shared_updates": _dict_keys(_first_dict(output_snapshot).get("shared_updates", {})),
                    "artifacts": _dict_keys(_first_dict(output_snapshot).get("artifacts", {})),
                    "private_context": _dict_keys(_first_dict(output_snapshot).get("private_context", {})),
                    "shared_context": _dict_keys(_first_dict(output_snapshot).get("shared_context", {})),
                },
                "error": getattr(record, "error", None),
            }
        )
    final_context = _extract_compact_final_context(_first_dict(getattr(trace, "final_shared_context", {})))
    qlib_experiment = _first_dict(final_context.get("qlib_factor_experiment"))
    started_values = [item.get("started_at") for item in records if item.get("started_at")]
    ended_values = [item.get("ended_at") for item in records if item.get("ended_at")]
    error_count = sum(1 for item in records if item.get("error"))
    summary = {
        "loop_index": loop_index,
        "status": final_context.get("round_status", "failed_incomplete" if error_count else "completed"),
        "phase": final_context.get("phase", qlib_experiment.get("phase", "")),
        "round_idx": int(final_context.get("round_idx", qlib_experiment.get("round_idx", loop_index)) or 0),
        "direction_id": int(final_context.get("direction_id", 0) or 0),
        "trajectory_id": str(final_context.get("trajectory_id", qlib_experiment.get("trajectory_id", ""))),
        "parent_ids": list(final_context.get("parent_ids", qlib_experiment.get("parent_ids", [])) or []),
        "hypothesis": final_context.get("hypothesis", qlib_experiment.get("target_hypothesis", "")),
        "factors": _factor_rows_from_context(final_context),
        "metrics": _first_dict(qlib_experiment.get("metrics"), final_context.get("metrics")),
        "per_factor_metrics": _first_dict(qlib_experiment.get("per_factor_metrics"), final_context.get("per_factor_metrics")),
        "feedback_summary": str(_first_dict(final_context.get("feedback"), qlib_experiment.get("feedback")).get("summary", "")),
        "error_count": error_count,
        "agent_count": len(records),
        "started_at": min(started_values) if started_values else "",
        "ended_at": max(ended_values) if ended_values else "",
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in records), 3),
        "agent_timings": [
            {
                "agent_name": item.get("agent_name"),
                "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                "error": item.get("error"),
            }
            for item in records
        ],
    }
    return summary, final_context, records


def _save_loop_trace(output_dir: Path, loop_index: int, trace: Any, *, debug_full_trace: bool = False) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, final_context, records = _build_loop_summary(loop_index, trace)
    payload = {
        "loop_index": loop_index,
        "storage_mode": "sqlite_primary_light_json",
        "record_count": len(records),
        "summary": summary,
        "records": records,
        "final_shared_context_keys": _dict_keys(getattr(trace, "final_shared_context", {})),
        "final_shared_context": final_context,
    }
    (output_dir / f"loop_{loop_index:03d}.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    debug_path: Path | None = None
    if debug_full_trace:
        full_records: list[dict[str, Any]] = []
        for record in getattr(trace, "records", []):
            full_records.append(
                {
                    "agent_name": getattr(record, "agent_name", ""),
                    "started_at": getattr(record, "started_at", None),
                    "ended_at": getattr(record, "ended_at", None),
                    "elapsed_seconds": getattr(record, "elapsed_seconds", 0.0),
                    "input_snapshot": getattr(record, "input_snapshot", {}),
                    "output_snapshot": getattr(record, "output_snapshot", {}),
                    "error": getattr(record, "error", None),
                }
            )
        debug_path = output_dir / f"loop_{loop_index:03d}.debug.json.gz"
        _write_json_gz(
            debug_path,
            {
                "loop_index": loop_index,
                "record_count": len(full_records),
                "records": full_records,
                "final_shared_context": getattr(trace, "final_shared_context", {}),
            },
        )
    return summary, final_context, records, debug_path


def _compact_progress_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"input_snapshot", "output_snapshot"}
    }
    if isinstance(payload.get("input_snapshot"), dict):
        compact["input_snapshot_keys"] = {
            "private_context": _dict_keys(payload["input_snapshot"].get("private_context", {})),
            "shared_context": _dict_keys(payload["input_snapshot"].get("shared_context", {})),
        }
    if isinstance(payload.get("output_snapshot"), dict):
        shared_updates = _first_dict(payload["output_snapshot"].get("shared_updates", {}))
        compact["output_snapshot_keys"] = {
            "shared_updates": _dict_keys(shared_updates),
            "artifacts": _dict_keys(payload["output_snapshot"].get("artifacts", {})),
            "private_context": _dict_keys(payload["output_snapshot"].get("private_context", {})),
            "shared_context": _dict_keys(payload["output_snapshot"].get("shared_context", {})),
        }
        progress_summary = _build_progress_summary_from_updates(shared_updates)
        if progress_summary:
            compact["progress_summary"] = progress_summary
    return compact


def _compact_llm_io_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"request", "response"}
    }
    request = payload.get("request")
    request_summary = payload.get("request_summary")
    if isinstance(request, dict):
        compact["request_summary"] = {
            key: value
            for key, value in request.items()
            if key != "messages"
        }
    elif isinstance(request_summary, dict):
        compact["request_summary"] = request_summary
    response = payload.get("response")
    if isinstance(response, dict):
        content = str(response.get("content", ""))
        compact["response_summary"] = {
            "content_chars": len(content),
            "content_preview": content[:500],
            "finish_reason": response.get("finish_reason"),
            "raw_keys": _dict_keys(response.get("raw", {})),
        }
    return compact


def _factor_expression_identity(expression: object) -> str:
    return re.sub(r"\s+", "", str(expression or "").strip()).upper()


def _factor_id_for_storage(trajectory_id: str, factor: dict[str, Any]) -> str:
    expression = str(factor.get("expression", factor.get("factor_expression", ""))).strip()
    expression_identity = _factor_expression_identity(expression)
    if expression_identity:
        raw = f"expression|{expression_identity}"
    else:
        raw = "|".join(
            [
                "fallback",
                str(factor.get("factor_name", factor.get("name", ""))),
                str(trajectory_id),
            ]
        )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _clear_factor_signal_cache() -> dict[str, Any]:
    cache_dir = PROJECT_ROOT / "factor_library" / "cache"
    result: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "removed_files": 0,
        "removed_dirs": 0,
        "removed_bytes": 0,
        "errors": [],
    }
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        result["status"] = "created_empty_cache_dir"
        return result
    for item in cache_dir.iterdir():
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
                result["removed_dirs"] = int(result["removed_dirs"]) + 1
                continue
            size = item.lstat().st_size
            item.unlink()
            result["removed_files"] = int(result["removed_files"]) + 1
            result["removed_bytes"] = int(result["removed_bytes"]) + int(size)
        except Exception as exc:
            errors = result.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append({"path": str(item), "error": str(exc)})
    result["status"] = "ok" if not result.get("errors") else "partial"
    return result


def _require_env(keys: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        raw = os.getenv(key, "").strip()
        if not raw:
            missing.append(key)
        else:
            values[key] = raw
    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")
    return values


def _build_real_model_client(
    raw_io_logger: Callable[[dict[str, Any]], None] | None = None,
) -> GatewayModelClient:
    provider_name = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    default_model = os.getenv("LLM_MODEL", "").strip() or None
    raw_agent_model_map = os.getenv("AGENT_MODEL_MAP", "").strip()
    agent_model_map: dict[str, str] = {}
    if raw_agent_model_map:
        parsed_agent_model_map = json.loads(raw_agent_model_map)
        if not isinstance(parsed_agent_model_map, dict):
            raise RuntimeError("AGENT_MODEL_MAP 必须是 JSON 对象")
        agent_model_map = {
            str(agent_name): str(model_name)
            for agent_name, model_name in parsed_agent_model_map.items()
            if str(model_name).strip()
        }
    try:
        experiment_designer_max_retries = max(
            1,
            int(os.getenv("EXPERIMENT_DESIGNER_LLM_MAX_RETRIES", "4").strip() or "4"),
        )
    except ValueError:
        experiment_designer_max_retries = 4
    agent_retry_policy_map = {
        "experiment_designer_agent": RetryPolicy(
            max_retries=experiment_designer_max_retries,
            wait_seconds=1.0,
        )
    }
    if provider_name == "zhipu":
        envs = _require_env(["ZHIPU_API_KEY"])
        model_name = os.getenv("ZHIPU_MODEL", "").strip() or default_model or "glm-4-flash"
        provider = ZhipuNativeProvider(api_key=envs["ZHIPU_API_KEY"])
        gateway = LLMGateway(
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
            default_model=model_name,
            raw_io_logger=raw_io_logger,
        )
        return GatewayModelClient(
            gateway,
            agent_model_overrides=agent_model_map,
            agent_retry_policy_overrides=agent_retry_policy_map,
        )
    if provider_name == "azure_openai":
        envs = _require_env(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"])
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or default_model
        if not model_name:
            raise RuntimeError("缺少环境变量: AZURE_OPENAI_DEPLOYMENT 或 LLM_MODEL")
        provider = AzureOpenAIProvider(
            api_key=envs["AZURE_OPENAI_API_KEY"],
            azure_endpoint=envs["AZURE_OPENAI_ENDPOINT"],
            api_version=envs["AZURE_OPENAI_API_VERSION"],
        )
        gateway = LLMGateway(
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
            default_model=model_name,
            raw_io_logger=raw_io_logger,
        )
        return GatewayModelClient(
            gateway,
            agent_model_overrides=agent_model_map,
            agent_retry_policy_overrides=agent_retry_policy_map,
        )
    envs = _require_env(["OPENAI_API_KEY", "LLM_MODEL"])
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    provider = OpenAICompatibleProvider(api_key=envs["OPENAI_API_KEY"], base_url=base_url)
    gateway = LLMGateway(
        provider=provider,
        retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
        default_model=envs["LLM_MODEL"],
        raw_io_logger=raw_io_logger,
    )
    return GatewayModelClient(
        gateway,
        agent_model_overrides=agent_model_map,
        agent_retry_policy_overrides=agent_retry_policy_map,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-agent-core alpha factor mining workflow.")
    parser.add_argument("--panel-data-path", default=str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument("--initial-direction", default="", help="Optional initial hypothesis direction.")
    parser.add_argument("--initial-hypothesis", default="", help="Optional initial hypothesis text used as first-round seed.")
    parser.add_argument("--research-report-pdf", default="", help="Optional research report PDF used to seed the first agent.")
    parser.add_argument(
        "--research-report-pdfs",
        nargs="+",
        default=[],
        help="Optional multiple research report PDFs. The PDFs are parsed into one compact combined research payload.",
    )
    parser.add_argument("--stop-on-error", action="store_true", default=False)
    parser.add_argument("--retry-per-round", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug-console-json", action="store_true", default=_env_bool("DEBUG_CONSOLE_JSON", False))
    parser.add_argument("--debug-full-trace", action="store_true", default=_env_bool("DEBUG_FULL_TRACE", False))
    parser.add_argument("--no-console-progress", action="store_true", default=False)
    parser.add_argument("--run-store-path", default=os.getenv("RUN_STORE_PATH", str(RunStore.default_path(PROJECT_ROOT))))
    parser.add_argument("--clear-factor-cache", action="store_true", default=_env_bool("CLEAR_FACTOR_CACHE", False))
    parser.add_argument("--no-report-pdf", action="store_true", default=False, help="Generate markdown report only and skip PDF conversion.")
    parser.add_argument(
        "--train-sharpe-threshold",
        type=float,
        default=_env_float_optional("TRAIN_SHARPE_THRESHOLD"),
        help="Override sample_windows.train_filter.min_value for train Sharpe filtering and holdout candidate selection.",
    )
    parser.add_argument(
        "--train-coverage-threshold",
        type=float,
        default=_env_float_optional("TRAIN_COVERAGE_THRESHOLD"),
        help="Override sample_windows.coverage_filter.min_value for train coverage filtering.",
    )
    parser.add_argument("--evolution-mode", action="store_true", default=_env_bool("EVOLUTION_MODE", False))
    parser.add_argument("--evolution-profile", default=os.getenv("EVOLUTION_PROFILE", "quantaalpha"))
    parser.add_argument("--evolution-iterations", type=int, default=_env_int("EVOLUTION_ITERATIONS", 5))
    parser.add_argument("--evolution-max-rounds", type=int, default=_env_int("EVOLUTION_MAX_ROUNDS", 6))
    parser.add_argument(
        "--evolution-num-initial-hypotheses",
        "--evolution-num-initial-directions",
        dest="evolution_num_initial_hypotheses",
        type=int,
        default=_env_int_alias("EVOLUTION_NUM_INITIAL_HYPOTHESES", "EVOLUTION_NUM_INITIAL_DIRECTIONS", 5),
        help="Number of diversified initial hypotheses H0 generated in evolution round 0. The old --evolution-num-initial-directions alias is still accepted.",
    )
    parser.add_argument("--evolution-mutation-top-n", type=int, default=_env_int("EVOLUTION_MUTATION_TOP_N", 3))
    parser.add_argument("--evolution-crossover-group-size", type=int, default=_env_int("EVOLUTION_CROSSOVER_GROUP_SIZE", 2))
    parser.add_argument("--evolution-crossover-n-groups", type=int, default=_env_int("EVOLUTION_CROSSOVER_N_GROUPS", 2))
    parser.add_argument("--evolution-mutation-tasks-per-round", type=int, default=_env_int("EVOLUTION_MUTATION_TASKS_PER_ROUND", 3))
    parser.add_argument("--evolution-crossover-tasks-per-round", type=int, default=_env_int("EVOLUTION_CROSSOVER_TASKS_PER_ROUND", 2))
    parser.add_argument("--evolution-evolved-directions-per-round", type=int, default=_env_int("EVOLUTION_EVOLVED_DIRECTIONS_PER_ROUND", 5))
    parser.add_argument("--evolution-mutation-parent-top-n", type=int, default=_env_int("EVOLUTION_MUTATION_PARENT_TOP_N", 10))
    parser.add_argument("--evolution-crossover-parent-top-n", type=int, default=_env_int("EVOLUTION_CROSSOVER_PARENT_TOP_N", 10))
    parser.add_argument("--bootstrap-refill-enabled", action="store_true", default=_env_bool("BOOTSTRAP_REFILL_ENABLED", True))
    parser.add_argument("--bootstrap-refill-disabled", dest="bootstrap_refill_enabled", action="store_false")
    parser.add_argument("--bootstrap-max-refill-rounds", type=int, default=_env_int("BOOTSTRAP_MAX_REFILL_ROUNDS", 5))
    parser.add_argument("--bootstrap-min-sharpe", type=float, default=_env_float("BOOTSTRAP_MIN_SHARPE", 1.0))
    parser.add_argument("--bootstrap-min-abs-ic", type=float, default=_env_float("BOOTSTRAP_MIN_ABS_IC", 0.02))
    parser.add_argument("--evolution-parallel-enabled", action="store_true", default=_env_bool("EVOLUTION_PARALLEL_ENABLED", True))
    parser.add_argument("--evolution-no-parallel", dest="evolution_parallel_enabled", action="store_false")
    parser.add_argument("--evolution-max-workers", type=int, default=_env_int("EVOLUTION_MAX_WORKERS", 5))
    parser.add_argument("--factor-pool-enabled", action="store_true", default=_env_bool("FACTOR_POOL_ENABLED", True))
    parser.add_argument("--factor-pool-disabled", dest="factor_pool_enabled", action="store_false")
    parser.add_argument("--factor-pool-corr-threshold", type=float, default=_env_float("FACTOR_POOL_CORR_THRESHOLD", 0.7))
    parser.add_argument("--factor-pool-max-fraction", type=float, default=_env_float("FACTOR_POOL_MAX_FRACTION", 0.5))
    parser.add_argument("--run-final-combined-backtest", action="store_true", default=_env_bool("RUN_FINAL_COMBINED_BACKTEST", True))
    parser.add_argument("--no-run-final-combined-backtest", dest="run_final_combined_backtest", action="store_false")
    return parser


def main() -> None:
    _load_dotenv_files()
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:])
    args = parser.parse_args()
    debug_enabled = bool(args.debug)
    debug_console_json = bool(args.debug_console_json)
    debug_full_trace = bool(args.debug_full_trace)
    console_progress_enabled = not bool(args.no_console_progress)
    if not debug_enabled:
        logging.getLogger("qlib").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"trading_agents\.factor_runtime\..*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*invalid value encountered.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Mean of empty slice.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Degrees of freedom <= 0.*")
    if bool(args.clear_factor_cache):
        atexit.register(_clear_factor_signal_cache)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["FACTOR_LIBRARY_RUN_ID"] = run_id
    if args.train_sharpe_threshold is not None:
        os.environ["TRAIN_SHARPE_THRESHOLD"] = str(float(args.train_sharpe_threshold))
    if args.train_coverage_threshold is not None:
        os.environ["TRAIN_COVERAGE_THRESHOLD"] = str(float(args.train_coverage_threshold))
    log_dir = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    progress_file = log_dir / "progress.jsonl"
    raw_llm_io_file = log_dir / "llm_raw_io.jsonl"
    raw_llm_io_debug_file = log_dir / "llm_raw_io.debug.jsonl.gz"
    summary_file = log_dir / "summary.json"
    run_store = RunStore(args.run_store_path)
    run_store.start_run(
        run_id=run_id,
        log_dir=str(log_dir),
        args={
            "loop_count": max(1, int(args.loop_count)),
            "evolution_mode": bool(args.evolution_mode),
            "evolution_profile": args.evolution_profile,
            "evolution_iterations": args.evolution_iterations,
            "debug": debug_enabled,
            "debug_console_json": debug_console_json,
            "debug_full_trace": debug_full_trace,
            "run_store_path": str(args.run_store_path),
            "clear_factor_cache": bool(args.clear_factor_cache),
            "train_sharpe_threshold": args.train_sharpe_threshold,
            "train_coverage_threshold": args.train_coverage_threshold,
            "research_report_pdf": args.research_report_pdf,
            "research_report_pdfs": list(args.research_report_pdfs or []),
            "factor_pool_enabled": bool(args.factor_pool_enabled),
            "factor_pool_corr_threshold": args.factor_pool_corr_threshold,
            "factor_pool_max_fraction": args.factor_pool_max_fraction,
            "run_final_combined_backtest": bool(args.run_final_combined_backtest),
        },
    )

    # 并行任务的 callback / raw_io_logger 会被多个线程同时调用，用同一把锁序列化文件写入
    _log_lock = threading.Lock()

    def _progress_logger(payload: dict[str, Any]) -> None:
        enriched = _compact_progress_payload(_enrich_progress_payload(payload))
        enriched.setdefault("timestamp", datetime.now().isoformat())
        with _log_lock:
            _append_jsonl(progress_file, enriched)
        if console_progress_enabled:
            _print_console_progress(enriched)
        if debug_console_json:
            _print_debug_event("workflow_event", enriched)

    def _raw_io_logger(payload: dict[str, Any]) -> None:
        compact = _compact_llm_io_payload(payload)
        with _log_lock:
            _append_raw_llm_io(raw_llm_io_file, compact)
            run_store.record_llm_event(run_id=run_id, payload=compact)
            if debug_full_trace:
                _append_jsonl_gz(raw_llm_io_debug_file, payload)
        if debug_console_json:
            _print_debug_event("llm_io", compact)

    def _emit_progress(payload: dict[str, Any]) -> None:
        _progress_logger(payload)

    _emit_progress(
        {
            "stage": "market_payload_started",
            "panel_data_path": str(args.panel_data_path),
        }
    )
    adapter = CryptoCrossSectionDomainAdapter(args.panel_data_path)
    payload = adapter.build_initial_payload()
    dataset_summary = payload.get("domain_dataset_summary", {})
    parametric_specs = payload.get("parametric_specifications", [])
    _emit_progress(
        {
            "stage": "market_payload_finished",
            "panel_data_path": str(args.panel_data_path),
            "feature_count": len(list(dataset_summary.get("feature_columns", [])))
            if isinstance(dataset_summary, dict)
            else 0,
            "target_count": len(list(dataset_summary.get("target_columns", [])))
            if isinstance(dataset_summary, dict)
            else 0,
            "row_count": int(dataset_summary.get("rows", 0) or 0) if isinstance(dataset_summary, dict) else 0,
            "symbol_count": int(dataset_summary.get("symbol_count", 0) or 0) if isinstance(dataset_summary, dict) else 0,
            "datetime_start": str(dataset_summary.get("datetime_start", "")) if isinstance(dataset_summary, dict) else "",
            "datetime_end": str(dataset_summary.get("datetime_end", "")) if isinstance(dataset_summary, dict) else "",
            "has_market_regime_summary": isinstance(payload.get("market_regime_summary"), dict)
            and bool(payload.get("market_regime_summary")),
            "parametric_specification_count": len(parametric_specs) if isinstance(parametric_specs, list) else 0,
        }
    )
    research_report_pdf = str(args.research_report_pdf or "").strip()
    research_report_pdfs = []
    for item in [research_report_pdf] + list(args.research_report_pdfs or []):
        text = str(item or "").strip()
        if text and text not in research_report_pdfs:
            research_report_pdfs.append(text)
    initial_direction = str(args.initial_direction or "").strip()
    initial_hypothesis = str(args.initial_hypothesis or "").strip()
    initial_seed_text = initial_hypothesis or initial_direction
    if initial_hypothesis:
        payload["initial_hypothesis"] = initial_hypothesis
    _append_jsonl(
        progress_file,
        {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "stage": "session_started",
            "loop_count": max(1, int(args.loop_count)),
            "evolution_mode": bool(args.evolution_mode),
            "evolution_profile": str(args.evolution_profile or ""),
            "evolution_iterations": int(args.evolution_iterations),
            "debug": debug_enabled,
            "train_sharpe_threshold": args.train_sharpe_threshold,
            "research_report_pdf": research_report_pdf or None,
            "research_report_pdfs": research_report_pdfs,
        },
    )
    workflow = AlphaFactorMiningWorkflow(
        model_client=_build_real_model_client(raw_io_logger=_raw_io_logger),
        progress_callback=_progress_logger,
    )
    def _arg_provided(option_name: str) -> bool:
        return any(item == option_name or item.startswith(f"{option_name}=") for item in raw_argv)

    evolution_profile = str(args.evolution_profile or "quantaalpha").strip().lower() or "quantaalpha"
    evolution_iterations = max(1, int(args.evolution_iterations))
    profile_is_quantaalpha = evolution_profile == "quantaalpha"
    default_initial_directions = 10 if profile_is_quantaalpha else 5
    default_mutation_children = 10 if profile_is_quantaalpha else 3
    default_crossover_children = 10 if profile_is_quantaalpha else 2
    default_max_rounds = evolution_iterations * 2 + 1 if profile_is_quantaalpha else 6
    effective_initial_directions = (
        int(args.evolution_num_initial_hypotheses)
        if (
            _arg_provided("--evolution-num-initial-hypotheses")
            or _arg_provided("--evolution-num-initial-directions")
        )
        else default_initial_directions
    )
    effective_mutation_tasks = (
        int(args.evolution_mutation_tasks_per_round)
        if _arg_provided("--evolution-mutation-tasks-per-round")
        else default_mutation_children
    )
    effective_crossover_tasks = (
        int(args.evolution_crossover_tasks_per_round)
        if _arg_provided("--evolution-crossover-tasks-per-round")
        else default_crossover_children
    )
    effective_max_rounds = (
        int(args.evolution_max_rounds)
        if _arg_provided("--evolution-max-rounds")
        else default_max_rounds
    )
    evolution_mode = bool(args.evolution_mode)
    if evolution_mode:
        if len(research_report_pdfs) > 1:
            _emit_progress(
                {
                    "stage": "research_report_payload_started",
                    "report_count": len(research_report_pdfs),
                    "research_report_pdfs": research_report_pdfs,
                }
            )
            payload = workflow.build_multi_report_driven_initial_payload(
                initial_shared_payload=payload,
                research_report_pdfs=research_report_pdfs,
            )
            research_report_payload = payload.get("research_report", {})
            _emit_progress(
                {
                    "stage": "research_report_payload_finished",
                    "report_count": len(research_report_pdfs),
                    "source_report_count": len(list(research_report_payload.get("source_reports", [])))
                    if isinstance(research_report_payload, dict)
                    else 0,
                    "multi_report_synthesis_status": str(
                        research_report_payload.get("multi_report_synthesis_status", "")
                    )
                    if isinstance(research_report_payload, dict)
                    else "",
                    **_research_report_progress_fields(research_report_payload),
                }
            )
        elif research_report_pdfs:
            _emit_progress(
                {
                    "stage": "research_report_payload_started",
                    "report_count": 1,
                    "research_report_pdfs": research_report_pdfs,
                }
            )
            payload = workflow.build_report_driven_initial_payload(
                initial_shared_payload=payload,
                research_report_pdf=research_report_pdfs[0],
            )
            research_report_payload = payload.get("research_report", {})
            _emit_progress(
                {
                    "stage": "research_report_payload_finished",
                    "report_count": 1,
                    "source_report_count": len(list(research_report_payload.get("source_reports", [])))
                    if isinstance(research_report_payload, dict)
                    else 1,
                    "multi_report_synthesis_status": str(
                        research_report_payload.get("multi_report_synthesis_status", "")
                    )
                    if isinstance(research_report_payload, dict)
                    else "",
                    **_research_report_progress_fields(research_report_payload),
                }
            )
        _emit_progress(
            {
                "stage": "initialization_payload_ready",
                "has_research_report": isinstance(payload.get("research_report"), dict)
                and bool(str(payload.get("research_report", {}).get("report_text", "")).strip()),
                "has_market_regime_summary": isinstance(payload.get("market_regime_summary"), dict)
                and bool(payload.get("market_regime_summary")),
                "has_dataset_summary": isinstance(payload.get("domain_dataset_summary"), dict)
                and bool(payload.get("domain_dataset_summary")),
                "parametric_specification_count": len(list(payload.get("parametric_specifications", [])))
                if isinstance(payload.get("parametric_specifications", []), list)
                else 0,
                "target_initial_hypotheses": max(1, effective_initial_directions),
            }
        )
        session = workflow.run_evolution_session(
            initial_shared_payload=payload,
            initial_direction=initial_seed_text or None,
            evolution_config=EvolutionConfig(
                profile=evolution_profile,
                iterations=evolution_iterations,
                phase_schedule="mutation_then_crossover" if profile_is_quantaalpha else "legacy",
                max_rounds=max(1, effective_max_rounds),
                num_initial_directions=max(1, effective_initial_directions),
                mutation_tasks_per_round=max(0, effective_mutation_tasks),
                crossover_tasks_per_round=max(0, effective_crossover_tasks),
                children_per_mutation_phase=max(0, effective_mutation_tasks),
                children_per_crossover_phase=max(0, effective_crossover_tasks),
                factor_expressions_per_hypothesis=2,
                evolved_directions_per_round=max(1, int(args.evolution_evolved_directions_per_round)),
                mutation_parent_top_n=max(1, int(args.evolution_mutation_parent_top_n)),
                crossover_parent_top_n=max(1, int(args.evolution_crossover_parent_top_n)),
                bootstrap_refill_enabled=bool(args.bootstrap_refill_enabled),
                bootstrap_max_refill_rounds=max(0, int(args.bootstrap_max_refill_rounds)),
                bootstrap_min_sharpe=float(args.bootstrap_min_sharpe),
                bootstrap_min_abs_ic=float(args.bootstrap_min_abs_ic),
                mutation_top_n=max(0, int(args.evolution_mutation_top_n)),
                crossover_group_size=max(1, int(args.evolution_crossover_group_size)),
                crossover_n_groups=max(0, int(args.evolution_crossover_n_groups)),
                parallel_enabled=bool(args.evolution_parallel_enabled),
                max_workers=max(1, int(args.evolution_max_workers)),
            ),
        )
    elif len(research_report_pdfs) > 1:
        _emit_progress(
            {
                "stage": "research_report_payload_started",
                "report_count": len(research_report_pdfs),
                "research_report_pdfs": research_report_pdfs,
            }
        )
        session = workflow.run_multi_report_driven_loop_session(
            research_report_pdfs=research_report_pdfs,
            loop_count=max(1, int(args.loop_count)),
            initial_shared_payload=payload,
            initial_direction=initial_seed_text or None,
            stop_on_error=bool(args.stop_on_error),
            retry_per_round=max(0, int(args.retry_per_round)),
        )
    elif research_report_pdfs:
        _emit_progress(
            {
                "stage": "research_report_payload_started",
                "report_count": 1,
                "research_report_pdfs": research_report_pdfs,
            }
        )
        session = workflow.run_report_driven_loop_session(
            research_report_pdf=research_report_pdfs[0],
            loop_count=max(1, int(args.loop_count)),
            initial_shared_payload=payload,
            initial_direction=initial_seed_text or None,
            stop_on_error=bool(args.stop_on_error),
            retry_per_round=max(0, int(args.retry_per_round)),
        )
    else:
        session = workflow.run_loop_session(
            loop_count=max(1, int(args.loop_count)),
            initial_shared_payload=payload,
            initial_direction=initial_seed_text or None,
            stop_on_error=bool(args.stop_on_error),
            retry_per_round=max(0, int(args.retry_per_round)),
        )
    for idx, trace in enumerate(session.loop_traces, start=1):
        loop_summary, compact_final_context, agent_records, debug_path = _save_loop_trace(
            log_dir,
            idx,
            trace,
            debug_full_trace=debug_full_trace,
        )
        run_store.record_loop(
            run_id=run_id,
            loop_index=idx,
            summary=loop_summary,
            final_context=compact_final_context,
        )
        for agent_record in agent_records:
            run_store.record_agent(
                run_id=run_id,
                loop_index=idx,
                agent_index=int(agent_record.get("agent_index", 0) or 0),
                payload=agent_record,
            )
        trajectory_id = str(loop_summary.get("trajectory_id", ""))
        for factor in list(loop_summary.get("factors", []) or []):
            if not isinstance(factor, dict):
                continue
            factor_id = _factor_id_for_storage(trajectory_id, factor)
            run_store.upsert_factor(
                run_id=run_id,
                loop_index=idx,
                trajectory_id=trajectory_id,
                factor_id=factor_id,
                factor=factor,
                metrics=_first_dict(factor.get("metrics", {})),
            )
        if debug_path is not None and debug_path.exists():
            run_store.record_artifact(
                run_id=run_id,
                loop_index=idx,
                kind="loop_debug_trace",
                path=str(debug_path),
                size_bytes=debug_path.stat().st_size,
                compressed=True,
                sha256=_artifact_sha256(debug_path),
            )
    for trajectory in session.trajectory_pool.get_all():
        run_store.upsert_trajectory(run_id=run_id, trajectory=trajectory)
    if debug_full_trace and raw_llm_io_debug_file.exists():
        run_store.record_artifact(
            run_id=run_id,
            kind="llm_raw_io_debug",
            path=str(raw_llm_io_debug_file),
            size_bytes=raw_llm_io_debug_file.stat().st_size,
            compressed=True,
            sha256=_artifact_sha256(raw_llm_io_debug_file),
        )
    _append_jsonl(
        progress_file,
        {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "stage": "session_finished",
            "status": session.status,
            "loop_count": len(session.loop_traces),
            "evolution_mode": evolution_mode,
            "total_errors": session.total_errors,
        },
    )
    final_trace = session.loop_traces[-1] if session.loop_traces else None
    holdout_validation_file = log_dir / "holdout_validation.json"
    holdout_validation: dict[str, Any] | None = None
    try:
        from trading_agents.infra.holdout_validation import run_holdout_validation

        session_experiment_ids = {
            str(trace.final_shared_context.get("experiment_id", "")).strip()
            for trace in session.loop_traces
            if str(trace.final_shared_context.get("experiment_id", "")).strip()
        }
        holdout_validation = run_holdout_validation(
            library_path=session.factor_library_path,
            panel_data_path=args.panel_data_path,
            output_path=holdout_validation_file,
            experiment_ids=session_experiment_ids,
            run_ids={run_id},
        )
        _append_jsonl(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "stage": "holdout_validation_finished",
                "status": holdout_validation.get("status"),
                "factor_count": holdout_validation.get("factor_count"),
                "output_file": str(holdout_validation_file),
            },
        )
    except Exception as exc:
        holdout_validation = {
            "status": "failed",
            "error": str(exc),
        }
        _append_jsonl(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "stage": "holdout_validation_failed",
                "error": str(exc),
                "output_file": str(holdout_validation_file),
            },
        )
    factor_pool_payload: dict[str, Any] = {"status": "disabled"}
    if bool(args.factor_pool_enabled):
        try:
            factor_pool_payload = FactorPoolManager(
                library_path=session.factor_library_path,
                panel_data_path=args.panel_data_path,
                output_dir=log_dir,
                config=FactorPoolConfig(
                    enabled=True,
                    corr_threshold=float(args.factor_pool_corr_threshold),
                    max_fraction=float(args.factor_pool_max_fraction),
                ),
            ).build(run_final_combined_backtest=bool(args.run_final_combined_backtest))
            _append_jsonl(
                progress_file,
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "stage": "factor_pool_finished",
                    "status": factor_pool_payload.get("status"),
                    "selected_count": factor_pool_payload.get("selected_count", 0),
                    "output_file": factor_pool_payload.get("factor_pool_path", ""),
                    "combined_factors_path": factor_pool_payload.get("combined_factors_path", ""),
                },
            )
        except Exception as exc:
            factor_pool_payload = {
                "status": "failed",
                "error": str(exc),
            }
            _append_jsonl(
                progress_file,
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "stage": "factor_pool_failed",
                    "error": str(exc),
                },
            )
    report_md_file: Path | None = None
    report_pdf_file: Path | None = None
    try:
        from scripts.generate_report import generate_report

        report_md_file, report_pdf_file = generate_report(log_dir=log_dir, no_pdf=bool(args.no_report_pdf))
    except Exception as exc:
        _append_jsonl(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "stage": "report_generation_failed",
                "error": str(exc),
            },
        )
    evolution_planning_summary = _summarize_evolution_planning(session)
    evolution_agent_plans = list(evolution_planning_summary.pop("evolution_agent_plans", []) or [])
    lineage_tree = _build_lineage_tree(session)
    detail_files: dict[str, str] = {}
    lineage_tree_file = log_dir / "lineage_tree.json"
    evolution_plans_file = log_dir / "evolution_plans.json"
    if lineage_tree:
        _write_detail_json(lineage_tree_file, lineage_tree)
        detail_files["lineage_tree"] = str(lineage_tree_file)
        run_store.record_artifact(
            run_id=run_id,
            kind="lineage_tree",
            path=str(lineage_tree_file),
            size_bytes=lineage_tree_file.stat().st_size,
            compressed=False,
            sha256=_artifact_sha256(lineage_tree_file),
        )
    if evolution_agent_plans:
        _write_detail_json(evolution_plans_file, evolution_agent_plans)
        detail_files["evolution_plans"] = str(evolution_plans_file)
        run_store.record_artifact(
            run_id=run_id,
            kind="evolution_plans",
            path=str(evolution_plans_file),
            size_bytes=evolution_plans_file.stat().st_size,
            compressed=False,
            sha256=_artifact_sha256(evolution_plans_file),
        )
    factor_pool_path_text = str(factor_pool_payload.get("factor_pool_path", "")).strip() if isinstance(factor_pool_payload, dict) else ""
    combined_factors_path_text = str(factor_pool_payload.get("combined_factors_path", "")).strip() if isinstance(factor_pool_payload, dict) else ""
    factor_pool_file = Path(factor_pool_path_text) if factor_pool_path_text else None
    combined_factors_file = Path(combined_factors_path_text) if combined_factors_path_text else None
    if factor_pool_file is not None and factor_pool_file.exists():
        detail_files["factor_pool"] = str(factor_pool_file)
        run_store.record_artifact(
            run_id=run_id,
            kind="factor_pool",
            path=str(factor_pool_file),
            size_bytes=factor_pool_file.stat().st_size,
            compressed=False,
            sha256=_artifact_sha256(factor_pool_file),
        )
    if combined_factors_file is not None and combined_factors_file.exists():
        detail_files["combined_factors"] = str(combined_factors_file)
        run_store.record_artifact(
            run_id=run_id,
            kind="combined_factors",
            path=str(combined_factors_file),
            size_bytes=combined_factors_file.stat().st_size,
            compressed=False,
            sha256=_artifact_sha256(combined_factors_file),
        )
    rounds_summary = _summarize_rounds(session)
    factor_summary = _summarize_factors_across_rounds(rounds_summary)
    run_factor_summary_file = PROJECT_ROOT / "factor_library" / f"run_factor_summary_{run_id}.csv"
    _export_run_factor_summary_csv(
        run_id=run_id,
        rounds=rounds_summary,
        holdout_validation=holdout_validation,
        output_path=run_factor_summary_file,
    )
    run_store.record_artifact(
        run_id=run_id,
        kind="run_factor_summary_csv",
        path=str(run_factor_summary_file),
        size_bytes=run_factor_summary_file.stat().st_size,
        compressed=False,
        sha256=_artifact_sha256(run_factor_summary_file),
    )
    segment_lineage = _summarize_segment_lineage(session)
    # ── 小轮次（轨迹）统计 ──────────────────────────────────────────────
    # loop_count / failed_trajectory_count：每条独立的 5-agent 流水线执行
    failed_trajectory_count = sum(
        1
        for trace, item in zip(session.loop_traces, session.rounds)
        if (
            item.error_count > 0
            or trace.final_shared_context.get("round_status") == "failed_incomplete"
        )
    )
    # 兼容旧字段名
    failed_round_count = failed_trajectory_count

    # ── 中轮次（进化阶段 / Evolution Phase）统计 ────────────────────────
    # 每个 round_idx 对应一个阶段：round 0=初始化, 1=mutation1, 2=crossover1, …
    # 阶段完成 = 该 round_idx 下至少有一条轨迹成功
    from collections import defaultdict as _defaultdict
    _phase_stats: dict[int, dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
    for _trace, _item in zip(session.loop_traces, session.rounds):
        _r = int(_trace.final_shared_context.get("round_idx", 0) or 0)
        _phase_stats[_r]["total"] += 1
        if _item.error_count == 0 and _trace.final_shared_context.get("round_status") != "failed_incomplete":
            _phase_stats[_r]["success"] += 1
    # 已运行阶段数（含初始化阶段 round 0）
    ran_phase_count = len(_phase_stats)
    # 完成的进化阶段数（round_idx >= 1，即不含初始化，且至少 1 条轨迹成功）
    completed_evolution_phase_count = sum(
        1 for r, s in _phase_stats.items() if r >= 1 and s["success"] > 0
    )
    # 全部失败的进化阶段数（round_idx >= 1，且所有轨迹均失败）
    failed_evolution_phase_count = sum(
        1 for r, s in _phase_stats.items() if r >= 1 and s["success"] == 0 and s["total"] > 0
    )

    # ── 大轮次（论文迭代 / Paper Iteration）统计 ─────────────────────────
    # paper_iteration i = mutation phase(2i-1) + crossover phase(2i)，i≥1
    # iteration 0 特指初始化（不计入"进化迭代"）
    _seen_iterations: set[int] = set()
    for row in rounds_summary:
        pi = int(row.get("paper_iteration", 0) or 0)
        if pi > 0:
            _seen_iterations.add(pi)
    ran_iteration_count = len(_seen_iterations)
    # 完整迭代 = 该 iteration 的 mutation phase 和 crossover phase 均至少有 1 条轨迹成功
    completed_iteration_count = 0
    for _pi in _seen_iterations:
        _mut_round = 2 * _pi - 1   # mutation round_idx for iteration _pi
        _cross_round = 2 * _pi     # crossover round_idx for iteration _pi
        _mut_ok = _phase_stats.get(_mut_round, {}).get("success", 0) > 0
        _cross_ok = _phase_stats.get(_cross_round, {}).get("success", 0) > 0
        if _mut_ok and _cross_ok:
            completed_iteration_count += 1
    artifacts = {
        "log_dir": str(log_dir),
        "summary": str(summary_file),
        "run_store": str(run_store.db_path),
        "progress": str(progress_file),
        "llm_raw_io": str(raw_llm_io_file),
        "llm_raw_io_debug": str(raw_llm_io_debug_file) if debug_full_trace else None,
        "holdout_validation": str(holdout_validation_file),
        "report_markdown": str(report_md_file) if report_md_file else None,
        "report_pdf": str(report_pdf_file) if report_pdf_file else None,
        "factor_library": FactorLibraryManager.to_portable_path(session.factor_library_path),
        "run_factor_summary_csv": FactorLibraryManager.to_portable_path(run_factor_summary_file),
        "factor_pool": str(factor_pool_file) if factor_pool_file is not None and factor_pool_file.exists() else None,
        "combined_factors": str(combined_factors_file) if combined_factors_file is not None and combined_factors_file.exists() else None,
        "trajectory_pool": FactorLibraryManager.to_portable_path(_trajectory_pool_path()),
        **detail_files,
    }
    artifacts = {key: value for key, value in artifacts.items() if value not in (None, "", [])}
    output = {
        "summary_version": 2,
        "run_id": run_id,
        "status": session.status,
        "overview": {
            "evolution_mode": evolution_mode,
            # ── 小轮次：轨迹（每条独立的 5-agent 流水线执行）
            "trajectory_count": len(session.loop_traces),
            "failed_trajectory_count": failed_trajectory_count,
            # ── 中轮次：进化阶段（每批同类轨迹的并行执行，round_idx 级别）
            "ran_phase_count": ran_phase_count,
            "completed_evolution_phase_count": completed_evolution_phase_count,
            "failed_evolution_phase_count": failed_evolution_phase_count,
            # ── 大轮次：论文迭代（一轮 mutation + 一轮 crossover = 论文中 iteration i）
            "target_iteration_count": evolution_iterations,
            "ran_iteration_count": ran_iteration_count,
            "completed_iteration_count": completed_iteration_count,
            # ── 兼容旧字段
            "loop_count": len(session.loop_traces),
            "failed_round_count": failed_round_count,
            "total_errors": session.total_errors,
            "trajectory_pool_stats": session.trajectory_pool.get_statistics(),
        },
        "config": {
            "storage_mode": "sqlite_primary_light_json",
            "quantaalpha_profile": evolution_profile,
            "phase_schedule": "mutation_then_crossover" if profile_is_quantaalpha else "legacy",
            "evolution_iterations": evolution_iterations,
            "evolution_num_initial_hypotheses": effective_initial_directions,
            "evolution_num_initial_directions": effective_initial_directions,
            "evolution_mutation_tasks_per_round": effective_mutation_tasks,
            "evolution_crossover_tasks_per_round": effective_crossover_tasks,
            "train_sharpe_threshold": args.train_sharpe_threshold,
            "research_report_pdfs": research_report_pdfs,
            "debug_full_trace": debug_full_trace,
            "factor_cache_cleanup_enabled": bool(args.clear_factor_cache),
            "factor_pool_enabled": bool(args.factor_pool_enabled),
            "factor_pool_corr_threshold": float(args.factor_pool_corr_threshold),
            "factor_pool_max_fraction": float(args.factor_pool_max_fraction),
            "run_final_combined_backtest": bool(args.run_final_combined_backtest),
        },
        "artifacts": artifacts,
        "holdout_validation": _compact_holdout_validation(holdout_validation),
        "best_trajectories": _summarize_best_trajectories(session),
        "factor_summary": factor_summary,
        "factor_pool": factor_pool_payload,
        "combined_backtest": dict(factor_pool_payload.get("combined_backtest", {}))
        if isinstance(factor_pool_payload, dict)
        else {},
        "rounds": rounds_summary,
        "segment_lineage": segment_lineage,
        "hypothesis_origin": {
            "direction_decomposer_mode": evolution_planning_summary.get("direction_decomposer_mode", ""),
            "initial_hypothesis_count": evolution_planning_summary.get("initial_hypothesis_count", 0),
            "initial_hypotheses": evolution_planning_summary.get("initial_hypotheses", []),
            "initial_direction_count": evolution_planning_summary.get("initial_direction_count", 0),
            "initial_directions": evolution_planning_summary.get("initial_directions", []),
        },
        "evolution": {
            "evolution_planner_mode": evolution_planning_summary.get("evolution_planner_mode", ""),
            "planned_mutation_count": evolution_planning_summary.get("planned_mutation_count", 0),
            "planned_crossover_count": evolution_planning_summary.get("planned_crossover_count", 0),
            "executed_evolved_count": evolution_planning_summary.get("executed_evolved_count", 0),
            "evolution_agent_plans": evolution_agent_plans,
            "detail_files": detail_files,
        },
    }
    if debug_enabled:
        output["debug"] = {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "") or os.getenv("ZHIPU_MODEL", ""),
            "agent_model_map": os.getenv("AGENT_MODEL_MAP", ""),
            "round_errors": [
                {
                    "loop_index": item.loop_index,
                    "attempt_index": item.attempt_index,
                    "errors": list(item.errors),
                }
                for item in session.rounds
                if item.errors
            ],
            "trace_errors": [
                {
                    "loop_index": idx + 1,
                    "agent_name": record.agent_name,
                    "error": str(record.error),
                }
                for idx, trace in enumerate(session.loop_traces)
                for record in trace.records
                if record.error
            ],
        }
    cache_cleanup = (
        _clear_factor_signal_cache()
        if bool(args.clear_factor_cache)
        else {
            "status": "skipped",
            "reason": "factor cache is retained by default; pass --clear-factor-cache to remove it",
            "cache_dir": str(PROJECT_ROOT / "factor_library" / "cache"),
        }
    )
    if isinstance(output.get("overview"), dict):
        output["overview"]["factor_cache_cleanup"] = cache_cleanup
    else:
        output["overview"] = {"factor_cache_cleanup": cache_cleanup}
    _append_jsonl(
        progress_file,
        {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "stage": "factor_cache_cleanup_finished",
            **cache_cleanup,
        },
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(_json_safe(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_store.finish_run(run_id=run_id, status=session.status, summary=output)
    _print_final_run_summary(output)


if __name__ == "__main__":
    main()
