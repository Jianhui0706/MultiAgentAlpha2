from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.infra.llm import GatewayModelClient, LLMGateway, RetryPolicy
from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.core.model_client import reset_current_loop_index, set_current_loop_index
from trading_agents.infra.llm.providers import (
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
    ZhipuNativeProvider,
)
from trading_agents.domains import CryptoCrossSectionDomainAdapter
from trading_agents.workflows import AlphaFactorMiningWorkflow, EvolutionConfig


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    """解析环境变量文件单行文本，返回键值对；注释、空行、非法行返回空值。"""
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
    """按顺序加载环境变量文件，并且不覆盖已有系统环境变量。"""
    env_candidates = [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]
    for env_file in env_candidates:
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
    """把路径、时间、元组等对象递归转换为可序列化结构。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，支持常见真值写法。"""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，非法或缺失时返回默认值。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _require_env(keys: list[str]) -> dict[str, str]:
    """批量读取必需环境变量，缺失时抛出带字段名的错误。"""
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        raw = os.getenv(key, "").strip()
        if not raw:
            missing.append(key)
        else:
            values[key] = raw
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"缺少环境变量: {missing_text}")
    return values


def _build_real_model_client(
    raw_io_logger: Callable[[dict[str, Any]], None] | None = None,
) -> GatewayModelClient:
    """根据模型提供方与相关环境变量构建真实可用的模型客户端。"""
    provider_name = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()
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
    if provider_name == "zhipu":
        envs = _require_env(["ZHIPU_API_KEY"])
        model_name = os.getenv("ZHIPU_MODEL", "").strip() or default_model or "glm-4-flash"
        provider = ZhipuNativeProvider(
            api_key=envs["ZHIPU_API_KEY"],
        )
        gateway = LLMGateway(
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
            default_model=model_name,
            raw_io_logger=raw_io_logger,
        )
        return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)
    if provider_name == "azure_openai":
        envs = _require_env(
            [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
            ]
        )
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
        return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)
    envs = _require_env(["OPENAI_API_KEY", "LLM_MODEL"])
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    provider = OpenAICompatibleProvider(
        api_key=envs["OPENAI_API_KEY"],
        base_url=base_url,
    )
    gateway = LLMGateway(
        provider=provider,
        retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
        default_model=envs["LLM_MODEL"],
        raw_io_logger=raw_io_logger,
    )
    return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)


def _save_trace_log(loop_trace: Any, output_dir: Path, loop_index: int) -> None:
    """把单轮循环的完整执行轨迹写入对应日志文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_payload = {
        "loop_index": loop_index,
        "record_count": len(loop_trace.records),
        "records": [
            {
                "agent_name": record.agent_name,
                "started_at": record.started_at.isoformat(),
                "ended_at": record.ended_at.isoformat(),
                "input_snapshot": _json_safe(record.input_snapshot),
                "output_snapshot": _json_safe(record.output_snapshot),
                "error": record.error,
            }
            for record in loop_trace.records
        ],
        "final_shared_context": _json_safe(loop_trace.final_shared_context),
    }
    output_path = output_dir / f"loop_{loop_index:03d}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(loop_payload, file, ensure_ascii=False, indent=2)


def _persist_factor_library(
    loop_trace: Any,
    output_file: Path,
    panel_data_path: Path,
    loop_index: int,
) -> None:
    """把每轮关键结果追加到因子库文件，便于后续检索与复用。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shared = loop_trace.final_shared_context
    experiment_spec = shared.get("experiment_spec", {})
    factors = experiment_spec.get("factors", []) if isinstance(experiment_spec, dict) else []
    factor_formulas: list[dict[str, Any]] = []
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        factor_formulas.append(
            {
                "factor_name": factor.get("factor_name"),
                "expression": factor.get("expression"),
                "formulation": factor.get("formulation"),
            }
        )
    error_count = len([record for record in loop_trace.records if record.error is not None])
    record = {
        "created_at": datetime.now().isoformat(),
        "loop_index": loop_index,
        "panel_data_path": str(panel_data_path),
        "hypothesis": shared.get("hypothesis"),
        "factor_formulas": factor_formulas,
        "metrics": shared.get("metrics"),
        "record_count": len(loop_trace.records),
        "error_count": error_count,
        "has_error": error_count > 0,
    }
    with output_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(record), ensure_ascii=False, indent=2))
        file.write("\n")
        file.write("\n")
    manager = FactorLibraryManager(FactorLibraryManager.default_library_path())
    manager.add_from_shared_payload(loop_trace.final_shared_context)


def _extract_keys(payload: Any) -> list[str]:
    """从字典中提取并排序键名，用于进度日志的轻量摘要。"""
    if not isinstance(payload, dict):
        return []
    return sorted(str(key) for key in payload.keys())


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


def _summarize_best_trajectories(session: Any, top_n: int = 5) -> list[dict[str, Any]]:
    """生成示例输出中的最佳轨迹摘要。"""
    return [
        {
            "trajectory_id": item.trajectory_id,
            "phase": item.phase.value if hasattr(item.phase, "value") else str(item.phase),
            "round_idx": item.round_idx,
            "direction_id": item.direction_id,
            "parent_ids": list(item.parent_ids),
            "reward_score": item.get_reward_score(),
            "metrics": dict(item.backtest_metrics),
        }
        for item in session.trajectory_pool.get_best_trajectories(top_n=top_n)
    ]


def _summarize_rounds(session: Any) -> list[dict[str, Any]]:
    """生成示例输出中的轮次与血缘字段摘要。"""
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(session.rounds):
        trace = session.loop_traces[idx] if idx < len(session.loop_traces) else None
        shared = getattr(trace, "final_shared_context", {}) if trace is not None else {}
        rows.append(
            {
                "loop_index": item.loop_index,
                "attempt_index": item.attempt_index,
                "phase": str(shared.get("phase", item.phase)),
                "round_idx": int(shared.get("round_idx", item.loop_index) or 0),
                "direction_id": int(shared.get("direction_id", 0) or 0),
                "trajectory_id": str(shared.get("trajectory_id", item.trajectory_id)),
                "parent_ids": [str(parent_id) for parent_id in list(shared.get("parent_ids", item.parent_ids))],
                "mutation_type": str(shared.get("mutation_type", "")),
                "evolution_planner_mode": str(shared.get("evolution_planner_mode", "")),
                "error_count": item.error_count,
                "agent_count": item.agent_count,
            }
        )
    return rows


def _build_lineage_tree(session: Any) -> list[dict[str, Any]]:
    """生成示例输出中的父子轨迹树。"""
    nodes: dict[str, dict[str, Any]] = {}
    for trajectory in session.trajectory_pool.get_all():
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
    for trajectory in session.trajectory_pool.get_all():
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


def _dominant_mode(values: Any) -> str:
    modes = [str(value).strip() for value in values if str(value).strip()]
    if not modes:
        return ""
    return "llm" if "llm" in modes else modes[0]


def _summarize_evolution_planning(session: Any) -> dict[str, Any]:
    trajectories = session.trajectory_pool.get_all()
    initial_directions: list[dict[str, Any]] = []
    plans_by_round: dict[int, dict[str, Any]] = {}
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
        }
        if phase == "original":
            row["direction_decomposer_mode"] = row["evolution_planner_mode"]
            initial_directions.append(row)
            continue
        bucket = plans_by_round.setdefault(
            int(trajectory.round_idx),
            {"round_idx": int(trajectory.round_idx), "mutation_tasks": [], "crossover_tasks": []},
        )
        if phase == "mutation":
            bucket["mutation_tasks"].append(row)
        elif phase == "crossover":
            bucket["crossover_tasks"].append(row)
    evolution_agent_plans = [
        {
            **bucket,
            "planned_mutation_count": len(bucket["mutation_tasks"]),
            "planned_crossover_count": len(bucket["crossover_tasks"]),
        }
        for _round_idx, bucket in sorted(plans_by_round.items())
    ]
    return {
        "direction_decomposer_mode": _dominant_mode(item.get("direction_decomposer_mode", "") for item in initial_directions),
        "initial_direction_count": len(initial_directions),
        "initial_directions": initial_directions,
        "evolution_planner_mode": _dominant_mode(
            task.get("evolution_planner_mode", "")
            for plan in evolution_agent_plans
            for task in [*plan["mutation_tasks"], *plan["crossover_tasks"]]
        ),
        "planned_mutation_count": sum(int(plan["planned_mutation_count"]) for plan in evolution_agent_plans),
        "planned_crossover_count": sum(int(plan["planned_crossover_count"]) for plan in evolution_agent_plans),
        "evolution_agent_plans": evolution_agent_plans,
    }


def _append_progress(progress_file: Path, payload: dict[str, Any]) -> None:
    """把进度事件追加到进度日志，同时输出到标准错误流做实时可视化。"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(payload), ensure_ascii=False))
        file.write("\n")
    print(json.dumps(_json_safe(payload), ensure_ascii=False), file=sys.stderr, flush=True)


def _append_raw_llm_io(raw_io_file: Path, payload: dict[str, Any]) -> None:
    raw_io_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_io_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n\n")


def main() -> None:
    """运行多轮因子挖掘工作流，并产出进度日志、轨迹日志和最终汇总。"""
    _load_dotenv_files()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    raw_llm_io_file = log_dir / "llm_raw_io.jsonl"
    model_client = _build_real_model_client(
        raw_io_logger=lambda payload: _append_raw_llm_io(raw_llm_io_file, payload)
    )
    panel_data_path = PROJECT_ROOT / "data" / "panel_data.parquet"
    adapter = CryptoCrossSectionDomainAdapter(panel_data_path)
    shared_payload = adapter.build_initial_payload()
    
    # 添加快速测试配置
    shared_payload["factor_coder_max_loop"] = int(os.getenv("FACTOR_CODER_MAX_LOOP", "1"))
    shared_payload["factor_coder_fail_task_trial_limit"] = int(os.getenv("FACTOR_CODER_FAIL_TASK_TRIAL_LIMIT", "1"))
    
    default_initial_direction = "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。"
    initial_direction_raw = os.getenv("INITIAL_DIRECTION", default_initial_direction)
    initial_direction = initial_direction_raw.strip() or None
    traces: list[Any] = []
    loop_count = int(os.getenv("LOOP_COUNT", "2"))  # 默认2个loop
    factor_library_file = PROJECT_ROOT / "factor_library" / "factor_library.jsonl"
    progress_file = log_dir / "progress.jsonl"
    progress_state: dict[str, int] = {"loop_index": 0}

    def on_progress(event: dict[str, object]) -> None:
        stage = str(event.get("stage", ""))
        agent_name = str(event.get("agent_name", ""))
        base = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "loop_index": progress_state["loop_index"],
            "agent_name": agent_name,
            "stage": stage,
        }
        base.update(_extract_evolution_metadata(event))
        if stage == "agent_started":
            input_snapshot = event.get("input_snapshot", {})
            private_context = input_snapshot.get("private_context", {}) if isinstance(input_snapshot, dict) else {}
            shared_context = input_snapshot.get("shared_context", {}) if isinstance(input_snapshot, dict) else {}
            base["input_private_keys"] = _extract_keys(private_context)
            base["input_shared_keys"] = _extract_keys(shared_context)
        elif stage == "agent_finished":
            output_snapshot = event.get("output_snapshot", {})
            shared_updates = output_snapshot.get("shared_updates", {}) if isinstance(output_snapshot, dict) else {}
            artifacts = output_snapshot.get("artifacts", {}) if isinstance(output_snapshot, dict) else {}
            base["output_shared_update_keys"] = _extract_keys(shared_updates)
            base["output_artifact_keys"] = _extract_keys(artifacts)
        elif stage == "agent_failed":
            output_snapshot = event.get("output_snapshot", {})
            shared_context = output_snapshot.get("shared_context", {}) if isinstance(output_snapshot, dict) else {}
            base["error"] = str(event.get("error", ""))
            base["output_shared_keys"] = _extract_keys(shared_context)
        _append_progress(progress_file, base)

    workflow = AlphaFactorMiningWorkflow(model_client=model_client, progress_callback=on_progress)
    evolution_mode = _env_bool("EVOLUTION_MODE", False)
    if evolution_mode:
        session = workflow.run_evolution_session(
            initial_shared_payload=shared_payload,
            initial_direction=initial_direction,
            evolution_config=EvolutionConfig(
                max_rounds=max(1, _env_int("EVOLUTION_MAX_ROUNDS", loop_count)),
                num_initial_directions=max(1, _env_int("EVOLUTION_NUM_INITIAL_DIRECTIONS", 5)),
                mutation_tasks_per_round=max(0, _env_int("EVOLUTION_MUTATION_TASKS_PER_ROUND", 3)),
                crossover_tasks_per_round=max(0, _env_int("EVOLUTION_CROSSOVER_TASKS_PER_ROUND", 2)),
                evolved_directions_per_round=max(1, _env_int("EVOLUTION_EVOLVED_DIRECTIONS_PER_ROUND", 5)),
                mutation_parent_top_n=max(1, _env_int("EVOLUTION_MUTATION_PARENT_TOP_N", 10)),
                crossover_parent_top_n=max(1, _env_int("EVOLUTION_CROSSOVER_PARENT_TOP_N", 10)),
                mutation_top_n=max(0, _env_int("EVOLUTION_MUTATION_TOP_N", 3)),
                crossover_group_size=max(1, _env_int("EVOLUTION_CROSSOVER_GROUP_SIZE", 2)),
                crossover_n_groups=max(0, _env_int("EVOLUTION_CROSSOVER_N_GROUPS", 2)),
            ),
        )
        for loop_index, trace in enumerate(session.loop_traces, start=1):
            _save_trace_log(trace, log_dir, loop_index)
        trace = session.loop_traces[-1] if session.loop_traces else None
        output = {
            "run_id": run_id,
            "evolution_mode": True,
            "status": session.status,
            "loop_count": len(session.loop_traces),
            "record_count_per_loop": [len(loop_trace.records) for loop_trace in session.loop_traces],
            "agents_per_loop": [[record.agent_name for record in loop_trace.records] for loop_trace in session.loop_traces],
            "final_shared_context_keys": sorted(list(trace.final_shared_context.keys())) if trace else [],
            "final_shared_context": trace.final_shared_context if trace else {},
            "initial_direction": initial_direction,
            "log_dir": str(log_dir),
            "raw_llm_io_file": str(raw_llm_io_file),
            "factor_library_file": str(factor_library_file),
            "trajectory_pool_path": str(PROJECT_ROOT / "artifacts" / "trajectory_pool.json"),
            "trajectory_pool_stats": session.trajectory_pool.get_statistics(),
            "best_trajectories": _summarize_best_trajectories(session),
            "rounds": _summarize_rounds(session),
            "lineage_tree": _build_lineage_tree(session),
            **_summarize_evolution_planning(session),
            "errors": [
                [record.error for record in loop_trace.records if record.error is not None]
                for loop_trace in session.loop_traces
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        _generate_final_report(log_dir)
        return

    for loop_index in range(1, loop_count + 1):
        progress_state["loop_index"] = loop_index
        _append_progress(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "loop_index": loop_index,
                "stage": "loop_started",
            },
        )
        loop_token = set_current_loop_index(loop_index)
        try:
            trace = workflow.run(
                shared_payload,
                initial_direction=initial_direction if loop_index == 1 else None,
            )
        finally:
            reset_current_loop_index(loop_token)
        traces.append(trace)
        _save_trace_log(trace, log_dir, loop_index)
        _persist_factor_library(trace, factor_library_file, panel_data_path, loop_index)
        shared_payload = dict(trace.final_shared_context)
        _append_progress(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "loop_index": loop_index,
                "stage": "loop_finished",
                "agent_count": len(trace.records),
                "error_count": len([record for record in trace.records if record.error is not None]),
            },
        )
    trace = traces[-1]
    output = {
        "run_id": run_id,
        "evolution_mode": False,
        "loop_count": len(traces),
        "record_count_per_loop": [len(loop_trace.records) for loop_trace in traces],
        "agents_per_loop": [[record.agent_name for record in loop_trace.records] for loop_trace in traces],
        "final_shared_context_keys": sorted(list(trace.final_shared_context.keys())),
        "final_shared_context": trace.final_shared_context,
        "initial_direction": initial_direction,
        "log_dir": str(log_dir),
        "raw_llm_io_file": str(raw_llm_io_file),
        "factor_library_file": str(factor_library_file),
        "errors": [
            [record.error for record in loop_trace.records if record.error is not None]
            for loop_trace in traces
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    _generate_final_report(log_dir)


def _generate_final_report(log_dir: Path) -> None:
    """在全部循环完成后生成最终报告。"""
    print("\n" + "=" * 60)
    print("Generating final report...")
    print("=" * 60)
    
    try:
        # Add PROJECT_ROOT to sys.path for scripts import
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        from scripts.generate_report import generate_report
        md_file, pdf_file = generate_report(log_dir, no_pdf=True)
        if md_file:
            print(f"\nReport generated successfully!")
            print(f"Markdown: {md_file}")
            if pdf_file and pdf_file.exists():
                print(f"PDF: {pdf_file}")
    except ImportError:
        print("Warning: Could not import generate_report module.")
        print("Run manually: python scripts/generate_report.py --log-dir <log_dir> --no-pdf")
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
        print("Run manually: python scripts/generate_report.py --log-dir <log_dir> --no-pdf")


if __name__ == "__main__":
    main()
