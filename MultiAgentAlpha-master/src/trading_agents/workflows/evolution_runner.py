from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from trading_agents.agents import DirectionDecomposerAgent, EvolutionAgent
from trading_agents.core import ContextStore, ModelClient, Orchestrator
from trading_agents.core.orchestrator import SequentialLoopOrchestrator
from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.infra.in_memory_context_store import InMemoryContextStore
from trading_agents.infra.in_memory_trace_store import InMemoryTraceStore
from trading_agents.infra.run_store import RunStore
from trading_agents.schemas import AgentContext, LoopTrace, QlibFactorExperiment, RoundPhase, SharedContext, StrategyTrajectory, TrajectoryPool
from trading_agents.workflows.evolution_guidance import EvolutionGuidanceBuilder
from trading_agents.workflows.loop_runtime import LoopPolicy, LoopRunner, LoopSession, RoundState


@dataclass(frozen=True)
class EvolutionConfig:
    """轨迹级遗传迭代调度配置。

    控制整个遗传进化流程的参数，包括每轮方向数量、变异/交叉任务配额、
    父代选择范围、早停策略等。可通过命令行参数或环境变量覆盖。
    """

    # ── QuantaAlpha profile ──
    profile: str = "quantaalpha"          # quantaalpha / legacy / custom
    # 论文式自进化迭代数。每个 iteration 对应「一个 mutation 轮 + 紧随其后的一个 crossover 轮」，
    # 即论文的 mutation→crossover→mutation→crossover… 交替结构。这两个阶段在【不同轮次】执行，
    # 而非同一轮内嵌套：第 0 轮为 ORIGINAL，之后奇数轮跑 mutation、偶数轮跑 crossover。
    # 因此 max_rounds = 1(original) + iterations*2 = 1 + 5*2 = 11。
    iterations: int = 5
    phase_schedule: str = "mutation_then_crossover"  # round>=1 起交替：奇数轮 mutation，偶数轮 crossover
    children_per_mutation_phase: int = 10
    children_per_crossover_phase: int = 10
    factor_expressions_per_hypothesis: int = 3

    # ── 初始假设 H0 ──
    num_initial_directions: int = 10      # 第 0 轮由初始化智能体 A0 直接生成的初始假设数量

    # ── 进化轮次 ──
    max_rounds: int = 11                  # 遗传迭代最大轮数（含第 0 轮）
    early_stop_patience: int = 2          # 连续无进展轮数达到此值时提前终止
    primary_metric: str = "Rank IC"       # 用于评估轨迹优劣的主要指标

    # ── 变异与交叉开关 ──
    mutation_enabled: bool = True         # 是否启用变异任务
    crossover_enabled: bool = True        # 是否启用交叉任务
    # 交叉状态级段继承（对齐论文 Eq.7：显式继承已验证决策）。
    # True：交叉任务冻结一个供体父代的 hypothesis 状态，从 experiment_designer 起跑，
    #       其余父代的因子构造模式经 fusion guidance 融入（状态级继承）。
    # False：回退为旧行为——从 hypothesis_agent 全量重跑，仅以融合文本引导（与官方仓库的
    #        hybrid-hypothesis 引导式交叉一致）。
    crossover_inherit_state: bool = True

    # ── 每轮任务配额（EvolutionAgent 的目标产出数）──
    mutation_tasks_per_round: int = 10    # 每轮规划的变异任务数
    crossover_tasks_per_round: int = 10   # 每轮规划的交叉任务数
    evolved_directions_per_round: int = 5 # 兼容旧 joint-seed 模式；默认调度直接执行 mutation/crossover 任务

    # ── 父代选择范围 ──
    mutation_parent_top_n: int = 10       # 提供给 EvolutionAgent 的变异候选父轨迹数
    crossover_parent_top_n: int = 10      # 提供给 EvolutionAgent 的交叉候选父轨迹数

    # ── 规则 fallback 参数（当 LLM 规划产出不足时使用）──
    mutation_top_n: int = 3               # 规则 fallback 的变异父轨迹数
    crossover_group_size: int = 2         # 规则 fallback 的每组交叉父轨迹数
    crossover_n_groups: int = 2           # 规则 fallback 的交叉组合组数

    # ── 轮内并行执行 ──
    parallel_enabled: bool = True         # 是否并行执行同一轮内的多个方向任务
    max_workers: int = 5                  # 并行执行的最大线程数（受 API 限流时可调低）

    def effective_max_rounds(self) -> int:
        return max(1, int(self.max_rounds))

    def target_counts_for_round(self, round_idx: int) -> tuple[int, int]:
        if round_idx <= 0:
            return 0, 0
        if self.profile.lower() == "quantaalpha" and self.phase_schedule == "mutation_then_crossover":
            mutation_children = (
                int(self.mutation_tasks_per_round)
                if int(self.children_per_mutation_phase) == 10 and int(self.mutation_tasks_per_round) != 10
                else int(self.children_per_mutation_phase)
            )
            crossover_children = (
                int(self.crossover_tasks_per_round)
                if int(self.children_per_crossover_phase) == 10 and int(self.crossover_tasks_per_round) != 10
                else int(self.children_per_crossover_phase)
            )
            if round_idx % 2 == 1:
                return (max(0, mutation_children if self.mutation_enabled else 0), 0)
            return (0, max(0, crossover_children if self.crossover_enabled else 0))
        return (
            max(0, int(self.mutation_tasks_per_round if self.mutation_enabled else 0)),
            max(0, int(self.crossover_tasks_per_round if self.crossover_enabled else 0)),
        )

    def phase_for_round(self, round_idx: int) -> str:
        if round_idx == 0:
            return RoundPhase.ORIGINAL.value
        if self.profile.lower() == "quantaalpha" and self.phase_schedule == "mutation_then_crossover":
            return RoundPhase.MUTATION.value if round_idx % 2 == 1 else RoundPhase.CROSSOVER.value
        return "evolved"


@dataclass(frozen=True)
class EvolutionTask:
    """传入现有编排器循环的单个可执行遗传任务。

    每个任务代表一次完整的 5-Agent 因子挖掘循环（假说→实验设计→编码→回测→反馈），
    携带遗传元数据（阶段、父代 ID、变异类型、进化引导等）供下游 Agent 参考。
    """

    # ── 身份标识 ──
    phase: RoundPhase                      # 遗传阶段：ORIGINAL / MUTATION / CROSSOVER / EVOLVED
    round_idx: int                         # 所属进化轮次（第 0 轮为原始探索，1+ 为进化轮）
    direction_id: int                      # 方向编号（同一轮内唯一）
    direction: str                         # 本次挖掘的研究方向文本
    trajectory_id: str                     # 轨迹唯一标识（格式: {phase}_{round}_{direction}）
    parent_ids: list[str]                  # 父代轨迹 ID 列表（原始任务为空列表）

    # ── 遗传引导信息 ──
    evolution_guidance: str = ""           # 注入给 HypothesisAgent 的进化引导文本
    parent_trajectory_summaries: list[dict[str, object]] = field(default_factory=list)  # 父代轨迹摘要（供 Agent 参考）
    mutation_type: str = ""                # 变异类型：hypothesis_mutation / expression_mutation / parameter_mutation / risk_mutation

    # ── 规划元数据 ──
    planner_mode: str = "fallback_rule"    # 规划模式："llm"（由 LLM 规划）或 "fallback_rule"（规则兜底）
    direction_title: str = ""              # 方向标题（可选）
    rationale: str = ""                    # 规划理由说明
    fusion_logic: str = ""                 # 交叉融合逻辑说明（仅 CROSSOVER 任务）
    evolution_seed: str = ""               # 进化种子（第 1+ 轮的综合方向文本）
    evolution_plan_summaries: list[dict[str, object]] = field(default_factory=list)  # 原始变异/交叉规划摘要
    target_segment: str = ""               # mutation 定位的失败 segment
    fault_diagnosis: str = ""              # mutation 自反思诊断
    preserve_segments: list[str] = field(default_factory=list)
    rewrite_instruction: str = ""
    source_segments: list[dict[str, object]] = field(default_factory=list)
    fusion_plan: str = ""
    start_agent_name: str = ""             # 局部重放从哪个 agent 开始
    frozen_prefix_payload: dict[str, object] = field(default_factory=dict)


@dataclass
class _TaskResult:
    """单个 EvolutionTask 完整执行后的结果容器，供主线程统一合并写入 session。

    worker 线程只负责填充此对象并返回，不直接写 session / trajectory_pool / factor_library。
    """

    task: EvolutionTask
    trace: LoopTrace
    round_state: RoundState
    private_snapshot: dict[str, Any]


class EvolutionRunner:
    """遗传进化调度器。

    核心职责：
    1. 第 0 轮：用初始化智能体 A0 直接生成 N 条初始假设 H0，每条假设进入实验设计和后续链路
    2. 第 1+ 轮：用 EvolutionAgent 分析轨迹池，规划变异/交叉任务
    3. 记录每轮轨迹到 TrajectoryPool，供后续轮次的父代选择使用

    整体流程: original → EvolutionAgent 规划 mutation/crossover → mutation/crossover 分支分别执行
    """

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        model_client: ModelClient,
        context_store: ContextStore | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        guidance_builder: EvolutionGuidanceBuilder | None = None,
    ) -> None:
        self.orchestrator = orchestrator          # 现有的 5-Agent 顺序编排器
        self.model_client = model_client           # LLM 客户端（用于 DirectionDecomposer 和 EvolutionAgent）
        self.context_store = context_store         # 上下文存储（可选）
        self.progress_callback = progress_callback # 进度回调（用于日志/UI 更新）
        self.guidance_builder = guidance_builder or EvolutionGuidanceBuilder()  # 进化引导信息生成器
        self.direction_decomposer_agent = DirectionDecomposerAgent("direction_decomposer_agent")  # 方向拆解智能体
        self.evolution_agent = EvolutionAgent("evolution_agent")  # 进化规划智能体

    def run(
        self,
        *,
        initial_shared_payload: dict[str, object] | None,
        initial_direction: str | None,
        config: EvolutionConfig,
    ) -> LoopSession:
        """运行遗传迭代会话的主入口。

        流程：
        1. 初始化 LoopSession、TrajectoryPool、FactorLibraryManager
        2. 遍历每一轮（round_idx 0 到 max_rounds-1）：
           a. 调用 _build_tasks 生成该轮的所有任务
           b. 如果没有任务生成（轨迹池为空等），累计无进展计数，达到 patience 时早停
           c. 对每个任务，转换为 SharedPayload 后交给 orchestrator.run_one_loop 执行
           d. 记录执行结果到 LoopSession 和 TrajectoryPool
        3. 返回包含完整血缘状态的 LoopSession

        Returns:
            LoopSession: 包含所有轮次的轨迹、执行记录和轨迹池状态
        """
        # 初始化会话：设置策略、共享上下文、轨迹池和因子库路径
        session = LoopSession(
            policy=LoopPolicy(max_loops=config.effective_max_rounds()),
            latest_shared_context=dict(initial_shared_payload or {}),
            trajectory_pool=TrajectoryPool(save_path=self._trajectory_state_path()),
            factor_library_path=self._factor_library_path(),
        )
        # 尝试从磁盘加载已有的轨迹池（支持断点续跑）
        self._load_trajectory_pool(session.trajectory_pool)
        factor_library = FactorLibraryManager(session.factor_library_path)
        base_payload = dict(initial_shared_payload or {})
        base_payload.setdefault("evolution_profile", config.profile)
        base_payload.setdefault("phase_schedule", config.phase_schedule)
        base_payload.setdefault("factor_expressions_per_hypothesis", int(config.factor_expressions_per_hypothesis))
        no_progress_rounds = 0   # 连续无进展轮数计数器
        executed_task_count = 0  # 已执行任务总数
        accumulated_execution_failures: list[dict[str, object]] = []  # 跨轮累积的执行错误，供下一轮 EvolutionAgent 参考

        for round_idx in range(config.effective_max_rounds()):
            # 根据轮次生成任务列表：第 0 轮生成原始方向，第 1+ 轮生成进化任务
            tasks = self._build_tasks(
                round_idx=round_idx,
                base_payload=base_payload,
                initial_direction=initial_direction,
                config=config,
                trajectory_pool=session.trajectory_pool,
                execution_failures=accumulated_execution_failures,
            )
            # 如果本轮没有生成任何任务，累计无进展计数
            if not tasks:
                no_progress_rounds += 1
                self._notify_progress(
                    {
                        "stage": "evolution_round_skipped",
                        "phase": self._phase_label(round_idx, config=config),
                        "round_idx": round_idx,
                        "task_count": 0,
                    }
                )
                if no_progress_rounds >= max(1, int(config.early_stop_patience)):
                    break
                continue

            # 本轮有任务，重置无进展计数
            no_progress_rounds = 0
            self._notify_progress(
                {
                    "stage": "evolution_round_planned",
                    "phase": self._phase_label(round_idx, config=config),
                    "round_idx": round_idx,
                    "task_count": len(tasks),
                    "mutation_task_count": len([task for task in tasks if task.phase == RoundPhase.MUTATION]),
                    "crossover_task_count": len([task for task in tasks if task.phase == RoundPhase.CROSSOVER]),
                    "original_task_count": len([task for task in tasks if task.phase == RoundPhase.ORIGINAL]),
                    "evolved_task_count": len([task for task in tasks if task.phase == RoundPhase.EVOLVED]),
                }
            )
            # 执行任务：串行或并行（由 config.parallel_enabled 控制）
            results = self._run_tasks_for_round(
                tasks=tasks,
                base_payload=base_payload,
                start_loop_index=executed_task_count + 1,
                config=config,
            )
            # 主线程统一合并结果，保证 trajectory_pool / factor_library 单线程写入
            for result in results:
                executed_task_count += 1
                session.rounds.append(result.round_state)
                session.loop_traces.append(result.trace)
                session.latest_shared_context = deepcopy(result.trace.final_shared_context)
                session.private_context_snapshots.append(result.private_snapshot)
                session.total_errors += result.round_state.error_count
                self._record_trajectory(session, result.trace.final_shared_context, result.task)
                factor_library.add_from_shared_payload(result.trace.final_shared_context)
                self._notify_task_progress(
                    "evolution_task_finished",
                    result.task,
                    result.round_state.loop_index,
                    result.round_state.ended_at,
                    extra={
                        "error_count": result.round_state.error_count,
                        "agent_count": result.round_state.agent_count,
                    },
                )
            # 判断本轮是否全部失败
            all_failed = all(r.round_state.error_count != 0 for r in results)
            if all_failed:
                # 提取本轮执行错误，累积供下一轮 EvolutionAgent 参考，驱动智能恢复
                new_failures = self._extract_execution_failures(results, round_idx=round_idx)
                accumulated_execution_failures.extend(new_failures)
                accumulated_execution_failures = accumulated_execution_failures[-20:]  # 滚动窗口，最多保留 20 条
                no_progress_rounds += 1
                self._notify_progress(
                    {
                        "stage": "evolution_round_all_failed",
                        "phase": self._phase_label(round_idx, config=config),
                        "round_idx": round_idx,
                        "task_count": len(results),
                        "new_failure_count": len(new_failures),
                        "accumulated_failure_count": len(accumulated_execution_failures),
                        "no_progress_rounds": no_progress_rounds,
                        "early_stop_patience": config.early_stop_patience,
                    }
                )
                if no_progress_rounds >= max(1, int(config.early_stop_patience)):
                    session.status = "failed"
                    return session
                # 继续下一轮，EvolutionAgent 将收到 accumulated_execution_failures 并据此调整方向
                continue
            else:
                # 至少一个任务成功，清空执行错误累积
                accumulated_execution_failures = []

        session.status = "completed"
        return session

    @staticmethod
    def _trajectory_state_path() -> str:
        """轨迹池持久化路径（artifacts/trajectory_pool.jsonl）。"""
        return os.getenv(
            "TRAJECTORY_POOL_PATH",
            str(FactorLibraryManager.project_root() / "artifacts" / "trajectory_pool.jsonl"),
        )

    @staticmethod
    def _factor_library_path() -> str:
        """因子库持久化路径。"""
        return str(FactorLibraryManager.default_library_path())

    @staticmethod
    def _load_trajectory_pool(pool: TrajectoryPool) -> None:
        store_path = RunStore.default_path(FactorLibraryManager.project_root())
        if store_path.exists():
            try:
                loaded = RunStore(store_path).load_trajectories()
                if loaded:
                    pool.trajectories = loaded
                    return
            except Exception:
                pass
        pool.load()

    def _notify_progress(self, payload: dict[str, object]) -> None:
        """转发进度事件到外部回调，并隔离回调异常对调度器的影响。"""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            return

    def _notify_task_progress(
        self,
        stage: str,
        task: EvolutionTask,
        loop_index: int,
        timestamp: datetime,
        *,
        extra: dict[str, object] | None = None,
    ) -> None:
        """发送包含遗传元数据的任务级进度事件（阶段、方向、父代等）。"""
        payload: dict[str, object] = {
            "stage": stage,
            "loop_index": loop_index,
            "phase": task.phase.value,
            "round_idx": task.round_idx,
            "direction_id": task.direction_id,
            "trajectory_id": task.trajectory_id,
            "parent_ids": list(task.parent_ids),
            "mutation_type": task.mutation_type,
            "target_segment": task.target_segment,
            "partial_replay_start_agent": task.start_agent_name,
            "timestamp": timestamp.isoformat(),
        }
        if extra:
            payload.update(extra)
        self._notify_progress(payload)

    def _build_tasks(
        self,
        *,
        round_idx: int,
        base_payload: dict[str, object],
        initial_direction: str | None,
        config: EvolutionConfig,
        trajectory_pool: TrajectoryPool,
        execution_failures: list[dict[str, object]] | None = None,
    ) -> list[EvolutionTask]:
        """根据轮次选择任务生成策略。

        第 0 轮：调用 _build_original_tasks，生成 N 条初始假设任务
        第 1+ 轮：调用 _build_evolved_tasks，基于轨迹池规划变异/交叉任务
        """
        if round_idx == 0:
            return self._build_original_tasks(
                base_payload=base_payload,
                initial_direction=initial_direction,
                count=max(1, int(config.num_initial_directions)),
            )
        return self._build_evolved_tasks(
            round_idx=round_idx,
            base_payload=base_payload,
            config=config,
            trajectory_pool=trajectory_pool,
            execution_failures=execution_failures or [],
        )

    def _build_original_tasks(
        self,
        *,
        base_payload: dict[str, object],
        initial_direction: str | None,
        count: int,
    ) -> list[EvolutionTask]:
        """为第 0 轮生成多样化初始假设任务。

        流程：
        1. 调用初始化智能体（LLM）直接提出 N 个互补初始假设 H0
        2. 如果 LLM 拆解不够 N 个，用确定性 fallback 规则补充（预设主题）
        3. 如果 fallback 后仍不够，生成 "initial_direction; fallback branch X" 格式的兜底方向
        4. 为每个方向创建 EvolutionTask（phase=ORIGINAL, parent_ids=[]）
        """
        target_count = max(1, int(count))
        direction_items = self._decompose_initial_directions(
            base_payload=base_payload,
            initial_direction=initial_direction,
            count=target_count,
        )
        if len(direction_items) < target_count:
            existing = {
                str(item.get("hypothesis", item.get("direction", ""))).strip().lower()
                for item in direction_items
                if str(item.get("hypothesis", item.get("direction", ""))).strip()
            }
            fallback_directions = self._diversified_directions(
                initial_direction=initial_direction or str(base_payload.get("direction", "")),
                count=target_count,
            )
            for direction in fallback_directions:
                if len(direction_items) >= target_count:
                    break
                if direction.strip().lower() in existing:
                    continue
                direction_items.append(self._build_fallback_initial_hypothesis_item(direction, idx=len(direction_items)))
                existing.add(direction.strip().lower())
            while len(direction_items) < target_count:
                idx = len(direction_items)
                direction_items.append(
                    self._build_fallback_initial_hypothesis_item(
                        f"{initial_direction or str(base_payload.get('direction', 'Explore cross-sectional alpha'))}; fallback branch {idx}",
                        idx=idx,
                    )
                )
        direction_items = direction_items[:target_count]
        return [
            EvolutionTask(
                phase=RoundPhase.ORIGINAL,
                round_idx=0,
                direction_id=idx,
                direction=str(item.get("hypothesis", item.get("direction", ""))),
                trajectory_id=StrategyTrajectory.generate_id(direction_id=idx, round_idx=0, phase=RoundPhase.ORIGINAL),
                parent_ids=[],
                evolution_guidance=self._build_original_guidance(item),
                planner_mode=str(item.get("direction_decomposer_mode", "llm")),
                direction_title=str(item.get("title", "")),
                rationale=str(item.get("rationale", "")),
                start_agent_name="experiment_designer_agent",
                frozen_prefix_payload=self._build_initial_hypothesis_prefix(item),
            )
            for idx, item in enumerate(direction_items)
            if str(item.get("hypothesis", item.get("direction", ""))).strip()
        ]

    def _decompose_initial_directions(
        self,
        *,
        base_payload: dict[str, object],
        initial_direction: str | None,
        count: int,
    ) -> list[dict[str, object]]:
        """用初始化智能体（LLM）直接生成 N 个聚焦初始假设。

        成功时返回假设列表（标记 direction_decomposer_mode="llm"）。
        失败时返回空列表，由调用方 _build_original_tasks 使用规则 fallback 补充。
        """
        planner_payload = dict(base_payload)
        planner_payload["initial_direction"] = initial_direction or str(base_payload.get("direction", ""))
        planner_payload["target_direction_count"] = int(count)
        started_at = datetime.now()
        self._notify_progress(
            {
                "stage": "direction_decomposer_started",
                "timestamp": started_at.isoformat(),
                "target_direction_count": int(count),
            }
        )
        try:
            result = self.direction_decomposer_agent.run(
                private_context=AgentContext(),
                shared_context=SharedContext(payload=planner_payload),
                model_client=self.model_client,
            )
            hypotheses = [
                dict(item)
                for item in list(result.shared_updates.get("initial_hypotheses", []))
                if isinstance(item, dict)
            ]
            if not hypotheses:
                hypotheses = [
                    self._build_fallback_initial_hypothesis_item(str(item.get("direction", "")), idx=idx, source_item=item)
                    for idx, item in enumerate(list(result.shared_updates.get("decomposed_directions", [])))
                    if isinstance(item, dict) and str(item.get("direction", "")).strip()
                ]
            for item in hypotheses:
                item["direction_decomposer_mode"] = "llm"
            self._notify_progress(
                {
                    "stage": "direction_decomposer_finished",
                    "timestamp": datetime.now().isoformat(),
                    "direction_count": len(hypotheses),
                    "initial_hypothesis_count": len(hypotheses),
                    "direction_decomposer_mode": "llm",
                }
            )
            return hypotheses
        except Exception as exc:
            self._notify_progress(
                {
                    "stage": "direction_decomposer_failed",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(exc),
                    "direction_decomposer_mode": "fallback_rule",
                }
            )
            return []

    @staticmethod
    def _build_original_guidance(direction_item: dict[str, object]) -> str:
        """为第 0 轮原始假设生成进化引导文本（包含标题、理由、信号族、风险提示）。"""
        parts = [
            "Evolution phase: diversified initial hypothesis generated by initialization agent A0.",
            f"Direction title: {str(direction_item.get('title', '')).strip()}",
            f"Initial hypothesis: {str(direction_item.get('hypothesis', direction_item.get('direction', ''))).strip()}",
            f"Mechanism: {str(direction_item.get('mechanism', '')).strip()}",
            f"Selected parametric specifications: {', '.join(str(item) for item in list(direction_item.get('selected_parametric_specifications', [])) if str(item).strip())}",
            f"Diversity axis: {str(direction_item.get('diversity_axis', '')).strip()}",
            f"Rationale: {str(direction_item.get('rationale', '')).strip()}",
            f"Expected signal family: {str(direction_item.get('expected_signal_family', '')).strip()}",
            f"Risk notes: {str(direction_item.get('risk_notes', '')).strip()}",
        ]
        return "\n".join(part for part in parts if part.split(":", 1)[-1].strip())

    @staticmethod
    def _build_fallback_initial_hypothesis_item(
        direction: str,
        *,
        idx: int,
        source_item: dict[str, object] | None = None,
    ) -> dict[str, object]:
        source = dict(source_item or {})
        hypothesis = str(source.get("hypothesis", direction)).strip() or direction
        return {
            "direction_id": idx,
            "title": str(source.get("title", f"fallback_hypothesis_{idx}")),
            "hypothesis": hypothesis,
            "thesis": str(source.get("thesis", hypothesis)),
            "mechanism": str(source.get("mechanism", source.get("rationale", ""))).strip(),
            "expected_direction": str(source.get("expected_direction", "")),
            "proxy_reconstruction_plan": dict(source.get("proxy_reconstruction_plan", {}))
            if isinstance(source.get("proxy_reconstruction_plan", {}), dict)
            else {},
            "selected_parametric_specifications": [
                str(item)
                for item in list(source.get("selected_parametric_specifications", []))
                if str(item).strip()
            ],
            "hypothesis_evolution_state": dict(source.get("hypothesis_evolution_state", {}))
            if isinstance(source.get("hypothesis_evolution_state", {}), dict)
            else {},
            "rationale": str(
                source.get(
                    "rationale",
                    "Fallback deterministic hypothesis generated because initialization output was incomplete.",
                )
            ),
            "concise_observation": str(source.get("concise_observation", "")),
            "concise_justification": str(source.get("concise_justification", source.get("rationale", ""))),
            "concise_specification": str(source.get("concise_specification", "")),
            "expected_signal_family": str(source.get("expected_signal_family", "")),
            "risk_notes": str(source.get("risk_notes", "")),
            "diversity_axis": str(source.get("diversity_axis", "")),
            "direction_decomposer_mode": str(source.get("direction_decomposer_mode", "fallback_rule")),
        }

    @staticmethod
    def _build_initial_hypothesis_prefix(item: dict[str, object]) -> dict[str, object]:
        hypothesis = str(item.get("hypothesis", item.get("direction", ""))).strip()
        proxy_plan = item.get("proxy_reconstruction_plan", {})
        if not isinstance(proxy_plan, dict):
            proxy_plan = {}
        evolution_state = item.get("hypothesis_evolution_state", {})
        if not isinstance(evolution_state, dict):
            evolution_state = {}
        structured = {
            "hypothesis": hypothesis,
            "thesis": str(item.get("thesis", hypothesis)).strip() or hypothesis,
            "mechanism": str(item.get("mechanism", "")).strip(),
            "expected_direction": str(item.get("expected_direction", "")).strip(),
            "selected_parametric_specifications": [
                str(spec)
                for spec in list(item.get("selected_parametric_specifications", []))
                if str(spec).strip()
            ],
            "proxy_reconstruction_plan": dict(proxy_plan),
            "concise_knowledge": str(item.get("concise_knowledge", "")).strip(),
            "concise_observation": str(item.get("concise_observation", "")).strip(),
            "concise_justification": str(item.get("concise_justification", item.get("rationale", ""))).strip(),
            "concise_specification": str(item.get("concise_specification", "")).strip(),
            "hypothesis_evolution_state": dict(evolution_state),
            "initialization_agent": {
                "title": str(item.get("title", "")).strip(),
                "diversity_axis": str(item.get("diversity_axis", "")).strip(),
                "expected_signal_family": str(item.get("expected_signal_family", "")).strip(),
                "risk_notes": str(item.get("risk_notes", "")).strip(),
            },
        }
        return {
            "hypothesis": hypothesis,
            "hypothesis_reasoning": structured["concise_justification"],
            "hypothesis_structured": structured,
            "initial_hypothesis_package": dict(item),
        }

    def _build_evolved_tasks(
        self,
        *,
        round_idx: int,
        base_payload: dict[str, object],
        config: EvolutionConfig,
        trajectory_pool: TrajectoryPool,
        execution_failures: list[dict[str, object]] | None = None,
    ) -> list[EvolutionTask]:
        """第 1+ 轮的进化任务生成。

        两阶段流程：
        1. 调用 EvolutionAgent（LLM）分析轨迹池，规划变异/交叉任务
        2. 如果 LLM 产出不足，用规则 fallback 补充（TrajectoryPool 父代选择）

        返回的每个任务都会携带父代摘要和进化引导信息。
        """
        mutation_target_count, crossover_target_count = config.target_counts_for_round(round_idx)
        if mutation_target_count <= 0 and crossover_target_count <= 0:
            return []
        planned_tasks = self._plan_evolved_tasks_with_llm(
            round_idx=round_idx,
            config=config,
            trajectory_pool=trajectory_pool,
            mutation_target_count=mutation_target_count,
            crossover_target_count=crossover_target_count,
            execution_failures=execution_failures or [],
        )
        mutation_tasks = [task for task in planned_tasks if task.phase == RoundPhase.MUTATION]
        crossover_tasks = [task for task in planned_tasks if task.phase == RoundPhase.CROSSOVER]

        fallback_tasks: list[EvolutionTask] = []
        if config.mutation_enabled and len(mutation_tasks) < mutation_target_count:
            parents = trajectory_pool.select_parents_for_mutation(top_n=max(mutation_target_count, int(config.mutation_top_n), int(config.mutation_parent_top_n)))
            used_parent_ids = {parent_id for task in mutation_tasks for parent_id in task.parent_ids}
            available = [parent for parent in parents if parent.trajectory_id not in used_parent_ids]
            for parent in available[: mutation_target_count - len(mutation_tasks)]:
                fallback_tasks.append(self._build_mutation_task(parent, round_idx=round_idx, offset=len(mutation_tasks) + len(fallback_tasks)))
        if config.crossover_enabled and len(crossover_tasks) < crossover_target_count:
            parent_groups = trajectory_pool.select_groups_for_crossover(
                group_size=config.crossover_group_size,
                n_groups=max(crossover_target_count, int(config.crossover_n_groups)),
            )
            for parents in parent_groups[: crossover_target_count - len(crossover_tasks)]:
                fallback_tasks.append(self._build_crossover_task(parents, round_idx=round_idx, offset=len(crossover_tasks) + len(fallback_tasks), config=config))
        planned_tasks = planned_tasks + fallback_tasks
        for task in planned_tasks:
            self._notify_progress(
                {
                    "stage": "evolution_task_planned",
                    "phase": task.phase.value,
                    "round_idx": round_idx,
                    "direction_id": task.direction_id,
                    "trajectory_id": task.trajectory_id,
                    "parent_ids": list(task.parent_ids),
                    "mutation_type": task.mutation_type,
                    "evolution_planner_mode": task.planner_mode,
                    "target_segment": task.target_segment,
                    "partial_replay_start_agent": task.start_agent_name,
                }
            )
        return planned_tasks

    def _decompose_evolved_seed_to_tasks(
        self,
        *,
        round_idx: int,
        base_payload: dict[str, object],
        config: EvolutionConfig,
        planned_tasks: list[EvolutionTask],
    ) -> list[EvolutionTask]:
        """将进化规划综合成一个种子方向，再拆解为下一代固定数量的方案。

        流程：
        1. 合并所有规划任务的父代 ID、父代摘要、融合逻辑为统一信息
        2. 将所有规划方向合成为一段 "进化种子" 文本
        3. 用 DirectionDecomposerAgent（LLM）将种子拆解为 N 个下一代方向
        4. 如果 LLM 拆解不足，用规则 fallback 补充
        5. 为每个方向创建 EvolutionTask（phase=EVOLVED, mutation_type=generation_branch）
        """
        target_count = max(1, int(config.evolved_directions_per_round or config.num_initial_directions))
        if not planned_tasks:
            return []
        evolution_seed = self._compose_evolution_seed(planned_tasks)
        parent_ids = self._merge_parent_ids(planned_tasks)
        parent_summaries = self._merge_parent_summaries(planned_tasks)
        plan_summaries = self._summarize_planned_tasks(planned_tasks)
        seed_mode = self._dominant_planner_mode(task.planner_mode for task in planned_tasks)
        direction_items = self._decompose_evolved_seed(
            base_payload=base_payload,
            evolution_seed=evolution_seed,
            count=target_count,
            round_idx=round_idx,
        )
        if len(direction_items) < target_count:
            existing = {str(item.get("direction", "")).strip().lower() for item in direction_items}
            fallback_directions = self._diversified_directions(initial_direction=evolution_seed, count=target_count)
            for direction in fallback_directions:
                if len(direction_items) >= target_count:
                    break
                if direction.strip().lower() in existing:
                    continue
                direction_items.append(
                    {
                        "title": f"evolved_fallback_{len(direction_items)}",
                        "direction": direction,
                        "rationale": "Fallback evolved branch generated from the EvolutionAgent seed.",
                        "direction_decomposer_mode": "fallback_rule",
                    }
                )
                existing.add(direction.strip().lower())
        generation_tasks: list[EvolutionTask] = []
        for idx, item in enumerate(direction_items[:target_count]):
            direction = str(item.get("direction", "")).strip()
            if not direction:
                continue
            mutation_type = "generation_branch"
            guidance = "\n".join(
                part
                for part in [
                    "Evolution phase: next-generation branch decomposed after joint mutation and crossover review.",
                    f"Evolution seed: {evolution_seed}",
                    f"Branch rationale: {str(item.get('rationale', '')).strip()}",
                    "Use this branch as one of the five next-round hypothesis directions.",
                ]
                if part.split(":", 1)[-1].strip()
            )
            generation_tasks.append(
                EvolutionTask(
                    phase=RoundPhase.EVOLVED,
                    round_idx=round_idx,
                    direction_id=round_idx * 1000 + idx,
                    direction=direction,
                    trajectory_id=StrategyTrajectory.generate_id(
                        direction_id=round_idx * 1000 + idx,
                        round_idx=round_idx,
                        phase=RoundPhase.EVOLVED,
                    ),
                    parent_ids=list(parent_ids),
                    evolution_guidance=guidance,
                    parent_trajectory_summaries=list(parent_summaries),
                    mutation_type=mutation_type,
                    planner_mode=seed_mode,
                    direction_title=str(item.get("title", "")),
                    rationale=str(item.get("rationale", "")),
                    fusion_logic=self._merge_fusion_logic(planned_tasks),
                    evolution_seed=evolution_seed,
                    evolution_plan_summaries=list(plan_summaries),
                )
            )
        self._notify_progress(
            {
                "stage": "evolved_seed_decomposed",
                "round_idx": int(round_idx),
                "direction_count": len(generation_tasks),
                "parent_ids": list(parent_ids),
                "evolution_planner_mode": seed_mode,
            }
        )
        return generation_tasks

    def _decompose_evolved_seed(
        self,
        *,
        base_payload: dict[str, object],
        evolution_seed: str,
        count: int,
        round_idx: int,
    ) -> list[dict[str, object]]:
        """用 DirectionDecomposerAgent（LLM）将进化种子拆解为 N 个下一代方向。

        与 _decompose_initial_directions 类似，但在 scenario 字段附加了进化轮次信息，
        提示 LLM 这是从上一代进化结果中衍生出的方向。
        """
        planner_payload = dict(base_payload)
        planner_payload["initial_direction"] = evolution_seed
        planner_payload["direction"] = evolution_seed
        planner_payload["target_direction_count"] = int(count)
        planner_payload["scenario"] = str(base_payload.get("scenario", "")) + f"\nGeneration round: {round_idx}"
        self._notify_progress(
            {
                "stage": "evolved_direction_decomposer_started",
                "timestamp": datetime.now().isoformat(),
                "round_idx": int(round_idx),
                "target_direction_count": int(count),
            }
        )
        try:
            result = self.direction_decomposer_agent.run(
                private_context=AgentContext(),
                shared_context=SharedContext(payload=planner_payload),
                model_client=self.model_client,
            )
            directions = [dict(item) for item in list(result.shared_updates.get("decomposed_directions", [])) if isinstance(item, dict)]
            for item in directions:
                item["direction_decomposer_mode"] = "llm"
            self._notify_progress(
                {
                    "stage": "evolved_direction_decomposer_finished",
                    "timestamp": datetime.now().isoformat(),
                    "round_idx": int(round_idx),
                    "direction_count": len(directions),
                    "direction_decomposer_mode": "llm",
                }
            )
            return directions
        except Exception as exc:
            self._notify_progress(
                {
                    "stage": "evolved_direction_decomposer_failed",
                    "timestamp": datetime.now().isoformat(),
                    "round_idx": int(round_idx),
                    "error": str(exc),
                    "direction_decomposer_mode": "fallback_rule",
                }
            )
            return []

    @staticmethod
    def _compose_evolution_seed(planned_tasks: list[EvolutionTask]) -> str:
        """将所有变异/交叉规划任务合并为一段综合的进化种子文本。

        格式为 "Joint mutation/crossover review of the previous generation:"
        后跟每个任务的阶段、方向和引导信息，最后附加下一代方向生成指令。
        此文本会作为 DirectionDecomposerAgent 的输入进行下一轮拆解。
        """
        lines = ["Joint mutation/crossover review of the previous generation:"]
        for task in planned_tasks:
            label = "Mutation" if task.phase == RoundPhase.MUTATION else "Crossover"
            lines.append(f"- {label}: {task.direction}")
            if task.evolution_guidance:
                lines.append(f"  Guidance: {task.evolution_guidance}")
        lines.append("Create five next-generation hypothesis directions that preserve useful evidence and diversify testable mechanisms.")
        return "\n".join(lines)

    @staticmethod
    def _merge_parent_ids(planned_tasks: list[EvolutionTask]) -> list[str]:
        """合并所有规划任务的父代轨迹 ID（去重保序）。"""
        parent_ids: list[str] = []
        for task in planned_tasks:
            for parent_id in task.parent_ids:
                if parent_id not in parent_ids:
                    parent_ids.append(parent_id)
        return parent_ids

    @staticmethod
    def _merge_parent_summaries(planned_tasks: list[EvolutionTask]) -> list[dict[str, object]]:
        """合并所有规划任务的父代轨迹摘要（按 trajectory_id 去重）。"""
        summaries: list[dict[str, object]] = []
        seen: set[str] = set()
        for task in planned_tasks:
            for summary in task.parent_trajectory_summaries:
                trajectory_id = str(summary.get("trajectory_id", ""))
                if trajectory_id and trajectory_id in seen:
                    continue
                if trajectory_id:
                    seen.add(trajectory_id)
                summaries.append(dict(summary))
        return summaries

    @staticmethod
    def _merge_fusion_logic(planned_tasks: list[EvolutionTask]) -> str:
        """合并所有交叉任务的融合逻辑文本。"""
        items = [task.fusion_logic for task in planned_tasks if task.fusion_logic]
        return "\n".join(items)

    @staticmethod
    def _summarize_planned_tasks(planned_tasks: list[EvolutionTask]) -> list[dict[str, object]]:
        """保留原始变异/交叉规划，供下游质量门判断子代是否真的变化。"""
        summaries: list[dict[str, object]] = []
        for task in planned_tasks:
            summaries.append(
                {
                    "phase": task.phase.value if hasattr(task.phase, "value") else str(task.phase),
                    "trajectory_id": task.trajectory_id,
                    "direction_id": int(task.direction_id),
                    "parent_ids": list(task.parent_ids),
                    "mutation_type": task.mutation_type,
                    "direction": task.direction,
                    "rationale": task.rationale,
                    "fusion_logic": task.fusion_logic,
                    "planner_mode": task.planner_mode,
                    "target_segment": task.target_segment,
                    "fault_diagnosis": task.fault_diagnosis,
                    "preserve_segments": list(task.preserve_segments),
                    "rewrite_instruction": task.rewrite_instruction,
                    "source_segments": list(task.source_segments),
                    "fusion_plan": task.fusion_plan,
                }
            )
        return summaries

    @staticmethod
    def _dominant_planner_mode(values: Any) -> str:
        """判断规划模式：只要有一个任务是 LLM 规划的，整体就标记为 "llm"。"""
        modes = [str(value).strip() for value in values if str(value).strip()]
        if not modes:
            return "fallback_rule"
        return "llm" if "llm" in modes else modes[0]

    @staticmethod
    def _extract_execution_failures(
        results: list[_TaskResult],
        *,
        round_idx: int,
    ) -> list[dict[str, object]]:
        """从全部失败的轮次中提取执行错误信息，供下一轮 EvolutionAgent 进行智能恢复规划。

        提取内容：
        - 哪条轨迹失败（trajectory_id、phase、mutation_type、target_segment）
        - 哪个 agent 崩溃（failed_agents）
        - 具体错误文本（errors，截断至 300 字符避免 prompt 膨胀）
        - 方向摘要（direction_summary，前 200 字符）
        """
        failures: list[dict[str, object]] = []
        for result in results:
            if result.round_state.error_count == 0:
                continue
            task = result.task
            # 从 trace.records 中收集崩溃的 agent 名称
            failed_agents = [
                record.agent_name
                for record in result.trace.records
                if record.error
            ]
            # 截断错误文本，避免 prompt 过长
            truncated_errors = [str(e)[:300] for e in result.round_state.errors]
            failures.append(
                {
                    "round_idx": int(round_idx),
                    "trajectory_id": task.trajectory_id,
                    "phase": task.phase.value,
                    "mutation_type": task.mutation_type or "",
                    "target_segment": task.target_segment or "",
                    "start_agent": task.start_agent_name or "",
                    "parent_ids": list(task.parent_ids),
                    "direction_summary": task.direction[:200],
                    "failed_agents": failed_agents,
                    "errors": truncated_errors,
                }
            )
        return failures

    def _plan_evolved_tasks_with_llm(
        self,
        *,
        round_idx: int,
        config: EvolutionConfig,
        trajectory_pool: TrajectoryPool,
        mutation_target_count: int,
        crossover_target_count: int,
        execution_failures: list[dict[str, object]] | None = None,
    ) -> list[EvolutionTask]:
        """调用 EvolutionAgent（LLM）分析轨迹池并规划变异/交叉任务。

        流程：
        1. 从轨迹池中提取 best/recent/failed 三类轨迹的摘要
        2. 将摘要注入 EvolutionAgent 的上下文（含跨轮执行错误，用于智能失败恢复）
        3. EvolutionAgent 返回 mutation_tasks 和 crossover_tasks
        4. 将 LLM 输出转换为 EvolutionTask 列表

        如果 LLM 调用失败，返回空列表，由调用方使用规则 fallback 补充。
        """
        trajectory_by_id = {item.trajectory_id: item for item in trajectory_pool.get_all()}
        if not trajectory_by_id:
            return []
        best = trajectory_pool.get_best_trajectories(top_n=max(1, int(config.mutation_parent_top_n)))
        recent = trajectory_pool.get_all()[-max(1, int(config.crossover_parent_top_n)) :]
        failed = [item for item in trajectory_pool.get_all() if not item.is_successful()][-10:]
        payload = {
            "round_idx": int(round_idx),
            "mutation_target_count": int(mutation_target_count),
            "crossover_target_count": int(crossover_target_count),
            "primary_metric": config.primary_metric,
            "evolution_profile": config.profile,
            "phase_schedule": config.phase_schedule,
            "best_trajectories": [self.guidance_builder.summarize_trajectory(item) for item in best],
            "recent_trajectories": [self.guidance_builder.summarize_trajectory(item) for item in recent],
            "failed_trajectories": [self.guidance_builder.summarize_trajectory(item) for item in failed],
            "execution_failures": list(execution_failures or []),  # 执行错误（agent 崩溃/LLM 失败），区别于回测失败
        }
        self._notify_progress(
            {
                "stage": "evolution_agent_started",
                "timestamp": datetime.now().isoformat(),
                "round_idx": int(round_idx),
                "mutation_target_count": int(mutation_target_count),
                "crossover_target_count": int(crossover_target_count),
            }
        )
        try:
            result = self.evolution_agent.run(
                private_context=AgentContext(),
                shared_context=SharedContext(payload=payload),
                model_client=self.model_client,
            )
            plan = dict(result.shared_updates.get("evolution_agent_plan", {}))
        except Exception as exc:
            self._notify_progress(
                {
                    "stage": "evolution_agent_failed",
                    "timestamp": datetime.now().isoformat(),
                    "round_idx": int(round_idx),
                    "error": str(exc),
                    "evolution_planner_mode": "fallback_rule",
                }
            )
            return []
        tasks = self._convert_evolution_plan_to_tasks(
            plan=plan,
            round_idx=round_idx,
            trajectory_by_id=trajectory_by_id,
            mutation_target_count=mutation_target_count,
            crossover_target_count=crossover_target_count,
            config=config,
        )
        self._notify_progress(
            {
                "stage": "evolution_agent_finished",
                "timestamp": datetime.now().isoformat(),
                "round_idx": int(round_idx),
                "mutation_task_count": len([task for task in tasks if task.phase == RoundPhase.MUTATION]),
                "crossover_task_count": len([task for task in tasks if task.phase == RoundPhase.CROSSOVER]),
                "evolution_planner_mode": "llm",
            }
        )
        return tasks

    def _convert_evolution_plan_to_tasks(
        self,
        *,
        plan: dict[str, object],
        round_idx: int,
        trajectory_by_id: dict[str, StrategyTrajectory],
        mutation_target_count: int,
        crossover_target_count: int,
        config: EvolutionConfig,
    ) -> list[EvolutionTask]:
        """将 EvolutionAgent 的规划输出转换为 EvolutionTask 列表。

        对每个变异任务：
        - 校验 parent_ids 存在于轨迹池中
        - 生成唯一的 trajectory_id
        - 附加父代摘要和变异引导信息

        对每个交叉任务：
        - 校验至少有 2 个有效父代
        - 附加融合逻辑和交叉引导信息

        无效任务（父代不存在、方向为空等）会被静默跳过。
        """
        tasks: list[EvolutionTask] = []
        used_ids: set[str] = set()
        mutation_items = list(plan.get("mutation_tasks", [])) if isinstance(plan.get("mutation_tasks", []), list) else []
        crossover_items = list(plan.get("crossover_tasks", [])) if isinstance(plan.get("crossover_tasks", []), list) else []
        for idx, item in enumerate(mutation_items[:mutation_target_count]):
            if not isinstance(item, dict):
                continue
            parent_ids = [str(parent_id) for parent_id in list(item.get("parent_ids", [])) if str(parent_id) in trajectory_by_id]
            if not parent_ids:
                continue
            direction = str(item.get("new_direction", "")).strip()
            if not direction:
                continue
            direction_id = self._coerce_direction_id(item.get("direction_id"), default=100 + idx)
            trajectory_id = self._unique_trajectory_id(direction_id=direction_id, round_idx=round_idx, phase=RoundPhase.MUTATION, used_ids=used_ids)
            parents = [trajectory_by_id[parent_id] for parent_id in parent_ids]
            summaries = [self.guidance_builder.summarize_trajectory(parent) for parent in parents]
            guidance = self._compose_llm_mutation_guidance(item)
            target_segment = self._normalize_target_segment(str(item.get("target_segment", "")), mutation_type=str(item.get("mutation_type", "")))
            tasks.append(
                EvolutionTask(
                    phase=RoundPhase.MUTATION,
                    round_idx=round_idx,
                    direction_id=direction_id,
                    direction=direction,
                    trajectory_id=trajectory_id,
                    parent_ids=parent_ids,
                    evolution_guidance=guidance,
                    parent_trajectory_summaries=summaries,
                    mutation_type=str(item.get("mutation_type", "hypothesis_mutation")),
                    planner_mode="llm",
                    rationale=str(item.get("rationale", "")),
                    target_segment=target_segment,
                    fault_diagnosis=str(item.get("fault_diagnosis", "")),
                    preserve_segments=[str(segment) for segment in list(item.get("preserve_segments", []))],
                    rewrite_instruction=str(item.get("rewrite_instruction", "")),
                    start_agent_name=self._start_agent_for_segment(target_segment),
                    frozen_prefix_payload=self._build_frozen_prefix_payload(parents[0], target_segment=target_segment),
                )
            )
        for idx, item in enumerate(crossover_items[:crossover_target_count]):
            if not isinstance(item, dict):
                continue
            parent_ids = [str(parent_id) for parent_id in list(item.get("parent_ids", [])) if str(parent_id) in trajectory_by_id]
            if len(parent_ids) < 2:
                continue
            direction = str(item.get("new_direction", "")).strip()
            if not direction:
                continue
            direction_id = self._coerce_direction_id(item.get("direction_id"), default=200 + idx)
            trajectory_id = self._unique_trajectory_id(direction_id=direction_id, round_idx=round_idx, phase=RoundPhase.CROSSOVER, used_ids=used_ids)
            parents = [trajectory_by_id[parent_id] for parent_id in parent_ids]
            summaries = [self.guidance_builder.summarize_trajectory(parent) for parent in parents]
            guidance = self._compose_llm_crossover_guidance(item)
            source_segments = [
                dict(segment)
                for segment in list(item.get("source_segments", []))
                if isinstance(segment, dict)
            ] or self._default_crossover_source_segments(parents)
            crossover_start_agent, crossover_prefix = self._build_crossover_inheritance(
                parents, source_segments, enabled=config.crossover_inherit_state
            )
            tasks.append(
                EvolutionTask(
                    phase=RoundPhase.CROSSOVER,
                    round_idx=round_idx,
                    direction_id=direction_id,
                    direction=direction,
                    trajectory_id=trajectory_id,
                    parent_ids=parent_ids,
                    evolution_guidance=guidance,
                    parent_trajectory_summaries=summaries,
                    planner_mode="llm",
                    rationale=str(item.get("rationale", "")),
                    fusion_logic=str(item.get("fusion_logic", "")),
                    source_segments=source_segments,
                    fusion_plan=str(item.get("fusion_plan", item.get("fusion_logic", ""))),
                    start_agent_name=crossover_start_agent,
                    frozen_prefix_payload=crossover_prefix,
                )
            )
        return tasks

    @staticmethod
    def _normalize_target_segment(value: str, *, mutation_type: str = "") -> str:
        text = str(value or "").strip()
        # evaluation_feedback 作为变异目标会从 feedback_summarizer_agent 起跑，
        # 复用父代的因子表达式与回测结果、仅改写反馈文本，产出与父代完全相同的因子，
        # 无法带来任何因子层面的改进。降级为 factor_expression，保证变异确实改变因子。
        if text == "evaluation_feedback":
            text = "factor_expression"
        if text in {"hypothesis_generation", "factor_expression", "implementation"}:
            return text
        if mutation_type == "hypothesis_mutation":
            return "hypothesis_generation"
        if mutation_type in {"expression_mutation", "parameter_mutation", "risk_mutation"}:
            return "factor_expression"
        return "factor_expression"

    @staticmethod
    def _start_agent_for_segment(segment: str) -> str:
        return {
            "hypothesis_generation": "hypothesis_agent",
            "factor_expression": "experiment_designer_agent",
            "implementation": "factor_coder_agent",
            "evaluation_feedback": "feedback_summarizer_agent",
        }.get(str(segment or "").strip(), "hypothesis_agent")

    @classmethod
    def _build_frozen_prefix_payload(cls, parent: StrategyTrajectory, *, target_segment: str) -> dict[str, object]:
        payload: dict[str, object] = {}
        if target_segment in {"factor_expression", "implementation", "evaluation_feedback"}:
            payload["hypothesis"] = parent.hypothesis
            if parent.hypothesis_details:
                payload["hypothesis_structured"] = dict(parent.hypothesis_details)
        if target_segment in {"implementation", "evaluation_feedback"}:
            factors = [dict(item) for item in parent.factors if isinstance(item, dict)]
            payload["experiment_spec"] = {
                "target_hypothesis": parent.hypothesis,
                "factors": factors,
                "factor_names": [
                    str(item.get("factor_name", item.get("name", ""))).strip()
                    for item in factors
                    if str(item.get("factor_name", item.get("name", ""))).strip()
                ],
                "deduplicated_against": [],
                "attempt_count": 0,
                "consistency_gate": {},
            }
        if target_segment == "evaluation_feedback":
            if isinstance(parent.backtest_result, dict):
                payload["backtest_report"] = dict(parent.backtest_result)
            payload["metrics"] = dict(parent.backtest_metrics)
            if parent.feedback_details:
                payload["feedback"] = dict(parent.feedback_details)
        return payload

    @classmethod
    def _build_crossover_inheritance(
        cls,
        parents: list[StrategyTrajectory],
        source_segments: list[dict[str, object]],
        *,
        enabled: bool,
    ) -> tuple[str, dict[str, object]]:
        """构造交叉任务的状态级继承（对齐论文 Eq.7）。

        选定一个 hypothesis 供体父代，冻结其 hypothesis 状态作为前缀，从
        experiment_designer_agent 起跑；其余父代的高分因子构造模式经 fusion
        guidance / parent_trajectory_summaries 注入 experiment_designer，由其在
        “已验证假说”之上融合出新的因子表达式。

        返回 (start_agent_name, frozen_prefix_payload)。未启用或缺乏可用供体时返回
        ("", {})，调用方据此回退到从 hypothesis_agent 全量重跑的引导式交叉。
        """
        if not enabled or not parents:
            return "", {}
        donor = cls._select_crossover_hypothesis_donor(parents, source_segments)
        if donor is None or not str(donor.hypothesis or "").strip():
            return "", {}
        frozen_prefix = cls._build_frozen_prefix_payload(donor, target_segment="factor_expression")
        if not frozen_prefix:
            return "", {}
        # 论文 Eq.7：把多个父代的高分 segment 拼接成子轨迹。
        #   - hypothesis_generation segment 来自 donor（上面已冻结其假说状态）。
        #   - factor_expression segment 来自其余父代，继承它们已验证的因子表达式作为锚点，
        #     供 experiment_designer 在 donor 假说之上融合，而非从零重写。
        factor_donors = cls._select_crossover_factor_donors(
            parents, source_segments, exclude_id=donor.trajectory_id
        )
        inherited_anchors = cls._collect_inherited_factor_anchors(factor_donors)
        frozen_prefix["crossover_inherited_factor_anchors"] = inherited_anchors
        frozen_prefix["crossover_fusion_blueprint"] = {
            "hypothesis_from": donor.trajectory_id,
            "factor_expression_from": [parent.trajectory_id for parent in factor_donors],
            "source_segments": [dict(item) for item in (source_segments or []) if isinstance(item, dict)],
        }
        return "experiment_designer_agent", frozen_prefix

    @staticmethod
    def _select_crossover_factor_donors(
        parents: list[StrategyTrajectory],
        source_segments: list[dict[str, object]],
        *,
        exclude_id: str,
    ) -> list[StrategyTrajectory]:
        """选择贡献 factor_expression segment 的父代（用于拼接其已验证因子表达式）。

        优先采用 source_segments 中显式标注 factor_expression / evaluation_feedback 段的父代；
        若没有显式标注，则退化为除 hypothesis 供体外所有持有有效因子的父代。
        """
        by_id = {parent.trajectory_id: parent for parent in parents}
        donors: list[StrategyTrajectory] = []
        seen: set[str] = set()
        for segment in source_segments or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("segment", "")).strip() not in {"factor_expression", "evaluation_feedback"}:
                continue
            parent_id = str(segment.get("parent_id", segment.get("trajectory_id", ""))).strip()
            if not parent_id or parent_id == exclude_id or parent_id in seen:
                continue
            parent = by_id.get(parent_id)
            if parent is not None and parent.factors:
                donors.append(parent)
                seen.add(parent_id)
        if not donors:
            for parent in parents:
                if parent.trajectory_id != exclude_id and parent.factors:
                    donors.append(parent)
        return donors

    @staticmethod
    def _collect_inherited_factor_anchors(
        factor_donors: list[StrategyTrajectory],
    ) -> list[dict[str, object]]:
        """从 factor segment 供体父代收集其已验证的因子表达式锚点。"""
        anchors: list[dict[str, object]] = []
        seen_expressions: set[str] = set()
        for parent in factor_donors:
            for factor in parent.factors:
                if not isinstance(factor, dict):
                    continue
                expression = str(factor.get("expression", factor.get("factor_expression", ""))).strip()
                if not expression or expression in seen_expressions:
                    continue
                seen_expressions.add(expression)
                anchors.append(
                    {
                        "parent_id": parent.trajectory_id,
                        "factor_name": str(factor.get("factor_name", factor.get("name", ""))).strip(),
                        "expression": expression,
                    }
                )
        return anchors

    @staticmethod
    def _select_crossover_hypothesis_donor(
        parents: list[StrategyTrajectory],
        source_segments: list[dict[str, object]],
    ) -> StrategyTrajectory | None:
        """选择交叉的 hypothesis 供体父代。

        优先采用 source_segments 中显式标注贡献 hypothesis_generation 段的父代；
        否则退化为奖励分数最高且具备有效假说的父代。
        """
        by_id = {parent.trajectory_id: parent for parent in parents}
        for segment in source_segments or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("segment", "")).strip() != "hypothesis_generation":
                continue
            parent_id = str(segment.get("parent_id", segment.get("trajectory_id", ""))).strip()
            donor = by_id.get(parent_id)
            if donor is not None and str(donor.hypothesis or "").strip():
                return donor
        candidates = [parent for parent in parents if str(parent.hypothesis or "").strip()]
        if not candidates:
            return None
        return max(candidates, key=lambda parent: parent.get_reward_score())

    @staticmethod
    def _default_crossover_source_segments(parents: list[StrategyTrajectory]) -> list[dict[str, object]]:
        defaults: list[dict[str, object]] = []
        segment_cycle = ["hypothesis_generation", "factor_expression", "evaluation_feedback"]
        for idx, parent in enumerate(parents):
            defaults.append(
                {
                    "parent_id": parent.trajectory_id,
                    "segment": segment_cycle[idx % len(segment_cycle)],
                    "inherit": "highest-reward reusable trajectory segment",
                }
            )
        return defaults

    @staticmethod
    def _coerce_direction_id(value: object, *, default: int) -> int:
        """安全地将 direction_id 转为整数，失败时使用默认值。"""
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _unique_trajectory_id(*, direction_id: int, round_idx: int, phase: RoundPhase, used_ids: set[str]) -> str:
        """生成唯一的 trajectory_id，如果冲突则递增 direction_id 重试。"""
        current_direction_id = int(direction_id)
        while True:
            trajectory_id = StrategyTrajectory.generate_id(direction_id=current_direction_id, round_idx=round_idx, phase=phase)
            if trajectory_id not in used_ids:
                used_ids.add(trajectory_id)
                return trajectory_id
            current_direction_id += 1

    @staticmethod
    def _compose_llm_mutation_guidance(item: dict[str, object]) -> str:
        """将 EvolutionAgent 的变异规划项转换为引导文本（变异类型、理由、保留项、变更项）。"""
        parts = [
            "Evolution phase: mutation planned by EvolutionAgent.",
            f"Mutation type: {str(item.get('mutation_type', '')).strip()}",
            f"Target segment: {str(item.get('target_segment', '')).strip()}",
            f"Fault diagnosis: {str(item.get('fault_diagnosis', '')).strip()}",
            f"Rationale: {str(item.get('rationale', '')).strip()}",
            f"Preserve: {str(item.get('preserve', '')).strip()}",
            f"Preserve segments: {', '.join(str(segment) for segment in list(item.get('preserve_segments', [])) if str(segment).strip())}",
            f"Change: {str(item.get('change', '')).strip()}",
            f"Rewrite instruction: {str(item.get('rewrite_instruction', '')).strip()}",
            "The child direction was created by LLM and should be translated into a simple, testable factor hypothesis.",
        ]
        return "\n".join(part for part in parts if part.split(":", 1)[-1].strip())

    @staticmethod
    def _compose_llm_crossover_guidance(item: dict[str, object]) -> str:
        """将 EvolutionAgent 的交叉规划项转换为引导文本（融合逻辑、理由，强调概念融合而非机械拼接）。"""
        parts = [
            "Evolution phase: crossover planned by EvolutionAgent.",
            f"Fusion logic: {str(item.get('fusion_logic', '')).strip()}",
            f"Fusion plan: {str(item.get('fusion_plan', '')).strip()}",
            f"Rationale: {str(item.get('rationale', '')).strip()}",
            "Fuse mechanisms conceptually; do not mechanically concatenate parent expressions.",
        ]
        return "\n".join(part for part in parts if part.split(":", 1)[-1].strip())

    def _build_mutation_task(self, parent: StrategyTrajectory, *, round_idx: int, offset: int) -> EvolutionTask:
        """基于单条父轨迹生成规则 fallback 的变异任务。

        调用 EvolutionGuidanceBuilder.build_mutation_guidance 推断变异类型并生成引导信息。
        标记 planner_mode="fallback_rule"，表示这不是由 LLM 规划的。
        """
        mutation_type, guidance, summaries = self.guidance_builder.build_mutation_guidance(parent)
        target_segment = self._normalize_target_segment("", mutation_type=mutation_type)
        return EvolutionTask(
            phase=RoundPhase.MUTATION,
            round_idx=round_idx,
            direction_id=parent.direction_id,
            direction=f"Mutate trajectory {parent.trajectory_id}: {parent.hypothesis}".strip(),
            trajectory_id=StrategyTrajectory.generate_id(
                direction_id=parent.direction_id * 100 + offset,
                round_idx=round_idx,
                phase=RoundPhase.MUTATION,
            ),
            parent_ids=[parent.trajectory_id],
            evolution_guidance=guidance,
            parent_trajectory_summaries=summaries,
            mutation_type=mutation_type,
            planner_mode="fallback_rule",
            rationale="Fallback rule mutation generated because EvolutionAgent did not provide enough valid mutation tasks.",
            target_segment=target_segment,
            fault_diagnosis="Fallback diagnosis inferred from parent metrics and feedback.",
            preserve_segments=["hypothesis_generation"] if target_segment != "hypothesis_generation" else [],
            rewrite_instruction="Apply the inferred mutation locally while preserving useful parent context.",
            start_agent_name=self._start_agent_for_segment(target_segment),
            frozen_prefix_payload=self._build_frozen_prefix_payload(parent, target_segment=target_segment),
        )

    def _build_crossover_task(
        self,
        parents: list[StrategyTrajectory],
        *,
        round_idx: int,
        offset: int,
        config: EvolutionConfig,
    ) -> EvolutionTask:
        """基于一组不重复父轨迹生成规则 fallback 的交叉任务。

        调用 EvolutionGuidanceBuilder.build_crossover_guidance 生成融合引导信息。
        标记 planner_mode="fallback_rule"。
        """
        guidance, summaries = self.guidance_builder.build_crossover_guidance(parents)
        direction_ids = [parent.direction_id for parent in parents]
        direction_id = min(direction_ids) * 100 + offset if direction_ids else offset
        parent_ids = [parent.trajectory_id for parent in parents]
        source_segments = self._default_crossover_source_segments(parents)
        crossover_start_agent, crossover_prefix = self._build_crossover_inheritance(
            parents, source_segments, enabled=config.crossover_inherit_state
        )
        return EvolutionTask(
            phase=RoundPhase.CROSSOVER,
            round_idx=round_idx,
            direction_id=direction_id,
            direction=f"Crossover trajectories: {', '.join(parent_ids)}",
            trajectory_id=StrategyTrajectory.generate_id(
                direction_id=direction_id,
                round_idx=round_idx,
                phase=RoundPhase.CROSSOVER,
            ),
            parent_ids=parent_ids,
            evolution_guidance=guidance,
            parent_trajectory_summaries=summaries,
            planner_mode="fallback_rule",
            fusion_logic="Fallback rule crossover generated because EvolutionAgent did not provide enough valid crossover tasks.",
            source_segments=source_segments,
            fusion_plan="Fallback fusion: inherit the strongest hypothesis and factor-expression anchors from selected parents.",
            start_agent_name=crossover_start_agent,
            frozen_prefix_payload=crossover_prefix,
        )

    @staticmethod
    def _phase_label(round_idx: int, *, config: EvolutionConfig | None = None) -> str:
        """返回轮次级标签：第 0 轮为 "original"，第 1+ 轮为 "evolved"。"""
        if config is not None:
            return config.phase_for_round(round_idx)
        return RoundPhase.ORIGINAL.value if round_idx == 0 else "evolved"

    @staticmethod
    def _diversified_directions(*, initial_direction: str | None, count: int) -> list[str]:
        """生成确定性的多样化备用方向。

        当 LLM 拆解失败或产出不足时使用。内置 6 个预设主题：
        价格动量反转、量价确认背离、波动率适配、流动性换手率、短期噪声过滤、长期趋势稳定性。
        """
        base = str(initial_direction or "").strip()
        fallback_themes = [
            "price momentum and reversal",
            "volume-price confirmation and divergence",
            "volatility regime adaptation",
            "liquidity and turnover control",
            "short-horizon noise filtering",
            "long-horizon trend stability",
            "funding and open-interest pressure",
            "intraday range expansion and compression",
            "cross-sectional crowding and de-crowding",
            "regime-conditioned risk control",
        ]
        directions: list[str] = []
        if base:
            directions.append(base)
        for theme in fallback_themes:
            if len(directions) >= count:
                break
            direction = f"{base}; explore {theme}" if base else f"Explore {theme} for cross-sectional alpha"
            if direction not in directions:
                directions.append(direction)
        while len(directions) < count:
            directions.append(f"{base or 'Explore cross-sectional alpha'}; diversified branch {len(directions)}")
        return directions[:count]

    @staticmethod
    def _task_to_shared_payload(*, base_payload: dict[str, object], task: EvolutionTask) -> dict[str, object]:
        """将遗传任务转换为单轮 5-Agent 循路使用的全新 SharedPayload。

        关键操作：
        1. 复制 base_payload 作为基础
        2. 清除上一轮的残留数据（hypothesis、experiment_spec、backtest_report 等）
        3. 注入遗传元数据（phase、direction_id、parent_ids、evolution_guidance 等）
        """
        payload = dict(base_payload)
        for stale_key in (
            "hypothesis",
            "hypothesis_details",
            "hypothesis_structured",
            "hypothesis_reasoning",
            "experiment_spec",
            "task_plan",
            "factor_implementation",
            "calculation_report",
            "backtest_report",
            "metrics",
            "per_factor_metrics",
            "feedback",
            "qlib_factor_experiment",
        ):
            payload.pop(stale_key, None)
        payload.update(dict(task.frozen_prefix_payload))
        payload.update(
            {
                "phase": task.phase.value,
                "round_idx": int(task.round_idx),
                "direction_id": int(task.direction_id),
                "direction": task.direction,
                "trajectory_id": task.trajectory_id,
                "parent_ids": list(task.parent_ids),
                "evolution_guidance": task.evolution_guidance,
                "parent_trajectory_summaries": list(task.parent_trajectory_summaries),
                "mutation_type": task.mutation_type,
                "evolution_planner_mode": task.planner_mode,
                "evolution_seed": task.evolution_seed,
                "evolution_plan_summaries": list(task.evolution_plan_summaries),
                "target_segment": task.target_segment,
                "fault_diagnosis": task.fault_diagnosis,
                "preserve_segments": list(task.preserve_segments),
                "rewrite_instruction": task.rewrite_instruction,
                "source_segments": list(task.source_segments),
                "fusion_plan": task.fusion_plan,
                "partial_replay_start_agent": task.start_agent_name,
            }
        )
        return payload

    @staticmethod
    def _record_trajectory(session: LoopSession, payload: dict[str, Any], task: EvolutionTask) -> None:
        """将已完成任务的执行结果写入当前会话的轨迹池。

        从 SharedPayload 中提取实验结果（假说、因子、回测指标、反馈），
        构造 StrategyTrajectory 对象并添加到 TrajectoryPool。
        如果 trajectory_id 已存在则跳过（防止重复记录）。
        """
        experiment = QlibFactorExperiment.from_shared_payload(payload)
        trajectory_id = experiment.trajectory_id or task.trajectory_id
        if not trajectory_id:
            return
        phase_text = str(experiment.phase or task.phase.value)
        if phase_text not in {item.value for item in RoundPhase}:
            phase_text = task.phase.value
        metric_payload: dict[str, float | None] = {}
        for key, value in experiment.metrics.items():
            try:
                metric_payload[str(key)] = float(value)
            except Exception:
                metric_payload[str(key)] = None
        trajectory = StrategyTrajectory(
            trajectory_id=trajectory_id,
            direction_id=int(task.direction_id),
            round_idx=int(experiment.round_idx or task.round_idx),
            phase=RoundPhase(phase_text),
            hypothesis=experiment.target_hypothesis,
            hypothesis_details=dict(payload.get("hypothesis_structured", {}))
            if isinstance(payload.get("hypothesis_structured", {}), dict)
            else {},
            factors=[item.to_dict() for item in experiment.factors],
            backtest_result=experiment.backtest_report,
            backtest_metrics=metric_payload,
            feedback=str(experiment.feedback.get("summary", "")),
            feedback_details=dict(experiment.feedback),
            parent_ids=list(experiment.parent_ids or task.parent_ids),
            extra_info={
                "experiment_id": experiment.experiment_id,
                "direction": task.direction,
                "mutation_type": task.mutation_type,
                "evolution_planner_mode": task.planner_mode,
                "direction_title": task.direction_title,
                "rationale": task.rationale,
                "fusion_logic": task.fusion_logic,
                "evolution_seed": task.evolution_seed,
                "evolution_plan_summaries": list(task.evolution_plan_summaries),
                "target_segment": task.target_segment,
                "fault_diagnosis": task.fault_diagnosis,
                "preserve_segments": list(task.preserve_segments),
                "rewrite_instruction": task.rewrite_instruction,
                "source_segments": list(task.source_segments),
                "fusion_plan": task.fusion_plan,
                "partial_replay_start_agent": task.start_agent_name,
                "segments": EvolutionRunner._build_trajectory_segments(payload, task),
            },
        )
        if session.trajectory_pool.get_by_id(trajectory.trajectory_id) is None:
            session.trajectory_pool.add(trajectory)

    @staticmethod
    def _build_trajectory_segments(payload: dict[str, Any], task: EvolutionTask) -> dict[str, dict[str, object]]:
        hypothesis_structured = dict(payload.get("hypothesis_structured", {})) if isinstance(payload.get("hypothesis_structured", {}), dict) else {}
        experiment_spec = dict(payload.get("experiment_spec", {})) if isinstance(payload.get("experiment_spec", {}), dict) else {}
        factors = [dict(item) for item in list(experiment_spec.get("factors", [])) if isinstance(item, dict)]
        factor_implementation = dict(payload.get("factor_implementation", {})) if isinstance(payload.get("factor_implementation", {}), dict) else {}
        calculation_report = dict(payload.get("calculation_report", {})) if isinstance(payload.get("calculation_report", {}), dict) else {}
        backtest_report = dict(payload.get("backtest_report", {})) if isinstance(payload.get("backtest_report", {}), dict) else {}
        metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics", {}), dict) else {}
        per_factor_metrics = dict(payload.get("per_factor_metrics", {})) if isinstance(payload.get("per_factor_metrics", {}), dict) else {}
        feedback = dict(payload.get("feedback", {})) if isinstance(payload.get("feedback", {}), dict) else {}
        expressions = [
            str(item.get("expression", item.get("factor_expression", ""))).strip()
            for item in factors
            if str(item.get("expression", item.get("factor_expression", ""))).strip()
        ]
        anchor_features = []
        anchors = hypothesis_structured.get("hypothesis_evolution_state", {})
        if isinstance(anchors, dict):
            verifiable = anchors.get("verifiable_anchors", {})
            if isinstance(verifiable, dict):
                anchor_features = [str(item) for item in list(verifiable.get("input_features", [])) if str(item).strip()]
        return {
            "hypothesis_generation": {
                "summary": str(payload.get("hypothesis", "")),
                "hypothesis_structured": hypothesis_structured,
                "reusable_anchors": anchor_features,
                "failure_label": str(feedback.get("semantic_diagnosis", feedback.get("Semantic Diagnosis", ""))),
            },
            "factor_expression": {
                "summary": f"{len(factors)} candidate factors",
                "factors": factors,
                "expressions": expressions,
                "reusable_anchors": expressions[:3],
                "failure_label": str(experiment_spec.get("consistency_gate", {}).get("status", ""))
                if isinstance(experiment_spec.get("consistency_gate", {}), dict)
                else "",
            },
            "implementation": {
                "summary": str(calculation_report.get("summary", "")) or str(factor_implementation.get("source", "")),
                "factor_implementation": factor_implementation,
                "calculation_report": calculation_report,
                "failure_label": ",".join(str(item) for item in list(calculation_report.get("failed_factors", [])))
                if isinstance(calculation_report.get("failed_factors", []), list)
                else "",
            },
            "evaluation_feedback": {
                "summary": str(feedback.get("summary", "")) or str(backtest_report.get("summary", "")),
                "metrics": metrics,
                "per_factor_metrics": per_factor_metrics,
                "backtest_report": backtest_report,
                "selected_factor_name": str(backtest_report.get("selected_factor_name", "")),
                "failure_label": str(backtest_report.get("status", "")),
            },
            "task_metadata": {
                "phase": task.phase.value,
                "target_segment": task.target_segment,
                "source_segments": list(task.source_segments),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 并行调度：核心方法
    # ─────────────────────────────────────────────────────────────────────────

    def _run_tasks_for_round(
        self,
        *,
        tasks: list[EvolutionTask],
        base_payload: dict[str, object],
        start_loop_index: int,
        config: EvolutionConfig,
    ) -> list[_TaskResult]:
        """根据 config 决定串行或并行执行本轮所有任务，结果按原始 tasks 顺序返回。"""
        if config.parallel_enabled and len(tasks) > 1:
            return self._run_tasks_parallel(
                tasks=tasks,
                base_payload=base_payload,
                start_loop_index=start_loop_index,
                max_workers=min(max(1, int(config.max_workers)), len(tasks)),
            )
        return self._run_tasks_serial(
            tasks=tasks,
            base_payload=base_payload,
            start_loop_index=start_loop_index,
        )

    def _run_tasks_serial(
        self,
        *,
        tasks: list[EvolutionTask],
        base_payload: dict[str, object],
        start_loop_index: int,
    ) -> list[_TaskResult]:
        """逐个执行任务，保留原有串行语义（parallel_enabled=False 时使用）。"""
        results: list[_TaskResult] = []
        for i, task in enumerate(tasks):
            loop_index = start_loop_index + i
            self._notify_task_progress("evolution_task_started", task, loop_index, datetime.now())
            result = self._run_single_task_isolated(
                task=task,
                base_payload=base_payload,
                loop_index=loop_index,
                safe_callback=self.progress_callback,
            )
            results.append(result)
        return results

    def _run_tasks_parallel(
        self,
        *,
        tasks: list[EvolutionTask],
        base_payload: dict[str, object],
        start_loop_index: int,
        max_workers: int,
    ) -> list[_TaskResult]:
        """用线程池并行执行本轮所有任务，结果按原始 tasks 顺序返回。

        关键设计：
        - 每个 worker 持有独立的 orchestrator / context_store / trace_store
        - 所有 progress callback 经过锁序列化，保证日志文件不交错
        - 每个 task 的 callback 自动注入 trajectory_id 等标识字段，便于日志追溯
        - worker 内部捕获所有异常并转换为 error_count != 0 的 _TaskResult，不抛出
        """
        _lock = threading.Lock()

        def _locked_callback(payload: dict[str, object]) -> None:
            if self.progress_callback is None:
                return
            with _lock:
                try:
                    self.progress_callback(payload)
                except Exception:
                    pass

        result_slots: list[_TaskResult | None] = [None] * len(tasks)
        future_to_idx: dict[Future[_TaskResult], int] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, task in enumerate(tasks):
                loop_index = start_loop_index + i
                # 在主线程通知 task 启动（无需锁）
                self._notify_task_progress("evolution_task_started", task, loop_index, datetime.now())
                future: Future[_TaskResult] = executor.submit(
                    self._run_single_task_isolated,
                    task=task,
                    base_payload=base_payload,
                    loop_index=loop_index,
                    safe_callback=self._make_tagged_callback(task, _locked_callback),
                )
                future_to_idx[future] = i
            # executor.__exit__ 阻塞直到所有 future 完成

        for future, idx in future_to_idx.items():
            result_slots[idx] = future.result()  # worker 内部已捕获异常，不会抛出

        return [r for r in result_slots if r is not None]

    def _run_single_task_isolated(
        self,
        *,
        task: EvolutionTask,
        base_payload: dict[str, object],
        loop_index: int,
        safe_callback: Callable[[dict[str, object]], None] | None,
    ) -> _TaskResult:
        """在完全隔离的环境中执行单个 EvolutionTask，内部异常被捕获并转为错误结果。

        隔离：每次调用创建独立的 InMemoryContextStore / InMemoryTraceStore /
        SequentialLoopOrchestrator，与其他并行任务无共享可变状态。
        private_snapshot 从 task-local context_store 提取，而非主 orchestrator 的 store。
        """
        task_payload = self._task_to_shared_payload(base_payload=base_payload, task=task)
        started_at = datetime.now()
        try:
            task_context_store = InMemoryContextStore()
            task_trace_store = InMemoryTraceStore()
            task_orchestrator = self._make_isolated_orchestrator(
                context_store=task_context_store,
                trace_store=task_trace_store,
                progress_callback=safe_callback,
            )
            start_agent = str(task.start_agent_name or "").strip()
            if start_agent and start_agent != "hypothesis_agent":
                trace = task_orchestrator.run_from_agent(start_agent, SharedContext(payload=dict(task_payload)))
            else:
                trace = task_orchestrator.run_one_loop(SharedContext(payload=dict(task_payload)))
            ended_at = datetime.now()
            errors = LoopRunner._extract_errors(trace)
            private_snapshot = LoopRunner._extract_private_snapshot(task_context_store, trace)
        except Exception as exc:
            ended_at = datetime.now()
            trace = LoopTrace()
            errors = [str(exc)]
            private_snapshot = {}

        round_state = RoundState(
            loop_index=loop_index,
            attempt_index=0,
            phase=task.phase.value,
            trajectory_id=task.trajectory_id,
            parent_ids=list(task.parent_ids),
            input_shared_keys=sorted(task_payload.keys()),
            output_shared_keys=sorted(trace.final_shared_context.keys()),
            agent_count=len(trace.records),
            error_count=len(errors),
            errors=errors,
            started_at=started_at,
            ended_at=ended_at,
        )
        return _TaskResult(task=task, trace=trace, round_state=round_state, private_snapshot=private_snapshot)

    def _make_isolated_orchestrator(
        self,
        *,
        context_store: InMemoryContextStore,
        trace_store: InMemoryTraceStore,
        progress_callback: Callable[[dict[str, object]], None] | None,
    ) -> SequentialLoopOrchestrator:
        """为单个并行任务创建独立 orchestrator，复用 agents 和 policy_registry（均无状态）。

        若主 orchestrator 不是 SequentialLoopOrchestrator，则抛出 TypeError，
        调用方 _run_tasks_for_round 会因 parallel_enabled 条件而走串行路径。
        """
        if not isinstance(self.orchestrator, SequentialLoopOrchestrator):
            raise TypeError(
                "并行执行要求 orchestrator 为 SequentialLoopOrchestrator，"
                "请将 parallel_enabled 设为 False 或改用 SequentialLoopOrchestrator。"
            )
        return SequentialLoopOrchestrator(
            agents=self.orchestrator.agents,
            context_store=context_store,
            model_client=self.model_client,
            policy_registry=self.orchestrator.policy_registry,
            trace_store=trace_store,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _make_tagged_callback(
        task: EvolutionTask,
        locked_callback: Callable[[dict[str, object]], None],
    ) -> Callable[[dict[str, object]], None]:
        """将 task 的身份字段（trajectory_id / direction_id / round_idx / phase）
        注入到每条 progress 事件中，使并行日志可按 task 过滤追溯。
        """
        def _tagged(payload: dict[str, object]) -> None:
            tagged = dict(payload)
            tagged.setdefault("trajectory_id", task.trajectory_id)
            tagged.setdefault("direction_id", task.direction_id)
            tagged.setdefault("round_idx", task.round_idx)
            tagged.setdefault("phase", task.phase.value)
            locked_callback(tagged)
        return _tagged
