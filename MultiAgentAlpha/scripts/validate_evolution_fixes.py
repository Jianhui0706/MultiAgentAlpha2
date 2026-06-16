"""独立验证脚本：核实进化机制与论文对齐的全部修复（无需 pytest）。

运行: python3 scripts/validate_evolution_fixes.py

验证项:
  1. evaluation_feedback 退化变异目标 -> 强制降级为 factor_expression
  2. mutation 父代选择多样性 tiebreaker -> 不再退化为纯按奖励排序
  3. crossover 状态级段继承 -> crossover_inherit_state=True 时从
     experiment_designer_agent 起跑并冻结供体假说前缀；False 时回退
  4. _build_crossover_task config 门控 start_agent / 前缀
  5. #1 父代选择以标准化 reward 为主导（get_best_trajectories / _selection_score）
  6. #4 crossover 多父代因子锚点 + 融合蓝图
  7. #5 轮次调度 mutation→crossover 交替（max_rounds=11）
  8. #6 方向拆解 normalize 容忍不足（仅 0 个时报错）
  9. #2 构造阶段冗余门（相关性过滤，默认启用）
 10. #3 最终组合回测 LGBM 优先、不可用时优雅回退 mean-rank
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading_agents.schemas.models import RoundPhase, StrategyTrajectory, TrajectoryPool
from trading_agents.workflows.evolution_runner import EvolutionConfig, EvolutionRunner

PASS = "PASS"
FAIL = "FAIL"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def make_traj(tid: str, direction_id: int, rank_ic: float, *, hypothesis: str = "", factors=None) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=tid,
        direction_id=direction_id,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis=hypothesis,
        factors=factors or [],
        backtest_metrics={"Rank IC": rank_ic, "Rank ICIR": rank_ic * 0.8, "coverage": 0.9},
    )


# ---------------------------------------------------------------------------
# 1. evaluation_feedback 退化变异目标
# ---------------------------------------------------------------------------
print("\n=== 1. _normalize_target_segment ===")
seg = EvolutionRunner._normalize_target_segment("evaluation_feedback", mutation_type="expression_mutation")
check("evaluation_feedback -> factor_expression", seg == "factor_expression", f"got {seg!r}")

seg = EvolutionRunner._normalize_target_segment("hypothesis_generation", mutation_type="hypothesis_mutation")
check("hypothesis_generation 保留", seg == "hypothesis_generation", f"got {seg!r}")

seg = EvolutionRunner._normalize_target_segment("", mutation_type="hypothesis_mutation")
check("空值按 mutation_type 回退", seg == "hypothesis_generation", f"got {seg!r}")

seg = EvolutionRunner._normalize_target_segment("factor_expression", mutation_type="")
check("factor_expression 保留", seg == "factor_expression", f"got {seg!r}")


# ---------------------------------------------------------------------------
# 2. mutation 父代选择多样性 tiebreaker
# ---------------------------------------------------------------------------
print("\n=== 2. select_parents_for_mutation ===")
# 两个高分但同方向(direction_id=1)，一个略低分但不同方向(direction_id=2)。
# 旧实现纯按奖励排序会选两个同方向；新实现应在并列附近偏向多样化方向。
pool = TrajectoryPool()
pool.trajectories = [
    make_traj("a", 1, 0.100),
    make_traj("b", 1, 0.099),
    make_traj("c", 2, 0.098),
]
selected = pool.select_parents_for_mutation(top_n=2)
ids = [t.trajectory_id for t in selected]
check("返回数量正确", len(selected) == 2, f"got {ids}")
check("首选最高奖励 a", ids[0] == "a", f"got {ids}")
check(
    "第二个偏向不同方向 c（多样性 tiebreaker 生效）",
    ids[1] == "c",
    f"got {ids}",
)

# 边界：top_n=0 返回空
check("top_n=0 -> []", pool.select_parents_for_mutation(top_n=0) == [])
# 边界：单条轨迹
single = TrajectoryPool()
single.trajectories = [make_traj("only", 1, 0.05)]
check("单轨迹 top_n=2 返回 1 条", len(single.select_parents_for_mutation(top_n=2)) == 1)


# ---------------------------------------------------------------------------
# 3. crossover 状态级段继承
# ---------------------------------------------------------------------------
print("\n=== 3. _build_crossover_inheritance ===")
p1 = make_traj("p1", 1, 0.12, hypothesis="动量在低波动期更稳定", factors=[{"factor_name": "mom20", "expression": "ts_mean(close,20)"}])
p2 = make_traj("p2", 2, 0.08, hypothesis="量价背离预示反转", factors=[{"factor_name": "vp", "expression": "corr(volume,close,10)"}])
parents = [p1, p2]
source_segments = EvolutionRunner._default_crossover_source_segments(parents)

# enabled=True -> 从 experiment_designer_agent 起跑，冻结供体假说
start_agent, prefix = EvolutionRunner._build_crossover_inheritance(parents, source_segments, enabled=True)
check("enabled=True start_agent=experiment_designer_agent", start_agent == "experiment_designer_agent", f"got {start_agent!r}")
check("冻结前缀含 hypothesis", prefix.get("hypothesis") == p1.hypothesis, f"got {prefix.get('hypothesis')!r}")
check("前缀不含 experiment_spec(仅冻结到假说)", "experiment_spec" not in prefix)

# 供体应是 source_segments 中标注 hypothesis_generation 的父代 p1
check("供体选 hypothesis_generation 标注父代 p1", prefix.get("hypothesis") == p1.hypothesis)

# enabled=False -> 回退到全量重跑
start_agent_off, prefix_off = EvolutionRunner._build_crossover_inheritance(parents, source_segments, enabled=False)
check("enabled=False -> 空 start_agent", start_agent_off == "", f"got {start_agent_off!r}")
check("enabled=False -> 空前缀", prefix_off == {})

# 无有效假说 -> 回退
no_hypo = [make_traj("x", 1, 0.1), make_traj("y", 2, 0.05)]
sa, pf = EvolutionRunner._build_crossover_inheritance(no_hypo, EvolutionRunner._default_crossover_source_segments(no_hypo), enabled=True)
check("无有效假说 -> 回退空", sa == "" and pf == {}, f"got {sa!r}, {pf}")


# ---------------------------------------------------------------------------
# 4. 端到端：_build_crossover_task 设置 start_agent / 前缀
# ---------------------------------------------------------------------------
print("\n=== 4. _build_crossover_task (config-gated) ===")
runner = EvolutionRunner.__new__(EvolutionRunner)
# guidance_builder 需要存在；用真实构造器
from trading_agents.workflows.evolution_guidance import EvolutionGuidanceBuilder
runner.guidance_builder = EvolutionGuidanceBuilder()

cfg_on = EvolutionConfig()
check("默认 crossover_inherit_state=True", cfg_on.crossover_inherit_state is True)
task_on = runner._build_crossover_task(parents, round_idx=2, offset=0, config=cfg_on)
check("on: phase=CROSSOVER", task_on.phase == RoundPhase.CROSSOVER)
check("on: start_agent=experiment_designer_agent", task_on.start_agent_name == "experiment_designer_agent", f"got {task_on.start_agent_name!r}")
check("on: frozen_prefix 含 hypothesis", task_on.frozen_prefix_payload.get("hypothesis") == p1.hypothesis)

cfg_off = EvolutionConfig(crossover_inherit_state=False)
task_off = runner._build_crossover_task(parents, round_idx=2, offset=0, config=cfg_off)
check("off: start_agent 为空(全量重跑)", (task_off.start_agent_name or "") == "", f"got {task_off.start_agent_name!r}")
check("off: frozen_prefix 为空", dict(task_off.frozen_prefix_payload) == {})


# ---------------------------------------------------------------------------
# 5. #1 父代选择改为标准化 reward 主导
# ---------------------------------------------------------------------------
print("\n=== 5. 选择以标准化 reward 为主导 (#1) ===")


def make_traj_metrics(tid: str, direction_id: int, metrics: dict) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=tid,
        direction_id=direction_id,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis="",
        factors=[],
        backtest_metrics=metrics,
    )


# high_rank_high_penalty：RankIC 较高但 drawdown/复杂度/冗余惩罚高；
# lower_rank_clean：RankIC 略低但更稳健，标准化 reward 更高。
high_rank_high_penalty = make_traj_metrics(
    "high_rank_high_penalty",
    1,
    {
        "Rank IC": 0.05,
        "max_drawdown": 0.3,
        "complexity_score": 30.0,
        "max_complexity_score": 30.0,
        "structural_similarity": 1.0,
        "max_abs_corr": 1.0,
    },
)
lower_rank_clean = make_traj_metrics("lower_rank_clean", 2, {"Rank IC": 0.04, "max_drawdown": 0.0})
check(
    "标准化 reward 会惩罚高回撤/复杂度/冗余",
    lower_rank_clean.get_reward_score() > high_rank_high_penalty.get_reward_score(),
    f"reward high_penalty={high_rank_high_penalty.get_reward_score():.4f} clean={lower_rank_clean.get_reward_score():.4f}",
)
sel_pool = TrajectoryPool()
sel_pool.trajectories = [high_rank_high_penalty, lower_rank_clean]
best = sel_pool.get_best_trajectories(top_n=2)
check("get_best_trajectories 首选标准化 reward 更高的 clean", best[0].trajectory_id == "lower_rank_clean", f"got {[t.trajectory_id for t in best]}")
check(
    "_selection_score 返回标准化 reward",
    abs(TrajectoryPool._selection_score(high_rank_high_penalty) - high_rank_high_penalty.get_reward_score()) < 1e-9,
    f"got {TrajectoryPool._selection_score(high_rank_high_penalty)}",
)


# ---------------------------------------------------------------------------
# 6. #4 crossover 多父代 segment 拼接：继承因子锚点 + 融合蓝图
# ---------------------------------------------------------------------------
print("\n=== 6. crossover 多父代因子锚点 (#4) ===")
sa6, pf6 = EvolutionRunner._build_crossover_inheritance(parents, source_segments, enabled=True)
anchors = pf6.get("crossover_inherited_factor_anchors", [])
blueprint = pf6.get("crossover_fusion_blueprint", {})
check("前缀含 crossover_inherited_factor_anchors", isinstance(anchors, list) and len(anchors) >= 1, f"got {anchors}")
check("锚点含表达式字段", all(isinstance(a, dict) and a.get("expression") for a in anchors), f"got {anchors}")
check("融合蓝图含 hypothesis_from", blueprint.get("hypothesis_from") == p1.trajectory_id, f"got {blueprint}")
check(
    "因子供体排除假说供体 p1（来自 p2）",
    p2.trajectory_id in [a.get("parent_id") for a in anchors],
    f"got {[a.get('parent_id') for a in anchors]}",
)
donors = EvolutionRunner._select_crossover_factor_donors(parents, source_segments, exclude_id=p1.trajectory_id)
check("_select_crossover_factor_donors 不含假说供体", all(d.trajectory_id != p1.trajectory_id for d in donors), f"got {[d.trajectory_id for d in donors]}")


# ---------------------------------------------------------------------------
# 7. #5 轮次调度语义对齐论文
# ---------------------------------------------------------------------------
print("\n=== 7. 轮次调度 mutation→crossover 交替 (#5) ===")
cfg5 = EvolutionConfig()
check("iterations=5", cfg5.iterations == 5)
check("max_rounds=11 (1 + 5*2)", cfg5.effective_max_rounds() == 11, f"got {cfg5.effective_max_rounds()}")
check("phase_schedule=mutation_then_crossover", cfg5.phase_schedule == "mutation_then_crossover")
check("round0=ORIGINAL", cfg5.phase_for_round(0) == RoundPhase.ORIGINAL.value)
check("round1=MUTATION", cfg5.phase_for_round(1) == RoundPhase.MUTATION.value)
check("round2=CROSSOVER", cfg5.phase_for_round(2) == RoundPhase.CROSSOVER.value)
check("round3=MUTATION", cfg5.phase_for_round(3) == RoundPhase.MUTATION.value)


# ---------------------------------------------------------------------------
# 8. #6 方向拆解放宽严格度
# ---------------------------------------------------------------------------
print("\n=== 8. 方向拆解 normalize 容忍不足 (#6) ===")
from trading_agents.agents.direction_decomposer_agent import DirectionDecomposerAgent

payload6 = {"directions": [
    {"direction": "动量延续", "title": "mom"},
    {"direction": "量价背离", "title": "vp"},
    {"direction": "动量延续", "title": "dup"},  # 与第一条重复，应去重
    {"foo": "bar"},  # 无效，应过滤
]}
dirs6 = DirectionDecomposerAgent.normalize_directions(payload6, target_count=5)
check("去重+过滤后得到 2 个有效方向（不足 target 不报错）", len(dirs6) == 2, f"got {len(dirs6)}")
check("direction_id 连续重编号", [d["direction_id"] for d in dirs6] == [0, 1], f"got {[d['direction_id'] for d in dirs6]}")
empty6 = DirectionDecomposerAgent.normalize_directions({"directions": []}, target_count=5)
check("空输入 -> 空列表（run 才抛错）", empty6 == [], f"got {empty6}")


# ---------------------------------------------------------------------------
# 9. #2 构造阶段冗余门
# ---------------------------------------------------------------------------
print("\n=== 9. 构造阶段冗余门 (#2) ===")
import yaml as _yaml
import numpy as _np
import pandas as _pd
from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine

_engine_cfg_path = ROOT / "src" / "trading_agents" / "configs" / "qlib" / "backtest_engine.yaml"
_engine_cfg = _yaml.safe_load(_engine_cfg_path.read_text(encoding="utf-8"))
_engine = QlibFactorBacktestEngine(_engine_cfg, _engine_cfg_path)
check("redundancy_filter 默认启用", _engine.redundancy_filter_enabled is True)
check("redundancy_corr_threshold=0.7", abs(_engine.redundancy_corr_threshold - 0.7) < 1e-9, f"got {_engine.redundancy_corr_threshold}")
_base = _pd.Series(_np.arange(100, dtype=float))
_corr_sig = _base + _np.random.RandomState(0).normal(0, 0.01, 100)  # 与 _base 高度相关
_indep_sig = _pd.Series(_np.random.RandomState(1).normal(0, 1, 100))  # 与 _base 几乎无关
check("高相关信号 max_abs_corr 接近 1", _engine._max_abs_signal_corr(_corr_sig, [_base]) > 0.9, f"got {_engine._max_abs_signal_corr(_corr_sig, [_base])}")
check("独立信号 max_abs_corr 低于阈值", _engine._max_abs_signal_corr(_indep_sig, [_base]) < 0.7, f"got {_engine._max_abs_signal_corr(_indep_sig, [_base])}")


# ---------------------------------------------------------------------------
# 10. #3 最终组合回测 LGBM 优先 / 优雅回退
# ---------------------------------------------------------------------------
print("\n=== 10. 组合回测 LGBM 优先 + 回退 (#3) ===")
from trading_agents.infra.factor_pool_manager import FactorPoolManager

_sig_frame = _pd.DataFrame({"f1": _np.arange(300, dtype=float), "f2": _np.random.RandomState(2).normal(0, 1, 300)})
_label = _pd.Series(_np.arange(300, dtype=float) * 0.5)
_lgb_out = FactorPoolManager._lgb_combined_signal(_sig_frame, _label)
try:
    import lightgbm  # noqa: F401
    _has_lgb = True
except Exception:
    _has_lgb = False
if _has_lgb:
    check("lightgbm 可用 -> 返回 (signal, 'factor_pool_lgbm')", _lgb_out is not None and _lgb_out[1] == "factor_pool_lgbm", f"got {_lgb_out[1] if _lgb_out else None}")
else:
    check("lightgbm 不可用 -> 优雅返回 None（由调用方回退 mean-rank）", _lgb_out is None, f"got {_lgb_out}")


# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
if _failures:
    print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
