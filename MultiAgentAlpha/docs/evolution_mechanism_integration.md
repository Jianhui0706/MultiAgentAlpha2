# QuantaAlpha 进化机制集成方案

本文档分析 QuantaAlpha 的突变和进化机制如何与当前 Multi-Agent Factor Mining 项目集成。

---

## 一、当前项目已有的基础设施

| 组件 | 当前状态 | QuantaAlpha 对应 |
|------|----------|------------------|
| `StrategyTrajectory` | ✅ 已有 | 策略轨迹 |
| `TrajectoryPool` | ✅ 已有 | 轨迹池 |
| `RoundPhase` | ✅ 已有（ORIGINAL, MUTATION, CROSSOVER） | 进化阶段 |
| `LoopRunner` | ✅ 已有 | 进化控制器 |
| `hypothesis_agent` | ✅ 已有 | 可改造为突变操作器 |

---

## 二、QuantaAlpha 进化机制概述

### 2.1 核心组件

#### 突变操作器 (Mutation Operator)
- **功能**：从父轨迹生成正交（突变）策略
- **实现原理**：
  - 正交性：生成与父策略几乎独立的新假设
  - 多样性：避免重复探索路径
  - 学习机制：利用父策略的反馈避免已知问题

#### 进化控制器 (Evolution Controller)
- **功能**：管理完整的进化周期
- **进化流程**：
  1. 原始阶段：在每个方向进行初始探索
  2. 突变阶段：从每个原始轨迹生成正交策略
  3. 交叉阶段：跨方向组合轨迹
  4. 重复：突变 → 交叉 → 突变 → 交叉 → ...

### 2.2 进化流程示例

```
轮次 0 (原始阶段):
  - 方向 0: 生成初始策略 A
  - 方向 1: 生成初始策略 B

轮次 1 (突变阶段):
  - 从策略 A 生成突变策略 A1
  - 从策略 B 生成突变策略 B1

轮次 2 (交叉阶段):
  - 策略 A + B → 交叉策略 AB
  - 策略 A1 + B1 → 交叉策略 A1B1

轮次 3 (突变阶段):
  - 从策略 AB 生成突变策略 AB1
  - 从策略 A1B1 生成突变策略 A1B1-1

轮次 4 (交叉阶段):
  - 策略 AB1 + A1B1-1 → 交叉策略 AB1-A1B1-1
  ...
```

---

## 三、当前项目与 QuantaAlpha 的对比

### 3.1 当前项目的进化流程（简化版）

```
当前流程：
Loop 1 → Loop 2 → Loop 3 → ... → Loop N
   ↓        ↓        ↓
假设演进   假设演进   假设演进
（线性）  （线性）   （线性）
```

### 3.2 QuantaAlpha 的进化流程

```
QuantaAlpha 流程：
Round 0 (ORIGINAL):
  Direction 0 → 策略 A
  Direction 1 → 策略 B
  
Round 1 (MUTATION):
  A → A' (突变)
  B → B' (突变)
  
Round 2 (CROSSOVER):
  A + B → AB (交叉)
  A' + B' → A'B' (交叉)
  
Round 3 (MUTATION):
  AB → AB' (突变)
  A'B' → A'B'' (突变)
...
```

### 3.3 关键差异

| 方面 | 当前项目 | QuantaAlpha |
|------|----------|-------------|
| **探索方向** | 单一方向（INITIAL_DIRECTION） | 多方向并行 |
| **进化阶段** | 无阶段区分 | ORIGINAL → MUTATION → CROSSOVER 循环 |
| **策略生成** | 线性演进（基于 feedback） | 突变（正交）+ 交叉（组合） |
| **父策略选择** | 仅看上一轮 | 多轮历史、多种选择策略 |
| **轨迹管理** | 简单追加 | 有向无环图（DAG）谱系 |

---

## 四、集成方案

### 4.1 需要新增的组件

```
新增架构：
┌─────────────────────────────────────────────────────────────┐
│                    EvolutionController                       │
│  ├── MutationOperator (突变操作器)                          │
│  │   └── 基于 hypothesis_agent 改造                         │
│  ├── CrossoverOperator (交叉操作器)                         │
│  │   └── 新增 Agent                                         │
│  └── SelectionStrategy (选择策略)                           │
│      ├── best / random / weighted / top_percent_plus_random │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 改造现有组件

#### (1) 改造 `LoopRunner`

```python
# 当前：线性执行
for loop_index in range(1, total_loops + 1):
    trace = self.orchestrator.run_one_loop(...)

# 改造后：阶段化执行
round_idx = 0
while round_idx < max_rounds:
    # Phase 1: ORIGINAL (仅第一轮)
    if round_idx == 0:
        for direction_id in range(num_directions):
            run_original_loop(direction_id)
    
    # Phase 2: MUTATION
    if mutation_enabled:
        for parent in select_mutation_targets():
            run_mutation_loop(parent)
    
    # Phase 3: CROSSOVER
    if crossover_enabled:
        for parents in select_crossover_targets():
            run_crossover_loop(parents)
    
    round_idx += 1
```

#### (2) 改造 `hypothesis_agent`

```python
# 当前：基于 feedback 演进
# 改造后：支持突变模式

class HypothesisAgent:
    def run(self, ..., phase: RoundPhase, parent_trajectories: list = None):
        if phase == RoundPhase.ORIGINAL:
            # 原始假设生成
            return self._generate_original_hypothesis(...)
        elif phase == RoundPhase.MUTATION:
            # 突变：生成正交假设
            return self._generate_mutation_hypothesis(parent_trajectories[0])
        elif phase == RoundPhase.CROSSOVER:
            # 交叉：组合多个父假设
            return self._generate_crossover_hypothesis(parent_trajectories)
```

---

## 五、QuantaAlpha 的不足与改进建议

### 5.1 不足之处

| 问题 | 说明 | 改进建议 |
|------|------|----------|
| **突变正交性验证不足** | 仅靠 LLM prompt 保证正交，无量化验证 | 添加假设相似度计算，确保正交性 |
| **交叉盲目性** | 简单组合父假设，可能产生无效策略 | 添加因子兼容性检查 |
| **选择策略简单** | 仅基于 IC 排序 | 多目标优化（IC + 稳定性 + 复杂度） |
| **无失败学习** | 失败轨迹被丢弃 | 从失败中学习，避免重复错误 |
| **无知识积累** | 每次进化从头开始 | 添加知识蒸馏，跨轮次积累经验 |

### 5.2 改进方案

#### 改进 1：量化正交性验证

```python
def verify_orthogonality(parent: StrategyTrajectory, child: StrategyTrajectory) -> float:
    """验证突变策略与父策略的正交性"""
    # 1. 假设文本相似度
    hypothesis_sim = cosine_similarity(
        embed(parent.hypothesis), 
        embed(child.hypothesis)
    )
    
    # 2. 因子表达式结构相似度
    factor_sim = structural_similarity(
        parent.factors, 
        child.factors
    )
    
    # 3. 特征使用相似度
    feature_overlap = jaccard_similarity(
        extract_features(parent.factors),
        extract_features(child.factors)
    )
    
    # 综合正交性得分（越低越好）
    orthogonality_score = 1 - (0.4 * hypothesis_sim + 0.3 * factor_sim + 0.3 * feature_overlap)
    return orthogonality_score
```

#### 改进 2：智能交叉策略

```python
def intelligent_crossover(parents: list[StrategyTrajectory]) -> dict:
    """智能交叉：基于因子兼容性组合"""
    # 1. 分析每个父策略的优势
    strengths = analyze_strengths(parents)
    # parent_A: 擅长捕捉动量
    # parent_B: 擅长过滤噪音
    
    # 2. 设计互补组合
    crossover_plan = design_complementary_combination(strengths)
    # 组合策略：用 A 的动量信号 + B 的过滤机制
    
    # 3. 生成交叉假设
    return generate_crossover_hypothesis(crossover_plan)
```

#### 改进 3：多目标选择

```python
def multi_objective_selection(
    trajectories: list[StrategyTrajectory],
    weights: dict = {"ic": 0.4, "stability": 0.3, "simplicity": 0.3}
) -> list[StrategyTrajectory]:
    """多目标优化选择"""
    scores = []
    for t in trajectories:
        ic_score = normalize(t.metrics["IC"])
        stability_score = normalize(t.metrics["ICIR"])
        simplicity_score = 1 - normalize(t.factor_complexity)
        
        total = (
            weights["ic"] * ic_score +
            weights["stability"] * stability_score +
            weights["simplicity"] * simplicity_score
        )
        scores.append((t, total))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

#### 改进 4：失败学习机制

```python
class FailureMemory:
    """失败记忆：避免重复错误"""
    
    def record_failure(self, trajectory: StrategyTrajectory, reason: str):
        self.failures.append({
            "hypothesis": trajectory.hypothesis,
            "factors": trajectory.factors,
            "reason": reason,
            "timestamp": datetime.now()
        })
    
    def check_similarity(self, new_hypothesis: str) -> list[str]:
        """检查新假设是否与历史失败相似"""
        warnings = []
        for failure in self.failures:
            sim = cosine_similarity(embed(new_hypothesis), embed(failure["hypothesis"]))
            if sim > 0.7:
                warnings.append(f"相似假设曾失败：{failure['reason']}")
        return warnings
```

#### 改进 5：知识蒸馏集成

```python
def distill_evolution_knowledge(trajectory_pool: TrajectoryPool) -> dict:
    """从进化历史中蒸馏知识"""
    knowledge = {
        "successful_patterns": [],  # 成功模式
        "failure_patterns": [],     # 失败模式
        "mutation_effectiveness": {},  # 突变效果统计
        "crossover_effectiveness": {}, # 交叉效果统计
    }
    
    # 分析成功轨迹的共同特征
    successful = [t for t in trajectory_pool.get_all() if t.metrics["IC"] > 0.05]
    knowledge["successful_patterns"] = extract_common_patterns(successful)
    
    # 分析突变效果
    mutations = [t for t in trajectory_pool.get_all() if t.phase == RoundPhase.MUTATION]
    knowledge["mutation_effectiveness"] = analyze_mutation_impact(mutations)
    
    return knowledge
```

---

## 六、集成可行性评估

### 6.1 高可行性（改动小）

| 功能 | 改动范围 | 工作量 |
|------|----------|--------|
| 多方向并行探索 | `LoopRunner` + 配置 | 2-3 天 |
| 阶段化执行 | `LoopRunner` | 1-2 天 |
| 突变操作器 | `hypothesis_agent` prompt | 1 天 |
| 选择策略 | 新增 `SelectionStrategy` 类 | 1 天 |

### 6.2 中等可行性（需要新增组件）

| 功能 | 改动范围 | 工作量 |
|------|----------|--------|
| 交叉操作器 | 新增 `crossover_agent` | 2-3 天 |
| 正交性验证 | 新增验证模块 | 2 天 |
| 多目标选择 | 改造选择逻辑 | 1-2 天 |
| 轨迹谱系管理 | `TrajectoryPool` 增强 | 2 天 |

### 6.3 需要深入设计

| 功能 | 复杂度 | 工作量 |
|------|--------|--------|
| 失败学习机制 | 需要持久化 + 检索 | 3-5 天 |
| 知识蒸馏集成 | 需要向量存储 | 3-5 天 |
| 智能交叉策略 | 需要因子分析能力 | 5-7 天 |

---

## 七、建议的实施路径

```
Phase 1: 基础进化框架（1周）
├── 多方向并行探索
├── 阶段化执行（ORIGINAL → MUTATION → CROSSOVER）
└── 基础突变操作器

Phase 2: 选择与验证（1周）
├── 多目标选择策略
├── 正交性验证
└── 交叉操作器

Phase 3: 智能化增强（1-2周）
├── 失败学习机制
├── 知识蒸馏集成
└── 智能交叉策略
```

---

## 八、关键问题讨论

### 8.1 是否需要多方向并行？

- **当前项目**：单方向线性演进
- **多方向优势**：可以探索更广的策略空间
- **多方向劣势**：会增加 API 成本
- **建议**：可配置，默认单方向，可选多方向

### 8.2 突变 vs 当前假设演进的区别？

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| 当前演进 | 基于 feedback 渐进式改进 | 微调、优化 |
| 突变 | 强制正交，探索新方向 | 跳出局部最优 |
| **建议** | 结合使用 | feedback 用于微调，突变用于跳转 |

### 8.3 交叉是否适合因子表达式？

- **问题**：因子表达式是代码，交叉可能产生语法错误
- **建议**：在假设层面交叉，而非表达式层面
- **替代方案**：交叉时重新生成表达式，仅继承假设思想

### 8.4 如何平衡探索与利用？

| 策略 | 探索比例 | 利用比例 |
|------|----------|----------|
| 当前项目 | 低 | 高（基于 feedback 改进） |
| 进化机制 | 高（突变 + 交叉） | 低 |
| **建议** | ε-greedy | 以一定概率探索 |

---

## 九、进化配置参数

### 9.1 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `num_directions` | 并行原始轮次数量 | 1-3 |
| `max_rounds` | 最大总轮次 | 10-20 |
| `mutation_enabled` | 是否启用突变 | True |
| `crossover_enabled` | 是否启用交叉 | True |
| `crossover_size` | 每次交叉的父轨迹数量 | 2 |
| `crossover_n` | 每轮交叉组合数量 | 2-3 |
| `exploration_rate` | 探索概率（ε-greedy） | 0.2-0.3 |

### 9.2 选择策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `best` | 选择表现最好的轨迹 | 利用阶段 |
| `random` | 随机选择 | 探索阶段 |
| `weighted` | 基于性能加权选择 | 平衡探索与利用 |
| `top_percent_plus_random` | 结合 top 表现和随机选择 | 推荐默认 |

---

## 十、性能评估指标

### 10.1 评估指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| IC (Information Coefficient) | 信息系数 | 高 |
| ICIR | 信息系数信息比率 | 高 |
| RankIC | 排名信息系数 | 高 |
| RankICIR | 排名信息系数信息比率 | 高 |
| 年化收益率 | 年化收益 | 中 |
| 信息比率 | 风险调整后收益 | 中 |
| 最大回撤 | 最大损失 | 中 |
| 因子复杂度 | 表达式复杂度 | 低 |

### 10.2 选择机制

- 基于主指标排序
- 选择表现最好的轨迹进行进一步进化
- 支持多目标优化

---

## 十一、技术亮点

1. **LLM 驱动的突变**：利用大语言模型生成创新的正交策略
2. **正交性保证**：确保新策略探索不同的市场维度
3. **多阶段进化**：通过原始、突变、交叉三个阶段全面探索策略空间
4. **并行执行**：支持并行处理多个策略方向
5. **状态管理**：完整记录策略进化历史和谱系
6. **自适应选择**：基于回测结果动态调整进化方向

---

*文档生成时间：2026-04-20*
