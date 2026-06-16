# 机制锚定、Proxy 可进化方案

## 一、目标

我们希望把研报用作：

- 研究方向的来源
- 市场机制直觉的来源
- proxy 构造灵感的来源

但不希望后续 loop 被强行锁死在研报的原始细节上。

我们真正想要的行为是：

- 前期 loop：尽量忠实复现研报
- 中期 loop：保持机制不变，允许 proxy 进化
- 后期 loop：如果反复证据表明原始 proxy family 很弱，则允许更大范围探索
- 全程：避免观点切换过快

一句话总结：

- **机制要锚定**
- **proxy 要允许进化**
- **thesis 的切换必须慢，而且要有证据**

---

## 二、核心原则

当前系统把几层东西混在了一起，我们需要把它们拆开。

### 1. 机制层

这是前几轮应该尽量保持稳定的部分。

例如：

- 价量背离
- 动量衰竭
- 趋势延续但量能不确认
- 波动扩张且参与度不足

这一层回答的是：

- 系统到底在建模什么市场行为
- 为什么它应该能预测未来收益

### 2. Proxy 层

这是应该被允许更自由进化的部分。

例如：

- `$volume_change_24h` 换成 `$volume_quote`
- 原始 return ratio 换成波动率归一化后的 return spread
- 简单 divergence score 换成 z-score / percentile / regime-filtered proxy
- 价量不匹配换成 spread-assisted liquidity stress proxy

这一层回答的是：

- 在当前特征和 runtime 限制下，如何把机制落成可计算表达式

### 3. 阶段控制层

这一层控制每个 loop 允许多大自由度。

这一层回答的是：

- 什么时候应该继续复现
- 什么时候应该调参
- 什么时候应该放宽 proxy 探索
- 什么时候才应该真正修 thesis

---

## 三、理想的 Loop 语义

### Loop 1：基础复现

行为要求：

- 强研报对齐
- 强机制锚定
- proxy 选择应尽量接近研报启发下的复现
- 表达式应尽量简单

允许的变化：

- crypto 语境下的直接改写
- 简单特征替换
- 必要时做轻量归一化

不允许的变化：

- 自由跳 proxy family
- 探索完全无关的变量族
- 大幅改写 hypothesis

### Loop 2：受控的 Proxy 进化

行为要求：

- 保持同一机制
- 允许中等程度的 proxy 变化
- 允许在同一机制簇里使用多个相近 proxy family

允许的变化：

- 窗口调整
- 阈值调整
- 归一化变化
- regime filter
- 波动率归一化
- volume confirmation 重写
- spread / liquidity proxy 替换

仍然不鼓励：

- 没有强证据时从 reversal 直接跳到 continuation
- 从 divergence thesis 跳到完全不同的市场机制

### Loop 3 及以后：基于证据的扩展探索

行为要求：

- 如果机制方向仍然大致合理，但信号很弱，则继续做 proxy 进化
- 如果反复证据表明原始 proxy family 已经很弱，则允许在同一机制邻域内做更广的探索
- 只有在多轮证据都不支持时，才进入 thesis revision

允许的变化：

- 更强的 proxy-family 替换
- 替代性 proxy 组合
- 更多过滤逻辑
- 更广的机制邻域探索

这之后才考虑：

- thesis revision
- 方向反转
- 机制替换

---

## 四、什么叫“观点不要切换太快”

我们不应该用“措辞是否完全一致”或者“proxy family 是否完全一致”来判断 continuity。

更合理的做法是判断：

- 系统是否仍然停留在同一个**机制邻域**里

### 可以接受的进化

- 价量背离 -> volume-confirmed exhaustion
- 动量衰竭 -> 波动率归一化后的动量衰竭
- divergence -> spread-assisted divergence
- reversal thesis -> 加了 regime filter 的更强 reversal proxy

### 可疑的漂移

- divergence exhaustion -> 纯 cross-sectional momentum continuation
- volume-confirmation thesis -> 完全无关的 liquidity factor
- reversal thesis -> 一步跳到 trend-following thesis

### 可以允许，但必须有证据

- reversal -> continuation
- divergence -> liquidity stress
- momentum exhaustion -> spread stress

这些变化至少要满足：

- 多轮弱信号或相反证据
- 或者有明确证据说明原始 proxy family 已经失效

---

## 五、具体要改什么

## 第一步：把“严格 continuity”改成“机制 continuity”

当前问题：

- continuity 现在过于绑定上一轮 proxy family 和 anchor token
- 这会误伤健康的 proxy 进化

改法：

- 在 hypothesis state 中新增 `mechanism_anchor`
- 把以下几层分开存：
  - `mechanism_anchor`
  - `proxy_family`
  - `proxy_family_tokens`
  - `verifiable_anchors`

实现方向：

- `hypothesis_agent` 需要额外输出：
  - `mechanism_anchor.name`
  - `mechanism_anchor.summary`
  - `mechanism_anchor.expected_direction`
  - `mechanism_anchor.core_concepts`
- `continuity_snapshot` 同时记录：
  - mechanism-level tokens
  - proxy-level tokens

预期结果：

- 后续 loop 优先看机制是否连续
- proxy continuity 退到次级约束

---

## 第二步：把 continuity score 拆成机制分和 proxy 分

当前问题：

- 现在一个低 overlap 分数就容易把健康进化误判成 drift

改法：

- 以后不要只算一个 continuity score
- 要拆成两个：
  - `mechanism_continuity_score`
  - `proxy_continuity_score`

建议解释方式：

- 机制高、proxy 低：
  - 这是健康的 proxy 进化
- 机制低、proxy 高：
  - 这只是表面连续，实际可能已经偏了
- 两者都低：
  - 更像真正 drift

实现方向：

- 机制分比较：
  - hypothesis / thesis / mechanism 文本 token
  - expected direction
  - core concepts
- proxy 分比较：
  - input features
  - proxy family tokens
  - expression patterns

预期结果：

- 系统可以自由优化实现方式，而不会被旧 proxy wording 卡死

---

## 第三步：按 stage 重定义 continuity gate

当前问题：

- 现在不同 stage 用的是同一种 continuity 逻辑，太硬

改法：

- 按 stage 分别定义 continuity 规则

### `base_reconstruction`

- 更强调 report alignment
- 对 proxy continuity 要求更强

### `parameter_tuning`

- 重点要求机制连续
- 允许中等程度的 proxy drift
- proxy-family 变化只做软惩罚，不直接重判失败

### `signal_enhancement` / `resonance_filtering`

- 仍要求机制连续
- 但允许更大的 proxy 进化
- 优先保留真正 runtime-valid 的候选，即使 proxy wording 变了

### `thesis_revision`

- continuity gate 退化为 advisory
- 允许机制变化，但必须说明原因

实现方向：

- `experiment_designer_agent._run_consistency_gate`
- `hypothesis_agent` 的 continuity enforcement
- `feedback_summarizer_agent` 的 next-stage transition logic

预期结果：

- continuity 约束会更符合研究目标，而不是阻碍有效探索

---

## 第四步：增加“允许 proxy 进化”的明确条件

当前问题：

- 系统现在没有清楚地区分：
  - “当前 proxy 很弱”
  - 和
  - “这个 thesis 本身错了”

改法：

- 在 feedback 里引入更细的诊断标签

建议新增：

- `mechanism_supported_proxy_weak`
- `mechanism_supported_proxy_exhausted`
- `mechanism_unclear`
- `mechanism_refuted`

解释方式：

- `mechanism_supported_proxy_weak`
  - 保持机制
  - 继续进化 proxy
- `mechanism_supported_proxy_exhausted`
  - 允许更大 proxy-family 变化
- `mechanism_refuted`
  - 才考虑 thesis revision

实现方向：

- `feedback_summarizer_agent` 输出更细的诊断
- `hypothesis_agent` 根据诊断决定是保机制还是修机制

预期结果：

- 后续 loop 可以更自由，但不会突然大跳

---

## 第五步：把硬拒绝尽量改成“降权 warning”

当前问题：

- 有些候选只是“不够理想”，但其实还能算
- 现在如果过早扔掉，会损失探索空间

改法：

- 严格拒绝只保留给：
  - parse failure
  - unsupported functions / features
  - 真正无法计算的表达式
- continuity / consistency / quality 相关问题：
  - 尽量保留
  - 记 warning
  - 排序降权
  - 仍然允许进入 backtest

实现方向：

- 保留当前的 `retained_with_warnings` 机制
- 但进一步调整 ranking，使“机制相近的 warning 候选”优先级高于“泛化但合法”的候选

预期结果：

- 探索空间保住了
- 选择质量也不会完全失控

---

## 第六步：把 candidate ranking 从“proxy 相似”升级成“机制相关性优先”

当前问题：

- 现在 ranking 比 FIFO 强，但还偏重 overlap 和 warning 数量
- 还不够强调机制相关性

改法：

- ranking 应优先奖励：
  - 机制相关性
  - expected-direction 一致性
  - mechanism token overlap
  - proxy 合理性
  - warning 更少
  - 复杂度更低

建议 ranking 组成：

- mechanism overlap score
- expected-direction consistency bonus
- proxy overlap score
- anchor / continuity feature match
- warning penalty
- complexity penalty

实现方向：

- 扩展 `experiment_designer_agent` 的 `selection_priority`
- 日志中完整记录 ranking 细节

预期结果：

- 最终选进 backtest 的因子会更贴近研究目标，而不是仅仅“能跑”

---

## 第七步：给 thesis 切换加“慢变量”规则

当前问题：

- 现在系统有时仍可能在证据不够时切换过快

改法：

- 机制级切换必须满足累计证据条件

建议规则：

- 只有满足以下之一，才允许机制切换：
  - 连续 2 轮出现 `mechanism_refuted`
  - 连续多轮方向与预期相反
  - 同一机制邻域下多个 proxy family 都失败

在此之前：

- 保持机制
- 主要调整 proxy

实现方向：

- `feedback_summarizer_agent` 需要积累：
  - diagnosis history
  - direction mismatch history
  - proxy-family exhaustion history
- `hypothesis_agent` 在切机制前先检查这些历史

预期结果：

- 观点切换会更慢、更可解释

---

## 第八步：让日志显式区分“机制进化”和“proxy 进化”

当前问题：

- 现在日志虽然有 continuity diagnostics
- 但还不能清楚区分：
  - 是机制没变、只是 proxy 变了
  - 还是 thesis 真改了

改法：

- 在日志里增加这些字段：
  - `mechanism_anchor`
  - `mechanism_continuity_score`
  - `proxy_continuity_score`
  - `proxy_evolution_mode`
  - `thesis_switch_blocked`
  - `thesis_switch_allowed`

建议的 `proxy_evolution_mode`：

- `report_reconstruction`
- `proxy_tuning`
- `proxy_substitution`
- `mechanism_adjacent_exploration`
- `thesis_revision`

预期结果：

- 后续查 log 会容易很多

---

## 第九步：增加一个不跑 qlib 的 smoke test

目标：

- 验证系统是否能在保持同一机制的同时，允许 proxy 自由进化

建议测试场景：

### Case A

- 上一轮机制：price-volume divergence reversal
- 新 proposal：volatility-normalized divergence
- 预期：允许

### Case B

- 上一轮机制：divergence reversal
- 新 proposal：spread-assisted divergence
- 预期：允许

### Case C

- 上一轮机制：divergence reversal
- 新 proposal：pure continuation momentum
- 预期：拦下，除非 revision 条件满足

### Case D

- 已经有多轮 proxy 弱证据
- 新 proposal：更广的机制邻域 proxy
- 预期：允许，但要记录原因

实现方向：

- 在 `scripts/` 下新增一个测试脚本
- 不依赖 qlib
- 只验证 hypothesis / designer 的 gating 和 ranking

---

## 六、预计会改哪些文件

- `src/trading_agents/agents/hypothesis/hypothesis_agent.py`
- `src/trading_agents/agents/experiment_designer_agent.py`
- `src/trading_agents/agents/feedback/feedback_summarizer_agent.py`
- `src/trading_agents/workflows/loop_runtime.py`
- 可能还要动：
  - `src/trading_agents/workflows/context_policies.py`
- 测试脚本：
  - `scripts/validate_thesis_continuity.py`
  - 或一个新的 mechanism-evolution smoke test

---

## 七、建议的实施顺序

1. 在 hypothesis 输出里加入 `mechanism_anchor`
2. 把 continuity 拆成 mechanism 和 proxy 两层
3. 把 continuity gate 改成 stage-aware
4. 把 feedback 诊断分类细化
5. 把 designer ranking 升级成 mechanism-aware
6. 加 thesis 切换的慢变量规则
7. 补日志字段
8. 加不跑 qlib 的 smoke test

---

## 八、验收标准

如果这个改造做对了，应该满足：

- loop1 看起来仍像研报复现
- loop2 保持同一机制，但可以用更好的 proxy family
- loop3 可以扩大 proxy 探索，但不会随机漂移 thesis
- 机制 / 方向不会在缺少证据时突然切换
- 日志能清楚区分：
  - 机制连续
  - proxy 进化
  - thesis revision
- 最终被选中的因子不仅“能跑”，而且更贴近预期机制

---

## 九、简短总结

我们的目标不是：

- “永远和研报一模一样”

也不是：

- “一轮弱信号后就自由乱跳”

真正的目标是：

- **机制锚定**
- **proxy 进化**
- **观点切换慢，而且必须基于累计证据**

