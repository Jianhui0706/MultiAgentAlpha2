# Research Report PDF Phase 1 Plan

## Scope

Phase 1 assumes the system input format is **PDF only**.

This phase does **not** change the final evaluation framework.

The goal is to make the upstream workflow stable:

`PDF report -> text extraction -> structured thesis extraction -> crypto adaptation -> proxy reconstruction -> base factor reconstruction`

## Phase 1 Checklist

### Product Scope

- Only support PDF as the input format in phase 1.
- Keep the existing backtest and evaluation framework unchanged.
- Focus on the first two agents first.
- Treat report-driven factor mining as a thesis reconstruction workflow, not a free-search workflow.

### Input Layer

- Confirm PDF is the only accepted input format in phase 1.
- Define a stable PDF-to-text extraction path.
- Allow the system to keep page metadata or report source metadata.
- Ensure extracted text can be passed into the first agent without manual copy-paste.
- Ensure parsing failure has a fallback path:
  - empty extraction detection
  - low-text PDF detection
  - optional manual summary fallback later

### First Agent

- Change the first agent role from free hypothesis generation to:
  - thesis extraction
  - mechanism summarization
  - crypto adaptation
  - proxy reconstruction planning
- Ensure the first agent outputs structured fields instead of only one hypothesis sentence.
- Ensure it can explicitly mark non-transferable market-specific logic.

### Crypto Adaptation

- Ensure A-share specific constructs are not copied directly into crypto logic.
- Ensure the first agent rewrites market-specific mechanisms into crypto-equivalent forms where possible.
- Ensure it can reject or soften non-transferable logic instead of hallucinating.

### Data Proxy Mapping

- Ensure the first agent only maps into currently available parquet features.
- Ensure `fwd_*` fields are never used as input features.
- Ensure proxy mapping is observable and testable in the existing data.

### Second Agent

- Ensure the second agent first round is `base_reconstruction`, not free expression exploration.
- Limit first-round factors to a small set of simple proxies.
- Delay thresholds, regime filters, and complex combinations until later loops.

### Loop Semantics

- Do not hardcode behavior by absolute loop number.
- Treat `loop_count` as budget, not as a fixed semantic schedule.
- Use a state-driven `research_stage` machine instead:
  - `base_reconstruction`
  - `parameter_tuning`
  - `resonance_filtering`
  - `thesis_revision`
- Ensure early loops usually begin with reconstruction, then move into tuning/filtering only when prior results support that transition.
- Ensure hypothesis mutation is delayed unless repeated evidence says the thesis is weak.

### Debugging

#### 第一步：只改输入层和第一个 agent

目标：

- 能吃研报
- 能输出结构化 thesis
- 能输出 crypto adaptation
- 能输出 proxy reconstruction plan

改动范围：

- `main.py` 或新的输入装配层
- 第一个 agent 的 prompt / schema / 上下文
- 不动第二个 agent 逻辑

这一步调试 1 次就够，重点看：

- 第一个 agent 是否真的提取出：
  - thesis
  - mechanism
  - crypto\_adaptation\_summary
  - proxy\_reconstruction\_plan
- 是否没有把 A 股特有机制原样带进来
- 是否只使用当前 parquet 支持的特征
- 输出 JSON 是否稳定可解析

这一步正确的标志：

- 你拿一篇 PDF，第一步输出的内容“像研究员写的结构化摘要”，而不是泛泛而谈
- `input_features` 里没有不存在的字段
- `fwd_*` 没有进入输入特征

#### 第二步：改第二个 agent，让第一轮变成“朴素复现”

目标：

- 第二个 agent 第一轮先按草案吐出 4 个基础因子
- 不自由发散

改动范围：

- `experiment_designer_agent`
- `experiment_designer_agent.yaml`
- 让它感知：
  - 当前是 `base_reconstruction`
  - 当前是第几 loop
  - 当前 thesis 来源于研报

这一步建议调试 1 到 2 次。

第一次调试重点看：

- `task_plan` 是否是“复现逻辑”，不是“自由搜索”
- 是否真的只生成那 4 个基础 proxy 风格因子
- 表达式是否简单、可解析、变量合法

第二次调试重点看：

- 如果第一轮结果一般，第二轮是否会先做参数扩展，而不是直接漂到别的逻辑
- 是否仍锚定原始 thesis

这一步正确的标志：

- 第一轮输出看起来像“研报复现实验”
- 第二轮输出看起来像“参数进化”
- 不会突然跳去做无关变量族

#### Loop_002 验收清单

目标不是“重新发明一个新 hypothesis”，而是确认系统已经从第一轮的报告复现，顺利进入了受控的参数进化。

- Hypothesis 是否仍锚定原研报
  - 看 `hypothesis / thesis / mechanism / crypto_adaptation_summary`
  - 通过标准：仍然在讲同一个核心报告机制，只是更聚焦或更精炼

- `research_stage` 是否进入 `parameter_tuning`
  - 看 shared context、feedback、experiment logs 中的 `research_stage`
  - 通过标准：`loop_001` 之后的下一轮优先进入 `parameter_tuning`，而不是重新 `base_reconstruction` 或直接自由探索

- `task_plan` 是否像参数扩展，不是自由搜索
  - 通过标准：计划里出现窗口、阈值、归一化、轻量组合，而不是新变量族探索

- 表达式是否仍围绕 `proxy_reconstruction_plan`
  - 通过标准：仍命中原来的 proxy family，只是在做窗口、标准化、确认项的变化

- 变量是否合法且没有 `fwd_*`
  - 通过标准：只用 parquet 真正存在的列；`fwd_*` 只允许当标签，不允许进入输入表达式

- 复杂度是否仍然“朴素可解释”
  - 通过标准：表达式仍简短、可解释、数量受 env 控制，看起来像“参数进化版复现”，而不是重新自由搜索

#### 第三步：保留后面 agent 流程不变，但补语义检查

目标：

- coder / backtest / feedback 不改框架
- 但要确认它们没有把新语义搞坏

改动范围：

- 可能只需要轻微改 feedback prompt
- coder / backtest 基本不动

这一步调试 1 次，重点看：

- coder 是否正常把这些基础因子落地
- backtest 是否仍然用 `fwd_returns_24h`
- feedback 是否能分清：
  - thesis 不成立
  - proxy 不好
  - 参数没调对

这一步正确的标志：

- 即使结果不好，反馈也能说明“是研报逻辑问题还是代理问题”
- 不会因为第一轮弱就直接建议完全换 hypothesis

#### 第四步：再改 loop 的控制语义

目标：

- 把 loop 从“自由换 hypothesis”
- 改成“复现 -> 参数 -> 共振 -> 再考虑修 thesis”

改动范围：

- 主要是第一、二个 agent之间的状态传递
- 可能要补一些 loop stage 字段
- 不一定立刻上 controller，大概率先轻量实现

这一步建议调试 1 次。

重点看：

- Loop 1 是否朴素复现
- Loop 2 是否参数进化
- Loop 3 是否共振/过滤
- 是否只有连续不支持才开始修 hypothesis

这一步正确的标志：

- loop 行为顺序符合你的预期
- 不会第一轮结束就改 thesis
- evolution 有层次感

## Phase 1 Implementation Sequence Table

| 文件                                                                                                                                                                                        | 目标                                                                   | 验证点                                                                               | 通过标准                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [main.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/main.py)                                                                                                               | 增加 PDF 输入入口，把 `materials/*.pdf` 作为正式研究输入源，而不是只靠 `initial_direction`。 | 能指定某个 PDF 文件进入工作流；路径被写入共享上下文；不会破坏现有运行方式。                                          | 运行时能识别 PDF 输入源，并把 PDF 路径和基础元信息传入后续 agent。                              |
| 新增 `src/trading_agents/infra/pdf_reader.py`                                                                                                                                               | 提供统一 PDF 文本抽取能力，封装 `pymupdf/pypdf` 调用，避免解析逻辑散落在 agent 里。             | 能稳定读取普通文本型 PDF；返回页数、全文文本、每页文本；遇到空文本能明确报错。                                         | 给定一篇 PDF，能输出结构化结果：`source_path / page_count / full_text / page_texts`。 |
| 新增 `src/trading_agents/infra/research_report_loader.py`                                                                                                                                   | 把 PDF 原始文本整理成研究输入对象，作为后续第一 agent 的标准输入。                              | 能生成统一 payload，如 `report_source`, `report_text`, `report_metadata`。                | 第一个 agent 不需要自己处理 PDF 文件，只接收已经清洗好的 report text。                        |
| [src/trading\_agents/workflows/alpha\_factor\_mining\_workflow.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/workflows/alpha_factor_mining_workflow.py) | 在工作流入口中接入 report payload，使“PDF 输入”成为正式分支。                            | PDF 输入与原始 `initial_direction` 输入不冲突；共享上下文里有 report 字段。                            | 单 loop 运行时，第一个 agent 能拿到 report 原文和来源信息。                               |
| [src/trading\_agents/agents/hypothesis/hypothesis\_agent.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/agents/hypothesis/hypothesis_agent.py)           | 让第一个 agent 从“自由生成 hypothesis”升级为“研报提取 + crypto 迁移 + proxy 规划”。       | 输出中新增 thesis / mechanism / crypto adaptation / proxy reconstruction 相关字段；仍能兼容旧结构。 | 给定 PDF 研报，第一个 agent 能输出结构化 hypothesis package，而不是只有一句泛化 hypothesis。    |
| [src/trading\_agents/prompts/hypothesis\_agent.json](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/prompts/hypothesis_agent.json)                           | 重写第一个 agent prompt，使其优先做：研报理解、市场机制迁移、数据代理映射。                         | 提示词中明确要求：先抽取 thesis，再做 crypto adaptation，再做 proxy reconstruction；禁止直接照搬 A 股制度概念。  | 在示例 PDF 上，输出结果能区分 transferable core 与 non-transferable parts。          |
| [src/trading\_agents/domains/crypto\_cross\_section\_adapter.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/domains/crypto_cross_section_adapter.py)     | 继续作为“数据现实约束层”，让第一 agent 能看到当前数据里有哪些可用特征。                             | `feature_columns`、`target_columns`、`validation_target_column` 在共享上下文中可读。          | 第一个 agent 的 proxy mapping 只落在真实可用特征上。                                  |
| [src/trading\_agents/agents/experiment\_designer\_agent.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/agents/experiment_designer_agent.py)              | 让第二个 agent 在 phase 1 第一轮只做“基础复现”，不做自由发散。                             | 能识别 `base_reconstruction` 语义；第一轮输出因子数量和复杂度受控；保持 thesis continuity。                | 第一个 loop 输出的是 4-6 个简单、可解释、与报告逻辑直接对应的基础因子。                              |
| [src/trading\_agents/prompts/experiment\_designer\_agent.yaml](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/prompts/experiment_designer_agent.yaml)        | 把第二个 agent prompt 改成 reconstruction-first 风格，先复现再进化。                 | 明确限制第一轮不要复杂阈值和过多变量；要求 task\_plan 反映“价格延续 / 动量衰减 / 确认项”的重建逻辑。                      | 给定第一 agent 的 hypothesis package，第二个 agent 第一轮输出看起来像“研报复现实验”，而不是自由搜索。   |
| [src/trading\_agents/prompts/feedback\_summarizer\_agent.json](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/prompts/feedback_summarizer_agent.json)        | 在不改整体框架的前提下，让反馈更偏向“thesis 是否成立 / proxy 是否有效”的判断。                     | 反馈中能区分 thesis validity 与 proxy quality；不会一轮弱信号就建议完全换方向。                           | 第一轮反馈能够明确说明：是逻辑本身弱，还是表达方式还需要继续复现。                                      |
| [scripts/generate\_report.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/scripts/generate_report.py)                                                                        | Phase 1 可先轻量增强 report，使最终输出能显示 report-driven 来源。                     | 报告里可看到 PDF 来源、提取 thesis、crypto adaptation 摘要。                                     | 最终 md 报告能体现“这次因子挖掘是由哪篇 PDF 驱动的”。                                       |
| 新增 `docs/research_report_drafts/` 下的样例草案                                                                                                                                                  | 用具体 PDF 样例驱动第一阶段调试，防止架构讨论太抽象。                                        | 样例里应包含第一 agent 输出、第二 agent 第一轮输出、loop 计划。                                         | 团队可以用固定 PDF 样例反复核对 phase 1 实施是否偏离设计。                                   |

## 总建议：调试 4 次最稳

我建议按下面这个节奏：

1. 调试第一个 agent 单独输出\
   检查：提取、迁移、proxy 映射是否对
2. 调试前两个 agent 串起来\
   检查：第二个 agent 第一轮是否按朴素版复现
3. 调试完整单 loop\
   检查：coder / backtest / feedback 全链路是否通
4. 调试多 loop\
   检查：loop 是否按“复现 -> 参数 -> 共振 -> 修 thesis”运行

## 每次调试时你重点看哪里

我建议每次都盯这 5 个点：

1. 输入是否干净
   - 研报摘要有没有抽歪
   - A 股机制有没有错误迁移
2. 变量是否合法
   - 是否只用 parquet 里真实存在的列
   - 是否没有 `fwd_*` 输入污染
3. 第二个 agent 是否漂移
   - 是否还围绕原 thesis
   - 是否第一轮过度复杂
4. 评估目标是否一致
   - target 是否始终是 `fwd_returns_24h`
5. feedback 是否说人话
   - 是逻辑错了
   - 还是 proxy 弱
   - 还是参数没调到位

## 如果按风险来排，最容易出 bug 的地方

- 第一个 agent 把研报说得很好听，但没落到你真实特征
- 第二个 agent 虽然接收了 thesis，但仍然自由发散
- feedback 看见 IC 弱就直接让 hypothesis 漂走
- loop stage 没控住，导致“第一轮复现”没站稳就开始乱进化

## 一句话落地建议

先拆成：

- 第一个 agent 调通
- 前两个 agent 调通
- 完整单 loop 调通
- 多 loop 调通

不要直接上完整多轮，不然你定位问题会很痛苦。

## Final Guidance

Phase 1 should be judged by one main criterion:

Can the system take a PDF research report and turn it into a legal, crypto-adapted, data-grounded first-round factor reconstruction experiment without drifting away from the original thesis?

If the answer is yes, phase 1 is successful.
