# Research Report PDF Phase 1 Status

## Scope Reminder

Phase 1 currently assumes:

- input format is **PDF only**
- the final evaluation framework stays unchanged
- the team will **manually pre-screen report quality**
- therefore, **manual summary fallback is intentionally out of scope**

The current target is to make this main path stable:

`PDF report -> text extraction -> structured thesis extraction -> crypto adaptation -> proxy reconstruction -> base factor reconstruction -> staged refinement`

## 已完成

### 输入层

- `main.py` 已支持 `--research-report-pdf`
- 已有统一 PDF 读取链路：
  - [main.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/main.py)
  - [pdf_reader.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/infra/pdf_reader.py)
  - [research_report_loader.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/infra/research_report_loader.py)
  - [report_cleaner.py](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/src/trading_agents/infra/report_cleaner.py)
- PDF 文本会被清洗后再送入后续 agent
- 已保留原始文本、清洗后文本、清洗统计、删减片段等信息
- 已有轻量检测：
  - 低文本页
  - image-heavy / text extraction may be incomplete

### 第一个 Agent

- 第一个 agent 已从“自由生成 hypothesis”升级为“研报提取 + crypto 迁移 + proxy 规划”
- 已输出结构化字段：
  - `thesis`
  - `mechanism`
  - `expected_direction`
  - `market_structure_dependencies`
  - `crypto_adaptation_summary`
  - `proxy_reconstruction_plan`
- hypothesis 输出链路已支持：
  - PDF 文本上下文
  - dataset feature 列表
  - `validation_target = fwd_returns_24h`

### Crypto Adaptation 与 Data Proxy Mapping

- 已能识别 A 股特有机制，并写入不可直接迁移部分
- 已能把可迁移逻辑改写成 crypto 横截面语言
- `proxy_reconstruction_plan.input_features` 已限制为 parquet 真实存在字段
- `fwd_*` 字段不会进入输入特征，只作为标签使用

### 第二个 Agent

- 第二个 agent 第一轮已不再是自由搜索
- 已支持 research-stage 驱动的行为：
  - `base_reconstruction`
  - `parameter_tuning`
  - `resonance_filtering`
  - `thesis_revision`
- 已能在 report-driven 模式下：
  - 第一轮先做 reconstruction-first
  - 后续进入 parameter tuning / resonance filtering
- 因子数量仍可由 env 控制

### Loop 语义

- 已不再按绝对 loop 编号硬编码语义
- `loop_count` 现在更接近预算
- `research_stage` 已成为主要状态机
- hypothesis mutation 已延后，不会第一轮弱信号就直接大改 thesis

### 语义检查与失败轮次处理

- feedback 已能区分：
  - `thesis_invalid`
  - `proxy_weak`
  - `parameter_not_tuned`
  - `implementation_unstable`
  - `direction_mismatch`
- `thesis_revision` 已加方向边界
- failed / partial 场景下不会再轻易替换 best 结果
- workflow 已在失败轮次清理 stale result，避免旧 experiment/backtest/feedback 污染下一轮

### 工程稳健性

- hypothesis JSON 解析稳健性已增强
- hypothesis 失败诊断日志已增强
- LLM provider / retry / error_details 观测已增强
- experiment designer 已增加复杂度预算与复杂度诊断

### Report 与文档

- report 已能体现 PDF 驱动来源
- report 主体已改成中文业务框架
- report 已展示：
  - 研报摘要
  - 当前阶段
  - 假设演化路径
  - 因子表达式
  - IC / Rank IC / Sharpe / Drawdown 等指标
  - 复杂度预算
  - 复杂度诊断
- 失败轮次已在 report 中明确标记“本轮实验未完成”
- 已有草案与规划文档：
  - [research_report_input_architecture.md](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/docs/research_report_input_architecture.md)
  - [research_report_pdf_phase1_plan.md](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/docs/research_report_pdf_phase1_plan.md)
  - [research_report_drafts](/Users/zhangzekun/Desktop/Multi-agent-factor-mining-main/docs/research_report_drafts)

## 已决定不做

### Manual Summary Fallback

- `manual summary fallback` 当前阶段明确不做
- 原因：
  - 团队会先人工筛选 PDF 质量
  - Phase 1 聚焦于“优质 PDF 主路径”做稳
  - 不希望为了兜底坏 PDF，把输入层做得过重

这意味着：

- Phase 1 默认只支持：
  - `PDF -> 自动抽取 -> 清洗 -> hypothesis package`
- 不额外增加：
  - 手工摘要输入分支
  - fallback summary loader
  - summary-based CLI 参数

## 仍待收口

### 1. workflow 层的 report-driven 分支抽象仍可更清晰

目前功能已经通了，但更多是：

- `main.py`
- shared context
- hypothesis / experiment / feedback agent

联合实现。

后续若要进一步收口，可以把“report-driven workflow branch”在 workflow 层表达得更明确，而不是主要依赖入口装配与 agent 约定。

### 2. feedback 判定边界仍需继续收稳

虽然现在已经能区分 thesis / proxy / parameter / implementation 问题，但从已有样本看：

- `thesis_invalid` 仍可能判得偏早
- 混合方向或弱信号情形下，仍值得进一步收紧判断条件

这一项不是“没有实现”，而是“已实现第一版，但还未完全稳定”。

### 3. crypto adaptation 的保守性还能再增强

现在第一个 agent 已能识别不可迁移机制，但在某些研报上仍可能保留部分原市场叙事惯性。

后续还可以继续优化：

- 更稳地拒绝不可迁移逻辑
- 更少地带入 A 股制度语言
- 更保守地重写成 crypto 等价机制

### 4. Phase 1 回归验收流程尚未完全固化

目前我们已经做了很多轮实际验收，但还没有彻底固化成：

- 固定输入样本
- 固定命令
- 固定检查项
- 固定通过标准

后续如果要进入更稳定的团队协作阶段，这一步会很有价值。

## 当前建议的下一步

在不扩大 Phase 1 范围的前提下，建议优先顺序为：

1. 收紧 feedback 判定边界，尤其是 `thesis_invalid` 的触发条件
2. 继续观察并优化 crypto adaptation 的保守性
3. 固化一套 Phase 1 回归验收流程
4. 如有需要，再考虑把 report-driven 分支在 workflow 层继续抽象清楚

## 一句话总结

当前 Phase 1 的主链路已经基本搭好：

`PDF -> 清洗 -> thesis extraction -> crypto adaptation -> proxy reconstruction -> reconstruction-first -> staged refinement`

剩下的工作重点，不再是“大框架是否存在”，而是：

- 语义边界是否更稳
- 状态流转是否更可靠
- 验收流程是否能固化
