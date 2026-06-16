# Prompt Engineering 修改总结

本文档总结了 Multi-Agent Factor Mining 项目在 Prompt Engineering 方面的所有修改，涵盖模型架构、Context Engineering、Harness Engineering 和 JSON 解析四个层面。

---

## 一、模型架构层面

### 1.1 函数库扩展

**原始版本：** 部分函数未实现（TS_QUANTILE, EMA, SMA, WMA, COUNT, SUMIF, FILTER, SEQUENCE, REGBETA, REGRESI, PROD, INV, FLOOR 等），prompt 中声明可用但实际执行会报错。

**修改后版本：** 全部实现 20+ 新函数，确保 prompt 声明的函数均可正常执行。

#### 新增函数列表

| 类别 | 函数 | 说明 |
|------|------|------|
| **时间序列函数** | `TS_QUANTILE(A, n, q)` | 滚动分位数，返回过去 n 期第 q 分位数的值 |
| | `TS_ARGMAX(A, n)` | 过去 n 期最大值的索引位置 |
| | `TS_ARGMIN(A, n)` | 过去 n 期最小值的索引位置 |
| | `TS_MAD(A, n)` | 滚动中位数绝对偏差 |
| | `HIGHDAY(A, n)` | 距离过去 n 期最高点的天数 |
| | `LOWDAY(A, n)` | 距离过去 n 期最低点的天数 |
| | `SUMAC(A, n)` | 累积求和 |
| **移动平均函数** | `SMA(A, n, m)` | 简单移动平均（带修饰符） |
| | `WMA(A, n)` | 加权移动平均，权重线性递减 |
| | `EMA(A, n)` | 指数移动平均 |
| | `DECAYLINEAR(A, d)` | 线性衰减移动平均 |
| **条件函数** | `COUNT(C, n)` | 统计过去 n 期满足条件 C 的数量 |
| | `SUMIF(A, n, C)` | 条件求和，过去 n 期满足条件 C 时对 A 求和 |
| | `FILTER(A, C)` | 条件过滤，不满足条件 C 的值设为 NaN |
| **回归函数** | `SEQUENCE(n)` | 生成 1 到 n 的序列 |
| | `REGBETA(A, B, n)` | 滚动回归系数，A 对 B 的过去 n 期回归 |
| | `REGRESI(A, B, n)` | 滚动回归残差 |
| **数学函数** | `PROD(A, n)` | 滚动乘积 |
| | `INV(A)` | 倒数 |
| | `FLOOR(A)` | 向下取整 |
| | `SCALE(A, target_sum)` | 截面缩放，使截面和为 target_sum |

### 1.2 RANK 函数边界处理

**原始版本：** 当截面数据所有值相同时，`rank(pct=True)` 会失败。

**修改后版本：** 添加边界检查，当 `std() < 1e-10` 时返回 0.5（中间排名），避免因数据无变化导致的计算失败。

```python
def RANK(series: pd.Series) -> pd.Series:
    def safe_rank(x):
        if x.isna().all():
            return x
        if x.std() < 1e-10:  # 所有值相同时返回 0.5
            return pd.Series(0.5, index=x.index)
        return x.rank(pct=True)
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(safe_rank)
    return safe_rank(series)
```

---

## 二、Context Engineering

### 2.1 添加 per_factor_metrics 字段

**原始版本：** 仅保存综合回测指标，所有因子共用同一组 IC/RankIC。

**修改后版本：** 每个因子单独保存指标，包含 IC、Rank IC、ICIR、Rank ICIR、layer_returns 等，便于分析单个因子表现。

#### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `models.py` | `QlibFactorExperiment` 类添加 `per_factor_metrics` 字段 |
| `backtest_runner_agent.py` | 设置 `experiment.per_factor_metrics = dict(per_factor_metrics)` |
| `feedback_summarizer_agent.py` | 从 context 提取并传递给 LLM |

#### 代码示例

```python
# models.py
@dataclass
class QlibFactorExperiment:
    per_factor_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

# backtest_runner_agent.py
experiment.per_factor_metrics = dict(per_factor_metrics)

# feedback_summarizer_agent.py
per_factor_metrics = dict(experiment.per_factor_metrics)
context["per_factor_metrics"] = per_factor_metrics
```

### 2.2 传递到 LLM Context

**修改后版本：** 将 `per_factor_metrics` 传递给 FeedbackSummarizerAgent，让 LLM 分析每个因子的单独表现。

```json
// feedback_summarizer_agent.json
"user_prompt": "Individual Factor Results (per_factor_metrics):
{{ per_factor_metrics }}

Analyze individual factors first to identify which factors are working well and which are not."
```

---

## 三、Harness Engineering

### 3.1 JSON 解析增强

**原始版本：** 单次尝试，失败即报错。

**修改后版本：** 5 层重试机制，大幅提高解析成功率。

| 层级 | 修复内容 | 函数 |
|------|----------|------|
| **预处理** | 移除思考块 | `_strip_thinking_blocks()` |
| | 提取 JSON 对象 | `_extract_json_object()` |
| | 移除 Markdown 代码块 | 内联处理 |
| **尝试 1** | 原始解析 | `json.loads()` |
| **尝试 2** | 修复尾随逗号 | `_fix_trailing_commas()` |
| **尝试 3** | 修复无效转义 | `_fix_invalid_escapes()` |
| **尝试 4** | 修复未加引号的键 | `_fix_unquoted_keys()` |
| **尝试 5** | 综合修复 | 组合以上修复 |

### 3.2 思考块处理

**原始版本：** 仅适配 zhipu 模型的 `{{-- --}}` 格式。

**修改后版本：** 考虑到 OpenAI/MiniMax 等 API 调用，用代码移除多种思考块格式，让系统适配更多模型，避免 JSON 解析失败。

```python
def _strip_thinking_blocks(text: str) -> str:
    # 移除 {{-- ... --}} 格式（zhipu 模型）
    text = re.sub(r'\{\{--.*?--\}\}', '', text, flags=re.DOTALL)
    # 移除 <think ...>...</think > 格式（OpenAI/其他模型）
    text = re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 移除自闭合 <think/> 标签
    text = re.sub(r'<think\s*/>', '', text, flags=re.IGNORECASE)
    return text.strip()
```

### 3.3 JSON 预处理

**修改后版本：** 添加 `_extract_json_object()` 从混合文本中提取 JSON 对象，自动移除 Markdown 代码块标记（```json ... ```），处理 LLM 返回非纯 JSON 的情况。

```python
def _extract_json_object(text: str) -> str:
    """从文本中提取第一个完整的 JSON 对象（处理嵌套花括号）。"""
    start_idx = text.find('{')
    if start_idx == -1:
        return text
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    
    return text[start_idx:]
```

### 3.4 调试信息打印

**修改后版本：** 添加调试打印，便于排查问题。

```python
print("[DEBUG parse_json_object] Final text to parse (first 500 chars):")
print(stripped[:500])
print(f"[DEBUG] 第一次解析失败: {exc}")
print("[DEBUG] 第二次解析成功（修复尾随逗号）")
```

---

## 四、Prompt 文件修改

### 4.1 factor_coder_agent.yaml

| 修改项 | 说明 |
|--------|------|
| ✅ 函数库描述 | 添加完整的函数库描述，包含所有新实现的函数 |
| ✅ PERCENTILE 说明 | 添加 `PERCENTILE` 详细使用说明，强调返回值而非布尔值 |
| ✅ 移除警告 | 移除"不可用函数"警告（因为已全部实现） |
| ✅ 数据频率 | 添加数据频率说明（4h/240min） |
| ✅ 除零保护 | 强制要求所有分母添加 `+ 1e-8` |

### 4.2 experiment_designer_agent.yaml

| 修改项 | 说明 |
|--------|------|
| ✅ task_plan 字段 | 添加 `task_plan` 字段让 LLM 自主决策因子构建策略 |
| ✅ JSON 格式 | 更新 JSON 输出格式，`factors` 放在对象内 |
| ✅ 指导原则 | 添加 `task_plan` 指导原则，解释数据预处理、特征选择等决策 |
| ✅ 简化示例 | 简化 JSON 示例，移除 LaTeX 公式避免解析错误 |
| ✅ 复杂度约束 | 添加因子复杂度约束（符号长度、基础特征、嵌套层级等） |
| ✅ 数据频率 | 添加数据频率说明和窗口参数建议 |

#### 新的 JSON 输出格式

```json
{
    "task_plan": [
        "Step 1: Identify overbought conditions using RSI indicator with 14-period window for 4h data",
        "Step 2: Combine RSI with volume surge detection to filter false signals",
        "Step 3: Apply cross-sectional ranking for comparability across assets"
    ],
    "factors": {
        "Factor_Name": {
            "description": "...",
            "variables": {...},
            "formulation": "...",
            "expression": "..."
        }
    }
}
```

### 4.3 feedback_summarizer_agent.json

| 修改项 | 说明 |
|--------|------|
| ✅ per_factor_metrics | 添加到 user_prompt，让 LLM 分析每个因子的单独表现 |
| ✅ 分析要求 | 要求 LLM 先分析单个因子，再综合分析 |

---

## 五、报告生成功能

### 5.1 创建 scripts/generate_report.py

**功能：**
- 从 log 文件提取信息
- 生成 Markdown 报告
- 包含：假设、实验设计、因子信息、反馈

### 5.2 集成到主脚本

```python
def _generate_final_report(log_dir: Path) -> None:
    """Generate final report after all loops complete."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from scripts.generate_report import generate_report
    md_file, pdf_file = generate_report(log_dir, no_pdf=True)
```

---

## 六、总结对比表

| 方面 | 原始版本 | 修改后版本 |
|------|----------|------------|
| **函数库** | 部分函数未实现，prompt 声明但执行报错 | 全部实现 20+ 新函数，确保声明即可用 |
| **思考块处理** | 仅适配 zhipu 的 `{{-- --}}` 格式 | 适配多种模型，移除多种思考块格式 |
| **JSON 解析** | 单次尝试，失败即报错 | 5 层重试机制，大幅提高成功率 |
| **JSON 预处理** | 无 | 提取 JSON 对象、移除 Markdown 代码块 |
| **因子指标** | 仅综合指标，所有因子共用 | 每个因子单独保存指标 |
| **LLM 自主决策** | 无 task_plan，使用默认计划 | LLM 自主决策 2-4 步构建策略 |
| **数据频率** | 未说明 | 明确 4h/240min，提供窗口参数建议 |
| **除零保护** | 提示添加，不强制 | 强制要求所有分母添加 `+ 1e-8` |
| **RANK 边界** | 数据无变化时失败 | 返回 0.5 中间排名 |
| **因子复杂度** | 无约束 | 硬性约束：长度 ≤ 250、特征 ≤ 6、嵌套 ≤ 3 |
| **变量限制** | 未明确限制 | 明确列出可用变量，禁止未来函数 |
| **反馈分析** | 仅分析综合结果 | 同时分析综合和单个因子表现 |
| **报告生成** | 无 | 自动生成 Markdown 报告 |
| **调试信息** | 无 | 显示解析过程和待解析文本 |
| **Prompt 格式** | JSON 示例含 LaTeX 公式 | 简化示例，降低解析错误率 |

---

## 七、文件修改清单

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `src/trading_agents/factor_runtime/factor_functions.py` | 新增 | 实现 20+ 新函数 |
| `src/trading_agents/infra/llm/parser.py` | 增强 | 5 层 JSON 解析重试机制 |
| `src/trading_agents/schemas/models.py` | 新增 | 添加 `per_factor_metrics` 字段 |
| `src/trading_agents/agents/backtest_runner_agent.py` | 修改 | 设置 `per_factor_metrics` |
| `src/trading_agents/agents/feedback/feedback_summarizer_agent.py` | 修改 | 传递 `per_factor_metrics` 给 LLM |
| `src/trading_agents/prompts/factor_coder_agent.yaml` | 更新 | 完善函数库描述和约束 |
| `src/trading_agents/prompts/experiment_designer_agent.yaml` | 更新 | 添加 task_plan 和复杂度约束 |
| `src/trading_agents/prompts/feedback_summarizer_agent.json` | 更新 | 添加 per_factor_metrics 分析 |
| `scripts/generate_report.py` | 新增 | 报告生成脚本 |
| `examples/run_alpha_factor_mining_loop.py` | 修改 | 集成报告生成功能 |

---

*文档生成时间：2026-04-20*
