# scripts/

辅助脚本集合，用于分析、调试和可视化因子挖掘结果。

---

## plot_group_returns.py

从 evolution 运行日志中提取因子，重新计算因子值，绘制 5 分组累积收益曲线。

### 依赖

使用项目 venv：`.venv/Scripts/python.exe`（需 `pandas`, `numpy`, `matplotlib`, `pyarrow`）。

### 用法

```bash
# 基本用法：自动找最新 log，画 sharpe >= 0.2 的全部因子
.venv/Scripts/python.exe scripts/plot_group_returns.py

# 只画 sharpe 最高的 9 个因子，限定近 1 年数据（推荐，速度快）
.venv/Scripts/python.exe scripts/plot_group_returns.py \
  --sharpe-min 0.2 --top 9 \
  --date-from 2025-04-01

# 指定具体 log 文件和输出路径
.venv/Scripts/python.exe scripts/plot_group_returns.py \
  --log log/evolution_debug_20260425_125836.log \
  --out results/group_returns_top9.png

# 全量历史数据（每因子约 100s，共约 15min）
.venv/Scripts/python.exe scripts/plot_group_returns.py \
  --sharpe-min 0.2 --top 9
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--log` | 自动找 `log/` 下最新的 `evolution_debug_*.log` | evolution debug log 路径 |
| `--panel` | `data/panel_data.parquet` | 面板数据路径 |
| `--sharpe-min` | `0.2` | 只画 sharpe 高于此阈值的因子 |
| `--top` | 不限 | 只取 sharpe 最高的 N 个因子 |
| `--date-from` | 全量 | 只用此日期之后的数据，格式 `YYYY-MM-DD` |
| `--label-col` | `fwd_returns_24h` | 标签列名 |
| `--n-layers` | `5` | 分组数 |
| `--out` | `results/group_returns.png` | 输出图片路径 |

### 输出说明

- 每个子图对应一个因子，标题显示 Sharpe 和 Rank ICIR
- G1–G5：5 个分位组的累积收益（G1=最低分位，G5=最高分位）
- L-S（虚线）：多空对冲组合（G5 - G1）
- 当累积收益跨越多个量级时自动切换为对数纵轴

### 已知问题

- **无 Sharpe > 1 的因子**：20260425 运行中最高 Sharpe 为 0.951。`--sharpe-min` 建议设为 0.2~0.5
- **计算速度**：panel 数据 250 万行，全量每因子约 100s。用 `--date-from` 裁剪后可降至 5~10s/因子
- **部分因子跳过**：`TS_QUANTILE` 参数顺序与函数签名相反的因子会自动 patch 修复；极少数语法不兼容的因子会打印 `[skip]` 并跳过

---

## generate_report.py

从 run summary 生成 Markdown / PDF 格式的因子挖掘报告。

## debug_metrics.py / debug_report.py

调试用脚本，打印单次运行的 metrics 和 backtest report 详情。

## validate_*.py

验证脚本，用于检查 hypothesis agent、counterfactual anchor binding、thesis continuity 等模块的稳定性。

### validate_factor_coder_skill_injection.py

专项验收 `FactorCoderAgent` 的规则层注入，不走完整 workflow，也不依赖外部 provider。

- 固定 `domain_dataset_summary` 和 `experiment_spec`
- 直接调用 `FactorCoderAgent.run()`
- 检查 generation / retry 的 skill metadata
- 检查 prompt 中是否出现 `Project-local FactorCoder rule context`
- 检查 retry 是否按需只注入相关 `function_rules`

```bash
python scripts/validate_factor_coder_skill_injection.py
```

通过时会输出：

- `factor_coder skill injection validation passed`
- `generation_summary=...`
- `retry_summary=...`
- `rule_files_used=...`
