# System Prompt

请在场景 {{scenario}} 下评估市场 {{market}} 的因子实现结果。
好因子达标口径固定为：净 Sharpe > 1.5，abs(IC) > 0.03，abs(Rank IC) > 0.03。
将计算状态作为硬约束，输出必须是严格 JSON：
{{backtest_output_format}}

# User Prompt

因子实现：
{{factor_implementation}}

计算报告：
{{calculation_report}}

因子来源：{{factor_source}}
成功生成因子数：{{generated_factors}}
失败因子数：{{failed_factors}}

仅返回 JSON。

# Backtest Output Format

{"backtest_report":{"status":"finished|partial|failed","summary":"string","factor_source":"string"},"metrics":{"sharpe":"number","turnover":"number","drawdown":"number","coverage":"number"}}
