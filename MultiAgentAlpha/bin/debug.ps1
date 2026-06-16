$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = ".\log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 新方案小规模遗传迭代测试：
# round 0: DirectionDecomposerAgent 将大方向拆成 5 条 original 方向
# round >= 1: EvolutionAgent 同时查看上一代 5 条结果，做 mutation + crossover 综合
#             然后 DirectionDecomposerAgent 再拆成 5 条下一代方案
# 旧 mutation_top_n / crossover_n_groups 仅作为 fallback 规则配置保留。
uv run .\main.py `
  --evolution-mode `
  --evolution-max-rounds 3 `
  --evolution-num-initial-hypotheses 5 `
  --evolution-mutation-tasks-per-round 2 `
  --evolution-crossover-tasks-per-round 1 `
  --evolution-mutation-parent-top-n 10 `
  --evolution-crossover-parent-top-n 10 `
  --evolution-mutation-top-n 2 `
  --evolution-crossover-group-size 2 `
  --evolution-crossover-n-groups 1 `
  --debug `
  --no-report-pdf `
  --initial-direction "关注成交量放大后短期收益反转的横截面效应" `
  --initial-hypothesis "在加密货币横截面中，近期异常放量且价格短期快速上行的标的，随后更容易出现短期收益反转。" `
  2>&1 | Tee-Object -FilePath "$logDir\evolution_debug_$ts.log"
