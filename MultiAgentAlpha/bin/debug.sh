#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="./log"
mkdir -p "${log_dir}"
python_bin="${PYTHON_BIN:-/opt/anaconda3/bin/python}"

# 快速 smoke test：先读取多份研报合成 research_report payload，再进入 gump evolution 入口。
# round 0: DirectionDecomposerAgent 将大方向拆成少量 original 方向
# round >= 1: EvolutionAgent 查看上一代结果，做 mutation + crossover 综合
# 这里故意压小规模，先确认链路；长跑时再调大 rounds/directions。
"${python_bin}" ./main.py \
  --evolution-mode \
  --evolution-max-rounds 1 \
  --evolution-num-initial-hypotheses 2 \
  --evolution-mutation-tasks-per-round 1 \
  --evolution-crossover-tasks-per-round 1 \
  --evolution-mutation-parent-top-n 4 \
  --evolution-crossover-parent-top-n 4 \
  --evolution-mutation-top-n 1 \
  --evolution-crossover-group-size 2 \
  --evolution-crossover-n-groups 1 \
  --retry-per-round 1 \
  --train-sharpe-threshold 2 \
  --research-report-pdfs materials/20260329-dingbeili.pdf materials/20150810-qushi.pdf \
  --debug \
  --no-report-pdf \
  --initial-direction "关注成交量放大后短期收益反转的横截面效应" \
  --initial-hypothesis "在加密货币横截面中，近期异常放量且价格短期快速上行的标的，随后更容易出现短期收益反转。" \
  2>&1 | tee "${log_dir}/evolution_debug_${ts}.log"
