const ACTION_LABELS: Record<string, string> = {
  continue: '继续当前方向',
  proxy_reconstruction: '重构代理变量',
  parameter_tuning: '参数调优',
  hypothesis_revision: '修订假说',
  hypothesis_replacement: '替换假说',
  hypothesis_reset: '重置假说',
}

const CHALLENGE_LABELS: Record<string, string> = {
  none: '无',
  weak: '弱',
  moderate: '中等',
  strong: '强',
}

const GROUP_LABELS: Record<string, string> = {
  mainline: '主线组',
  counterfactual: '反事实组',
  adjacent: '相邻代理组',
}

function label(map: Record<string, string>, value: string) {
  return map[value] ?? value
}

function translateGroupDiagnostics(text: string) {
  let next = text.replace(
    /Group-level diagnosis:\s*counterfactual_challenge_level=([^,.\s]+),\s*recommended_action=([^.\s]+)\./g,
    (_match, challenge: string, action: string) =>
      `组别诊断：反事实挑战=${label(CHALLENGE_LABELS, String(challenge))}，建议动作=${label(ACTION_LABELS, String(action))}。`,
  )
  next = next.replace(
    /\b(mainline|counterfactual|adjacent):\s*count=([^,]+),\s*best_factor=([^,]+),\s*best_abs_rank_ic=([^,]+),\s*median_abs_rank_ic=([^.\n]+)\./g,
    (_match, group: string, count: string, bestFactor: string, bestRankIc: string, medianRankIc: string) =>
      `${label(GROUP_LABELS, String(group))}：数量=${count}，最佳因子=${bestFactor}，最佳 |Rank IC|=${bestRankIc}，中位 |Rank IC|=${medianRankIc}。`,
  )
  return next
}

export function formatFeedbackSummary(value: string | null | undefined) {
  const text = String(value ?? '').trim()
  if (!text) return ''
  return translateGroupDiagnostics(text)
}
