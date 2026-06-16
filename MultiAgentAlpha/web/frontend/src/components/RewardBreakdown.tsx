import { Tag, Typography } from 'antd'

const { Text } = Typography

export interface RewardTerm {
  name: string
  value: number
  weight: number
  contribution: number
}

export interface RewardBreakdownData {
  score?: number | null
  terms?: RewardTerm[]
  target_status?: Record<string, unknown>
}

const TERM_LABELS: Record<string, string> = {
  sharpe_progress: 'Sharpe 进度',
  ic_progress: '|IC| 进度',
  rank_ic_progress: '|RankIC| 进度',
  icir_progress: '|ICIR| 进度',
  joint_progress: '联合进度',
  drawdown_penalty: '回撤惩罚',
  turnover_penalty: '换手惩罚',
  complexity_penalty: '复杂度惩罚',
  redundancy_penalty: '冗余惩罚',
  corr_penalty: '相关性惩罚',
}

function finite(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function fmt(v: unknown, digits = 4): string {
  const n = finite(v)
  return n === null ? '-' : n.toFixed(digits)
}

export default function RewardBreakdown({ breakdown, compact = false }: { breakdown?: RewardBreakdownData | null; compact?: boolean }) {
  const terms = (breakdown?.terms ?? []).filter((term) => finite(term.contribution) !== null)
  if (!terms.length) return null

  const total = finite(breakdown?.score) ?? terms.reduce((sum, term) => sum + Number(term.contribution || 0), 0)
  const positives = terms.filter((term) => Number(term.contribution) >= 0)
  const negatives = terms.filter((term) => Number(term.contribution) < 0)
  const status = breakdown?.target_status ?? {}

  return (
    <div style={{
      marginTop: compact ? 6 : 10,
      padding: compact ? '7px 9px' : '10px 12px',
      borderRadius: 6,
      border: '1px solid #f0f0f0',
      background: '#fff',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        <Text strong style={{ fontSize: compact ? 12 : 13 }}>Reward 分解</Text>
        <Tag color={total >= 0 ? 'blue' : 'red'} style={{ marginInlineEnd: 0 }}>
          总分 {fmt(total, 4)}
        </Tag>
        {typeof status.passed === 'boolean' && (
          <Tag color={status.passed ? 'green' : 'gold'} style={{ marginInlineEnd: 0 }}>
            硬目标{status.passed ? '已达成' : '未达成'}
          </Tag>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: compact ? '1fr' : 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
        {[...positives, ...negatives].map((term) => {
          const isPenalty = Number(term.contribution) < 0
          return (
            <div key={term.name} style={{
              display: 'grid',
              gridTemplateColumns: '1fr auto',
              gap: 6,
              alignItems: 'baseline',
              padding: '6px 8px',
              borderRadius: 5,
              background: isPenalty ? '#fff1f0' : '#f6ffed',
              border: `1px solid ${isPenalty ? '#ffccc7' : '#b7eb8f'}`,
            }}>
              <Text style={{ fontSize: 12, color: isPenalty ? '#a8071a' : '#237804', fontWeight: 600 }}>
                {TERM_LABELS[term.name] ?? term.name}
              </Text>
              <Text code style={{ fontSize: 11 }}>
                {fmt(term.value, 3)} x {fmt(term.weight, 2)} = {fmt(term.contribution, 4)}
              </Text>
            </div>
          )
        })}
      </div>
    </div>
  )
}
