import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import {
  Collapse, Tag, Typography, Space, Button, Empty, Spin,
  Statistic, Row, Col, Card, Tooltip, Table, Badge,
} from 'antd'
import {
  CheckCircleFilled, CloseCircleFilled, MinusCircleFilled,
  FolderOpenOutlined, ExperimentOutlined, ApartmentOutlined,
  FieldTimeOutlined,
} from '@ant-design/icons'
import type { Run, Factor } from '../types'
import FactorMiniChart from '../components/FactorMiniChart'
import { formatFeedbackSummary } from '../utils/feedbackText'

const { Text, Title } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, d = 3) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(d)
}

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? '—' : (Number(v) * 100).toFixed(2) + '%'
}

// 与因子运行记录页（FactorDashboard）保持同一三态口径（资格线 OR）：
// 成功 = （Sharpe>1.0 或 |IC|>0.02）且 通过入库标准；部分 = 过资格线但未入库；失败 = 其余
const SUCCESS_SHARPE_THRESHOLD = 1.0
const SUCCESS_ABS_IC_THRESHOLD = 0.02

function factorStatus(f: Factor): 'success' | 'partial' | 'failed' {
  const impl = (f.implementation_status ?? '').toLowerCase()
  if (impl === 'failed' || impl === 'error') return 'failed'
  const sharpeOk = f.sharpe !== null && f.sharpe !== undefined && f.sharpe > SUCCESS_SHARPE_THRESHOLD
  const icOk = f.IC !== null && f.IC !== undefined && Math.abs(f.IC) > SUCCESS_ABS_IC_THRESHOLD
  if (!sharpeOk && !icOk) return 'failed'
  return f.admission_passed ? 'success' : 'partial'
}

const STATUS_ICON = {
  success: <CheckCircleFilled style={{ color: '#52c41a' }} />,
  partial: <MinusCircleFilled style={{ color: '#faad14' }} />,
  failed:  <CloseCircleFilled  style={{ color: '#f5222d' }} />,
}
const STATUS_LABEL = { success: '成功', partial: '达标未入库', failed: '失败' }

const PHASE_COLOR: Record<string, string> = {
  original: 'blue', mutation: 'orange', crossover: 'green', evolved: 'purple',
}
const PHASE_LABEL: Record<string, string> = {
  original: '初始化', mutation: 'Mutation', crossover: 'Crossover', evolved: '进化',
}

function fmtTime(iso: string | null) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}

// ── factor mini-table inside a loop panel ────────────────────────────────────

function LoopFactorList({ factors }: { factors: Factor[] }) {
  const columns = [
    {
      title: '状态',
      width: 70,
      render: (_: unknown, f: Factor) => {
        const s = factorStatus(f)
        return <Space size={4}>{STATUS_ICON[s]}<Text style={{ fontSize: 11 }}>{STATUS_LABEL[s]}</Text></Space>
      },
    },
    {
      title: '因子名',
      dataIndex: 'factor_name',
      render: (n: string, f: Factor) => (
        <Tooltip title={f.expression || '无表达式'}>
          <Space size={6} wrap>
            <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{n}</Text>
            {(f.duplicate_expression_count ?? 0) > 1 && (
              <Tag color="orange" style={{ marginInlineEnd: 0 }}>重复 x{f.duplicate_expression_count}</Tag>
            )}
          </Space>
        </Tooltip>
      ),
    },
    {
      title: 'Reward',
      dataIndex: 'reward_score',
      render: (v: number | null) => v === null || v === undefined ? <Text type="secondary">-</Text>
        : <Text style={{ color: v >= 0.3 ? '#52c41a' : '#faad14', fontWeight: 700 }}>{fmt(v, 4)}</Text>,
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe',
      render: (v: number | null) => v === null ? <Text type="secondary">—</Text>
        : <Text style={{ color: v >= 1 ? '#52c41a' : '#faad14', fontWeight: 700 }}>{fmt(v)}</Text>,
    },
    { title: 'IC', dataIndex: 'IC', render: (v: number | null) => fmt(v, 4) },
    { title: 'ICIR', dataIndex: 'icir', render: (v: number | null) => fmt(v, 4) },
    {
      title: '收益率',
      render: (_: unknown, f: Factor) => {
        const v = f.ls_return ?? f.ann_return
        if (v === null || v === undefined) return <Text type="secondary">—</Text>
        return <Text style={{ color: v > 0 ? '#52c41a' : '#f5222d' }}>{pct(v)}</Text>
      },
    },
    {
      title: '表达式',
      dataIndex: 'expression',
      ellipsis: true,
      render: (e: string) => <Text style={{ fontFamily: 'monospace', fontSize: 11 }}>{e || '—'}</Text>,
    },
  ]

  return (
    <Table
      rowKey="factor_id"
      columns={columns}
      dataSource={factors}
      size="small"
      pagination={false}
      locale={{ emptyText: <Empty description="本轮无因子" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
      rowClassName={(f: Factor) => {
        const s = factorStatus(f)
        return s === 'success' ? 'row-success' : s === 'failed' ? 'row-failed' : ''
      }}
      expandable={{
        expandedRowRender: (f: Factor) => (
          <div style={{ padding: '6px 16px 12px', background: '#fafafa', fontSize: 12 }}>
            {f.hypothesis && (
              <div style={{ marginBottom: 6 }}>
                <Text strong>来源假说：</Text>
                <Text type="secondary">{f.hypothesis}</Text>
              </div>
            )}
            {f.feedback_summary && (
              <div style={{ marginBottom: 6 }}>
                <Text strong>问题总结：</Text>
                <Text type="secondary">{formatFeedbackSummary(f.feedback_summary)}</Text>
              </div>
            )}
            {f.expression && (
              <div style={{ marginTop: 6, background: '#1a1a2e', borderRadius: 4, padding: '6px 10px', marginBottom: 8 }}>
                <Text code style={{ color: '#a8d8ff', fontSize: 11 }}>{f.expression}</Text>
              </div>
            )}
            <FactorMiniChart metrics={{
              sharpe: f.sharpe, IC: f.IC, rank_ic: f.rank_ic,
              icir: f.icir, rank_icir: f.rank_icir,
              coverage: f.coverage, turnover: f.turnover,
              valid_cross_section_mean: f.valid_cross_section_mean,
              cross_section_universe_size: f.cross_section_universe_size,
              ic_win_rate: f.ic_win_rate,
              rank_ic_win_rate: f.rank_ic_win_rate,
              drawdown: f.drawdown,
              long_short_positive_ratio: f.long_short_positive_ratio,
              ls_return: f.ls_return, ann_return: f.ann_return,
              yearly_returns: f.yearly_returns,
              reward_score: f.factor_reward_score ?? f.reward_score,
              reward_breakdown: f.reward_breakdown,
              layer_1: f.layer_1, layer_2: f.layer_2, layer_3: f.layer_3,
              layer_4: f.layer_4, layer_5: f.layer_5,
              layer_monotonicity: f.layer_monotonicity,
              ls_cumulative_series: f.ls_cumulative_series,
              layer_1_cumulative_series: f.layer_1_cumulative_series,
              layer_2_cumulative_series: f.layer_2_cumulative_series,
              layer_3_cumulative_series: f.layer_3_cumulative_series,
              layer_4_cumulative_series: f.layer_4_cumulative_series,
              layer_5_cumulative_series: f.layer_5_cumulative_series,
              benchmark_cumulative_series: f.benchmark_cumulative_series,
              ic_cumulative_series: f.ic_cumulative_series,
              rank_ic_cumulative_series: f.rank_ic_cumulative_series,
            }} />
          </div>
        ),
        rowExpandable: (f: Factor) => !!(f.hypothesis || f.feedback_summary || f.expression || f.sharpe !== null),
      }}
    />
  )
}

// ── single run panel ─────────────────────────────────────────────────────────

function RunPanel({ run }: { run: Run }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const { data: factors, isLoading } = useSWR<Factor[]>(
    open ? `/api/runs/${run.run_id}/factors` : null,
    fetcher
  )

  // Group factors by loop_index
  const byLoop: Record<number, Factor[]> = {}
  for (const f of factors ?? []) {
    ;(byLoop[f.loop_index] ??= []).push(f)
  }
  const loops = Object.keys(byLoop).map(Number).sort((a, b) => a - b)

  const succeeded = (factors ?? []).filter(f => factorStatus(f) === 'success').length
  const failed    = (factors ?? []).filter(f => factorStatus(f) === 'failed').length

  const statusColor = run.status === 'completed' ? 'success' : run.status === 'failed' ? 'error' : 'processing'

  return (
    <Collapse
      size="small"
      style={{ marginBottom: 10 }}
      onChange={(keys) => setOpen(keys.length > 0)}
      items={[{
        key: run.run_id,
        label: (
          <Row align="middle" gutter={16} style={{ width: '100%' }}>
            <Col flex="auto">
              <Space size={10}>
                <Text strong style={{ fontFamily: 'monospace', fontSize: 13 }}>{run.run_id}</Text>
                <Tag color={statusColor}>{run.status}</Tag>
                {run.evolution_mode && <Tag color="purple">进化模式</Tag>}
              </Space>
              <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
                {fmtTime(run.started_at)}
                {run.ended_at ? ` → ${fmtTime(run.ended_at)}` : ' → 运行中'}
              </div>
            </Col>
            <Col>
              <Space size={16}>
                <span>
                  <CheckCircleFilled style={{ color: '#52c41a', marginRight: 4 }} />
                  <Text>{run.factor_count ?? 0} 因子</Text>
                </span>
                {succeeded > 0 && <Text style={{ color: '#52c41a' }}>✓ {succeeded} 成功</Text>}
                {failed > 0 && <Text style={{ color: '#f5222d' }}>✗ {failed} 失败</Text>}
                <Text type="secondary" style={{ fontSize: 12 }}>{run.trajectory_count ?? run.loop_count ?? 0} 条轨迹</Text>
              </Space>
            </Col>
            <Col>
              <Space onClick={e => e.stopPropagation()}>
                <Button size="small" icon={<FieldTimeOutlined />}
                  onClick={() => navigate(`/runs/${run.run_id}`)}>事件时间线</Button>
                <Button size="small" icon={<ExperimentOutlined />}
                  onClick={() => navigate(`/runs/${run.run_id}/factors`)}>因子详情</Button>
                <Button size="small" icon={<ApartmentOutlined />}
                  onClick={() => navigate(`/runs/${run.run_id}/lineage`)}>进化树</Button>
              </Space>
            </Col>
          </Row>
        ),
        children: isLoading ? (
          <Spin style={{ display: 'block', margin: '16px auto' }} />
        ) : loops.length === 0 ? (
          <Empty description="暂无因子记录" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : (
          <Collapse
            ghost
            size="small"
            defaultActiveKey={loops.map(String)}
            items={loops.map((loopIdx) => {
              const loopFactors = byLoop[loopIdx]
              const phase = loopFactors[0]?.phase ?? ''
              const roundIdx = loopFactors[0]?.round_idx
              const ok = loopFactors.filter(f => factorStatus(f) === 'success').length
              const fail = loopFactors.filter(f => factorStatus(f) === 'failed').length

              return {
                key: String(loopIdx),
                label: (
                  <Space size={10}>
                    <Text strong style={{ fontSize: 12 }}>轨迹评估 {loopIdx}</Text>
                    {phase && <Tag color={PHASE_COLOR[phase] ?? 'default'} style={{ fontSize: 11 }}>{PHASE_LABEL[phase] ?? phase}</Tag>}
                    {roundIdx !== undefined && <Text type="secondary" style={{ fontSize: 11 }}>Round {roundIdx}</Text>}
                    <Badge count={ok} color="#52c41a" style={{ fontSize: 10 }} />
                    {fail > 0 && <Badge count={fail} color="#f5222d" style={{ fontSize: 10 }} />}
                    <Text type="secondary" style={{ fontSize: 11 }}>{loopFactors.length} 个因子</Text>
                  </Space>
                ),
                children: <LoopFactorList factors={loopFactors} />,
              }
            })}
          />
        ),
      }]}
    />
  )
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function HistoryRuns() {
  const navigate = useNavigate()
  const { data: runs, isLoading } = useSWR<Run[]>('/api/runs', fetcher, { refreshInterval: 10000 })

  const history = (runs ?? [])
    .filter(r => r.status !== 'running')
    .sort((a, b) => (b.started_at > a.started_at ? 1 : -1))

  const completed = history.filter(r => r.status === 'completed').length
  const failed    = history.filter(r => r.status === 'failed').length

  if (isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />

  return (
    <div>
      <Row align="middle" justify="space-between" style={{ marginBottom: 20 }}>
        <Col><Title level={4} style={{ margin: 0 }}>历史运行</Title></Col>
        <Col>
          <Button type="primary" onClick={() => navigate('/runs/new')}>+ 新建运行</Button>
        </Col>
      </Row>

      <Row gutter={12} style={{ marginBottom: 20 }}>
        <Col span={5}><Card size="small"><Statistic title="历史总数" value={history.length} /></Card></Col>
        <Col span={5}>
          <Card size="small">
            <Statistic title="已完成" value={completed} valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleFilled />} />
          </Card>
        </Col>
        <Col span={5}>
          <Card size="small">
            <Statistic title="失败" value={failed} valueStyle={{ color: '#f5222d' }}
              prefix={<CloseCircleFilled />} />
          </Card>
        </Col>
      </Row>

      {history.length === 0 ? (
        <Empty
          image={<FolderOpenOutlined style={{ fontSize: 56, color: '#d9d9d9' }} />}
          description={<Text type="secondary">暂无历史运行，点击「新建运行」开始第一次因子挖掘</Text>}
          style={{ padding: '60px 0' }}
        >
          <Button type="primary" onClick={() => navigate('/runs/new')}>新建运行</Button>
        </Empty>
      ) : (
        <>
          {history.map(run => <RunPanel key={run.run_id} run={run} />)}
          <style>{`
            .row-success td { background: #f6ffed !important; }
            .row-failed td  { background: #fff2f0 !important; }
          `}</style>
        </>
      )}
    </div>
  )
}
