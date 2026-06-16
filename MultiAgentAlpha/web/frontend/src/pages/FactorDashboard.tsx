import React, { useMemo } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import {
  Table, Tag, Typography, Row, Col, Card, Statistic, Empty,
  Collapse, Descriptions, Badge, Space, Tooltip, Divider, Button,
} from 'antd'
import {
  CheckCircleFilled, CloseCircleFilled, MinusCircleFilled,
  BulbOutlined, CommentOutlined, CodeOutlined, BarChartOutlined,
  FileTextOutlined, BranchesOutlined, ExperimentOutlined, MonitorOutlined,
} from '@ant-design/icons'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip,
  ResponsiveContainer, ScatterChart, Scatter, LineChart, Line, Legend, Cell,
} from 'recharts'
import type { Factor, RunDetail } from '../types'
import FactorMiniChart from '../components/FactorMiniChart'
import { formatFeedbackSummary } from '../utils/feedbackText'

const { Title, Text, Paragraph } = Typography

const fetcher = (url: string) => fetch(url).then((r) => r.json())

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return '—'
  return Number(v).toFixed(digits)
}

function pct(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return (Number(v) * 100).toFixed(2) + '%'
}

type FactorStatusKey = 'success' | 'partial' | 'failed'

// 三态分类（与顶部计数卡一一对应，成功+部分+失败=总数）：
// 成功 = 过资格线（Sharpe>1.0 或 |IC|>0.02）且 通过 9 项入库标准（已录入因子库）
// 部分 = 过资格线 但 未通过入库标准（指标达标但被入库门拦下）
// 失败 = 其余（实现失败 / 两个指标都不达标 / 无指标）
// 资格线与假说资格赛、时间线"过资格线"徽章、因子库排行榜完全同口径（OR）。
const SUCCESS_SHARPE_THRESHOLD = 1.0
const SUCCESS_ABS_IC_THRESHOLD = 0.02

function passesMetricBar(f: Factor): boolean {
  const sharpeOk = f.sharpe !== null && f.sharpe !== undefined && f.sharpe > SUCCESS_SHARPE_THRESHOLD
  const icOk = f.IC !== null && f.IC !== undefined && Math.abs(f.IC) > SUCCESS_ABS_IC_THRESHOLD
  return sharpeOk || icOk
}

function factorStatus(f: Factor): FactorStatusKey {
  const impl = (f.implementation_status ?? '').toLowerCase()
  if (impl === 'failed' || impl === 'error') return 'failed'
  if (!passesMetricBar(f)) return 'failed'
  return f.admission_passed ? 'success' : 'partial'
}

const STATUS_META: Record<FactorStatusKey, { color: string; icon: React.ReactNode; label: string }> = {
  success: { color: '#52c41a', icon: <CheckCircleFilled style={{ color: '#52c41a' }} />, label: '成功（达标且入库）' },
  partial: { color: '#faad14', icon: <MinusCircleFilled style={{ color: '#faad14' }} />, label: '部分（达标未入库）' },
  failed:  { color: '#f5222d', icon: <CloseCircleFilled style={{ color: '#f5222d' }} />, label: '失败' },
}

const PHASE_COLOR: Record<string, string> = {
  original: 'blue',
  mutation: 'orange',
  crossover: 'green',
  evolved: 'purple',
}

const PHASE_BAR_COLOR: Record<string, string> = {
  original: '#1677ff',
  mutation: '#fa8c16',
  crossover: '#52c41a',
  evolved: '#722ed1',
}

// ── chart helpers ─────────────────────────────────────────────────────────────

function useChartData(factors: Factor[]) {
  return useMemo(() => {
    // 图表纳入「成功 + 部分」：指标达标即可参与展示，不要求已入库
    const succeeded = factors.filter((f) => {
      const s = factorStatus(f)
      return s === 'success' || s === 'partial'
    })

    // Top 15 by Sharpe for bar chart
    const topBySharepe = [...succeeded]
      .sort((a, b) => (b.sharpe ?? -99) - (a.sharpe ?? -99))
      .slice(0, 15)
      .map((f) => ({
        name: f.factor_name.length > 18 ? f.factor_name.slice(0, 18) + '…' : f.factor_name,
        fullName: f.factor_name,
        sharpe: Number((f.sharpe ?? 0).toFixed(3)),
        phase: f.phase ?? 'original',
      }))

    // Mean metrics per loop
    const byLoop: Record<number, { count: number; shSum: number; icSum: number; rankIcSum: number }> = {}
    for (const f of succeeded) {
      const li = f.loop_index ?? 0
      if (!byLoop[li]) byLoop[li] = { count: 0, shSum: 0, icSum: 0, rankIcSum: 0 }
      byLoop[li].count += 1
      byLoop[li].shSum += f.sharpe ?? 0
      byLoop[li].icSum += f.IC ?? 0
      byLoop[li].rankIcSum += f.rank_ic ?? 0
    }
    const loopTrend = Object.entries(byLoop)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([loop, agg]) => ({
        loop: Number(loop),
        avgSharpe: Number((agg.shSum / agg.count).toFixed(4)),
        avgIC: Number((agg.icSum / agg.count).toFixed(4)),
        avgRankIC: Number((agg.rankIcSum / agg.count).toFixed(4)),
        count: agg.count,
      }))

    // IC vs Sharpe scatter
    const scatter = succeeded
      .filter((f) => f.IC !== null && f.sharpe !== null)
      .map((f) => ({
        IC: Number((f.IC ?? 0).toFixed(4)),
        Sharpe: Number((f.sharpe ?? 0).toFixed(3)),
        phase: f.phase ?? 'original',
        name: f.factor_name,
      }))

    return { topBySharepe, loopTrend, scatter }
  }, [factors])
}

// ── chart components ──────────────────────────────────────────────────────────

function SharpeBarChart({ data }: { data: ReturnType<typeof useChartData>['topBySharepe'] }) {
  if (!data.length) return <Empty description="暂无指标达标的因子" style={{ padding: 32 }} />
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 10 }} />
        <ReTooltip
          formatter={(v, _n, props) => [v, `Sharpe — ${props.payload?.fullName ?? ''}`]}
          labelFormatter={() => ''}
        />
        <Bar dataKey="sharpe" name="Sharpe" radius={[0, 3, 3, 0]}>
          {data.map((entry) => (
            <Cell key={entry.fullName} fill={PHASE_BAR_COLOR[entry.phase] ?? '#1677ff'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function LoopTrendChart({ data }: { data: ReturnType<typeof useChartData>['loopTrend'] }) {
  if (data.length < 2) return <Empty description="需要至少 2 次轨迹评估才能展示趋势" style={{ padding: 32 }} />
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 24, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="loop" label={{ value: '轨迹评估', position: 'insideBottomRight', offset: -4, fontSize: 11 }} tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <ReTooltip />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="avgSharpe" name="均值 Sharpe" stroke="#1677ff" strokeWidth={2} dot />
        <Line type="monotone" dataKey="avgIC" name="均值 IC" stroke="#52c41a" strokeWidth={2} dot />
        <Line type="monotone" dataKey="avgRankIC" name="均值 Rank IC" stroke="#fa8c16" strokeWidth={2} dot />
      </LineChart>
    </ResponsiveContainer>
  )
}

function ICScatterChart({ data }: { data: ReturnType<typeof useChartData>['scatter'] }) {
  if (!data.length) return <Empty description="暂无数据" style={{ padding: 32 }} />
  const phases = [...new Set(data.map((d) => d.phase))]
  const byPhase = Object.fromEntries(phases.map((p) => [p, data.filter((d) => d.phase === p)]))
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 4, right: 24, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="IC" name="IC" type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} label={{ value: 'IC', position: 'insideBottomRight', offset: -4, fontSize: 11 }} />
        <YAxis dataKey="Sharpe" name="Sharpe" type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} label={{ value: 'Sharpe', angle: -90, position: 'insideLeft', fontSize: 11 }} />
        <ReTooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
          if (!payload?.length) return null
          const d = payload[0]?.payload as { name: string; IC: number; Sharpe: number; phase: string }
          return (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 6, padding: '6px 10px', fontSize: 12 }}>
              <div style={{ fontFamily: 'monospace', fontWeight: 600, marginBottom: 2 }}>{d.name}</div>
              <div>IC: {d.IC} &nbsp;|&nbsp; Sharpe: {d.Sharpe}</div>
              <div>阶段: {d.phase}</div>
            </div>
          )
        }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {phases.map((phase) => (
          <Scatter key={phase} name={phase} data={byPhase[phase]} fill={PHASE_BAR_COLOR[phase] ?? '#8884d8'} opacity={0.75} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ── expanded row detail ───────────────────────────────────────────────────────

function FactorDetail({ f }: { f: Factor }) {
  const status = factorStatus(f)
  const returnVal = f.ann_return ?? f.ls_return

  const metricsContent = status === 'failed' && !f.sharpe ? (
    <Text type="secondary">因子未通过回测，无指标数据</Text>
  ) : (
    <>
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
      <Divider style={{ margin: '10px 0' }} />
      <Descriptions size="small" column={4} bordered items={[
        {
          key: 'ann_return',
          label: '年化收益（L/S）',
          children: (
            <Text style={{ color: returnVal && returnVal > 0 ? '#52c41a' : '#f5222d', fontWeight: 600 }}>
              {returnVal !== null && returnVal !== undefined ? pct(returnVal) : '—'}
            </Text>
          ),
        },
        {
          key: 'sharpe',
          label: 'Sharpe',
          children: (
            <Text strong style={{ color: (f.sharpe ?? 0) >= 1 ? '#52c41a' : '#faad14' }}>
              {fmt(f.sharpe, 3)}
            </Text>
          ),
        },
        { key: 'IC',        label: 'IC',        children: fmt(f.IC, 4) },
        { key: 'ICIR',      label: 'ICIR',      children: fmt(f.icir, 4) },
        { key: 'rank_ic',   label: 'Rank IC',   children: fmt(f.rank_ic, 4) },
        { key: 'rank_icir', label: 'Rank ICIR', children: fmt(f.rank_icir, 4) },
        { key: 'coverage',  label: 'Coverage',  children: fmt(f.coverage, 3) },
        { key: 'turnover',  label: 'Turnover',  children: fmt(f.turnover, 3) },
      ]} />
    </>
  )

  const exprContent = (
    <>
      {f.expression ? (
        <div style={{ background: '#1a1a2e', borderRadius: 6, padding: '10px 14px', marginBottom: 8 }}>
          <Text code style={{ color: '#a8d8ff', fontSize: 12, fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {f.expression}
          </Text>
        </div>
      ) : (
        <Text type="secondary">无表达式记录</Text>
      )}
      {f.formulation && (
        <>
          <Text type="secondary" style={{ fontSize: 12 }}>公式描述：</Text>
          <Paragraph style={{ fontSize: 12, marginTop: 4, marginBottom: 0 }}>{f.formulation}</Paragraph>
        </>
      )}
      {f.description && (
        <>
          <Divider style={{ margin: '8px 0' }} />
          <Text type="secondary" style={{ fontSize: 12 }}>因子说明：</Text>
          <Paragraph style={{ fontSize: 12, marginTop: 4, marginBottom: 0 }}>{f.description}</Paragraph>
        </>
      )}
    </>
  )

  const collapseItems = [
    {
      key: 'metrics',
      label: <Space><BarChartOutlined style={{ color: '#1677ff' }} /><Text strong>指标详情</Text></Space>,
      children: metricsContent,
    },
    {
      key: 'expr',
      label: <Space><CodeOutlined style={{ color: '#722ed1' }} /><Text strong>因子组成 / 表达式</Text></Space>,
      children: exprContent,
    },
    ...(f.hypothesis ? [{
      key: 'source',
      label: <Space><BulbOutlined style={{ color: '#fa8c16' }} /><Text strong>来源假说</Text></Space>,
      children: (
        <>
          <Paragraph style={{ fontSize: 12, margin: 0 }}>{f.hypothesis}</Paragraph>
          <div style={{ marginTop: 6 }}>
            <Tag color={PHASE_COLOR[f.phase] ?? 'default'}>{f.phase || '—'}</Tag>
            <Text type="secondary" style={{ fontSize: 11 }}>轨迹评估 {f.loop_index}  ·  Round {f.round_idx}  ·  轨迹 {f.trajectory_id?.slice(-8) ?? '—'}</Text>
          </div>
        </>
      ),
    }] : []),
    ...(f.feedback_summary ? [{
      key: 'feedback',
      label: <Space><CommentOutlined style={{ color: '#eb2f96' }} /><Text strong>问题总结 / 反馈</Text></Space>,
      children: <Paragraph style={{ fontSize: 12, margin: 0 }}>{formatFeedbackSummary(f.feedback_summary)}</Paragraph>,
    }] : []),
  ]

  return (
    <div style={{ padding: '8px 16px 16px', background: '#fafafa' }}>
      <Collapse ghost size="small" defaultActiveKey={['metrics', 'source']} items={collapseItems} />
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function FactorDashboard() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const { data: run } = useSWR<RunDetail>(
    runId ? `/api/runs/${runId}` : null,
    fetcher,
    { refreshInterval: 5000 },
  )
  const refreshWhileRunning = run?.status === 'running' ? 3000 : 0
  const { data: factors, isLoading } = useSWR<Factor[]>(
    runId ? `/api/runs/${runId}/factors` : null,
    fetcher,
    { refreshInterval: refreshWhileRunning }
  )

  const all = factors ?? []
  const succeeded = all.filter((f) => factorStatus(f) === 'success')
  const partial   = all.filter((f) => factorStatus(f) === 'partial')
  const failed    = all.filter((f) => factorStatus(f) === 'failed')
  // 指标达标集合（成功 + 部分）：图表与最优因子都基于它
  const metricPassed = [...succeeded, ...partial]

  const bestSharpe = metricPassed.reduce<Factor | null>(
    (acc, f) => (!acc || (f.sharpe ?? -99) > (acc.sharpe ?? -99) ? f : acc), null
  )

  const { topBySharepe, loopTrend, scatter } = useChartData(all)

  const columns = [
    {
      title: '状态',
      width: 90,
      render: (_: unknown, f: Factor) => {
        const s = factorStatus(f)
        const m = STATUS_META[s]
        const failedChecks = (f.admission_failed_checks ?? []).filter(Boolean)
        const tooltip = s === 'partial' && failedChecks.length
          ? `${m.label} · 未过入库检查：${failedChecks.join('、')}`
          : m.label
        return (
          <Tooltip title={tooltip}>
            <Space size={4}>
              {m.icon}
              <Text style={{ fontSize: 12, color: m.color }}>{m.label.split('（')[0]}</Text>
            </Space>
          </Tooltip>
        )
      },
      filters: [
        { text: '成功（达标且入库）', value: 'success' },
        { text: '部分（达标未入库）', value: 'partial' },
        { text: '失败', value: 'failed' },
      ],
      onFilter: (v: unknown, f: Factor) => factorStatus(f) === v,
    },
    {
      title: '因子名',
      dataIndex: 'factor_name',
      render: (name: string, f: Factor) => (
        <Space size={6} wrap>
          <Text strong style={{ fontFamily: 'monospace', fontSize: 12 }}>{name}</Text>
          {(f.duplicate_expression_count ?? 0) > 1 && (
            <Tag color="orange" style={{ marginInlineEnd: 0 }}>重复 x{f.duplicate_expression_count}</Tag>
          )}
        </Space>
      ),
    },
    {
      title: 'Reward',
      dataIndex: 'reward_score',
      render: (v: number | null) => {
        if (v === null || v === undefined) return <Text type="secondary">-</Text>
        const color = v >= 0.5 ? '#389e0d' : v >= 0.3 ? '#52c41a' : v >= 0 ? '#faad14' : '#f5222d'
        return <Text style={{ color, fontWeight: 700 }}>{fmt(v, 4)}</Text>
      },
      sorter: (a: Factor, b: Factor) => (a.reward_score ?? -99) - (b.reward_score ?? -99),
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe',
      render: (v: number | null) => {
        if (v === null) return <Text type="secondary">—</Text>
        const color = v >= 1.5 ? '#52c41a' : v >= 0.8 ? '#faad14' : '#f5222d'
        return <Text style={{ color, fontWeight: 700 }}>{fmt(v, 3)}</Text>
      },
      sorter: (a: Factor, b: Factor) => (a.sharpe ?? -99) - (b.sharpe ?? -99),
      defaultSortOrder: 'descend' as const,
    },
    {
      title: '年化收益',
      render: (_: unknown, f: Factor) => {
        const v = f.ann_return ?? f.ls_return
        if (v === null || v === undefined) return <Text type="secondary">—</Text>
        const color = v > 0 ? '#52c41a' : '#f5222d'
        return <Text style={{ color, fontWeight: 600 }}>{pct(v)}</Text>
      },
    },
    { title: 'IC', dataIndex: 'IC', render: (v: number | null) => fmt(v, 4) },
    { title: 'ICIR', dataIndex: 'icir', render: (v: number | null) => fmt(v, 4) },
    { title: 'Rank IC', dataIndex: 'rank_ic', render: (v: number | null) => fmt(v, 4) },
    {
      title: '阶段',
      dataIndex: 'phase',
      render: (p: string) => p ? <Tag color={PHASE_COLOR[p] ?? 'default'} style={{ fontSize: 11 }}>{p}</Tag> : '—',
      filters: ['original', 'mutation', 'crossover', 'evolved'].map((p) => ({ text: p, value: p })),
      onFilter: (v: unknown, f: Factor) => f.phase === v,
    },
    {
      title: '轨迹评估',
      dataIndex: 'loop_index',
      sorter: (a: Factor, b: Factor) => a.loop_index - b.loop_index,
    },
  ]

  return (
    <div>
      <Row align="middle" justify="space-between" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={4} style={{ margin: 0 }}>
            因子运行记录 <Text code style={{ fontSize: 13 }}>{runId}</Text>
          </Title>
        </Col>
        <Col>
          <Space>
            <Button icon={<MonitorOutlined />} onClick={() => navigate(`/runs/${runId}`)}>监控</Button>
            <Button icon={<ExperimentOutlined />} type="primary" onClick={() => navigate('/factor-library')}>因子库</Button>
            <Button icon={<BranchesOutlined />} onClick={() => navigate(`/runs/${runId}/lineage`)}>进化树</Button>
            <Button icon={<FileTextOutlined />} onClick={() => navigate(`/runs/${runId}/report`)}>报告</Button>
          </Space>
        </Col>
      </Row>

      {/* Summary cards */}
      <Row gutter={12} style={{ marginBottom: 20 }}>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#f0f0f0' }}>
            <Statistic title="总因子数" value={all.length} loading={isLoading} />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#b7eb8f' }}>
            <Statistic
              title={(
                <Tooltip title={`Sharpe>${SUCCESS_SHARPE_THRESHOLD} 或 |IC|>${SUCCESS_ABS_IC_THRESHOLD}，且通过 9 项入库标准（已录入因子库）`}>
                  <Space size={4}><CheckCircleFilled style={{ color: '#52c41a' }} />成功（入库）</Space>
                </Tooltip>
              )}
              value={succeeded.length}
              valueStyle={{ color: '#52c41a' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#ffe58f' }}>
            <Statistic
              title={(
                <Tooltip title={`Sharpe>${SUCCESS_SHARPE_THRESHOLD} 或 |IC|>${SUCCESS_ABS_IC_THRESHOLD}，但未通过入库标准`}>
                  <Space size={4}><MinusCircleFilled style={{ color: '#faad14' }} />部分（达标未入库）</Space>
                </Tooltip>
              )}
              value={partial.length}
              valueStyle={{ color: '#faad14' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#ffccc7' }}>
            <Statistic
              title={(
                <Tooltip title="实现失败，或 Sharpe 与 |IC| 两个指标都未达标">
                  <Space size={4}><CloseCircleFilled style={{ color: '#f5222d' }} />失败</Space>
                </Tooltip>
              )}
              value={failed.length}
              valueStyle={{ color: '#f5222d' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small" style={{ borderColor: '#d9d9d9' }}>
            {bestSharpe ? (
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>最优因子</Text>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Tooltip title={bestSharpe.factor_name}>
                    <Text strong style={{ fontFamily: 'monospace', fontSize: 12, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'inline-block' }}>
                      {bestSharpe.factor_name}
                    </Text>
                  </Tooltip>
                  <Tag color="success" style={{ flexShrink: 0, marginLeft: 0 }}>Sharpe {fmt(bestSharpe.sharpe, 3)}</Tag>
                </div>
              </Space>
            ) : (
              <Text type="secondary">暂无达标因子</Text>
            )}
          </Card>
        </Col>
      </Row>

      {/* Charts section：成功 + 部分（指标达标即展示，不要求已入库） */}
      {metricPassed.length > 0 && (
        <Collapse
          defaultActiveKey={['charts']}
          style={{ marginBottom: 20, borderRadius: 8 }}
          items={[{
            key: 'charts',
            label: <Space><BarChartOutlined style={{ color: '#1677ff' }} /><Text strong>图表概览（成功 + 部分）</Text><Badge count={metricPassed.length} style={{ backgroundColor: '#1677ff' }} /></Space>,
            children: (
              <>
                <Row gutter={16}>
                  <Col span={10}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                      Top 15 因子 Sharpe（按阶段着色）
                    </Text>
                    <SharpeBarChart data={topBySharepe} />
                  </Col>
                  <Col span={7}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                      各轨迹评估均值指标趋势
                    </Text>
                    <LoopTrendChart data={loopTrend} />
                  </Col>
                  <Col span={7}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                      IC vs Sharpe 散点（按阶段着色）
                    </Text>
                    <ICScatterChart data={scatter} />
                  </Col>
                </Row>
                <div style={{ marginTop: 10, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  {(['original', 'mutation', 'crossover', 'evolved'] as const).map((phase) => (
                    <Space key={phase} size={4}>
                      <div style={{ width: 10, height: 10, borderRadius: 2, background: PHASE_BAR_COLOR[phase] }} />
                      <Text type="secondary" style={{ fontSize: 11 }}>{phase}</Text>
                    </Space>
                  ))}
                </div>
              </>
            ),
          }]}
        />
      )}

      <Table
        rowKey="factor_id"
        columns={columns}
        dataSource={all}
        loading={isLoading}
        size="small"
        pagination={{ pageSize: 30, showSizeChanger: true }}
        locale={{ emptyText: <Empty description="暂无因子记录" /> }}
        expandable={{
          expandedRowRender: (f: Factor) => <FactorDetail f={f} />,
          rowExpandable: () => true,
          expandRowByClick: false,
        }}
        rowClassName={(f: Factor) => {
          const s = factorStatus(f)
          if (s === 'failed') return 'row-failed'
          if (s === 'success') return 'row-success'
          if (s === 'partial') return 'row-partial'
          return ''
        }}
      />

      <style>{`
        .row-failed td { background: #fff2f0 !important; }
        .row-success td { background: #f6ffed !important; }
        .row-partial td { background: #fffbe6 !important; }
        .row-failed:hover td { background: #ffe7e0 !important; }
        .row-success:hover td { background: #edfcd5 !important; }
        .row-partial:hover td { background: #fff3bf !important; }
      `}</style>
    </div>
  )
}
