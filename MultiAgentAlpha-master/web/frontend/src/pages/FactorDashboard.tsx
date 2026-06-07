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
import type { Factor } from '../types'
import FactorMiniChart from '../components/FactorMiniChart'

const { Title, Text, Paragraph } = Typography
const { Panel } = Collapse

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

type FactorStatusKey = 'success' | 'failed' | 'partial' | 'below_threshold'

function factorStatus(f: Factor): FactorStatusKey {
  const impl = (f.implementation_status ?? '').toLowerCase()
  if (impl === 'failed' || impl === 'error') return 'failed'
  // 质量门槛：quality_gate_passed === 0 表示 Sharpe 未达到用户设定阈值
  if (f.quality_gate_passed !== null && f.quality_gate_passed !== undefined) {
    if (f.quality_gate_passed === 0) return 'below_threshold'
  }
  if (f.sharpe !== null && f.sharpe !== undefined) return 'success'
  if (impl === 'success' || impl === 'generated') return 'partial'
  return 'failed'
}

const STATUS_META: Record<FactorStatusKey, { color: string; icon: React.ReactNode; label: string }> = {
  success:         { color: '#52c41a', icon: <CheckCircleFilled style={{ color: '#52c41a' }} />, label: '成功' },
  partial:         { color: '#faad14', icon: <MinusCircleFilled style={{ color: '#faad14' }} />, label: '编码成功 / 未回测' },
  failed:          { color: '#f5222d', icon: <CloseCircleFilled style={{ color: '#f5222d' }} />, label: '失败' },
  below_threshold: { color: '#fa8c16', icon: <CloseCircleFilled style={{ color: '#fa8c16' }} />, label: '质量未达标' },
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
    const succeeded = factors.filter((f) => factorStatus(f) === 'success')

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
  if (!data.length) return <Empty description="暂无回测成功的因子" style={{ padding: 32 }} />
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
  if (data.length < 2) return <Empty description="需要至少 2 个 Loop 才能展示趋势" style={{ padding: 32 }} />
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 24, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="loop" label={{ value: 'Loop', position: 'insideBottomRight', offset: -4, fontSize: 11 }} tick={{ fontSize: 11 }} />
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

  return (
    <div style={{ padding: '8px 16px 16px', background: '#fafafa' }}>
      <Collapse ghost size="small" defaultActiveKey={['metrics', 'source']}>

        <Panel
          key="metrics"
          header={<Space><BarChartOutlined style={{ color: '#1677ff' }} /><Text strong>指标详情</Text></Space>}
        >
          {status === 'failed' && !f.sharpe ? (
            <Text type="secondary">因子未通过回测，无指标数据</Text>
          ) : (
            <>
            <FactorMiniChart metrics={{
              sharpe: f.sharpe, IC: f.IC, rank_ic: f.rank_ic,
              icir: f.icir, rank_icir: f.rank_icir,
              coverage: f.coverage, turnover: f.turnover,
              ls_return: f.ls_return, ann_return: f.ann_return,
              yearly_returns: f.yearly_returns,
              layer_1: f.layer_1, layer_2: f.layer_2, layer_3: f.layer_3,
              layer_4: f.layer_4, layer_5: f.layer_5,
              layer_monotonicity: f.layer_monotonicity,
              ls_cumulative_series: f.ls_cumulative_series,
              layer_1_cumulative_series: f.layer_1_cumulative_series,
              layer_5_cumulative_series: f.layer_5_cumulative_series,
            }} />
            <Divider style={{ margin: '10px 0' }} />
            <Descriptions size="small" column={4} bordered>
              <Descriptions.Item label="年化收益（L/S）">
                <Text style={{ color: returnVal && returnVal > 0 ? '#52c41a' : '#f5222d', fontWeight: 600 }}>
                  {returnVal !== null && returnVal !== undefined ? pct(returnVal) : '—'}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Sharpe">
                <Text strong style={{ color: (f.sharpe ?? 0) >= 1 ? '#52c41a' : '#faad14' }}>
                  {fmt(f.sharpe, 3)}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="IC">{fmt(f.IC, 4)}</Descriptions.Item>
              <Descriptions.Item label="ICIR">{fmt(f.icir, 4)}</Descriptions.Item>
              <Descriptions.Item label="Rank IC">{fmt(f.rank_ic, 4)}</Descriptions.Item>
              <Descriptions.Item label="Rank ICIR">{fmt(f.rank_icir, 4)}</Descriptions.Item>
              <Descriptions.Item label="Coverage">{fmt(f.coverage, 3)}</Descriptions.Item>
              <Descriptions.Item label="Turnover">{fmt(f.turnover, 3)}</Descriptions.Item>
            </Descriptions>
            </>
          )}
        </Panel>

        <Panel
          key="expr"
          header={<Space><CodeOutlined style={{ color: '#722ed1' }} /><Text strong>因子组成 / 表达式</Text></Space>}
        >
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
        </Panel>

        {f.hypothesis && (
          <Panel
            key="source"
            header={<Space><BulbOutlined style={{ color: '#fa8c16' }} /><Text strong>来源假说</Text></Space>}
          >
            <Paragraph style={{ fontSize: 12, margin: 0 }}>{f.hypothesis}</Paragraph>
            <div style={{ marginTop: 6 }}>
              <Tag color={PHASE_COLOR[f.phase] ?? 'default'}>{f.phase || '—'}</Tag>
              <Text type="secondary" style={{ fontSize: 11 }}>Loop {f.loop_index}  ·  Round {f.round_idx}  ·  轨迹 {f.trajectory_id?.slice(-8) ?? '—'}</Text>
            </div>
          </Panel>
        )}

        {f.feedback_summary && (
          <Panel
            key="feedback"
            header={<Space><CommentOutlined style={{ color: '#eb2f96' }} /><Text strong>问题总结 / 反馈</Text></Space>}
          >
            <Paragraph style={{ fontSize: 12, margin: 0 }}>{f.feedback_summary}</Paragraph>
          </Panel>
        )}

      </Collapse>
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function FactorDashboard() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const { data: factors, isLoading } = useSWR<Factor[]>(
    runId ? `/api/runs/${runId}/factors` : null,
    fetcher
  )

  const all = factors ?? []
  const succeeded = all.filter((f) => factorStatus(f) === 'success')
  const partial   = all.filter((f) => factorStatus(f) === 'partial')
  const failed    = all.filter((f) => factorStatus(f) === 'failed')

  const bestSharpe = succeeded.reduce<Factor | null>(
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
        return (
          <Tooltip title={m.label}>
            <Space size={4}>
              {m.icon}
              <Text style={{ fontSize: 12, color: m.color }}>{m.label.split(' ')[0]}</Text>
            </Space>
          </Tooltip>
        )
      },
      filters: [
        { text: '成功', value: 'success' },
        { text: '失败', value: 'failed' },
        { text: '部分', value: 'partial' },
      ],
      onFilter: (v: unknown, f: Factor) => factorStatus(f) === v,
    },
    {
      title: '因子名',
      dataIndex: 'factor_name',
      render: (name: string) => (
        <Text strong style={{ fontFamily: 'monospace', fontSize: 12 }}>{name}</Text>
      ),
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
      title: 'Loop',
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
            <Button icon={<ExperimentOutlined />} type="primary">因子库</Button>
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
              title={<Space size={4}><CheckCircleFilled style={{ color: '#52c41a' }} />成功</Space>}
              value={succeeded.length}
              valueStyle={{ color: '#52c41a' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#ffe58f' }}>
            <Statistic
              title={<Space size={4}><MinusCircleFilled style={{ color: '#faad14' }} />部分</Space>}
              value={partial.length}
              valueStyle={{ color: '#faad14' }}
              loading={isLoading}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small" style={{ borderColor: '#ffccc7' }}>
            <Statistic
              title={<Space size={4}><CloseCircleFilled style={{ color: '#f5222d' }} />失败</Space>}
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
              <Text type="secondary">暂无成功因子</Text>
            )}
          </Card>
        </Col>
      </Row>

      {/* Charts section */}
      {succeeded.length > 0 && (
        <Collapse
          defaultActiveKey={['charts']}
          style={{ marginBottom: 20, borderRadius: 8 }}
        >
          <Panel
            key="charts"
            header={<Space><BarChartOutlined style={{ color: '#1677ff' }} /><Text strong>图表概览</Text><Badge count={succeeded.length} style={{ backgroundColor: '#1677ff' }} /></Space>}
          >
            <Row gutter={16}>
              <Col span={10}>
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                  Top 15 因子 Sharpe（按阶段着色）
                </Text>
                <SharpeBarChart data={topBySharepe} />
              </Col>
              <Col span={7}>
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                  各 Loop 均值指标趋势
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
          </Panel>
        </Collapse>
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
          return ''
        }}
      />

      <style>{`
        .row-failed td { background: #fff2f0 !important; }
        .row-success td { background: #f6ffed !important; }
        .row-failed:hover td { background: #ffe7e0 !important; }
        .row-success:hover td { background: #edfcd5 !important; }
      `}</style>
    </div>
  )
}
