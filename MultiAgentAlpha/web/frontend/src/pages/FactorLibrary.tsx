import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import {
  Table, Tag, Typography, Card, Statistic, Row, Col,
  Tooltip, Empty, Button, Space, Popconfirm, message,
} from 'antd'
import { TrophyOutlined, ReloadOutlined, EyeInvisibleOutlined, CalculatorOutlined, SafetyCertificateOutlined, CheckCircleOutlined } from '@ant-design/icons'
import FactorMiniChart from '../components/FactorMiniChart'
import type { RewardBreakdownData } from '../components/RewardBreakdown'

const { Title, Text } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

interface TopFactor {
  rank: number
  admission_passed?: boolean
  admission_failed_checks?: string[]
  factor_id: string
  run_id: string
  run_name: string
  loop_index: number
  factor_name: string
  expression: string
  description: string
  implementation_status: string
  reward_score: number | null
  sharpe: number | null
  IC: number | null
  rank_ic: number | null
  icir: number | null
  rank_icir: number | null
  coverage: number | null
  turnover: number | null
  valid_cross_section_mean: number | null
  cross_section_universe_size: number | null
  ic_win_rate: number | null
  rank_ic_win_rate: number | null
  drawdown: number | null
  long_short_positive_ratio: number | null
  ls_return: number | null
  ann_return: number | null
  yearly_returns: Record<string, number> | null
  layer_1: number | null
  layer_2: number | null
  layer_3: number | null
  layer_4: number | null
  layer_5: number | null
  layer_monotonicity: number | null
  ls_cumulative_series: [string, number][] | null
  layer_1_cumulative_series: [string, number][] | null
  layer_2_cumulative_series: [string, number][] | null
  layer_3_cumulative_series: [string, number][] | null
  layer_4_cumulative_series: [string, number][] | null
  layer_5_cumulative_series: [string, number][] | null
  benchmark_cumulative_series: [string, number][] | null
  ic_cumulative_series: [string, number][] | null
  rank_ic_cumulative_series: [string, number][] | null
  duplicate_expression_count?: number | null
  duplicate_trajectory_ids?: string[] | null
  factor_reward_score?: number | null
  reward_breakdown?: RewardBreakdownData | null
}

function fmt(v: number | null | undefined, d = 4) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(d)
}
function pct(v: number | null | undefined) {
  return v === null || v === undefined ? '—' : (Number(v) * 100).toFixed(2) + '%'
}

function factorReward(f: TopFactor): number | null {
  return f.factor_reward_score ?? f.reward_score
}

// 入库硬门标准（与后端 evaluate_admission_gate 对齐）
const ADMISSION_TIERS = [
  {
    idx: '①', title: '基础有效性', color: '#fa541c', bg: '#fff7f0', border: '#ffd8bf',
    items: [
      { name: '有效截面数均值', op: '≥ 30', note: '或池 50%' },
      { name: '|IC| 均值', op: '> 0.03' },
      { name: 'IC 胜率', op: '> 52%' },
    ],
  },
  {
    idx: '②', title: '预测稳定性', color: '#1677ff', bg: '#f0f7ff', border: '#bcdcff',
    items: [
      { name: '|ICIR|', op: '> 0.05' },
      { name: '|RankICIR|', op: '> 0.05' },
      { name: 'RankIC 胜率', op: '> 55%' },
    ],
  },
  {
    idx: '③', title: '策略表现', color: '#389e0d', bg: '#f5fff0', border: '#c5ecab',
    items: [
      { name: '多空夏普', op: '> 1.0', note: '可调' },
      { name: '最大回撤', op: '< 30%' },
      { name: '胜率', op: '> 55%' },
    ],
  },
] as const

export default function FactorLibrary() {
  const navigate = useNavigate()
  const [clearing, setClearing] = useState(false)
  const [hidingKey, setHidingKey] = useState<string | null>(null)
  const { data: factors, isLoading, mutate } = useSWR<TopFactor[]>(
    '/api/factors/top?limit=30',
    fetcher,
    { revalidateOnFocus: false }
  )

  const best = factors?.[0]

  const hideFactor = async (f: TopFactor) => {
    const key = `${f.run_id}:${f.factor_id}`
    setHidingKey(key)
    try {
      const res = await fetch(`/api/factors/${encodeURIComponent(f.run_id)}/${encodeURIComponent(f.factor_id)}/hide`, {
        method: 'POST',
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.detail ?? data.message ?? '移出排行榜失败')
      }
      message.success(data.message ?? '已从排行榜隐藏，历史记录仍保留')
      mutate()
    } catch (err) {
      message.error(err instanceof Error ? err.message : '移出排行榜失败')
    } finally {
      setHidingKey(null)
    }
  }

  const columns = [
    {
      title: '排名',
      dataIndex: 'rank',
      width: 60,
      render: (r: number) => (
        <Text strong style={{
          color: r === 1 ? '#faad14' : r === 2 ? '#8c8c8c' : r === 3 ? '#ad6800' : '#595959',
          fontSize: r <= 3 ? 15 : 13,
        }}>
          {r <= 3 ? ['🥇', '🥈', '🥉'][r - 1] : `#${r}`}
        </Text>
      ),
    },
    {
      title: '状态',
      width: 100,
      render: (_: unknown, f: TopFactor) => {
        // 旧数据无 admission_passed 字段时按已入库处理（旧排行榜只返回入库因子）
        const admitted = f.admission_passed !== false
        if (admitted) return <Tag color="green" style={{ marginInlineEnd: 0 }}>已入库</Tag>
        const failedChecks = (f.admission_failed_checks ?? []).filter(Boolean)
        return (
          <Tooltip title={failedChecks.length ? `未过入库检查：${failedChecks.join('、')}` : '过资格线（Sharpe>1.0 或 |IC|>0.02）但未通过 9 项入库标准'}>
            <Tag color="gold" style={{ marginInlineEnd: 0 }}>达标未入库</Tag>
          </Tooltip>
        )
      },
      filters: [
        { text: '已入库', value: 'admitted' },
        { text: '达标未入库', value: 'qualified' },
      ],
      onFilter: (v: unknown, f: TopFactor) => (v === 'admitted') === (f.admission_passed !== false),
    },
    {
      title: '因子名',
      dataIndex: 'factor_name',
      width: 200,
      render: (name: string, f: TopFactor) => (
        <Tooltip title={f.expression || '无表达式'}>
          <Space size={6} wrap>
            <Text strong style={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>{name}</Text>
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
      width: 90,
      render: (_v: number | null, f: TopFactor) => {
        const v = factorReward(f)
        if (v === null) return <Text type="secondary">—</Text>
        const color = v >= 0.5 ? '#389e0d' : v >= 0.3 ? '#52c41a' : v >= 0 ? '#faad14' : '#f5222d'
        return <Text style={{ color, fontWeight: 700, fontSize: 14 }}>{fmt(v, 4)}</Text>
      },
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe',
      width: 85,
      render: (v: number | null) => {
        if (v === null) return <Text type="secondary">—</Text>
        const color = v >= 2 ? '#389e0d' : v >= 1 ? '#52c41a' : '#faad14'
        return <Text style={{ color, fontWeight: 600 }}>{fmt(v, 3)}</Text>
      },
    },
    {
      title: '年化收益',
      dataIndex: 'ann_return',
      width: 90,
      render: (v: number | null, f: TopFactor) => {
        const val = v ?? f.ls_return
        if (val === null) return '—'
        return <Text style={{ color: val > 0 ? '#52c41a' : '#f5222d', fontWeight: 600 }}>{pct(val)}</Text>
      },
    },
    { title: 'IC',        dataIndex: 'IC',        width: 75, render: (v: number | null) => fmt(v, 4) },
    { title: 'ICIR',      dataIndex: 'icir',      width: 75, render: (v: number | null) => fmt(v, 4) },
    { title: 'Rank IC',   dataIndex: 'rank_ic',   width: 85, render: (v: number | null) => fmt(v, 4) },
    { title: 'Rank ICIR', dataIndex: 'rank_icir', width: 90, render: (v: number | null) => fmt(v, 4) },
    { title: 'Coverage',  dataIndex: 'coverage',  width: 80, render: (v: number | null) => fmt(v, 3) },
    {
      title: '来源',
      dataIndex: 'run_name',
      width: 140,
      render: (name: string, f: TopFactor) => (
        <Button type="link" size="small" style={{ padding: 0, fontFamily: 'monospace', fontSize: 11 }}
          onClick={() => navigate(`/runs/${f.run_id}/factors`)}>
          {name !== f.run_id ? name : name.slice(0, 15)}
        </Button>
      ),
    },
    {
      title: '',
      width: 64,
      render: (_: unknown, f: TopFactor) => {
        const key = `${f.run_id}:${f.factor_id}`
        return (
          <Popconfirm
            title="从排行榜隐藏这个因子？"
            description="只是不再显示在因子库排行榜里，历史记录和运行详情仍会保留。"
            okText="隐藏"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => hideFactor(f)}
          >
            <Button
              danger
              type="text"
              size="small"
              icon={<EyeInvisibleOutlined />}
              loading={hidingKey === key}
            />
          </Popconfirm>
        )
      },
    },
  ]

  return (
    <div>
      <Row align="middle" justify="space-between" style={{ marginBottom: 20 }}>
        <Col>
          <Space align="center">
            <TrophyOutlined style={{ fontSize: 22, color: '#faad14' }} />
            <Title level={4} style={{ margin: 0 }}>因子库 — 全局排行榜</Title>
          </Space>
          <div><Text type="secondary" style={{ fontSize: 12 }}>跨所有历史运行，按 Reward 综合评分排名前 30 的因子（含「已入库」与「达标未入库」，评分公式见下）</Text></div>
        </Col>
        <Col>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => mutate()}>刷新</Button>
            <Popconfirm
              title="清空排行榜？"
              description="只会从因子库排行榜隐藏全部因子，历史记录和运行详情仍会保留。"
              onConfirm={async () => {
                setClearing(true)
                try {
                  const res = await fetch('/api/factors', { method: 'DELETE' })
                  const data = await res.json()
                  message.success(data.message ?? '排行榜已清空，历史记录仍保留')
                  mutate()
                } catch {
                  message.error('清空排行榜失败')
                } finally {
                  setClearing(false)
                }
              }}
              okText="确认隐藏"
              cancelText="取消"
              okButtonProps={{ danger: true }}
            >
              <Button danger icon={<EyeInvisibleOutlined />} loading={clearing}>清空排行榜</Button>
            </Popconfirm>
          </Space>
        </Col>
      </Row>

      <Card
        size="small"
        style={{
          marginBottom: 16,
          borderRadius: 12,
          border: '1px solid #e6ebf5',
          background: 'linear-gradient(135deg, #f4fbf6 0%, #f5f9ff 100%)',
          boxShadow: '0 1px 3px rgba(20,40,80,0.04)',
        }}
        styles={{ body: { padding: '14px 18px' } }}
      >
        <Space align="center" style={{ marginBottom: 12 }} wrap>
          <SafetyCertificateOutlined style={{ color: '#52c41a', fontSize: 16 }} />
          <Text strong style={{ fontSize: 14 }}>入库标准 · 硬门</Text>
          <Tag color="error" style={{ marginInlineEnd: 0, fontWeight: 400 }}>任一不达标直接淘汰</Tag>
          <Tag color="default" style={{ marginInlineEnd: 0, fontWeight: 400 }}>缺数据则跳过该项</Tag>
        </Space>
        <Row gutter={[12, 12]}>
          {ADMISSION_TIERS.map((tier) => (
            <Col xs={24} md={8} key={tier.title}>
              <div style={{
                background: tier.bg,
                border: `1px solid ${tier.border}`,
                borderRadius: 9,
                padding: '10px 12px',
                height: '100%',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <span style={{ color: tier.color, fontWeight: 700, fontSize: 15 }}>{tier.idx}</span>
                  <Text strong style={{ color: tier.color, fontSize: 13 }}>{tier.title}</Text>
                </div>
                {tier.items.map((item, i) => (
                  <div key={item.name} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '5px 0',
                    borderTop: i > 0 ? '1px dashed rgba(0,0,0,0.06)' : 'none',
                  }}>
                    <Text style={{ fontSize: 12.5 }}>
                      <CheckCircleOutlined style={{ color: tier.color, fontSize: 11, marginRight: 6 }} />
                      {item.name}
                    </Text>
                    <span style={{ fontFamily: 'monospace', fontWeight: 700, color: tier.color, fontSize: 12.5, whiteSpace: 'nowrap' }}>
                      {item.op}
                      {'note' in item && item.note && (
                        <Text type="secondary" style={{ fontSize: 10, marginLeft: 4, fontWeight: 400 }}>{item.note}</Text>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </Col>
          ))}
        </Row>
        <div style={{ marginTop: 12, fontSize: 11.5, color: '#8c8c8c', lineHeight: 1.7 }}>
          <Text type="secondary" style={{ fontSize: 11.5 }}>
            共 3 档 9 项必要条件，<Text strong style={{ fontSize: 11.5 }}>任一「有数据且不达标」即淘汰</Text>；某项<Text strong style={{ fontSize: 11.5 }}>缺数据则跳过</Text>该项，不因缺数据误杀好因子。
            预测能力（IC/ICIR）按<Text strong style={{ fontSize: 11.5 }}>绝对值</Text>判定；IC/ICIR/RankICIR 均按<Text strong style={{ fontSize: 11.5 }}>预测窗口非重叠采样</Text>计算（与回测 / Reward 同口径）。「鲁棒性」（参数平原 / 分箱单调性）暂不强制。
          </Text>
        </div>
      </Card>

      <Card
        size="small"
        style={{
          marginBottom: 20,
          borderRadius: 12,
          border: '1px solid #e6ebf5',
          background: 'linear-gradient(135deg, #f5f9ff 0%, #fffaf4 100%)',
          boxShadow: '0 1px 3px rgba(20,40,80,0.04)',
        }}
        styles={{ body: { padding: '14px 18px' } }}
      >
        <Space align="center" style={{ marginBottom: 12 }} wrap>
          <CalculatorOutlined style={{ color: '#1677ff', fontSize: 16 }} />
          <Text strong style={{ fontSize: 14 }}>Reward 综合评分公式</Text>
          <Tag color="blue" style={{ marginInlineEnd: 0, fontWeight: 400 }}>主指标归一化到 [0,1]，再加联合进度与惩罚</Tag>
        </Space>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, fontFamily: 'monospace' }}>
          <Text strong style={{ fontSize: 15, marginInlineEnd: 4 }}>Reward =</Text>
          {[
            { w: '+1.0', label: 'Sharpe归一化' },
            { w: '+1.0', label: '|IC|归一化' },
            { w: '+1.0', label: '|RankIC|归一化' },
            { w: '+1.0', label: '|ICIR|归一化' },
            { w: '+1.0', label: '联合进度' },
          ].map((t) => (
            <span key={t.label} style={{
              display: 'inline-flex', alignItems: 'baseline', gap: 5,
              padding: '4px 10px', borderRadius: 7,
              background: '#f6ffed', border: '1px solid #b7eb8f',
            }}>
              <Text strong style={{ color: '#389e0d', fontSize: 13 }}>{t.w}</Text>
              <Text style={{ color: '#8c8c8c', fontSize: 11 }}>×</Text>
              <Text style={{ color: '#237804', fontSize: 13, fontWeight: 600 }}>{t.label}</Text>
            </span>
          ))}
          {[
            { w: '−0.35', label: '回撤' },
            { w: '−0.25', label: '换手' },
            { w: '−0.25', label: '复杂度' },
            { w: '−0.35', label: '冗余' },
            { w: '−0.25', label: '相关性' },
          ].map((t) => (
            <span key={t.label} style={{
              display: 'inline-flex', alignItems: 'baseline', gap: 5,
              padding: '4px 10px', borderRadius: 7,
              background: '#fff1f0', border: '1px solid #ffccc7',
            }}>
              <Text strong style={{ color: '#cf1322', fontSize: 13 }}>{t.w}</Text>
              <Text style={{ color: '#8c8c8c', fontSize: 11 }}>×</Text>
              <Text style={{ color: '#a8071a', fontSize: 13, fontWeight: 600 }}>{t.label}</Text>
            </span>
          ))}
        </div>
        <div style={{ marginTop: 12, fontSize: 11.5, color: '#8c8c8c', lineHeight: 1.7 }}>
          <Text type="secondary" style={{ fontSize: 11.5 }}>主目标归一化（绿，相对硬目标 Sharpe&gt;1.5、|IC|&gt;0.03、|RankIC|&gt;0.03）：</Text>
          <Text code style={{ fontSize: 11 }}>Sharpe归一化 = Sharpe/1.5 截[-1,1]</Text>（负 Sharpe 真实扣分）、
          <Text code style={{ fontSize: 11 }}>|IC|归一化 = |IC|/0.03 截[0,1]</Text>、
          <Text code style={{ fontSize: 11 }}>|RankIC|归一化 = |RankIC|/0.03 截[0,1]</Text>、
          <Text code style={{ fontSize: 11 }}>|ICIR|归一化 = |ICIR|/0.05 截[0,1]</Text>；
          <Text code style={{ fontSize: 11 }}>联合进度 = min(max(Sharpe归一化,0), |IC|归一化, |RankIC|归一化)</Text> 防止单指标偏科。
          <br />
          <Text type="secondary" style={{ fontSize: 11.5 }}>观察项：</Text>
          RankICIR、Coverage 仍展示并参与入库检查，但当前不进入 Reward 加权（ICIR 已计权 +1.0）。
          <br />
          <Text type="secondary" style={{ fontSize: 11.5 }}>惩罚（红）：</Text>
          回撤 <Text code style={{ fontSize: 11 }}>/0.30</Text>、换手 <Text code style={{ fontSize: 11 }}>/1.0</Text>，复杂度/冗余/相关性截 <Text code style={{ fontSize: 11 }}>[0,1]</Text>。
          入库分 = <Text code style={{ fontSize: 11 }}>0.75×训练 + 0.25×验证</Text>。
        </div>
      </Card>

      {best && (
        <Row gutter={12} style={{ marginBottom: 20 }}>
          <Col span={6}>
            <Card size="small" style={{ borderColor: '#faad14', background: '#fffbe6' }}>
              <Statistic title="🥇 最优 Reward" value={fmt(factorReward(best), 4)}
                valueStyle={{ color: '#d48806', fontWeight: 700 }} />
              <Text style={{ fontSize: 11, fontFamily: 'monospace' }}>{best.factor_name}</Text>
            </Card>
          </Col>
          <Col span={4}>
            <Card size="small">
              <Statistic title="最优 Sharpe" value={fmt(best.sharpe, 3)} />
            </Card>
          </Col>
          <Col span={4}>
            <Card size="small">
              <Statistic title="最优年化收益" value={pct(best.ann_return ?? best.ls_return)} />
            </Card>
          </Col>
          <Col span={4}>
            <Card size="small">
              <Statistic title="最优 IC" value={fmt(factors?.reduce((b, f) =>
                (f.IC ?? -99) > (b.IC ?? -99) ? f : b, best).IC, 4)} />
            </Card>
          </Col>
          <Col span={4}>
            <Card size="small">
              <Statistic title="上榜因子数" value={factors?.length ?? 0} />
            </Card>
          </Col>
        </Row>
      )}

      <Table
        rowKey={(f) => `${f.run_id}:${f.factor_id}`}
        columns={columns}
        dataSource={factors ?? []}
        loading={isLoading}
        size="small"
        pagination={false}
        scroll={{ x: 1020 }}
        locale={{ emptyText: <Empty description="暂无因子数据，完成运行后自动更新" /> }}
        rowClassName={(_, idx) => idx === 0 ? 'row-gold' : ''}
        expandable={{
          expandedRowRender: (f: TopFactor) => (
            <div style={{ padding: '6px 16px 12px', background: '#fafafa' }}>
              {f.expression && (
                <div style={{ background: '#1a1a2e', borderRadius: 5, padding: '8px 12px', marginBottom: 8 }}>
                  <Text code style={{ color: '#a8d8ff', fontSize: 12, fontFamily: 'monospace' }}>
                    {f.expression}
                  </Text>
                </div>
              )}
              {f.description && (
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>{f.description}</Text>
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
              <div style={{ marginTop: 10 }}>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  来源：运行 <Text code style={{ fontSize: 11 }}>{f.run_id}</Text> · 轨迹评估 {f.loop_index}
                </Text>
              </div>
            </div>
          ),
          rowExpandable: (f: TopFactor) => !!(f.expression || f.description || f.sharpe !== null),
        }}
      />

      <style>{`.row-gold td { background: #fffbe6 !important; }`}</style>
    </div>
  )
}
