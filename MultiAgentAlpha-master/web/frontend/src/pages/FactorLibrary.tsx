import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import {
  Table, Tag, Typography, Card, Statistic, Row, Col,
  Tooltip, Empty, Button, Space,
} from 'antd'
import { TrophyOutlined, ReloadOutlined } from '@ant-design/icons'
import FactorMiniChart from '../components/FactorMiniChart'

const { Title, Text } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

interface TopFactor {
  rank: number
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
  layer_5_cumulative_series: [string, number][] | null
}

function fmt(v: number | null | undefined, d = 4) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(d)
}
function pct(v: number | null | undefined) {
  return v === null || v === undefined ? '—' : (Number(v) * 100).toFixed(2) + '%'
}

export default function FactorLibrary() {
  const navigate = useNavigate()
  const { data: factors, isLoading, mutate } = useSWR<TopFactor[]>(
    '/api/factors/top?limit=30',
    fetcher,
    { revalidateOnFocus: false }
  )

  const best = factors?.[0]

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
      title: '因子名',
      dataIndex: 'factor_name',
      width: 200,
      render: (name: string, f: TopFactor) => (
        <Tooltip title={f.expression || '无表达式'}>
          <Text strong style={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>{name}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'Reward',
      dataIndex: 'reward_score',
      width: 90,
      render: (v: number | null) => {
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
  ]

  return (
    <div>
      <Row align="middle" justify="space-between" style={{ marginBottom: 20 }}>
        <Col>
          <Space align="center">
            <TrophyOutlined style={{ fontSize: 22, color: '#faad14' }} />
            <Title level={4} style={{ margin: 0 }}>因子库 — 全局排行榜</Title>
          </Space>
          <div><Text type="secondary" style={{ fontSize: 12 }}>跨所有历史运行，按 Reward 综合评分排名前 30（Rank IC 35% + ICIR 15% + Sharpe 10% + 覆盖率 10% − 换手/复杂度/相关性惩罚）</Text></div>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={() => mutate()}>刷新</Button>
        </Col>
      </Row>

      {best && (
        <Row gutter={12} style={{ marginBottom: 20 }}>
          <Col span={6}>
            <Card size="small" style={{ borderColor: '#faad14', background: '#fffbe6' }}>
              <Statistic title="🥇 最优 Reward" value={fmt(best.reward_score, 4)}
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
        scroll={{ x: 950 }}
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
                ls_return: f.ls_return, ann_return: f.ann_return,
                yearly_returns: f.yearly_returns,
                reward_score: f.reward_score,
                layer_1: f.layer_1, layer_2: f.layer_2, layer_3: f.layer_3,
                layer_4: f.layer_4, layer_5: f.layer_5,
                layer_monotonicity: f.layer_monotonicity,
                ls_cumulative_series: f.ls_cumulative_series,
                layer_1_cumulative_series: f.layer_1_cumulative_series,
                layer_5_cumulative_series: f.layer_5_cumulative_series,
              }} />
              <div style={{ marginTop: 10 }}>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  来源：运行 <Text code style={{ fontSize: 11 }}>{f.run_id}</Text> · Loop {f.loop_index}
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
