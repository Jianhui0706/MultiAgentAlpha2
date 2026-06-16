import { Table, Tag, Button, Statistic, Row, Col, Card, Space, Tooltip } from 'antd'
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { Link, useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import type { Run } from '../types'

const fetcher = (url: string) => fetch(url).then((r) => r.json())

const STATUS_COLOR: Record<string, string> = {
  running: 'processing',
  completed: 'success',
  failed: 'error',
}

function fmt(iso: string | null) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}

export default function RunList() {
  const { data: runs, isLoading, mutate } = useSWR<Run[]>('/api/runs', fetcher, { refreshInterval: 5000 })
  const navigate = useNavigate()

  const columns = [
    {
      title: '运行 ID',
      dataIndex: 'run_id',
      render: (id: string) => (
        <Link to={`/runs/${id}`} style={{ fontFamily: 'monospace' }}>{id}</Link>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      render: (s: string) => <Tag color={STATUS_COLOR[s] ?? 'default'}>{s}</Tag>,
      filters: [
        { text: '运行中', value: 'running' },
        { text: '已完成', value: 'completed' },
        { text: '失败', value: 'failed' },
      ],
      onFilter: (v: unknown, r: Run) => r.status === v,
    },
    {
      title: '模式',
      dataIndex: 'evolution_mode',
      render: (v: boolean) => <Tag color={v ? 'purple' : 'blue'}>{v ? '进化模式' : '单循环'}</Tag>,
    },
    {
      title: '因子数',
      dataIndex: 'factor_count',
      sorter: (a: Run, b: Run) => a.factor_count - b.factor_count,
    },
    {
      title: '轨迹数',
      render: (_: unknown, r: Run) => r.trajectory_count ?? r.loop_count ?? 0,
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      render: fmt,
      sorter: (a: Run, b: Run) => (a.started_at > b.started_at ? 1 : -1),
      defaultSortOrder: 'descend' as const,
    },
    {
      title: '结束时间',
      dataIndex: 'ended_at',
      render: fmt,
    },
    {
      title: '操作',
      render: (_: unknown, r: Run) => (
        <Space>
          <Button size="small" onClick={() => navigate(`/runs/${r.run_id}`)}>详情</Button>
          <Button size="small" onClick={() => navigate(`/runs/${r.run_id}/factors`)}>因子</Button>
          <Tooltip title="进化树">
            <Button size="small" onClick={() => navigate(`/runs/${r.run_id}/lineage`)}>进化树</Button>
          </Tooltip>
        </Space>
      ),
    },
  ]

  const total = runs?.length ?? 0
  const running = runs?.filter((r) => r.status === 'running').length ?? 0
  const completed = runs?.filter((r) => r.status === 'completed').length ?? 0

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}><Card size="small"><Statistic title="总运行次数" value={total} /></Card></Col>
        <Col span={6}><Card size="small"><Statistic title="运行中" value={running} valueStyle={{ color: '#1677ff' }} /></Card></Col>
        <Col span={6}><Card size="small"><Statistic title="已完成" value={completed} valueStyle={{ color: '#52c41a' }} /></Card></Col>
        <Col span={6}>
          <Card size="small">
            <Space>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/runs/new')}>新建运行</Button>
              <Button icon={<ReloadOutlined />} onClick={() => mutate()}>刷新</Button>
            </Space>
          </Card>
        </Col>
      </Row>
      <Table
        rowKey="run_id"
        columns={columns}
        dataSource={runs ?? []}
        loading={isLoading}
        size="small"
        pagination={{ pageSize: 20 }}
      />
    </div>
  )
}
