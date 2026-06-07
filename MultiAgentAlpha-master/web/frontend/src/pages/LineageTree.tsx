import { useParams, useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { useMemo, useCallback, useEffect } from 'react'
import { Typography, Row, Col, Card, Tag, Empty, Spin, Space, Button } from 'antd'
import {
  MonitorOutlined, ExperimentOutlined, BranchesOutlined, FileTextOutlined,
} from '@ant-design/icons'
import {
  ReactFlow, Background, Controls, MiniMap,
  addEdge, useNodesState, useEdgesState,
  type Node, type Edge, type Connection,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { Trajectory } from '../types'

const { Title, Text } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

const PHASE_COLOR: Record<string, string> = {
  original: '#1677ff',
  mutation: '#fa8c16',
  crossover: '#52c41a',
  evolved: '#722ed1',
}

const PHASE_LABEL: Record<string, string> = {
  original: '初始',
  mutation: 'Mutation',
  crossover: 'Crossover',
  evolved: '进化',
}

function buildGraph(trajectories: Trajectory[]): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  // Group by round to determine layout
  const byRound: Record<number, Trajectory[]> = {}
  for (const t of trajectories) {
    ;(byRound[t.round_idx] ??= []).push(t)
  }

  const rounds = Object.keys(byRound)
    .map(Number)
    .sort((a, b) => a - b)

  for (const round of rounds) {
    const items = byRound[round]
    items.forEach((t, col) => {
      const sharpe = typeof t.metrics?.sharpe === 'number' ? t.metrics.sharpe.toFixed(3) : null
      nodes.push({
        id: t.trajectory_id,
        position: { x: col * 220 + 40, y: round * 180 + 40 },
        data: {
          label: (
            <div style={{ fontSize: 11, textAlign: 'left', lineHeight: 1.5 }}>
              <div><Tag color={PHASE_COLOR[t.phase] ?? '#aaa'} style={{ fontSize: 10 }}>{PHASE_LABEL[t.phase] ?? t.phase}</Tag></div>
              <div style={{ fontFamily: 'monospace', marginTop: 2 }}>#{t.trajectory_id.slice(-6)}</div>
              {t.hypothesis && (
                <div style={{ maxWidth: 160, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', color: '#555' }}>
                  {t.hypothesis.slice(0, 40)}
                </div>
              )}
              {sharpe && <div style={{ color: Number(sharpe) >= 1 ? '#52c41a' : '#faad14', fontWeight: 600 }}>Sharpe: {sharpe}</div>}
              {t.reward_score !== null && <div style={{ color: '#888' }}>Score: {Number(t.reward_score).toFixed(3)}</div>}
            </div>
          ),
        },
        style: {
          background: '#fff',
          border: `2px solid ${PHASE_COLOR[t.phase] ?? '#d9d9d9'}`,
          borderRadius: 8,
          padding: '6px 10px',
          width: 200,
          minHeight: 80,
        },
      })

      for (const parentId of t.parent_ids) {
        edges.push({
          id: `${parentId}->${t.trajectory_id}`,
          source: String(parentId),
          target: t.trajectory_id,
          animated: t.phase === 'mutation',
          style: { stroke: PHASE_COLOR[t.phase] ?? '#aaa', strokeWidth: 1.5 },
          label: t.phase === 'crossover' ? '×' : undefined,
        })
      }
    })
  }

  return { nodes, edges }
}

export default function LineageTree() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const { data: trajectories, isLoading } = useSWR<Trajectory[]>(
    runId ? `/api/runs/${runId}/lineage` : null,
    fetcher
  )

  const { nodes: initNodes, edges: initEdges } = useMemo(
    () => (trajectories ? buildGraph(trajectories) : { nodes: [], edges: [] }),
    [trajectories]
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(initNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges)
  const onConnect = useCallback((c: Connection) => setEdges((e) => addEdge(c, e)), [setEdges])

  useEffect(() => { setNodes(initNodes) }, [initNodes, setNodes])
  useEffect(() => { setEdges(initEdges) }, [initEdges, setEdges])

  const phaseCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const t of trajectories ?? []) counts[t.phase] = (counts[t.phase] ?? 0) + 1
    return counts
  }, [trajectories])

  if (isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />

  return (
    <div>
      <Row align="middle" justify="space-between" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={4} style={{ margin: 0 }}>进化树 <Text code>{runId}</Text></Title>
        </Col>
        <Col>
          <Space>
            <Button icon={<MonitorOutlined />} onClick={() => navigate(`/runs/${runId}`)}>监控</Button>
            <Button icon={<ExperimentOutlined />} onClick={() => navigate(`/runs/${runId}/factors`)}>因子库</Button>
            <Button icon={<BranchesOutlined />} type="primary">进化树</Button>
            <Button icon={<FileTextOutlined />} onClick={() => navigate(`/runs/${runId}/report`)}>报告</Button>
          </Space>
        </Col>
      </Row>

      <Row gutter={8} style={{ marginBottom: 16 }}>
        {Object.entries(PHASE_LABEL).map(([phase, label]) => (
          <Col key={phase}>
            <Card size="small" style={{ borderColor: PHASE_COLOR[phase] }}>
              <Tag color={PHASE_COLOR[phase]}>{label}</Tag>
              <Text strong> {phaseCounts[phase] ?? 0}</Text>
            </Card>
          </Col>
        ))}
        <Col>
          <Card size="small">
            <Text>总轨迹 <strong>{trajectories?.length ?? 0}</strong></Text>
          </Card>
        </Col>
      </Row>

      {!trajectories?.length ? (
        <Empty description="暂无轨迹数据" />
      ) : (
        <div style={{ height: 600, border: '1px solid #f0f0f0', borderRadius: 8 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
            attributionPosition="bottom-left"
          >
            <Background />
            <Controls />
            <MiniMap nodeColor={(n) => {
              const traj = trajectories?.find((t) => t.trajectory_id === n.id)
              return PHASE_COLOR[traj?.phase ?? ''] ?? '#aaa'
            }} />
          </ReactFlow>
        </div>
      )}
    </div>
  )
}
