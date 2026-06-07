import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { Button, Result, Spin, Space, Typography } from 'antd'
import { PlusCircleOutlined, SyncOutlined } from '@ant-design/icons'
import RunProgressView from '../components/RunProgressView'
import { useRunWebSocket } from '../hooks/useRunWebSocket'
import type { ProgressEvent, Run } from '../types'
import { mergeProgressEvents } from '../utils/progressEvents'

const { Text } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

function RunningRunCard({ run }: { run: Run }) {
  const { events: wsEvents, connected } = useRunWebSocket(run.run_id)
  const [initEvents, setInitEvents] = useState<ProgressEvent[]>([])

  useEffect(() => {
    fetch(`/api/runs/${run.run_id}/progress?limit=200`)
      .then((r) => r.json())
      .then(setInitEvents)
      .catch(() => {})
  }, [run.run_id])

  const events = useMemo(() => mergeProgressEvents(initEvents, wsEvents), [initEvents, wsEvents])

  return (
    <RunProgressView
      run={run}
      runId={run.run_id}
      events={events}
      connected={connected}
      compact
    />
  )
}

export default function ActiveRun() {
  const navigate = useNavigate()
  const { data: runs, isLoading } = useSWR<Run[]>('/api/runs', fetcher, { refreshInterval: 3000 })

  if (isLoading) {
    return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  }

  const running = (runs ?? [])
    .filter((run) => run.status === 'running')
    .sort((a, b) => (b.started_at > a.started_at ? 1 : -1))

  if (running.length === 0) {
    return (
      <Result
        icon={<SyncOutlined style={{ color: '#d9d9d9', fontSize: 56 }} />}
        title="当前无运行中的任务"
        subTitle={<Text type="secondary">启动新运行后，Agent 执行进度会实时显示在这里</Text>}
        extra={(
          <Space>
            <Button type="primary" icon={<PlusCircleOutlined />} onClick={() => navigate('/runs/new')}>新建运行</Button>
            <Button onClick={() => navigate('/history')}>历史运行</Button>
          </Space>
        )}
      />
    )
  }

  return (
    <div>
      {running.length > 1 && (
        <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
          共 {running.length} 个任务运行中
        </Text>
      )}
      {running.map((run) => <RunningRunCard key={run.run_id} run={run} />)}
    </div>
  )
}
