import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import useSWR from 'swr'
import { Spin } from 'antd'
import RunProgressView from '../components/RunProgressView'
import { useRunWebSocket } from '../hooks/useRunWebSocket'
import type { ProgressEvent, RunDetail } from '../types'
import { mergeProgressEvents } from '../utils/progressEvents'

const fetcher = (url: string) => fetch(url).then((r) => r.json())

export default function RunMonitor() {
  const { runId } = useParams<{ runId: string }>()
  const { data: run, isLoading } = useSWR<RunDetail>(
    runId ? `/api/runs/${runId}` : null,
    fetcher,
    { refreshInterval: 5000 }
  )
  const { events: wsEvents, connected } = useRunWebSocket(runId ?? null)
  const [initEvents, setInitEvents] = useState<ProgressEvent[]>([])

  useEffect(() => {
    if (!runId) return
    // 历史回放：加载全量事件（含已结束运行），按阶段分组展示完整流程
    fetch(`/api/runs/${runId}/progress?limit=20000`)
      .then((r) => r.json())
      .then(setInitEvents)
      .catch(() => {})
  }, [runId])

  const events = useMemo(() => mergeProgressEvents(initEvents, wsEvents), [initEvents, wsEvents])

  if (isLoading) {
    return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  }

  return (
    <RunProgressView
      run={run}
      runId={runId ?? ''}
      events={events}
      connected={connected}
    />
  )
}
