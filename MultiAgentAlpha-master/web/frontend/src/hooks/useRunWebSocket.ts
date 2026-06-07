import { useEffect, useRef, useState } from 'react'
import type { ProgressEvent } from '../types'

const MAX_EVENTS = 500

export function useRunWebSocket(runId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!runId) return

    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/runs/${runId}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (e) => {
      try {
        const event: ProgressEvent = JSON.parse(e.data)
        setEvents((prev) => {
          const next = [...prev, event]
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next
        })
      } catch {
        // ignore malformed
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [runId])

  return { events, connected }
}
