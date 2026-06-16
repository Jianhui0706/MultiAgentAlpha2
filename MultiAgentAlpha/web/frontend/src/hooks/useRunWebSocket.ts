import { useEffect, useRef, useState, useCallback } from 'react'
import type { ProgressEvent } from '../types'

const MAX_EVENTS = 500
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30000

export function useRunWebSocket(runId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryCountRef = useRef(0)
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const destroyedRef = useRef(false)

  const connect = useCallback(() => {
    if (!runId || destroyedRef.current) return

    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/runs/${runId}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      retryCountRef.current = 0
      setConnected(true)
    }

    ws.onclose = () => {
      setConnected(false)
      if (destroyedRef.current) return
      // exponential backoff: 1s, 2s, 4s, ... up to 30s
      const delay = Math.min(RECONNECT_BASE_MS * 2 ** retryCountRef.current, RECONNECT_MAX_MS)
      retryCountRef.current += 1
      retryTimerRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }

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
  }, [runId])

  useEffect(() => {
    if (!runId) return
    destroyedRef.current = false
    retryCountRef.current = 0
    connect()

    return () => {
      destroyedRef.current = true
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [runId, connect])

  return { events, connected }
}
