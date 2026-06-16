import type { ProgressEvent } from '../types'

function eventKey(event: ProgressEvent) {
  const parts = [
    event.timestamp ?? '',
    event.stage ?? '',
    event.agent_name ?? '',
    event.loop_index ?? '',
    event.round_idx ?? '',
    event.phase ?? '',
    event.trajectory_id ?? '',
    event.error ?? '',
    event.sequence ?? '',
  ]
  return parts.map(String).join('|')
}

function eventTime(event: ProgressEvent) {
  const parsed = Date.parse(String(event.timestamp ?? ''))
  return Number.isFinite(parsed) ? parsed : 0
}

export function mergeProgressEvents(...groups: ProgressEvent[][]) {
  const byKey = new Map<string, ProgressEvent>()
  for (const event of groups.flat()) {
    byKey.set(eventKey(event), event)
  }
  return [...byKey.values()].sort((a, b) => {
    const timeDiff = eventTime(a) - eventTime(b)
    if (timeDiff !== 0) return timeDiff
    const aSeq = typeof a.sequence === 'number' ? a.sequence : 0
    const bSeq = typeof b.sequence === 'number' ? b.sequence : 0
    return aSeq - bSeq
  })
}
