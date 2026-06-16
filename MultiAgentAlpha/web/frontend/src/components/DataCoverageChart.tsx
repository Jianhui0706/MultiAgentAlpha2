import { useEffect, useMemo, useRef, useState } from 'react'
import { Space, Tag, Tooltip, Typography, Spin, Alert, Slider, DatePicker, Button } from 'antd'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'

const { Text } = Typography
const CHINESE_DATE_FMT = 'YYYY年MM月DD日 HH:mm'

interface CoveragePoint {
  timestamp: string
  count: number
}

interface CoverageInfo {
  union_start_datetime?: string
  union_end_datetime?: string
  recommended_start_datetime?: string
  recommended_end_datetime?: string
  sampled_coverage?: CoveragePoint[]
  min_cross_section?: number
  total_symbols?: number
  errors?: Array<{ path: string; error: string }>
}

interface Props {
  paths: string[]
  /** When true, render a time-window selector (slider + presets + range picker). */
  selectable?: boolean
  /** Controlled selected range (only used when selectable). */
  range?: [Dayjs, Dayjs] | null
  /** Called whenever the selected range changes, and once to auto-init on load. */
  onRangeChange?: (range: [Dayjs, Dayjs] | null) => void
}

/**
 * Data distribution view: given panel file paths, fetches per-timestamp
 * cross-sectional coverage and renders a histogram + summary tags. When
 * `selectable`, it also renders a time-window picker (slider / presets / range
 * picker) and reports the chosen window via `onRangeChange` — used by the
 * Preprocess page to crop the saved panel to a sub-range.
 */
export default function DataCoverageChart({ paths, selectable = false, range = null, onRangeChange }: Props) {
  const [info, setInfo] = useState<CoverageInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const initSigRef = useRef<string>('')

  useEffect(() => {
    if (!paths.length) {
      setInfo(null)
      setFailed(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setFailed(false)
    fetch('/api/data-files/coverage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths }),
    })
      .then((res) => {
        if (!res.ok) throw new Error('coverage request failed')
        return res.json()
      })
      .then((data: CoverageInfo) => {
        if (!cancelled) setInfo(data)
      })
      .catch(() => {
        if (!cancelled) {
          setInfo(null)
          setFailed(true)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [paths.join('|')])

  const start = useMemo(() => (info?.union_start_datetime ? dayjs(info.union_start_datetime) : null), [info])
  const end = useMemo(() => (info?.union_end_datetime ? dayjs(info.union_end_datetime) : null), [info])

  // Auto-initialize the selected window to the recommended/union range once per load.
  useEffect(() => {
    if (!selectable || !info || !onRangeChange) return
    const sig = `${info.union_start_datetime ?? ''}|${info.union_end_datetime ?? ''}`
    if (initSigRef.current === sig) return
    const s = dayjs(info.recommended_start_datetime || info.union_start_datetime)
    const e = dayjs(info.recommended_end_datetime || info.union_end_datetime)
    if (s.isValid() && e.isValid()) {
      initSigRef.current = sig
      onRangeChange([s, e])
    }
  }, [info, selectable])

  const maxCount = Math.max(1, info?.total_symbols ?? 1)
  const minCross = Number(info?.min_cross_section ?? 5)

  const rangeMs = start && end ? ([start.valueOf(), end.valueOf()] as [number, number]) : null
  const selectedMs = range ? ([range[0].valueOf(), range[1].valueOf()] as [number, number]) : rangeMs

  function applyPreset(preset: 'all' | 'recommended' | '30d' | '90d' | '1y') {
    if (!start || !end || !onRangeChange) return
    if (preset === 'all') return onRangeChange([start, end])
    if (preset === 'recommended') {
      const s = dayjs(info?.recommended_start_datetime || info?.union_start_datetime)
      const e = dayjs(info?.recommended_end_datetime || info?.union_end_datetime)
      if (s.isValid() && e.isValid()) onRangeChange([s, e])
      return
    }
    const amount = preset === '30d' ? 30 : preset === '90d' ? 90 : 365
    const s = end.subtract(amount, 'day')
    onRangeChange([s.isBefore(start) ? start : s, end])
  }

  if (!paths.length) return null
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '20px 0' }}>
        <Spin tip="读取数据时间分布…" size="small" />
      </div>
    )
  }
  if (failed) {
    return <Alert type="error" showIcon style={{ fontSize: 12 }} message="数据时间分布读取失败" />
  }
  if (!info) return null

  const points = info.sampled_coverage ?? []

  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 12, background: '#fbfdff' }}>
      <Space wrap size={8} style={{ marginBottom: 10 }}>
        {start && end && (
          <Tag color="blue">数据范围 {start.format(CHINESE_DATE_FMT)} ~ {end.format(CHINESE_DATE_FMT)}</Tag>
        )}
        <Tag>标的 {info.total_symbols ?? 0}</Tag>
        <Tag>最低横截面 {minCross}</Tag>
        {selectable && range && (
          <Tag color="green">已选 {range[0].format(CHINESE_DATE_FMT)} ~ {range[1].format(CHINESE_DATE_FMT)}</Tag>
        )}
      </Space>

      {points.length > 0 ? (
        <>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 56, marginBottom: 6 }}>
            {points.map((point, idx) => {
              const ratio = Math.max(0.08, Math.min(1, Number(point.count || 0) / maxCount))
              const ok = Number(point.count || 0) >= minCross
              const inWindow = !selectable || !range
                || (dayjs(point.timestamp).valueOf() >= range[0].valueOf() && dayjs(point.timestamp).valueOf() <= range[1].valueOf())
              return (
                <Tooltip
                  key={`${point.timestamp}-${idx}`}
                  title={`${dayjs(point.timestamp).format(CHINESE_DATE_FMT)} · ${point.count} 标的`}
                >
                  <div
                    style={{
                      flex: 1,
                      height: `${Math.round(ratio * 100)}%`,
                      minWidth: 2,
                      borderRadius: 3,
                      background: ok ? '#1677ff' : '#f59e0b',
                      opacity: inWindow ? (ok ? 0.85 : 0.6) : 0.18,
                    }}
                  />
                </Tooltip>
              )
            })}
          </div>
          <Space size={16} style={{ marginBottom: selectable ? 10 : 0 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#1677ff', marginRight: 4 }} />
              横截面 ≥ {minCross}（可回测）
            </Text>
            <Text type="secondary" style={{ fontSize: 11 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#f59e0b', marginRight: 4 }} />
              标的不足
            </Text>
            {selectable
              ? <Text type="secondary" style={{ fontSize: 11 }}>浅色 = 不在所选区间</Text>
              : <Text type="secondary" style={{ fontSize: 11 }}>柱高 = 该时刻有数据的标的数 / 总标的数</Text>}
          </Space>

          {selectable && (
            <Space orientation="vertical" size={10} style={{ width: '100%' }}>
              {rangeMs && selectedMs && (
                <Slider
                  range={{ draggableTrack: true }}
                  min={rangeMs[0]}
                  max={rangeMs[1]}
                  value={selectedMs}
                  tooltip={{ formatter: (v?: number) => (v == null ? '' : dayjs(v).format(CHINESE_DATE_FMT)) }}
                  onChange={(v: number | number[]) => {
                    const [s, e] = v as [number, number]
                    onRangeChange?.([dayjs(s), dayjs(e)])
                  }}
                />
              )}
              <Space wrap size={8}>
                <Button size="small" onClick={() => applyPreset('recommended')}>推荐区间</Button>
                <Button size="small" onClick={() => applyPreset('all')}>全部数据</Button>
                <Button size="small" onClick={() => applyPreset('30d')}>最近 30 天</Button>
                <Button size="small" onClick={() => applyPreset('90d')}>最近 90 天</Button>
                <Button size="small" onClick={() => applyPreset('1y')}>最近 1 年</Button>
              </Space>
              <DatePicker.RangePicker
                showTime={{ format: 'HH:mm' }}
                value={range}
                onChange={(v) => onRangeChange?.(v as [Dayjs, Dayjs] | null)}
                style={{ width: '100%' }}
                format="YYYY年MM月DD日 HH:mm"
                placeholder={['开始时间', '结束时间']}
              />
            </Space>
          )}
        </>
      ) : (
        <Text type="secondary" style={{ fontSize: 12 }}>暂无可绘制的时间分布点</Text>
      )}

      {!!info.errors?.length && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 10, fontSize: 12 }}
          message={`${info.errors.length} 个文件读取异常`}
          description={
            <div style={{ fontSize: 11 }}>
              {info.errors.slice(0, 5).map((e, i) => (
                <div key={i}><Text code style={{ fontSize: 11 }}>{e.path.split(/[\\/]/).pop()}</Text>：{e.error}</div>
              ))}
            </div>
          }
        />
      )}
    </div>
  )
}
