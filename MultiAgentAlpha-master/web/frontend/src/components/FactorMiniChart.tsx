import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell,
  LineChart, Line, ReferenceLine,
  Tooltip as ReTooltip, ResponsiveContainer,
} from 'recharts'
import { Typography, Button } from 'antd'
import { BarChartOutlined, DownOutlined, RightOutlined } from '@ant-design/icons'

const { Text } = Typography

export interface FactorMetrics {
  sharpe?: number | null
  IC?: number | null
  rank_ic?: number | null
  icir?: number | null
  rank_icir?: number | null
  coverage?: number | null
  turnover?: number | null
  ls_return?: number | null
  ann_return?: number | null
  yearly_returns?: Record<string, number> | null
  reward_score?: number | null
  layer_1?: number | null
  layer_2?: number | null
  layer_3?: number | null
  layer_4?: number | null
  layer_5?: number | null
  layer_monotonicity?: number | null
  ls_cumulative_series?: [string, number][] | null
  layer_1_cumulative_series?: [string, number][] | null
  layer_5_cumulative_series?: [string, number][] | null
}

function num(v: number | null | undefined): number | null {
  return v === null || v === undefined || !Number.isFinite(v) ? null : v
}
function fmt(v: number | null, d = 4): string { return v === null ? '—' : v.toFixed(d) }
function pct(v: number | null): string { return v === null ? '—' : (v * 100).toFixed(2) + '%' }

function barColor(name: string, value: number): string {
  if (name === 'Turnover') return value > 0.5 ? '#ff7875' : '#91caff'
  return value >= 0 ? '#1677ff' : '#ff7875'
}
function layerColor(idx: number): string {
  return ['#cf1322', '#fa8c16', '#faad14', '#52c41a', '#237804'][Math.min(idx, 4)]
}

function mergeSeries(
  ls: [string, number][] | null | undefined,
  l1: [string, number][] | null | undefined,
  l5: [string, number][] | null | undefined,
): { t: string; ls?: number; l1?: number; l5?: number }[] {
  const allTimes = new Set<string>()
  ;[ls, l1, l5].forEach((s) => s?.forEach(([t]) => allTimes.add(t)))
  if (allTimes.size === 0) return []
  const mapLs = new Map(ls?.map(([t, v]) => [t, v]) ?? [])
  const mapL1 = new Map(l1?.map(([t, v]) => [t, v]) ?? [])
  const mapL5 = new Map(l5?.map(([t, v]) => [t, v]) ?? [])
  return Array.from(allTimes).sort().map((t) => ({
    t: t.slice(0, 16).replace('T', ' '),
    ...(mapLs.has(t) ? { ls: Math.round((mapLs.get(t) ?? 1) * 10000) / 10000 } : {}),
    ...(mapL1.has(t) ? { l1: Math.round((mapL1.get(t) ?? 1) * 10000) / 10000 } : {}),
    ...(mapL5.has(t) ? { l5: Math.round((mapL5.get(t) ?? 1) * 10000) / 10000 } : {}),
  }))
}

export default function FactorMiniChart({ metrics }: { metrics: FactorMetrics }) {
  const [open, setOpen] = useState(false)

  const BARS = [
    { key: 'sharpe',    label: 'Sharpe',   v: num(metrics.sharpe) },
    { key: 'IC',        label: 'IC',        v: num(metrics.IC) },
    { key: 'rank_ic',   label: 'Rank IC',   v: num(metrics.rank_ic) },
    { key: 'icir',      label: 'ICIR',      v: num(metrics.icir) },
    { key: 'rank_icir', label: 'Rank ICIR', v: num(metrics.rank_icir) },
    { key: 'coverage',  label: 'Coverage',  v: num(metrics.coverage) },
    { key: 'turnover',  label: 'Turnover',  v: num(metrics.turnover) },
  ].filter((d) => d.v !== null) as { key: string; label: string; v: number }[]

  const LAYER_BARS = [
    { label: 'L1', v: num(metrics.layer_1), idx: 0 },
    { label: 'L2', v: num(metrics.layer_2), idx: 1 },
    { label: 'L3', v: num(metrics.layer_3), idx: 2 },
    { label: 'L4', v: num(metrics.layer_4), idx: 3 },
    { label: 'L5', v: num(metrics.layer_5), idx: 4 },
  ].filter((d) => d.v !== null) as { label: string; v: number; idx: number }[]

  const monotonicity = num(metrics.layer_monotonicity)

  // 逐年收益
  const yearlyData = metrics.yearly_returns
    ? Object.entries(metrics.yearly_returns)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([year, ret]) => ({ year, ret: Math.round(ret * 10000) / 10000 }))
    : []

  // 累计曲线
  const curveData = mergeSeries(
    metrics.ls_cumulative_series,
    metrics.layer_1_cumulative_series,
    metrics.layer_5_cumulative_series,
  )
  const hasL1Curve = (metrics.layer_1_cumulative_series?.length ?? 0) > 1
  const hasL5Curve = (metrics.layer_5_cumulative_series?.length ?? 0) > 1
  const hasLsCurve = (metrics.ls_cumulative_series?.length ?? 0) > 1
  const hasCurve = curveData.length > 1

  // 年化收益优先 ann_return，兜底用 ls_return（单期）
  const annReturn = num(metrics.ann_return)
  const returnVal = annReturn ?? num(metrics.ls_return)
  const returnLabel = annReturn !== null ? '年化收益' : 'L/S 单期均值'

  const hasData = BARS.length > 0

  return (
    <div style={{ marginTop: 8 }}>
      <Button
        type="text" size="small"
        icon={<BarChartOutlined style={{ color: hasData ? '#1677ff' : '#bfbfbf' }} />}
        onClick={() => setOpen((v) => !v)}
        style={{ padding: '0 4px', color: hasData ? '#1677ff' : '#bfbfbf', fontSize: 12 }}
      >
        {open ? <DownOutlined style={{ fontSize: 10 }} /> : <RightOutlined style={{ fontSize: 10 }} />}
        {hasData ? '指标图表' : '暂无指标'}
      </Button>

      {open && (hasData ? (
        <div style={{ marginTop: 8, padding: '10px 12px', background: '#fafafa', borderRadius: 6, border: '1px solid #f0f0f0' }}>

          {/* Row 1: 基础指标 bar + 数值卡片 */}
          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', marginBottom: 14 }}>
            <div style={{ flex: '0 0 320px' }}>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>基础指标</div>
              <ResponsiveContainer width="100%" height={110}>
                <BarChart data={BARS} margin={{ top: 2, right: 4, left: -28, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={0} />
                  <YAxis tick={{ fontSize: 9 }} />
                  <ReTooltip
                    formatter={(v, _n, p) => [typeof v === 'number' ? v.toFixed(4) : String(v), (p.payload as { label?: string })?.label ?? '']}
                    labelFormatter={() => ''} contentStyle={{ fontSize: 11 }}
                  />
                  <Bar dataKey="v" radius={[3, 3, 0, 0]} maxBarSize={28}>
                    {BARS.map((e) => <Cell key={e.key} fill={barColor(e.label, e.v)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(4, minmax(76px, 1fr))', gap: '6px 10px' }}>
              {BARS.map(({ label, v }) => {
                const color = label === 'Turnover' ? (v > 0.5 ? '#cf1322' : '#0958d9') : v >= 0 ? '#237804' : '#cf1322'
                return (
                  <div key={label} style={{ background: '#fff', borderRadius: 6, padding: '5px 8px', border: '1px solid #f0f0f0' }}>
                    <div style={{ fontSize: 10, color: '#8c8c8c', marginBottom: 1 }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color, fontFamily: 'monospace' }}>{fmt(v, label === 'Sharpe' ? 3 : 4)}</div>
                  </div>
                )
              })}
              {returnVal !== null && (
                <div style={{ background: '#fff', borderRadius: 6, padding: '5px 8px', border: '1px solid #f0f0f0' }}>
                  <div style={{ fontSize: 10, color: '#8c8c8c', marginBottom: 1 }}>{returnLabel}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: returnVal > 0 ? '#237804' : '#cf1322', fontFamily: 'monospace' }}>{pct(returnVal)}</div>
                </div>
              )}
              {num(metrics.reward_score) !== null && (
                <div style={{ background: '#fffbe6', borderRadius: 6, padding: '5px 8px', border: '1px solid #ffe58f' }}>
                  <div style={{ fontSize: 10, color: '#8c8c8c', marginBottom: 1 }}>Reward</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#d48806', fontFamily: 'monospace' }}>{fmt(num(metrics.reward_score), 4)}</div>
                </div>
              )}
            </div>
          </div>

          {/* Row 2: 分层检验柱状图 */}
          {LAYER_BARS.length >= 3 && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>
                分层检验（L1 = 因子值最低组 → L5 = 因子值最高组）
                {monotonicity !== null && (
                  <span style={{ marginLeft: 8, fontWeight: 600, color: monotonicity >= 0.75 ? '#237804' : monotonicity >= 0.5 ? '#faad14' : '#cf1322' }}>
                    单调性 {(monotonicity * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={LAYER_BARS} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `${(v * 100).toFixed(2)}%`} />
                  <ReTooltip formatter={(v) => [`${(Number(v) * 100).toFixed(4)}%`, '各期平均收益']} contentStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#d9d9d9" />
                  <Bar dataKey="v" radius={[3, 3, 0, 0]} maxBarSize={48}>
                    {LAYER_BARS.map((e) => <Cell key={e.label} fill={layerColor(e.idx)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Row 3: 逐年收益柱状图 */}
          {yearlyData.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>各年度多空收益率（L5−L1）</div>
              <ResponsiveContainer width="100%" height={130}>
                <BarChart data={yearlyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <ReTooltip formatter={(v) => [`${(Number(v) * 100).toFixed(2)}%`, '年度收益']} contentStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#d9d9d9" />
                  <Bar dataKey="ret" name="年度收益" radius={[3, 3, 0, 0]} maxBarSize={60}>
                    {yearlyData.map((e) => (
                      <Cell key={e.year} fill={e.ret >= 0 ? '#52c41a' : '#ff4d4f'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Row 4: 做多/做空/多空价差三线累计净值图 */}
          {hasCurve && (
            <div>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>
                累计净值曲线
                {hasL5Curve && <span style={{ marginLeft: 8, color: '#237804' }}>● 做多组(L5)</span>}
                {hasL1Curve && <span style={{ marginLeft: 8, color: '#cf1322' }}>● 做空组(L1)</span>}
                {hasLsCurve && <span style={{ marginLeft: 8, color: '#1677ff' }}>● 多空价差(L5−L1)</span>}
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={curveData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" tick={{ fontSize: 8 }} interval={Math.floor(curveData.length / 6)} tickFormatter={(v) => String(v).slice(0, 10)} />
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => v.toFixed(2)} />
                  <ReTooltip
                    formatter={(v, name) => {
                      const labels: Record<string, string> = { l5: '做多组(L5)', l1: '做空组(L1)', ls: '多空价差' }
                      return [Number(v).toFixed(4), labels[String(name)] ?? String(name)]
                    }}
                    labelFormatter={(l) => `时间: ${l}`} contentStyle={{ fontSize: 11 }}
                  />
                  <ReferenceLine y={1} stroke="#bfbfbf" strokeDasharray="4 2" />
                  {hasL5Curve && <Line type="monotone" dataKey="l5" stroke="#237804" dot={false} strokeWidth={1.5} name="l5" connectNulls />}
                  {hasL1Curve && <Line type="monotone" dataKey="l1" stroke="#cf1322" dot={false} strokeWidth={1.5} name="l1" connectNulls />}
                  {hasLsCurve && <Line type="monotone" dataKey="ls" stroke="#1677ff" dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="ls" connectNulls />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

        </div>
      ) : (
        <div style={{ marginTop: 6, padding: '6px 12px', background: '#fafafa', borderRadius: 6, border: '1px dashed #d9d9d9' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>因子未完成回测，暂无指标数据</Text>
        </div>
      ))}
    </div>
  )
}
