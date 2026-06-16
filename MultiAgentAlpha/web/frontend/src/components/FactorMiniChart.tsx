import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell,
  LineChart, Line, ReferenceLine,
  Tooltip as ReTooltip, ResponsiveContainer,
} from 'recharts'
import { Typography, Button, Tag } from 'antd'
import { BarChartOutlined, DownOutlined, RightOutlined } from '@ant-design/icons'
import RewardBreakdown, { type RewardBreakdownData } from './RewardBreakdown'

const { Text } = Typography

export interface FactorMetrics {
  sharpe?: number | null
  IC?: number | null
  rank_ic?: number | null
  icir?: number | null
  rank_icir?: number | null
  coverage?: number | null
  turnover?: number | null
  valid_cross_section_mean?: number | null
  cross_section_universe_size?: number | null
  ic_win_rate?: number | null
  rank_ic_win_rate?: number | null
  drawdown?: number | null
  long_short_positive_ratio?: number | null
  ls_return?: number | null
  ann_return?: number | null
  yearly_returns?: Record<string, number> | null
  reward_score?: number | null
  reward_breakdown?: RewardBreakdownData | null
  layer_1?: number | null
  layer_2?: number | null
  layer_3?: number | null
  layer_4?: number | null
  layer_5?: number | null
  layer_monotonicity?: number | null
  ls_cumulative_series?: [string, number][] | null
  layer_1_cumulative_series?: [string, number][] | null
  layer_2_cumulative_series?: [string, number][] | null
  layer_3_cumulative_series?: [string, number][] | null
  layer_4_cumulative_series?: [string, number][] | null
  layer_5_cumulative_series?: [string, number][] | null
  benchmark_cumulative_series?: [string, number][] | null
  ic_cumulative_series?: [string, number][] | null
  rank_ic_cumulative_series?: [string, number][] | null
}

function num(v: number | null | undefined): number | null {
  return v === null || v === undefined || !Number.isFinite(v) ? null : v
}

/**
 * Backend metrics are not always the shape the types promise — e.g. a
 * *_cumulative_series can come back as a bare scalar (0.0) instead of a
 * [time, value][] array. Coerce anything that isn't a valid pair-array to
 * null so chart code never calls .forEach/.map on a non-array and crashes.
 */
function asSeries(v: unknown): [string, number][] | null {
  if (!Array.isArray(v)) return null
  const ok = v.filter(
    (p): p is [string, number] =>
      Array.isArray(p) && p.length >= 2 && typeof p[0] === 'string' && Number.isFinite(p[1]),
  )
  return ok.length ? ok : null
}

function asYearly(v: unknown): Record<string, number> | null {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return null
  const entries = Object.entries(v as Record<string, unknown>).filter(
    ([, val]) => Number.isFinite(val),
  ) as [string, number][]
  return entries.length ? Object.fromEntries(entries) : null
}
function fmt(v: number | null, d = 4): string { return v === null ? '—' : v.toFixed(d) }
function pct(v: number | null): string { return v === null ? '—' : (v * 100).toFixed(2) + '%' }

type GateRow = {
  group: string
  name: string
  value: number | null
  valueText: string
  threshold: string
  passed: boolean | null
  weight?: string
}

function gateRow(
  group: string,
  name: string,
  value: number | null,
  threshold: string,
  passed: boolean | null,
  valueText: string,
  weight?: string,
): GateRow {
  return { group, name, value, valueText, threshold, passed, weight }
}

function barColor(name: string, value: number): string {
  if (name === 'Turnover') return value > 0.5 ? '#ff7875' : '#91caff'
  return value >= 0 ? '#1677ff' : '#ff7875'
}
function layerColor(idx: number): string {
  return ['#cf1322', '#fa8c16', '#faad14', '#52c41a', '#237804'][Math.min(idx, 4)]
}

/** Merge any number of named [time, value] series into rows keyed by time. */
function mergeNamed(
  series: Record<string, [string, number][] | null>,
): Record<string, number | string>[] {
  const allTimes = new Set<string>()
  Object.values(series).forEach((s) => s?.forEach(([t]) => allTimes.add(t)))
  if (allTimes.size === 0) return []
  const maps = Object.fromEntries(
    Object.entries(series).map(([name, s]) => [name, new Map(s?.map(([t, v]) => [t, v]) ?? [])]),
  ) as Record<string, Map<string, number>>
  return Array.from(allTimes).sort().map((t) => {
    const row: Record<string, number | string> = { t: t.slice(0, 16).replace('T', ' ') }
    for (const [name, m] of Object.entries(maps)) {
      if (m.has(t)) row[name] = Math.round((m.get(t) ?? 0) * 10000) / 10000
    }
    return row
  })
}

export default function FactorMiniChart({ metrics }: { metrics: FactorMetrics }) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setOpen(true), 50)
    return () => clearTimeout(t)
  }, [])

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

  // 逐年收益（兜底：后端偶尔返回标量而非对象）
  const yearly = asYearly(metrics.yearly_returns)
  const yearlyData = yearly
    ? Object.entries(yearly)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([year, ret]) => ({ year, ret: Math.round(ret * 10000) / 10000 }))
    : []

  // ── 五分组净值曲线：G1..G5 + 多空 + 基准 ──
  // asSeries 把标量/脏数据统一归一成 null，避免对非数组调用 .forEach 崩溃。
  const groupSeriesMap: Record<string, [string, number][] | null> = {
    g1: asSeries(metrics.layer_1_cumulative_series),
    g2: asSeries(metrics.layer_2_cumulative_series),
    g3: asSeries(metrics.layer_3_cumulative_series),
    g4: asSeries(metrics.layer_4_cumulative_series),
    g5: asSeries(metrics.layer_5_cumulative_series),
    ls: asSeries(metrics.ls_cumulative_series),
    bench: asSeries(metrics.benchmark_cumulative_series),
  }
  const groupCurveData = mergeNamed(groupSeriesMap)
  const GROUP_LINES: { key: string; label: string; color: string; width: number; dash?: string }[] = [
    { key: 'g1', label: 'G1（最低）', color: layerColor(0), width: 1.2 },
    { key: 'g2', label: 'G2', color: layerColor(1), width: 1.2 },
    { key: 'g3', label: 'G3', color: layerColor(2), width: 1.2 },
    { key: 'g4', label: 'G4', color: layerColor(3), width: 1.2 },
    { key: 'g5', label: 'G5（最高）', color: layerColor(4), width: 1.2 },
    { key: 'ls', label: '多空', color: '#1677ff', width: 2, dash: '4 2' },
    { key: 'bench', label: '基准', color: '#8c8c8c', width: 1.2, dash: '5 3' },
  ].filter((l) => (groupSeriesMap[l.key]?.length ?? 0) > 1)
  const hasGroupCurve = groupCurveData.length > 1 && GROUP_LINES.length > 0

  // ── 累计 IC / 累计 RankIC 曲线 ──
  const icSeriesMap: Record<string, [string, number][] | null> = {
    ic: asSeries(metrics.ic_cumulative_series),
    rankic: asSeries(metrics.rank_ic_cumulative_series),
  }
  const icCurveData = mergeNamed(icSeriesMap)
  const hasIcCurve = (icSeriesMap.ic?.length ?? 0) > 1 || (icSeriesMap.rankic?.length ?? 0) > 1

  // 年化收益优先 ann_return，兜底用 ls_return（单期）
  const annReturn = num(metrics.ann_return)
  const returnVal = annReturn ?? num(metrics.ls_return)
  const returnLabel = annReturn !== null ? '年化收益' : 'L/S 单期均值'
  const validCrossSection = num(metrics.valid_cross_section_mean)
  const universeSize = num(metrics.cross_section_universe_size)
  const validCrossSectionThreshold = universeSize !== null && universeSize > 0 ? Math.min(30, 0.5 * universeSize) : 30
  const ic = num(metrics.IC)
  const icir = num(metrics.icir)
  const rankIcir = num(metrics.rank_icir)
  const sharpe = num(metrics.sharpe)
  const drawdown = num(metrics.drawdown)
  const icWinRate = num(metrics.ic_win_rate)
  const rankIcWinRate = num(metrics.rank_ic_win_rate)
  const longShortWinRate = num(metrics.long_short_positive_ratio)
  const gateRows: GateRow[] = [
    gateRow(
      '基础有效性',
      '有效截面数均值',
      validCrossSection,
      `>= ${validCrossSectionThreshold.toFixed(validCrossSectionThreshold % 1 === 0 ? 0 : 1)}`,
      validCrossSection === null ? null : validCrossSection >= validCrossSectionThreshold,
      fmt(validCrossSection, 2),
    ),
    gateRow(
      '基础有效性',
      '|IC| 均值',
      ic,
      '> 0.03',
      ic === null ? null : Math.abs(ic) > 0.03,
      ic === null ? '—' : fmt(Math.abs(ic), 4),
    ),
    gateRow(
      '基础有效性',
      'IC 胜率',
      icWinRate,
      '> 52%',
      icWinRate === null ? null : icWinRate > 0.52,
      pct(icWinRate),
    ),
    gateRow(
      '预测稳定性',
      '|ICIR|',
      icir,
      '> 0.05',
      icir === null ? null : Math.abs(icir) > 0.05,
      icir === null ? '—' : fmt(Math.abs(icir), 4),
      '核心',
    ),
    gateRow(
      '预测稳定性',
      '|RankICIR|',
      rankIcir,
      '> 0.05',
      rankIcir === null ? null : Math.abs(rankIcir) > 0.05,
      rankIcir === null ? '—' : fmt(Math.abs(rankIcir), 4),
      '核心',
    ),
    gateRow(
      '预测稳定性',
      'RankIC 胜率',
      rankIcWinRate,
      '> 55%',
      rankIcWinRate === null ? null : rankIcWinRate > 0.55,
      pct(rankIcWinRate),
      '重点',
    ),
    gateRow(
      '策略表现',
      '多空 Sharpe',
      sharpe,
      '> 1.0',
      sharpe === null ? null : sharpe > 1.0,
      fmt(sharpe, 3),
      '核心',
    ),
    gateRow(
      '策略表现',
      '最大回撤',
      drawdown,
      '< 30%',
      drawdown === null ? null : Math.abs(drawdown) < 0.30,
      drawdown === null ? '—' : pct(Math.abs(drawdown)),
      '重点',
    ),
    gateRow(
      '策略表现',
      '胜率',
      longShortWinRate,
      '> 55%',
      longShortWinRate === null ? null : longShortWinRate > 0.55,
      pct(longShortWinRate),
    ),
  ]
  const gateGroups = Array.from(new Set(gateRows.map((row) => row.group)))
  const hasGateData = gateRows.some((row) => row.value !== null)

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

          <RewardBreakdown breakdown={metrics.reward_breakdown} />

          {hasGateData && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 6 }}>
                入库硬门与诊断值（缺失项跳过判断，未达项会阻止进入全因子库排行榜）
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(180px, 1fr))', gap: 10 }}>
                {gateGroups.map((group) => (
                  <div key={group} style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 6, padding: '8px 10px' }}>
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{group}</div>
                    <div style={{ display: 'grid', gap: 5 }}>
                      {gateRows.filter((row) => row.group === group).map((row) => {
                        const color = row.passed === null ? 'default' : row.passed ? 'success' : 'error'
                        const label = row.passed === null ? '缺失' : row.passed ? '通过' : '未达'
                        return (
                          <div key={`${row.group}:${row.name}`} style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', alignItems: 'center', gap: 6 }}>
                            <Tag color={color} style={{ margin: 0, minWidth: 42, textAlign: 'center' }}>{label}</Tag>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ fontSize: 11, color: '#595959', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {row.name}
                                {row.weight ? <Text type="secondary" style={{ fontSize: 10 }}> · {row.weight}</Text> : null}
                              </div>
                              <div style={{ fontSize: 10, color: '#8c8c8c' }}>{row.threshold}</div>
                            </div>
                            <Text code style={{ fontSize: 11 }}>{row.valueText}</Text>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

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

          {/* Row 4: 累计 IC / 累计 RankIC 双线图 */}
          {hasIcCurve && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>
                累计 IC 曲线（IC 信息系数逐期累加，向上 = 因子持续有效）
                {(icSeriesMap.ic?.length ?? 0) > 1 && <span style={{ marginLeft: 8, color: '#1677ff' }}>● IC累计</span>}
                {(icSeriesMap.rankic?.length ?? 0) > 1 && <span style={{ marginLeft: 8, color: '#fa8c16' }}>● RankIC累计</span>}
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={icCurveData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" tick={{ fontSize: 8 }} interval={Math.floor(icCurveData.length / 6)} tickFormatter={(v) => String(v).slice(0, 10)} />
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => v.toFixed(1)} />
                  <ReTooltip
                    formatter={(v, name) => {
                      const labels: Record<string, string> = { ic: 'IC累计', rankic: 'RankIC累计' }
                      return [Number(v).toFixed(3), labels[String(name)] ?? String(name)]
                    }}
                    labelFormatter={(l) => `时间: ${l}`} contentStyle={{ fontSize: 11 }}
                  />
                  <ReferenceLine y={0} stroke="#bfbfbf" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="ic" stroke="#1677ff" dot={false} strokeWidth={1.5} name="ic" connectNulls />
                  <Line type="monotone" dataKey="rankic" stroke="#fa8c16" dot={false} strokeWidth={1.5} name="rankic" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Row 5: 五分组净值图 G1..G5 + 多空 + 基准 */}
          {hasGroupCurve && (
            <div>
              <div style={{ fontSize: 11, color: '#8c8c8c', marginBottom: 4 }}>
                五分组净值曲线（G1=因子值最低组 → G5=最高组；理想因子各组单调分层）
                {GROUP_LINES.map((l) => (
                  <span key={l.key} style={{ marginLeft: 8, color: l.color }}>● {l.label}</span>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={170}>
                <LineChart data={groupCurveData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" tick={{ fontSize: 8 }} interval={Math.floor(groupCurveData.length / 6)} tickFormatter={(v) => String(v).slice(0, 10)} />
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => v.toFixed(2)} />
                  <ReTooltip
                    formatter={(v, name) => {
                      const m = GROUP_LINES.find((l) => l.key === String(name))
                      return [Number(v).toFixed(4), m?.label ?? String(name)]
                    }}
                    labelFormatter={(l) => `时间: ${l}`} contentStyle={{ fontSize: 11 }}
                  />
                  <ReferenceLine y={1} stroke="#bfbfbf" strokeDasharray="4 2" />
                  {GROUP_LINES.map((l) => (
                    <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} dot={false} strokeWidth={l.width} strokeDasharray={l.dash} name={l.key} connectNulls />
                  ))}
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
