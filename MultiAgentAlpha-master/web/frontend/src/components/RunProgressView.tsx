import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Alert,
  Badge,
  Button,
  Collapse,
  message,
  Popconfirm,
  Progress,
  Space,
  Tag,
  Timeline,
  Tooltip,
  Typography,
} from 'antd'
import {
  ApiOutlined,
  BranchesOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CodeOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FieldTimeOutlined,
  FileSearchOutlined,
  FileTextOutlined,
  ForkOutlined,
  FunctionOutlined,
  LineChartOutlined,
  NodeIndexOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  StopOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import type { ProgressEvent, Run, RunDetail } from '../types'

const { Text, Title } = Typography

type AgentStatus = 'wait' | 'process' | 'finish' | 'error'
type StageStatus = 'wait' | 'process' | 'finish' | 'error'

interface RunProgressViewProps {
  run?: Run | RunDetail
  runId: string
  events: ProgressEvent[]
  connected?: boolean
  compact?: boolean
}

const AGENT_ORDER = [
  'direction_decomposer_agent',
  'hypothesis_agent',
  'experiment_designer_agent',
  'factor_coder_agent',
  'backtest_runner_agent',
  'feedback_summarizer_agent',
]

const AGENT_META: Record<string, { label: string; action: string; icon: React.ReactNode }> = {
  direction_decomposer_agent: {
    label: '初始化假设',
    action: '生成初始研究方向',
    icon: <BranchesOutlined />,
  },
  hypothesis_agent: {
    label: '假说生成',
    action: '形成可测试假说',
    icon: <FunctionOutlined />,
  },
  experiment_designer_agent: {
    label: '实验设计',
    action: '设计候选因子',
    icon: <ExperimentOutlined />,
  },
  factor_coder_agent: {
    label: '因子编码',
    action: '生成表达式与信号',
    icon: <CodeOutlined />,
  },
  backtest_runner_agent: {
    label: '回测执行',
    action: '计算指标并筛选',
    icon: <LineChartOutlined />,
  },
  feedback_summarizer_agent: {
    label: '反馈总结',
    action: '诊断与下一步建议',
    icon: <SafetyCertificateOutlined />,
  },
  evolution_agent: {
    label: '进化规划',
    action: '规划 mutation/crossover',
    icon: <ForkOutlined />,
  },
}

const STAGE_LABELS: Record<string, string> = {
  market_payload_started: '读取市场数据',
  market_payload_finished: '市场摘要完成',
  research_report_payload_started: '读取研报',
  research_report_payload_finished: '研报合成完成',
  initialization_payload_ready: '初始化输入完成',
  direction_decomposer_started: '生成初始假设',
  direction_decomposer_finished: '初始假设完成',
  evolved_direction_decomposer_started: '生成下一代假设',
  evolved_direction_decomposer_finished: '下一代假设完成',
  evolved_seed_decomposed: '下一代假设完成',
  evolution_round_planned: '进化轮次规划',
  evolution_round_skipped: '进化轮次跳过',
  evolution_agent_started: '进化规划开始',
  evolution_agent_finished: '进化规划完成',
  evolution_task_planned: '轨迹任务规划',
  evolution_task_started: '轨迹任务开始',
  evolution_task_finished: '轨迹任务完成',
  agent_started: 'Agent 开始',
  agent_finished: 'Agent 完成',
  agent_failed: 'Agent 失败',
  round_finished: '轮次完成',
  holdout_validation_finished: '留样验证完成',
  holdout_validation_failed: '留样验证失败',
  factor_pool_finished: '因子池完成',
  factor_pool_failed: '因子池失败',
  factor_cache_cleanup_finished: '缓存清理完成',
  report_generation_failed: '报告生成失败',
  session_finished: '会话完成',
}

STAGE_LABELS.run_stop_requested = '终止请求已发送'
STAGE_LABELS.run_cancelled = '运行已终止'

const PHASE_LABELS: Record<string, string> = {
  original: '初始化',
  mutation: 'Mutation',
  crossover: 'Crossover',
  evolved: '进化分支',
}

const STATUS_LABELS: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
}

STATUS_LABELS.cancelled = '已终止'

AGENT_META.direction_decomposer_agent.label = '初始假说'
AGENT_META.direction_decomposer_agent.action = '生成初始研究假说'
AGENT_META.hypothesis_agent.label = '假说生成'
AGENT_META.hypothesis_agent.action = '生成可检验假说'
AGENT_META.experiment_designer_agent.label = '实验设计'
AGENT_META.experiment_designer_agent.action = '设计候选因子'
AGENT_META.factor_coder_agent.label = '因子实现'
AGENT_META.factor_coder_agent.action = '实现并校验因子表达式'
AGENT_META.backtest_runner_agent.label = '因子评估'
AGENT_META.backtest_runner_agent.action = '计算因子指标并筛选'
AGENT_META.feedback_summarizer_agent.label = '反馈诊断'
AGENT_META.feedback_summarizer_agent.action = '诊断失败原因和下一步'
AGENT_META.evolution_agent.label = '进化规划'
AGENT_META.evolution_agent.action = '规划变异/交叉任务'

Object.assign(STAGE_LABELS, {
  panel_data_loaded: '数据文件读取完成',
  market_payload_started: '数据准备开始',
  market_payload_finished: '数据摘要完成',
  data_filter_applied: '时间过滤完成',
  panel_data_normalized: '数据标准化完成',
  research_report_payload_started: '研报输入开始',
  research_report_payload_finished: '研报输入完成',
  initialization_payload_ready: '初始化输入完成',
  direction_decomposer_started: '初始假说开始',
  direction_decomposer_finished: '初始假说完成',
  direction_decomposer_failed: '初始假说失败',
  evolved_seed_decomposed: '子代假说完成',
  evolved_direction_decomposer_started: '子代假说开始',
  evolved_direction_decomposer_finished: '子代假说完成',
  evolved_direction_decomposer_failed: '子代假说失败',
  evolution_round_planned: '轮次规划完成',
  evolution_round_skipped: '轮次跳过',
  evolution_round_all_failed: '本轮全失败',
  evolution_agent_started: '进化规划开始',
  evolution_agent_finished: '进化规划完成',
  evolution_agent_failed: '进化规划失败',
  evolution_task_planned: '轨迹任务已规划',
  evolution_task_started: '轨迹任务开始',
  evolution_task_finished: '轨迹任务完成',
  agent_started: '步骤开始',
  agent_finished: '步骤完成',
  agent_failed: '步骤失败',
  round_finished: '轨迹完成',
  holdout_validation_finished: '留样验证完成',
  holdout_validation_failed: '留样验证失败',
  factor_pool_finished: '因子入库完成',
  factor_pool_failed: '因子入库失败',
  factor_cache_cleanup_finished: '缓存清理完成',
  report_generation_failed: '报告生成失败',
  session_finished: '运行完成',
  session_failed: '运行失败',
  run_stop_requested: '停止请求已发送',
  run_cancelled: '运行已终止',
})

Object.assign(PHASE_LABELS, {
  original: '初始探索',
  mutation: '变异轮',
  crossover: '交叉轮',
  evolved: '进化轮',
})

Object.assign(STATUS_LABELS, {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已终止',
})

function numberValue(value: unknown, fallback = 0) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value : ''
}

function formatTime(value?: string) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

function formatDuration(seconds: unknown) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return ''
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const min = Math.floor(seconds / 60)
  const sec = Math.round(seconds % 60)
  return `${min}m ${sec}s`
}

function paperIteration(roundIdx: unknown) {
  const value = numberValue(roundIdx, 0)
  if (value <= 0) return 0
  return Math.floor((value + 1) / 2)
}

function phaseLabel(phase?: unknown, roundIdx?: unknown) {
  const raw = stringValue(phase)
  if (raw) return PHASE_LABELS[raw] ?? raw
  const iteration = paperIteration(roundIdx)
  return iteration > 0 ? `第 ${iteration} 次进化` : '初始化'
}

function evolutionStageLabel(roundIdx: unknown, phase?: unknown) {
  const value = numberValue(roundIdx, Number.NaN)
  if (!Number.isFinite(value)) return '-'
  if (value <= 0) return '初始探索'
  const rawPhase = stringValue(phase)
  const fallbackPhase = value % 2 === 1 ? '变异轮' : '交叉轮'
  const phaseText = rawPhase ? PHASE_LABELS[rawPhase] ?? rawPhase : fallbackPhase
  return `第${paperIteration(value)}次 · ${phaseText}`
}

function compactId(value: unknown, max = 18) {
  const text = String(value ?? '').trim()
  if (!text) return ''
  return text.length > max ? `${text.slice(0, max)}...` : text
}

function eventSeverity(event: ProgressEvent): 'success' | 'error' | 'processing' | 'default' {
  const stage = event.stage ?? ''
  if (stage.includes('failed') || event.error) return 'error'
  if (stage === 'run_cancelled') return 'default'
  if (stage.includes('finished') || stage === 'agent_finished' || stage === 'session_finished') return 'success'
  if (stage.includes('started') || stage.includes('planned') || stage === 'agent_started') return 'processing'
  return 'default'
}

function buildEventText(event: ProgressEvent) {
  const stage = event.stage ?? ''
  const agentName = stringValue(event.agent_name)
  const agent = AGENT_META[agentName]?.label ?? agentName
  const summary = typeof event.progress_summary === 'object' && event.progress_summary
    ? stringValue((event.progress_summary as { text?: unknown }).text)
    : ''
  const elapsed = formatDuration(event.elapsed_seconds)
  const phase = phaseLabel(event.phase, event.round_idx)
  const loop = event.loop_index !== undefined ? `Loop ${event.loop_index}` : ''
  const context = [phase, loop].filter(Boolean).join(' / ')

  if (stage === 'panel_data_loaded') {
    const files: Array<{ filename: string; rows: number; symbols: string[]; symbol_count: number; start: string; end: string }> = event.files ?? []
    if (!files.length) return `数据文件读取完成 · 共 ${event.file_count ?? 0} 个文件 · ${event.total_rows ?? 0} 行`
    const lines = files.map((f) => {
      const sym = f.symbol_count === 1 ? f.symbols[0] : `${f.symbol_count} 个标的`
      const start = (f.start ?? '').slice(0, 10)
      const end   = (f.end   ?? '').slice(0, 10)
      return `${f.filename}  [${sym}]  ${start} → ${end}  (${f.rows.toLocaleString()} 行)`
    })
    return `读取 ${files.length} 个文件 · ${event.total_rows?.toLocaleString() ?? 0} 行\n${lines.join('\n')}`
  }
  if (stage === 'data_filter_applied') {
    return `时间过滤完成 · ${event.start_date ?? '-'} ~ ${event.end_date ?? '-'} · ${event.filtered_rows ?? 0}/${event.original_rows ?? 0} 行`
  }
  if (stage === 'panel_data_normalized') {
    return `数据标准化完成 · 行数 ${event.row_count ?? 0} · 日期 ${event.date_source ?? '-'} · 标的 ${event.symbol_source ?? '-'}`
  }
  if (stage === 'market_payload_finished') {
    return `数据摘要完成 · 特征 ${event.feature_count ?? 0} · 标的 ${event.symbol_count ?? 0} · 行数 ${event.row_count ?? 0}`
  }
  if (stage === 'research_report_payload_finished') {
    return `研报输入完成 · 来源 ${event.source_report_count ?? event.report_count ?? 0} 份${event.multi_report_synthesis_status ? ` · ${event.multi_report_synthesis_status}` : ''}`
  }
  if (stage === 'initialization_payload_ready') {
    return `初始化输入完成 · 初始假说 ${event.target_initial_hypotheses ?? 0} 条 · 参数设定 ${event.parametric_specification_count ?? 0} 组`
  }
  if (stage === 'direction_decomposer_started' || stage === 'evolved_direction_decomposer_started') {
    return `${stage === 'direction_decomposer_started' ? '初始假说' : '子代假说'}开始 · 目标 ${event.target_direction_count ?? 0} 条`
  }
  if (stage === 'direction_decomposer_finished' || stage === 'evolved_direction_decomposer_finished') {
    return `${stage === 'direction_decomposer_finished' ? '初始假说' : '子代假说'}完成 · 生成 ${event.initial_hypothesis_count ?? event.direction_count ?? 0} 条`
  }
  if (stage === 'evolution_round_planned') {
    return `${phase}规划完成 · 轨迹 ${event.task_count ?? 0} 条 · 初始 ${event.original_task_count ?? 0} / 变异 ${event.mutation_task_count ?? 0} / 交叉 ${event.crossover_task_count ?? 0}`
  }
  if (stage === 'evolution_round_all_failed') {
    return `${phase}全部失败 · 失败轨迹 ${event.new_failure_count ?? event.task_count ?? 0} 条 · 无进展轮次 ${event.no_progress_rounds ?? 0}/${event.early_stop_patience ?? 0}`
  }
  if (stage === 'evolution_agent_started') {
    return `进化规划开始 · 目标变异 ${event.mutation_target_count ?? 0} / 交叉 ${event.crossover_target_count ?? 0}`
  }
  if (stage === 'evolution_agent_finished') {
    return `进化规划完成 · 变异 ${event.mutation_task_count ?? 0} / 交叉 ${event.crossover_task_count ?? 0}`
  }
  if (stage === 'evolution_agent_failed') return `进化规划失败：${event.error ?? ''}`
  if (stage === 'evolution_task_started') {
    return `${phase}轨迹开始 · #${event.direction_id ?? '-'} · ${compactId(event.trajectory_id, 24)}`
  }
  if (stage === 'evolution_task_finished') {
    return `${phase}轨迹完成 · 步骤 ${event.agent_count ?? 0} · 错误 ${event.error_count ?? 0}`
  }
  if (stage === 'session_failed') return `运行失败：${event.error ?? ''}`
  if (stage === 'run_stop_requested') return '停止请求已发送 · 将在下一个安全检查点停止'
  if (stage === 'run_cancelled') return `运行已终止${event.error ? `：${event.error}` : ''}`

  if (stage === 'agent_started') return `${context ? `${context} · ` : ''}${agent}开始`
  if (stage === 'agent_finished') {
    return `${context ? `${context} · ` : ''}${agent}完成${elapsed ? ` · ${elapsed}` : ''}${summary ? ` · ${summary}` : ''}`
  }
  if (stage === 'agent_failed') return `${agent}失败：${event.error ?? ''}`
  if (stage === 'evolution_round_planned') {
    return `${phase}规划完成 · 任务 ${event.task_count ?? 0} 个 · 原始 ${event.original_task_count ?? 0} / 变异 ${event.mutation_task_count ?? 0} / 交叉 ${event.crossover_task_count ?? 0}`
  }
  if (stage === 'evolution_agent_started') {
    return `进化规划开始 · 目标变异 ${event.mutation_target_count ?? 0} / 交叉 ${event.crossover_target_count ?? 0}`
  }
  if (stage === 'evolution_agent_finished') {
    return `进化规划完成 · 变异 ${event.mutation_task_count ?? 0} / 交叉 ${event.crossover_task_count ?? 0}`
  }
  if (stage === 'evolution_task_started') {
    return `${phase}任务开始 · direction ${event.direction_id ?? '-'} · 轨迹 ${compactId(event.trajectory_id, 24)}`
  }
  if (stage === 'evolution_task_finished') {
    return `${phase}任务完成 · Agent ${event.agent_count ?? 0} 个 · 错误 ${event.error_count ?? 0} 个`
  }
  if (stage === 'market_payload_finished') {
    return `市场摘要完成 · 特征 ${event.feature_count ?? 0} · 标的 ${event.symbol_count ?? 0} · 行数 ${event.row_count ?? 0}`
  }
  if (stage === 'research_report_payload_finished') {
    return `研报合成完成 · 来源 ${event.source_report_count ?? event.report_count ?? 0} 份 · ${event.multi_report_synthesis_status ?? ''}`
  }
  if (stage === 'initialization_payload_ready') {
    return `初始化输入完成 · 初始假设 ${event.target_initial_hypotheses ?? 0} 条 · 参数设定 ${event.parametric_specification_count ?? 0} 组`
  }
  if (stage === 'holdout_validation_finished') {
    return `留样验证完成 · ${event.status ?? ''} · 因子 ${event.factor_count ?? 0} 个`
  }
  if (stage === 'factor_pool_finished') {
    return `因子池完成 · ${event.status ?? ''} · 入池 ${event.selected_count ?? 0} 个`
  }
  if (stage === 'factor_cache_cleanup_finished') {
    const cleanup = typeof event.cleanup === 'object' && event.cleanup ? event.cleanup as Record<string, unknown> : {}
    return `缓存清理完成 · 删除 ${cleanup.deleted_files ?? 0} 个文件`
  }
  if (stage === 'session_finished') {
    return `会话完成 · ${event.status ?? ''} · Loop ${event.loop_count ?? 0} · 错误 ${event.total_errors ?? 0}`
  }
  if (summary) return `${STAGE_LABELS[stage] ?? stage} · ${summary}`
  return STAGE_LABELS[stage] ?? stage
}

function latestEvent(events: ProgressEvent[], predicate: (event: ProgressEvent) => boolean) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (predicate(events[i])) return events[i]
  }
  return undefined
}

function activeEventKey(event: ProgressEvent) {
  const stage = event.stage ?? ''
  if (stage === 'agent_started' || stage === 'agent_finished' || stage === 'agent_failed') {
    return `agent:${event.agent_name ?? ''}:${event.trajectory_id ?? ''}`
  }
  if (stage === 'evolution_task_started' || stage === 'evolution_task_finished') {
    return `task:${event.trajectory_id ?? ''}`
  }
  if (
    stage === 'direction_decomposer_started'
    || stage === 'direction_decomposer_finished'
    || stage === 'direction_decomposer_failed'
    || stage === 'evolved_direction_decomposer_started'
    || stage === 'evolved_direction_decomposer_finished'
    || stage === 'evolved_direction_decomposer_failed'
  ) {
    return 'direction_decomposer'
  }
  if (stage === 'evolution_agent_started' || stage === 'evolution_agent_finished' || stage === 'evolution_agent_failed') {
    return 'evolution_agent'
  }
  return stage
}

function activeEventKeys(events: ProgressEvent[]) {
  const active = new Set<string>()
  for (const event of events) {
    const stage = event.stage ?? ''
    const key = activeEventKey(event)
    if (
      stage === 'agent_started'
      || stage === 'evolution_task_started'
      || stage === 'direction_decomposer_started'
      || stage === 'evolved_direction_decomposer_started'
      || stage === 'evolution_agent_started'
    ) {
      active.add(key)
    }
    if (stage.endsWith('_finished') || stage.endsWith('_failed') || stage === 'agent_finished' || stage === 'agent_failed') {
      active.delete(key)
    }
  }
  return active
}

function compactTimelineEvents(events: ProgressEvent[], compact: boolean, activeKeys: Set<string>) {
  const visible = events.filter((event, index, all) => {
    if (event.stage === 'evolution_task_started') return false
    if (event.stage === 'agent_started') {
      const key = activeEventKey(event)
      if (!activeKeys.has(key)) return false
      return !all.slice(index + 1).some((next) => next.stage === 'agent_started' && next.agent_name === event.agent_name)
    }
    return true
  })
  return visible.slice(compact ? -6 : -28).reverse()
}

function deriveAgentStates(events: ProgressEvent[], currentLoop: number | null) {
  const scopedEvents = currentLoop === null
    ? events
    : events.filter((event) => event.loop_index === currentLoop || event.loop_index === undefined)
  const states: Record<string, { status: AgentStatus; summary?: string; elapsed?: string }> = {}
  let current = 0

  for (const event of scopedEvents) {
    const stage = event.stage ?? ''
    const syntheticName = (stage.startsWith('direction_decomposer_') || stage.startsWith('evolved_direction_decomposer_'))
      ? 'direction_decomposer_agent'
      : ''
    const name = syntheticName || stringValue(event.agent_name)
    if (!name || !AGENT_ORDER.includes(name)) continue
    const idx = AGENT_ORDER.indexOf(name)
    const summary = typeof event.progress_summary === 'object' && event.progress_summary
      ? stringValue((event.progress_summary as { text?: unknown }).text)
      : ''

    if (event.stage === 'direction_decomposer_started' || event.stage === 'evolved_direction_decomposer_started') {
      states.direction_decomposer_agent = { status: 'process' }
      current = idx
    } else if (event.stage === 'direction_decomposer_finished' || event.stage === 'evolved_direction_decomposer_finished') {
      const count = event.initial_hypothesis_count ?? event.direction_count ?? 0
      states.direction_decomposer_agent = { status: 'finish', summary: `生成 ${count} 个初始方向` }
      states.hypothesis_agent = { status: 'finish', summary: `由初始化假设生成 ${count} 条候选假说` }
      current = Math.max(current, AGENT_ORDER.indexOf('experiment_designer_agent'))
    } else if (event.stage === 'direction_decomposer_failed' || event.stage === 'evolved_direction_decomposer_failed') {
      states.direction_decomposer_agent = { status: 'error', summary: String(event.error ?? '') }
      current = idx
    } else if (event.stage === 'agent_started') {
      states[name] = { status: 'process' }
      current = idx
    } else if (event.stage === 'agent_finished') {
      states[name] = { status: 'finish', summary, elapsed: formatDuration(event.elapsed_seconds) }
      current = Math.min(idx + 1, AGENT_ORDER.length - 1)
    } else if (event.stage === 'agent_failed') {
      states[name] = { status: 'error', summary: String(event.error ?? '') }
      current = idx
    }
  }

  return { states, current }
}

function deriveStageStatuses(events: ProgressEvent[], runStatus?: string) {
  const has = (names: string[]) => events.some((event) => names.includes(event.stage))
  const failed = (names: string[]) => events.some((event) => names.includes(event.stage) || (event.error && names.includes(event.stage)))

  const stages = [
    {
      key: 'prepare',
      label: '数据准备',
      detail: '行情数据、研报、参数设定',
      icon: <DatabaseOutlined />,
      started: has(['market_payload_started', 'research_report_payload_started']),
      finished: has(['initialization_payload_ready', 'market_payload_finished']) && !latestEvent(events, (e) => e.stage === 'research_report_payload_started' && !has(['research_report_payload_finished'])),
      failed: failed([]),
    },
    {
      key: 'plan',
      label: '假说进化',
      detail: '初始假说、变异、交叉',
      icon: <ForkOutlined />,
      started: has(['direction_decomposer_started', 'evolution_agent_started', 'evolution_round_planned']),
      finished: has(['evolution_round_planned', 'direction_decomposer_finished']),
      failed: failed(['direction_decomposer_failed', 'evolution_agent_failed']),
    },
    {
      key: 'execute',
      label: '轨迹执行',
      detail: '实验设计、因子实现、因子评估、反馈',
      icon: <NodeIndexOutlined />,
      started: has(['evolution_task_started', 'agent_started']),
      finished: has(['evolution_task_finished', 'round_finished']),
      failed: failed(['agent_failed']),
    },
    {
      key: 'validate',
      label: '验证入库',
      detail: '留样验证、因子入库、报告',
      icon: <FileSearchOutlined />,
      started: has(['holdout_validation_finished', 'factor_pool_finished', 'session_finished']),
      finished: runStatus === 'completed' || has(['session_finished']),
      failed: runStatus === 'failed' || failed(['holdout_validation_failed', 'factor_pool_failed', 'report_generation_failed']),
    },
  ]

  return stages.map((stage) => {
    let status: StageStatus = 'wait'
    if (stage.failed) status = 'error'
    else if (stage.finished) status = 'finish'
    else if (stage.started) status = 'process'
    return { ...stage, status }
  })
}

function deriveProgress(events: ProgressEvent[], run?: Run | RunDetail) {
  const roundEvents = events.filter((event) => event.stage === 'evolution_round_planned')
  const plannedTasks = roundEvents.reduce((sum, event) => sum + numberValue(event.task_count), 0)
  const finishedTasks = events.filter((event) => event.stage === 'evolution_task_finished').length
  const finishedAgents = events.filter((event) => event.stage === 'agent_finished').length
  const totalAgents = Math.max(AGENT_ORDER.length, finishedAgents + Number(Boolean(latestEvent(events, (e) => e.stage === 'agent_started'))))

  if (plannedTasks > 0) return Math.min(99, Math.round((finishedTasks / plannedTasks) * 100))
  if (run?.status === 'completed') return 100
  if (run?.status === 'failed') return Math.max(5, Math.min(95, Math.round((finishedAgents / totalAgents) * 100)))
  if (finishedAgents > 0) return Math.min(95, Math.round((finishedAgents / totalAgents) * 100))
  return events.length > 0 ? 8 : 0
}

function currentLoopIndex(events: ProgressEvent[], run?: Run | RunDetail) {
  const fromEvents = events.reduce<number | null>((max, event) => {
    if (typeof event.loop_index !== 'number') return max
    return max === null || event.loop_index > max ? event.loop_index : max
  }, null)
  if (fromEvents !== null) return fromEvents
  const loops = 'loops' in (run ?? {}) ? (run as RunDetail).loops : []
  if (loops?.length) return Math.max(...loops.map((loop) => loop.loop_index))
  return null
}

function currentRoundIndex(events: ProgressEvent[]) {
  return events.reduce<number | null>((max, event) => {
    if (typeof event.round_idx !== 'number') return max
    return max === null || event.round_idx > max ? event.round_idx : max
  }, null)
}

function statusTag(status?: string) {
  const color = status === 'completed' ? 'success' : status === 'failed' ? 'error' : status === 'running' ? 'processing' : status === 'cancelled' ? 'warning' : 'default'
  return <Tag color={color}>{STATUS_LABELS[status ?? ''] ?? status ?? '加载中'}</Tag>
}

function StageDot({ status }: { status: StageStatus }) {
  if (status === 'finish') return <CheckCircleOutlined />
  if (status === 'error') return <CloseCircleOutlined />
  if (status === 'process') return <ReloadOutlined spin />
  return <FieldTimeOutlined />
}

function AgentNode({ name, state }: { name: string; state?: { status: AgentStatus; summary?: string; elapsed?: string } }) {
  const meta = AGENT_META[name]
  const status = state?.status ?? 'wait'
  return (
    <Tooltip title={state?.summary || meta.action}>
      <div className={`run-agent-node is-${status}`}>
        <div className="run-agent-icon">{status === 'process' ? <ReloadOutlined spin /> : meta.icon}</div>
        <div className="run-agent-text">
          <Text strong className="run-agent-title">{meta.label}</Text>
          <Text type="secondary" className="run-agent-sub">
            {state?.elapsed || meta.action}
          </Text>
        </div>
      </div>
    </Tooltip>
  )
}

function TimelineDot({ event, active }: { event: ProgressEvent; active: boolean }) {
  const severity = eventSeverity(event)
  if (severity === 'error') return <CloseCircleOutlined style={{ color: '#cf1322' }} />
  if (severity === 'success') return <CheckCircleOutlined style={{ color: '#237804' }} />
  if (severity === 'processing' && active) return <ReloadOutlined spin style={{ color: '#0958d9' }} />
  return <FieldTimeOutlined style={{ color: '#8c8c8c' }} />
}

export default function RunProgressView({ run, runId, events, connected = false, compact = false }: RunProgressViewProps) {
  const navigate = useNavigate()
  const [stopping, setStopping] = useState(false)
  const currentLoop = currentLoopIndex(events, run)
  const currentRound = currentRoundIndex(events)
  const progress = deriveProgress(events, run)
  const agentStates = deriveAgentStates(events, currentLoop)
  const stageStatuses = deriveStageStatuses(events, run?.status)
  const activeAgent = latestEvent(events, (event) => event.stage === 'agent_started')
  const activeTask = latestEvent(events, (event) => event.stage === 'evolution_task_started' || event.stage === 'evolution_task_planned')
  const latest = latestEvent(events, () => true)
  const errors = events.filter((event) => event.error || event.stage?.includes('failed'))
  const plannedTasks = events
    .filter((event) => event.stage === 'evolution_round_planned')
    .reduce((sum, event) => sum + numberValue(event.task_count), 0)
  const finishedTasks = events.filter((event) => event.stage === 'evolution_task_finished').length
  const latestMarket = latestEvent(events, (event) => event.stage === 'market_payload_finished')
  const latestSystem = latestEvent(events, (event) => (
    event.stage === 'holdout_validation_finished'
    || event.stage === 'factor_pool_finished'
    || event.stage === 'session_finished'
    || event.stage === 'factor_cache_cleanup_finished'
  ))
  const activeKeys = activeEventKeys(events)
  const timeline = compactTimelineEvents(events, compact, activeKeys)
  const currentPhase = activeTask?.phase ?? latest?.phase
  const currentEvolutionStage = evolutionStageLabel(currentRound, currentPhase)
  const currentTrajectory = activeTask?.trajectory_id ?? latest?.trajectory_id
  const stopRequested = events.some((event) => event.stage === 'run_stop_requested')

  async function handleStopRun() {
    setStopping(true)
    try {
      const res = await fetch(`/api/runs/${runId}/stop`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      message.success('已请求终止运行')
    } catch {
      message.error('终止请求失败，请检查后端服务')
    } finally {
      setStopping(false)
    }
  }

  return (
    <section className={`run-progress-shell ${compact ? 'is-compact' : ''}`}>
      <div className="run-progress-header">
        <div className="run-title-block">
          <Space size={10} align="center" wrap>
            <ThunderboltOutlined className="run-title-icon" />
            <Title level={compact ? 5 : 4} style={{ margin: 0 }}>运行进度</Title>
            <Text code className="run-id-text">{runId}</Text>
            {statusTag(run?.status)}
            {connected ? <Badge status="processing" text="实时" /> : <Badge status="default" text="离线" />}
          </Space>
          {!compact && (
            <Text type="secondary" className="run-subtitle">
              {run?.evolution_mode ? 'QuantaAlpha evolution workflow' : 'Single-loop factor mining workflow'}
            </Text>
          )}
        </div>
        <Space wrap>
          {run?.status === 'running' && (
            <Popconfirm
              title="终止当前运行？"
              description="会在下一个安全检查点停止，并保留已有日志。"
              okText="终止"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={handleStopRun}
            >
              <Button danger icon={<StopOutlined />} loading={stopping} disabled={stopRequested}>
                {stopRequested ? '终止中' : '终止'}
              </Button>
            </Popconfirm>
          )}
          <Button icon={<PlayCircleOutlined />} type={compact ? 'default' : 'primary'} onClick={() => navigate(`/runs/${runId}`)}>
            监控
          </Button>
          <Button icon={<ExperimentOutlined />} onClick={() => navigate(`/runs/${runId}/factors`)}>
            因子库
          </Button>
          <Button icon={<BranchesOutlined />} onClick={() => navigate(`/runs/${runId}/lineage`)}>
            进化树
          </Button>
          <Button icon={<FileTextOutlined />} onClick={() => navigate(`/runs/${runId}/report`)}>
            报告
          </Button>
        </Space>
      </div>

      <div className="run-progress-meter">
        <Progress
          percent={progress}
          status={run?.status === 'failed' ? 'exception' : run?.status === 'completed' ? 'success' : 'active'}
          strokeColor={run?.status === 'failed' ? undefined : { from: '#1677ff', to: '#52c41a' }}
        />
      </div>

      <div className="run-kpi-grid">
        <div className="run-kpi">
          <Text type="secondary">当前轨迹</Text>
          <strong>{currentLoop ?? '-'}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">进化阶段</Text>
          <strong>{currentEvolutionStage}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">已规划轨迹</Text>
          <strong>{plannedTasks ? `${finishedTasks}/${plannedTasks}` : `${finishedTasks}`}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">事件</Text>
          <strong>{events.length}</strong>
        </div>
        {!compact && (
          <>
            <div className="run-kpi">
              <Text type="secondary">特征</Text>
              <strong>{latestMarket?.feature_count === undefined ? '-' : String(latestMarket.feature_count)}</strong>
            </div>
            <div className="run-kpi">
              <Text type="secondary">错误</Text>
              <strong className={errors.length ? 'danger-text' : ''}>{errors.length}</strong>
            </div>
          </>
        )}
      </div>

      {!compact && (
      <div className="run-stage-rail">
        {stageStatuses.map((stage) => (
          <div key={stage.key} className={`run-stage-item is-${stage.status}`}>
            <div className="run-stage-icon">
              <StageDot status={stage.status} />
            </div>
            <div>
              <Text strong>{stage.label}</Text>
              <Text type="secondary">{stage.detail}</Text>
            </div>
          </div>
        ))}
      </div>
      )}

      <div className="run-current-strip">
        <div className="run-current-main">
          <Text type="secondary">当前步骤</Text>
          <Text strong>
            {activeAgent?.agent_name
              ? `${AGENT_META[stringValue(activeAgent.agent_name)]?.label ?? activeAgent.agent_name} · ${AGENT_META[stringValue(activeAgent.agent_name)]?.action ?? ''}`
              : activeTask
                ? buildEventText(activeTask)
                : latest
                  ? buildEventText(latest)
                  : '等待事件'}
          </Text>
        </div>
        <div className="run-current-meta">
          {currentPhase && <Tag color="blue">{phaseLabel(currentPhase, currentRound)}</Tag>}
          {currentTrajectory && <Tag>{compactId(currentTrajectory, 26)}</Tag>}
          {latest?.timestamp && <Text type="secondary">{formatTime(latest.timestamp)}</Text>}
        </div>
      </div>

      <div className="run-body-grid">
        {!compact && (
        <div className="run-panel">
          <div className="run-panel-title">
            <ApiOutlined />
            <Text strong>Agent 链路</Text>
          </div>
          <div className="run-agent-grid">
            {AGENT_ORDER.map((name) => (
              <AgentNode key={name} name={name} state={agentStates.states[name]} />
            ))}
          </div>
        </div>
        )}

        <div className="run-panel">
          <div className="run-panel-title">
            <FieldTimeOutlined />
            <Text strong>{compact ? '最近事件' : '事件时间线'}</Text>
          </div>
          {errors.length > 0 && !compact && (
            <Alert
              type="error"
              showIcon
              message={`检测到 ${errors.length} 个失败事件`}
              description={
                <Collapse
                  ghost
                  size="small"
                  items={[{
                    key: 'errors',
                    label: <Text style={{ fontSize: 12, color: '#cf1322' }}>点击展开查看详情</Text>,
                    children: (
                      <div style={{ maxHeight: 260, overflowY: 'auto' }}>
                        {errors.map((e, i) => (
                          <div key={i} style={{ borderBottom: '1px solid #ffccc7', padding: '6px 0', fontSize: 12 }}>
                            <Space size={6}>
                              <Text type="secondary" style={{ fontSize: 11 }}>{formatTime(e.timestamp)}</Text>
                              <Tag color="red" style={{ fontSize: 11 }}>{STAGE_LABELS[e.stage] ?? e.stage}</Tag>
                            </Space>
                            <div style={{ color: '#cf1322', marginTop: 2, wordBreak: 'break-all' }}>
                              {String(e.error ?? buildEventText(e))}
                            </div>
                          </div>
                        ))}
                      </div>
                    ),
                  }]}
                  style={{ marginTop: 4 }}
                />
              }
              style={{ marginBottom: 12 }}
            />
          )}
          {latestSystem && !compact && (
            <Alert
              type={eventSeverity(latestSystem) === 'error' ? 'error' : 'success'}
              showIcon
              message={buildEventText(latestSystem)}
              style={{ marginBottom: 12 }}
            />
          )}
          <Timeline
            className="run-timeline"
            items={timeline.map((event, index) => ({
              key: `${event.timestamp ?? index}-${event.stage}-${index}`,
              dot: <TimelineDot event={event} active={activeKeys.has(activeEventKey(event))} />,
              children: (
                <div className="run-timeline-row">
                  <div className="run-timeline-head">
                    <Text type="secondary">{formatTime(event.timestamp)}</Text>
                    <Tag color={eventSeverity(event) === 'error' ? 'red' : eventSeverity(event) === 'success' ? 'green' : eventSeverity(event) === 'processing' && activeKeys.has(activeEventKey(event)) ? 'blue' : 'default'}>
                      {STAGE_LABELS[event.stage] ?? event.stage}
                    </Tag>
                  </div>
                  <Text style={{ whiteSpace: 'pre-wrap', fontFamily: event.stage === 'panel_data_loaded' ? 'monospace' : undefined, fontSize: event.stage === 'panel_data_loaded' ? 12 : undefined }}>{buildEventText(event)}</Text>
                </div>
              ),
            }))}
          />
          {timeline.length === 0 && <Text type="secondary">等待进度事件</Text>}
        </div>
      </div>

      {!compact && (
        <div className="run-link-row">
          <Link to={`/runs/${runId}/factors`}>查看本次因子结果</Link>
          <Link to={`/runs/${runId}/lineage`}>查看进化树</Link>
          <Link to={`/runs/${runId}/report`}>查看分析报告</Link>
        </div>
      )}

      <style>{`
        .run-progress-shell {
          border: 1px solid #e5e7eb;
          background: #ffffff;
          border-radius: 8px;
          padding: 18px;
          box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
        }
        .run-progress-shell.is-compact {
          margin-bottom: 16px;
        }
        .run-progress-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 12px;
        }
        .run-title-block {
          display: flex;
          flex-direction: column;
          gap: 4px;
          min-width: 0;
        }
        .run-title-icon {
          color: #1677ff;
          font-size: 20px;
        }
        .run-id-text {
          max-width: 280px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .run-subtitle {
          font-size: 12px;
        }
        .run-progress-meter {
          margin-bottom: 14px;
        }
        .run-kpi-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 10px;
          margin-bottom: 14px;
        }
        .run-progress-shell.is-compact .run-kpi-grid {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .run-kpi {
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-height: 64px;
          border: 1px solid #eef0f3;
          background: #fafafa;
          border-radius: 8px;
          padding: 10px 12px;
        }
        .run-kpi strong {
          font-size: 20px;
          line-height: 1.25;
          color: #111827;
        }
        .danger-text {
          color: #cf1322 !important;
        }
        .run-stage-rail {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin-bottom: 14px;
        }
        .run-stage-item {
          display: grid;
          grid-template-columns: 34px minmax(0, 1fr);
          gap: 10px;
          align-items: center;
          border: 1px solid #eef0f3;
          border-radius: 8px;
          padding: 10px;
          background: #ffffff;
        }
        .run-stage-item > div:last-child {
          display: flex;
          min-width: 0;
          flex-direction: column;
          gap: 2px;
        }
        .run-stage-icon {
          width: 34px;
          height: 34px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          background: #f5f5f5;
          color: #8c8c8c;
        }
        .run-stage-item.is-process {
          border-color: #91caff;
          background: #f0f7ff;
        }
        .run-stage-item.is-process .run-stage-icon {
          background: #e6f4ff;
          color: #0958d9;
        }
        .run-stage-item.is-finish {
          border-color: #b7eb8f;
          background: #f6ffed;
        }
        .run-stage-item.is-finish .run-stage-icon {
          background: #d9f7be;
          color: #237804;
        }
        .run-stage-item.is-error {
          border-color: #ffa39e;
          background: #fff1f0;
        }
        .run-stage-item.is-error .run-stage-icon {
          background: #ffccc7;
          color: #cf1322;
        }
        .run-current-strip {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          border: 1px solid #d6e4ff;
          background: #f5f8ff;
          border-radius: 8px;
          padding: 12px 14px;
          margin-bottom: 14px;
        }
        .run-current-main {
          display: flex;
          min-width: 0;
          flex-direction: column;
          gap: 2px;
        }
        .run-current-main .ant-typography {
          overflow-wrap: anywhere;
        }
        .run-current-meta {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          flex-wrap: wrap;
          gap: 6px;
        }
        .run-body-grid {
          display: grid;
          grid-template-columns: minmax(300px, 0.95fr) minmax(360px, 1.35fr);
          gap: 14px;
        }
        .run-progress-shell.is-compact .run-body-grid {
          grid-template-columns: minmax(0, 1fr);
        }
        .run-panel {
          border: 1px solid #eef0f3;
          border-radius: 8px;
          padding: 14px;
          background: #ffffff;
          min-width: 0;
        }
        .run-panel-title {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
          color: #111827;
        }
        .run-agent-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 8px;
        }
        .run-agent-node {
          display: grid;
          grid-template-columns: 36px minmax(0, 1fr);
          gap: 10px;
          align-items: center;
          min-height: 54px;
          border: 1px solid #eef0f3;
          border-radius: 8px;
          padding: 8px 10px;
          background: #fafafa;
        }
        .run-agent-icon {
          width: 36px;
          height: 36px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          background: #ffffff;
          color: #8c8c8c;
        }
        .run-agent-text {
          display: flex;
          min-width: 0;
          flex-direction: column;
          gap: 1px;
        }
        .run-agent-title,
        .run-agent-sub {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .run-agent-node.is-process {
          border-color: #91caff;
          background: #f0f7ff;
        }
        .run-agent-node.is-process .run-agent-icon {
          color: #0958d9;
        }
        .run-agent-node.is-finish {
          border-color: #b7eb8f;
          background: #f6ffed;
        }
        .run-agent-node.is-finish .run-agent-icon {
          color: #237804;
        }
        .run-agent-node.is-error {
          border-color: #ffa39e;
          background: #fff1f0;
        }
        .run-agent-node.is-error .run-agent-icon {
          color: #cf1322;
        }
        .run-timeline {
          max-height: 560px;
          overflow: auto;
          padding: 2px 4px 0 0;
        }
        .run-progress-shell.is-compact .run-timeline {
          max-height: 290px;
        }
        .run-timeline-row {
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding-bottom: 2px;
        }
        .run-timeline-head {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: 6px;
        }
        .run-link-row {
          display: flex;
          gap: 14px;
          margin-top: 12px;
        }
        @media (max-width: 1100px) {
          .run-kpi-grid,
          .run-progress-shell.is-compact .run-kpi-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
          .run-stage-rail {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .run-body-grid,
          .run-progress-shell.is-compact .run-body-grid {
            grid-template-columns: 1fr;
          }
        }
        @media (max-width: 720px) {
          .run-progress-header,
          .run-current-strip {
            align-items: stretch;
            flex-direction: column;
          }
          .run-kpi-grid,
          .run-progress-shell.is-compact .run-kpi-grid,
          .run-stage-rail {
            grid-template-columns: 1fr;
          }
          .run-id-text {
            max-width: 220px;
          }
        }
      `}</style>
    </section>
  )
}
