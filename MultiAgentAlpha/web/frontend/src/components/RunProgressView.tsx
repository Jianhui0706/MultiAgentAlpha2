import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Alert,
  Badge,
  Button,
  Collapse,
  Drawer,
  List,
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
import RewardBreakdown, { type RewardBreakdownData } from './RewardBreakdown'

const { Paragraph, Text, Title } = Typography

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
  preprocessing_required: '数据预处理中（计算预测标签）',
  preprocessing_complete: '数据预处理完成',
  preprocessing_failed: '数据预处理失败（缺少预测标签）',
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
  factor_coder_started: '因子实现准备',
  factor_coder_panel_load_started: '读取因子数据',
  factor_coder_panel_load_finished: '因子数据读取完成',
  factor_coder_panel_load_failed: '因子数据读取失败',
  factor_coder_panel_load_skipped: '跳过因子数据读取',
  factor_coder_config_ready: '因子实现配置完成',
  factor_coder_loop_started: '因子实现轮次开始',
  factor_coder_loop_finished: '因子实现轮次完成',
  factor_coder_factor_started: '因子实现开始',
  factor_coder_factor_evaluation_started: '因子校验开始',
  factor_coder_value_started: '因子值计算开始',
  factor_coder_value_finished: '因子值计算完成',
  factor_coder_factor_finished: '因子实现完成',
  factor_coder_finished: '因子实现汇总完成',
  report_generation_failed: '报告生成失败',
  session_finished: '运行完成',
  session_failed: '运行失败',
  run_stop_requested: '停止请求已发送',
  run_cancelled: '运行已终止',
  bootstrap_quality_gate_decision: '假说资格判定',
  bootstrap_refill_round_started: '假说重生轮开始',
  bootstrap_quality_gate_finished: '资格赛收官',
  bootstrap_all_rejected: '资格赛全军覆没',
  novelty_context_prepared: '新颖性上下文就绪',
  report_generated: '报告已生成',
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

function optionalNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
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

function formatScore(value: unknown, digits = 4) {
  const num = optionalNumber(value)
  return num === null ? '' : num.toFixed(digits)
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
  return iteration > 0 ? `第 ${iteration} 次进化` : '初始探索'
}

function evolutionStageLabel(roundIdx: unknown, phase?: unknown) {
  const value = numberValue(roundIdx, Number.NaN)
  if (!Number.isFinite(value)) return '-'
  if (value <= 0) return '初始探索'
  const rawPhase = stringValue(phase)
  const fallbackPhase = value % 2 === 1 ? '变异轮' : '交叉轮'
  const phaseText = rawPhase ? PHASE_LABELS[rawPhase] ?? rawPhase : fallbackPhase
  return `第 ${paperIteration(value)} 次 · ${phaseText}`
}

function compactId(value: unknown, max = 18) {
  const text = String(value ?? '').trim()
  if (!text) return ''
  return text.length > max ? `${text.slice(0, max)}...` : text
}

function eventSeverity(event: ProgressEvent): 'success' | 'error' | 'processing' | 'default' {
  const stage = event.stage ?? ''
  if (stage === 'bootstrap_all_rejected') return 'error'
  if (stage === 'bootstrap_quality_gate_decision') return event.rejected === true ? 'default' : 'success'
  if (stage.includes('failed') || event.error) return 'error'
  if (stage === 'run_cancelled') return 'default'
  if (stage.includes('finished') || stage === 'agent_finished' || stage === 'session_finished') return 'success'
  if (stage.includes('started') || stage.includes('planned') || stage === 'agent_started') return 'processing'
  return 'default'
}

function translateFactorGroupShortfall(text: string) {
  const groupLabels: Record<string, string> = {
    mainline: '主线因子组',
    counterfactual: '反事实因子组',
    adjacent: '邻近探索因子组',
  }
  const segments = text.split(';').map((item) => item.trim()).filter(Boolean)
  const translated = segments.map((segment) => {
    const match = segment.match(/^(\w+)\s+actual=(\d+)\s+required=(\d+)$/)
    if (!match) return segment
    const [, group, actual, required] = match
    const label = groupLabels[group] ?? group
    return `${label}（${group}）实际 ${actual} 个，要求 ${required} 个`
  })
  return translated.join('；')
}

function translateErrorText(value: unknown) {
  const text = String(value ?? '').trim()
  if (!text) return ''

  const contextLimitMatch = text.match(/Input tokens exceed the configured limit of (\d+) tokens\. Your messages resulted in (\d+) tokens\./)
  if (contextLimitMatch) {
    const [, limit, actual] = contextLimitMatch
    return `输入上下文过长：本次请求约 ${Number(actual).toLocaleString()} tokens，超过模型配置上限 ${Number(limit).toLocaleString()} tokens。请压缩进化规划上下文或减少注入的历史轨迹内容。`
  }

  const requiredGroupMatch = text.match(
    /^Experiment designer failed to satisfy required factor groups after (\d+) attempts: MISSING REQUIRED FACTOR GROUPS: (.+?)\. Retry the designer rather than accepting an incomplete mainline\/counterfactual group\.$/,
  )
  if (requiredGroupMatch) {
    const [, attempts, shortfalls] = requiredGroupMatch
    return `实验设计连续 ${attempts} 次未满足必需因子组：${translateFactorGroupShortfall(shortfalls)}。请重试实验设计，不要接受缺少必需因子组的不完整方案。`
  }

  const shortfallOnlyMatch = text.match(/^MISSING REQUIRED FACTOR GROUPS: (.+)$/)
  if (shortfallOnlyMatch) {
    return `缺少必需因子组：${translateFactorGroupShortfall(shortfallOnlyMatch[1])}`
  }

  const noValidFactorsMatch = text.match(
    /^Experiment designer failed to generate valid factors after (\d+) attempts\. Feedback: (.+)$/,
  )
  if (noValidFactorsMatch) {
    const [, attempts, feedback] = noValidFactorsMatch
    return `实验设计连续 ${attempts} 次未能生成通过校验的有效因子。反馈：${feedback}`
  }

  return text
}

function metricValue(metrics: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const value = optionalNumber(metrics[key])
    if (value !== null) return value
  }
  return null
}

function summarizeBestFactor(event: ProgressEvent) {
  const factorSummaries = Array.isArray(event.factor_summaries)
    ? event.factor_summaries.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
    : []
  if (!factorSummaries.length) return ''

  const scored = factorSummaries
    .map((factor) => ({
      factor,
      reward: optionalNumber(factor.reward_score),
      metrics: factor.metrics && typeof factor.metrics === 'object'
        ? factor.metrics as Record<string, unknown>
        : {},
    }))
    // mainline 优先：adjacent/counterfactual 对照因子不代表假说成绩，不抢"最佳因子"展示位
    .sort((a, b) => {
      const aMain = optionalNumber(a.metrics.mainline_factor) ?? 1
      const bMain = optionalNumber(b.metrics.mainline_factor) ?? 1
      if (aMain !== bMain) return bMain - aMain
      return (b.reward ?? Number.NEGATIVE_INFINITY) - (a.reward ?? Number.NEGATIVE_INFINITY)
    })
  const best = scored[0]
  const name = stringValue(best.factor.factor_name) || stringValue(best.factor.name)
  const pieces: string[] = []
  if (name) pieces.push(`最佳因子 ${compactId(name, 48)}`)
  if (best.reward !== null) pieces.push(`Reward=${formatScore(best.reward)}`)
  const rankIc = metricValue(best.metrics, 'Rank IC', 'RankIC', 'rank_ic')
  const rankIcir = metricValue(best.metrics, 'Rank ICIR', 'RankICIR', 'rank_icir')
  if (rankIc !== null) pieces.push(`Rank IC=${formatScore(rankIc)}`)
  if (rankIcir !== null) pieces.push(`Rank ICIR=${formatScore(rankIcir)}`)
  return pieces.join('，')
}

function summarizeReward(event: ProgressEvent) {
  const pieces: string[] = []
  const reward = optionalNumber(event.reward_score)
  if (reward !== null) pieces.push(`轨迹 Reward=${formatScore(reward)}`)
  const bestFactor = summarizeBestFactor(event)
  if (bestFactor) pieces.push(bestFactor)
  return pieces.join('；')
}

// 统一的事件文案生成：每条「结果短语 · 指标=值 · 指标=值」，
// 中文术语与单位保持一致（步骤/轨迹/因子/条/个/份/组/行），失败用「…：原因」。
function reportSummaryText(event: ProgressEvent) {
  const direct = stringValue(event.report_summary_text)
  if (direct) return direct
  const summary = event.report_summary
  if (!summary || typeof summary !== 'object') return ''
  const record = summary as Record<string, unknown>
  const proxy = Array.isArray(record.proxy_candidates)
    ? (record.proxy_candidates as unknown[]).map((item) => stringValue(item)).filter(Boolean).slice(0, 5).join(', ')
    : ''
  return [
    stringValue(record.thesis_summary),
    stringValue(record.mechanism_summary),
    proxy ? `Proxy: ${proxy}` : '',
    stringValue(record.compact_excerpt),
  ].filter(Boolean).join(' / ')
}

interface ResearchReportDetail {
  run_id: string
  report_count: number
  reports: Array<Record<string, unknown>>
  combined?: Record<string, unknown>
  errors?: Array<Record<string, unknown>>
  direction_decomposer?: Record<string, unknown>
  factor_reference_chain?: string[]
}

function reportSeedText(event: ProgressEvent) {
  const seed = event.report_initial_seed
  if (!seed || typeof seed !== 'object') return ''
  const record = seed as Record<string, unknown>
  const seedType = stringValue(record.seed_type)
  const seedText = stringValue(record.seed_text) || stringValue(record.hypothesis_hint) || stringValue(record.expression)
  if (!seedText) return seedType
  return seedType ? `${seedType}: ${seedText}` : seedText
}

function hypothesisListText(event: ProgressEvent) {
  const raw = Array.isArray(event.initial_hypotheses)
    ? event.initial_hypotheses
    : Array.isArray(event.directions)
      ? event.directions
      : []
  const items = raw
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .slice(0, 5)
    .map((item, index) => {
      const title = stringValue(item.title)
      const hypothesis = stringValue(item.hypothesis) || stringValue(item.direction) || stringValue(item.thesis)
      const proxy = stringValue(item.core_proxy_family)
      const body = [title, hypothesis].filter(Boolean).join(': ')
      return `${index + 1}. ${body || 'unnamed hypothesis'}${proxy ? ` (${proxy})` : ''}`
    })
  if (!items.length) return ''
  const more = raw.length > items.length ? ` 等 ${raw.length} 条` : ''
  return `${items.join('；')}${more}`
}

function reportSuggestedHypothesesText(event: ProgressEvent) {
  const raw = Array.isArray(event.report_suggested_hypotheses) ? event.report_suggested_hypotheses : []
  const items = raw
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .slice(0, 4)
    .map((item, index) => {
      const title = stringValue(item.title)
      const hypothesis = stringValue(item.hypothesis)
      return `${index + 1}. ${[title, hypothesis].filter(Boolean).join(': ') || 'report hypothesis'}`
    })
  if (!items.length) return ''
  return items.join('；')
}

function reportCountFromRun(run?: Run | RunDetail) {
  const args = (run as RunDetail | undefined)?.args
  const fromArgs = Array.isArray(args?.research_report_pdfs) ? args?.research_report_pdfs.length : 0
  const fromRun = Array.isArray(run?.research_report_pdfs) ? run?.research_report_pdfs.length : 0
  return Math.max(
    fromArgs ?? 0,
    fromRun ?? 0,
    stringValue(run?.research_report_pdf) ? 1 : 0,
  )
}

function listText(value: unknown) {
  if (Array.isArray(value)) return value.map((item) => stringValue(item)).filter(Boolean).join(', ')
  return stringValue(value)
}

function renderHypothesisItems(items: unknown) {
  const rows = Array.isArray(items) ? items.filter((item) => item && typeof item === 'object') as Record<string, unknown>[] : []
  if (!rows.length) return <Text type="secondary">暂无分条假设</Text>
  return (
    <List
      size="small"
      dataSource={rows}
      renderItem={(item, index) => (
        <List.Item>
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Text strong>{index + 1}. {stringValue(item.title) || stringValue(item.source_report_name) || 'hypothesis'}</Text>
            <Text>{stringValue(item.hypothesis)}</Text>
            {stringValue(item.source_report_name) && (
              <Text type="secondary">来源：{stringValue(item.source_report_name)}</Text>
            )}
          </Space>
        </List.Item>
      )}
    />
  )
}

function renderReportDetail(detail: ResearchReportDetail | null) {
  if (!detail) return <Text type="secondary">暂无研报分析内容</Text>
  const reports = Array.isArray(detail.reports) ? detail.reports : []
  const combined = detail.combined && Object.keys(detail.combined).length ? detail.combined : null
  return (
    <Space direction="vertical" size={14} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="报告如何影响因子实现"
        description={(detail.factor_reference_chain ?? []).join(' / ') || '研报进入假设生成和实验设计，因子实现继承实验设计产出的 experiment_spec。'}
      />
      {combined && (
        <Collapse
          defaultActiveKey={['combined']}
          items={[{
            key: 'combined',
            label: `多研报综合 (${detail.report_count} 份)`,
            children: (
              <Space direction="vertical" size={10} style={{ width: '100%' }}>
                <Paragraph>{stringValue(combined.summary_text)}</Paragraph>
                <div>
                  <Text strong>综合可测假设</Text>
                  {renderHypothesisItems(combined.suggested_hypotheses)}
                </div>
                <Paragraph style={{ whiteSpace: 'pre-wrap', maxHeight: 280, overflow: 'auto' }}>
                  {stringValue(combined.report_text)}
                </Paragraph>
              </Space>
            ),
          }]}
        />
      )}
      <Collapse
        defaultActiveKey={reports.slice(0, 1).map((item) => String(item.index ?? 1))}
        items={reports.map((report, idx) => {
          const summary = (report.summary && typeof report.summary === 'object' ? report.summary : {}) as Record<string, unknown>
          const source = (report.source && typeof report.source === 'object' ? report.source : {}) as Record<string, unknown>
          return {
            key: String(report.index ?? idx + 1),
            label: `${report.index ?? idx + 1}. ${stringValue(report.source_report_name) || stringValue(source.file_name) || 'research report'}`,
            children: (
              <Space direction="vertical" size={10} style={{ width: '100%' }}>
                <Space wrap>
                  <Tag>页数 {stringValue(source.page_count) || '-'}</Tag>
                  {listText(summary.proxy_candidates) && <Tag color="blue">Proxy {listText(summary.proxy_candidates)}</Tag>}
                </Space>
                <div>
                  <Text strong>研报摘要</Text>
                  <Paragraph>{stringValue(report.summary_text) || stringValue(summary.compact_excerpt)}</Paragraph>
                </div>
                <div>
                  <Text strong>分条可测假设</Text>
                  {renderHypothesisItems(report.suggested_hypotheses)}
                </div>
                <div>
                  <Text strong>A0 初始 Seed</Text>
                  <Paragraph>{stringValue((report.initial_seed as Record<string, unknown> | undefined)?.seed_text) || stringValue((report.initial_seed as Record<string, unknown> | undefined)?.hypothesis_hint)}</Paragraph>
                </div>
                <div>
                  <Text strong>研报全文/清洗文本</Text>
                  <Paragraph style={{ whiteSpace: 'pre-wrap', maxHeight: 320, overflow: 'auto' }}>
                    {stringValue(report.report_text)}
                  </Paragraph>
                </div>
              </Space>
            ),
          }
        })}
      />
      <div>
        <Text strong>A0 实际生成的初始假设</Text>
        {Array.isArray((detail.direction_decomposer as Record<string, unknown> | undefined)?.initial_hypotheses)
          && ((detail.direction_decomposer as Record<string, unknown>).initial_hypotheses as unknown[]).length > 0
          ? renderHypothesisItems((detail.direction_decomposer as Record<string, unknown>).initial_hypotheses)
          : (
            <Text type="secondary" style={{ display: 'block', marginTop: 6 }}>
              暂未读取到 A0 假设明细。若运行已过“初始假设完成”，请点击刷新；旧运行可能只记录了假设数量，没有记录分条文本。
            </Text>
          )}
      </div>
      {(detail.errors ?? []).length > 0 && (
        <Alert
          type="warning"
          showIcon
          message="部分研报解析失败"
          description={(detail.errors ?? []).map((item) => `${stringValue(item.path)}: ${stringValue(item.error)}`).join('；')}
        />
      )}
    </Space>
  )
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
  const loop = event.loop_index !== undefined ? `轨迹评估 ${event.loop_index}` : ''
  const context = [phase, loop].filter(Boolean).join(' / ')
  const join = (...parts: Array<string | false | null | undefined>) => parts.filter(Boolean).join(' · ')
  const n = (value: unknown) => {
    const x = optionalNumber(value)
    return x === null ? '0' : x.toLocaleString('zh-CN')
  }

  switch (stage) {
    // ── 数据准备 / 预处理 ──
    case 'preprocessing_required':
      return `缺少预测标签 ${event.prediction_horizon ?? ''}，正在从 close 价格计算 fwd_returns…`
    case 'preprocessing_failed': {
      const avail = Array.isArray(event.available_fwd_columns) && event.available_fwd_columns.length
        ? `（现有标签：${(event.available_fwd_columns as string[]).join(', ')}）`
        : ''
      return `❌ ${event.reason ?? '数据缺少预测标签，运行已终止'}${avail}`
    }
    case 'preprocessing_complete': {
      const barMin = event.bar_minutes ? `K线 ${event.bar_minutes}min` : ''
      const saved = event.saved_to ? `已存 ${String(event.saved_to).split(/[\\/]/).pop()}` : ''
      if (event.was_computed === false) {
        const base = typeof event.message === 'string' && event.message ? `✅ ${event.message}` : '✅ 数据已含 fwd 标签，无需计算'
        return join(base, barMin)
      }
      const cols = event.computed_columns ? String(event.computed_columns).replace(/^derived from close[^,]*,\s*computed=/, '') : ''
      return join('✅ 预处理完成，已计算 fwd 标签', barMin, cols, saved)
    }
    case 'panel_data_loaded': {
      const files = Array.isArray(event.files)
        ? event.files as Array<{ filename: string; rows: number; symbols: string[]; symbol_count: number; start: string; end: string }>
        : []
      if (!files.length) return join('数据文件读取完成', `${event.file_count ?? 0} 个文件`, `${n(event.total_rows)} 行`)
      const lines = files.map((f) => {
        const sym = f.symbol_count === 1 ? f.symbols[0] : `${f.symbol_count} 个标的`
        const start = (f.start ?? '').slice(0, 10)
        const end = (f.end ?? '').slice(0, 10)
        return `${f.filename}  [${sym}]  ${start} → ${end}  (${f.rows.toLocaleString()} 行)`
      })
      return `读取 ${files.length} 个文件 · ${n(event.total_rows)} 行\n${lines.join('\n')}`
    }
    case 'data_filter_applied':
      return join('时间过滤完成', `区间 ${event.start_date ?? '-'} ~ ${event.end_date ?? '-'}`, `保留 ${n(event.filtered_rows)}/${n(event.original_rows)} 行`)
    case 'panel_data_normalized': {
      const effFile = event.effective_panel_data_path
        ? `回测读取 ${String(event.effective_panel_data_path).split(/[\\/]/).pop()}`
        : ''
      return join('数据标准化完成', `${n(event.row_count)} 行`, `日期源 ${event.date_source ?? '-'}`, `标的源 ${event.symbol_source ?? '-'}`, effFile)
    }
    case 'market_payload_finished':
      return join('数据摘要完成', `特征 ${event.feature_count ?? 0} 个`, `标的 ${event.symbol_count ?? 0} 个`, `${n(event.row_count)} 行`)
    case 'research_report_payload_finished': {
      const names = Array.isArray(event.source_report_names)
        ? (event.source_report_names as unknown[]).map((item) => stringValue(item)).filter(Boolean).slice(0, 3).join(', ')
        : ''
      const summaryText = reportSummaryText(event)
      const seedText = reportSeedText(event)
      const suggestions = reportSuggestedHypothesesText(event)
      return join(
        '研报输入完成',
        `来源 ${event.source_report_count ?? event.report_count ?? 0} 份`,
        names ? `文件 ${compactId(names, 80)}` : '',
        summaryText ? `摘要 ${compactId(summaryText, 260)}` : '',
        suggestions ? `可测假设 ${compactId(suggestions, 320)}` : '',
        seedText ? `初始 Seed ${compactId(seedText, 220)}` : '',
        event.multi_report_synthesis_status ? String(event.multi_report_synthesis_status) : '',
      )
    }
    case 'initialization_payload_ready':
      return join('初始化输入完成', `初始假说 ${event.target_initial_hypotheses ?? 0} 条`, `参数设定 ${event.parametric_specification_count ?? 0} 组`)

    // ── 假说进化 ──
    case 'direction_decomposer_started':
    case 'evolved_direction_decomposer_started':
      return join(stage === 'direction_decomposer_started' ? '初始假说生成开始' : '子代假说生成开始', `目标 ${event.target_direction_count ?? 0} 条`)
    case 'direction_decomposer_finished':
    case 'evolved_direction_decomposer_finished':
      return join(
        stage === 'direction_decomposer_finished' ? '初始假设生成完成' : '子代假设生成完成',
        `生成 ${event.initial_hypothesis_count ?? event.direction_count ?? 0} 条`,
        hypothesisListText(event) ? `假设 ${compactId(hypothesisListText(event), 360)}` : '',
      )
    case 'direction_decomposer_failed':
    case 'evolved_direction_decomposer_failed':
      return `${stage === 'direction_decomposer_failed' ? '初始假说生成失败' : '子代假说生成失败'}：${translateErrorText(event.error)}`
    case 'evolution_round_planned':
      return join(`${phase}规划完成`, `轨迹 ${event.task_count ?? 0} 条`, `初始 ${event.original_task_count ?? 0}`, `变异 ${event.mutation_task_count ?? 0}`, `交叉 ${event.crossover_task_count ?? 0}`)
    case 'evolution_round_all_failed':
      return join(`${phase}全部失败`, `失败轨迹 ${event.new_failure_count ?? event.task_count ?? 0} 条`, `无进展 ${event.no_progress_rounds ?? 0}/${event.early_stop_patience ?? 0} 轮`)
    case 'evolution_agent_started':
      return join('进化规划开始', `目标变异 ${event.mutation_target_count ?? 0}`, `目标交叉 ${event.crossover_target_count ?? 0}`)
    case 'evolution_agent_finished':
      return join('进化规划完成', `变异 ${event.mutation_task_count ?? 0}`, `交叉 ${event.crossover_task_count ?? 0}`)
    case 'evolution_agent_failed':
      return `进化规划失败：${translateErrorText(event.error)}`

    // ── 初始假说资格赛（宁缺毋滥）──
    case 'bootstrap_quality_gate_decision': {
      const rejected = event.rejected === true
      const checked = numberValue(event.checked_factor_count, 0)
      const qualifying = numberValue(event.qualifying_factor_count, 0)
      const sharpe = formatScore(event.sharpe, 2)
      const absIc = formatScore(event.abs_ic, 4)
      const hypo = stringValue(event.hypothesis)
      if (rejected) {
        return join(
          `❌ 资格出局 · 方向 #${event.direction_id ?? '-'}`,
          checked > 0 ? `${qualifying}/${checked} 因子过线` : '',
          sharpe ? `Sharpe=${sharpe}` : '',
          absIc ? `|IC|=${absIc}` : '',
          hypo ? compactId(hypo, 60) : '',
        )
      }
      const best = Array.isArray(event.qualifying_factors) && event.qualifying_factors.length
        ? event.qualifying_factors[0] as Record<string, unknown>
        : null
      return join(
        `✅ 资格合格 · 方向 #${event.direction_id ?? '-'}`,
        checked > 0 ? `${qualifying}/${checked} 因子过线` : '',
        best ? `最佳 ${compactId(best.factor_name, 36)} Sharpe=${formatScore(best.sharpe, 2)} |IC|=${formatScore(best.abs_ic, 4)}` : '',
        hypo ? compactId(hypo, 60) : '',
      )
    }
    case 'bootstrap_refill_round_started':
      return join(
        `假说重生 第 ${event.refill_round ?? 0}/${event.max_refill_rounds ?? 10} 轮`,
        `重生 ${event.task_count ?? 0} 个名额`,
        `资格线 Sharpe>${event.min_sharpe ?? 1.0} 或 |IC|>${event.min_abs_ic ?? 0.02}`,
      )
    case 'bootstrap_quality_gate_finished':
      return join(
        '资格赛收官',
        `合格 ${event.qualified_count ?? '-'} 个`,
        numberValue(event.remaining_rejected_slots, 0) > 0 ? `作废 ${event.remaining_rejected_slots} 个名额` : '',
        `重生 ${event.refill_rounds_used ?? 0} 轮`,
        `资格线 Sharpe>${event.min_sharpe ?? 1.0} 或 |IC|>${event.min_abs_ic ?? 0.02}`,
      )
    case 'bootstrap_all_rejected':
      return `❌ ${stringValue(event.message) || '所有初始假说均未通过资格赛，已终止运行（宁缺毋滥）'}`

    // ── 轨迹执行 ──
    case 'evolution_task_started':
      return join(`${phase}轨迹开始`, `#${event.direction_id ?? '-'}`, compactId(event.trajectory_id, 24))
    case 'evolution_task_finished':
    case 'round_finished':
      return join(stage === 'round_finished' ? '轨迹完成' : `${phase}轨迹完成`, `步骤 ${event.agent_count ?? 0} 个`, `错误 ${event.error_count ?? 0} 个`, summarizeReward(event))
    case 'factor_coder_started':
      return join('因子实现准备', event.has_panel_data_path ? '将读取本地 panel 数据' : '未提供 panel 数据路径')
    case 'factor_coder_panel_load_started':
      return join('读取因子实现数据', compactId(String(event.panel_data_path ?? '').split(/[\\/]/).pop(), 48))
    case 'factor_coder_panel_load_finished':
      return join('因子实现数据读取完成', `${n(event.panel_row_count)} 行`, `${n(event.panel_column_count)} 列`, elapsed)
    case 'factor_coder_panel_load_failed':
      return `因子实现数据读取失败：${translateErrorText(event.error)}`
    case 'factor_coder_panel_load_skipped':
      return '跳过因子值预计算 · 未提供 panel 数据路径'
    case 'factor_coder_config_ready':
      return join(
        '因子实现配置完成',
        `候选因子 ${event.factor_count ?? 0} 个`,
        `修复轮次上限 ${event.max_loop ?? 0}`,
        `失败阈值 ${event.fail_task_trial_limit ?? 0}`,
        `历史表达式 ${event.library_expression_count ?? 0} 条`,
        event.llm_eval_enabled ? 'LLM 评估开启' : '本地门控',
      )
    case 'factor_coder_loop_started':
      return join(`因子实现第 ${event.factor_coder_loop_index ?? 0}/${event.max_loop ?? 0} 轮`, `待处理 ${event.pending_factor_count ?? 0}/${event.factor_count ?? 0} 个`)
    case 'factor_coder_loop_finished':
      return join(
        `因子实现第 ${event.factor_coder_loop_index ?? 0} 轮完成`,
        `尝试 ${event.attempted_factor_count ?? 0} 个`,
        `通过 ${event.accepted_factor_count ?? 0} 个`,
        `失败 ${event.failed_factor_count ?? 0} 个`,
        `丢弃 ${event.dropped_factor_count ?? 0} 个`,
        elapsed,
      )
    case 'factor_coder_factor_started':
      return join(
        `实现因子 ${event.factor_index ?? 0}/${event.factor_total ?? 0}`,
        compactId(event.factor_name, 48),
        `第 ${event.attempt_number ?? 0} 次尝试`,
        event.expression ? compactId(event.expression, 80) : '',
      )
    case 'factor_coder_factor_evaluation_started':
      return join('因子表达式校验开始', compactId(event.factor_name, 48), event.source ? `来源 ${event.source}` : '')
    case 'factor_coder_value_started':
      return join('因子值计算开始', compactId(event.factor_name, 48), `${n(event.panel_row_count)} 行`, event.expression ? compactId(event.expression, 80) : '')
    case 'factor_coder_value_finished':
      return join(
        event.value_generated ? '因子值计算完成' : '因子值计算未通过',
        compactId(event.factor_name, 48),
        event.signal_cache_status ? `缓存 ${event.signal_cache_status}` : '',
        elapsed,
      )
    case 'factor_coder_factor_finished':
      return join(
        event.accepted ? '因子实现通过' : '因子实现未通过',
        compactId(event.factor_name, 48),
        event.quality_stage ? `门控 ${event.quality_stage}` : '',
        `本地检查 ${event.local_gate_count ?? 0}`,
        event.llm_eval_call_count ? `LLM ${event.llm_eval_call_count}` : '',
        elapsed,
      )
    case 'factor_coder_finished':
      return join(
        '因子实现汇总完成',
        `成功 ${event.generated_factors ?? 0}/${event.total_factors ?? 0}`,
        `失败 ${event.failed_factor_count ?? 0}`,
        `缓存命中 ${event.signal_cache_hits ?? 0}`,
        `LLM 调用 ${event.llm_call_count ?? 0}`,
      )
    case 'agent_started':
      return join(`${agent}开始`, context)
    case 'agent_finished':
      return join(`${agent}完成`, context, elapsed, summary)
    case 'agent_failed':
      return `${agent}失败：${translateErrorText(event.error)}`

    // ── 验证 / 入库 / 收尾 ──
    case 'holdout_validation_finished':
      return join('留样验证完成', event.status ? String(event.status) : '', `因子 ${event.factor_count ?? 0} 个`)
    case 'factor_pool_finished':
      return join('因子入库完成', event.status ? String(event.status) : '', `入库 ${event.selected_count ?? 0} 个`)
    case 'factor_cache_cleanup_finished': {
      const cleanup = typeof event.cleanup === 'object' && event.cleanup ? event.cleanup as Record<string, unknown> : {}
      return join('缓存清理完成', `删除 ${cleanup.deleted_files ?? 0} 个文件`)
    }
    case 'session_finished':
      return join('运行完成', event.status ? String(event.status) : '', `轨迹评估 ${event.loop_count ?? 0} 次`, `错误 ${event.total_errors ?? 0} 个`)
    case 'session_failed':
      return `运行失败：${translateErrorText(event.error)}`
    case 'run_stop_requested':
      return '停止请求已发送 · 将在下一个安全检查点停止'
    case 'run_cancelled':
      return `运行已终止${event.error ? `：${translateErrorText(event.error)}` : ''}`
  }

  if (summary) return join(STAGE_LABELS[stage] ?? stage, summary)
  return STAGE_LABELS[stage] ?? stage
}

function latestEvent(events: ProgressEvent[], predicate: (event: ProgressEvent) => boolean) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (predicate(events[i])) return events[i]
  }
  return undefined
}

function currentPhaseEvent(events: ProgressEvent[]) {
  return latestEvent(events, (event) => (
    typeof event.round_idx === 'number'
    && (
      event.stage === 'evolution_task_started'
      || event.stage === 'agent_started'
      || event.stage?.startsWith('factor_coder_')
      || event.stage === 'evolution_task_finished'
      || event.stage === 'evolution_round_planned'
      || event.stage === 'evolution_agent_started'
      || event.stage === 'evolution_agent_finished'
    )
  ))
}

function activeEventKey(event: ProgressEvent) {
  const stage = event.stage ?? ''
  if (stage === 'agent_started' || stage === 'agent_finished' || stage === 'agent_failed') {
    return `agent:${event.agent_name ?? ''}:${event.trajectory_id ?? ''}`
  }
  if (stage === 'evolution_task_started' || stage === 'evolution_task_finished') {
    return `task:${event.trajectory_id ?? ''}`
  }
  if (stage.startsWith('factor_coder_')) {
    return `factor_coder:${event.trajectory_id ?? ''}`
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
      || (
        stage.startsWith('factor_coder_')
        && !stage.endsWith('_finished')
        && !stage.endsWith('_failed')
        && stage !== 'factor_coder_config_ready'
        && stage !== 'factor_coder_panel_load_skipped'
      )
    ) {
      active.add(key)
    }
    if (stage.endsWith('_finished') || stage.endsWith('_failed') || stage === 'agent_finished' || stage === 'agent_failed') {
      active.delete(key)
    }
  }
  return active
}

const TIMELINE_LIMIT_COMPACT = 120
const TIMELINE_LIMIT_FULL = 2000

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
  return visible.slice(compact ? -TIMELINE_LIMIT_COMPACT : -TIMELINE_LIMIT_FULL).reverse()
}

// ── 事件分级：关键事件（里程碑）默认展示，过程事件折叠在「全部事件」模式 ──
const MILESTONE_STAGES = new Set([
  'session_started',
  'panel_data_loaded',
  'preprocessing_required',
  'preprocessing_complete',
  'preprocessing_failed',
  'data_filter_applied',
  'panel_data_normalized',
  'market_payload_finished',
  'research_report_payload_finished',
  'initialization_payload_ready',
  'direction_decomposer_started',
  'direction_decomposer_finished',
  'direction_decomposer_failed',
  'evolved_direction_decomposer_finished',
  'evolved_direction_decomposer_failed',
  'evolution_round_planned',
  'evolution_round_all_failed',
  'evolution_agent_finished',
  'evolution_agent_failed',
  'evolution_task_finished',
  'round_finished',
  'bootstrap_quality_gate_decision',
  'bootstrap_refill_round_started',
  'bootstrap_quality_gate_finished',
  'bootstrap_all_rejected',
  'factor_coder_finished',
  'agent_failed',
  'holdout_validation_finished',
  'holdout_validation_failed',
  'factor_pool_finished',
  'factor_pool_failed',
  'report_generated',
  'report_generation_failed',
  'session_finished',
  'session_failed',
  'run_stop_requested',
  'run_cancelled',
])

function isMilestoneEvent(event: ProgressEvent) {
  return MILESTONE_STAGES.has(event.stage ?? '') || Boolean(event.error)
}

// ── 时间线分组：准备阶段 → 第0轮初始假说 → 资格赛重生轮 → 进化轮次 → 收尾 ──
interface TimelineTrajectory {
  trajectoryId: string
  directionId: number | null
  hypothesis: string
  status: 'running' | 'finished' | 'error' | 'qualified' | 'rejected'
  reward: number | null
  events: ProgressEvent[]
}

interface TimelineGroup {
  key: string
  title: string
  kind: 'prepare' | 'round' | 'refill' | 'final'
  events: ProgressEvent[]
  trajectories: TimelineTrajectory[]
  hasError: boolean
}

const FINAL_GROUP_STAGES = new Set([
  'holdout_validation_finished',
  'holdout_validation_failed',
  'factor_pool_finished',
  'factor_pool_failed',
  'factor_cache_cleanup_finished',
  'report_generated',
  'report_generation_failed',
  'session_finished',
  'session_failed',
])

function groupTimelineEvents(events: ProgressEvent[]): TimelineGroup[] {
  const groups: TimelineGroup[] = []
  const trajectoryMaps = new Map<string, Map<string, TimelineTrajectory>>()
  const groupByKey = new Map<string, TimelineGroup>()

  const newGroup = (key: string, title: string, kind: TimelineGroup['kind']): TimelineGroup => {
    const group: TimelineGroup = { key, title, kind, events: [], trajectories: [], hasError: false }
    groups.push(group)
    trajectoryMaps.set(key, new Map())
    groupByKey.set(key, group)
    return group
  }

  const getOrCreateGroup = (key: string, title: string, kind: TimelineGroup['kind']): TimelineGroup => (
    groupByKey.get(key) ?? newGroup(key, title, kind)
  )

  const roundGroupTitle = (roundIdx: number, phase: unknown): string => (
    roundIdx <= 0 ? '第 0 轮 · 初始假说' : evolutionStageLabel(roundIdx, phase)
  )

  const groupForTrajectory = (trajectoryId: string): TimelineGroup | null => {
    for (const group of groups) {
      if (trajectoryMaps.get(group.key)?.has(trajectoryId)) return group
    }
    return null
  }

  let current = newGroup('prepare', '准备阶段 · 数据与研报', 'prepare')
  let finalGroup: TimelineGroup | null = null

  for (const event of events) {
    const stage = event.stage ?? ''

    // ── 组边界 ──
    if (stage === 'direction_decomposer_started' && current.kind === 'prepare') {
      current = getOrCreateGroup('round:0', '第 0 轮 · 初始假说', 'round')
    } else if (stage === 'bootstrap_refill_round_started') {
      const n = numberValue(event.refill_round, 0)
      current = getOrCreateGroup(`refill:${n}`, `资格赛 · 假说重生 第 ${n}/${event.max_refill_rounds ?? 10} 轮`, 'refill')
    } else if (stage === 'evolution_round_planned' && typeof event.round_idx === 'number') {
      const idx = numberValue(event.round_idx, 0)
      current = getOrCreateGroup(`round:${idx}`, roundGroupTitle(idx, event.phase), 'round')
    } else if (
      // 普通（非进化）模式没有轮次规划事件，按 loop_index 分组
      typeof event.loop_index === 'number'
      && !stringValue(event.trajectory_id)
      && (stage === 'agent_started' || stage === 'round_finished')
      && (current.kind === 'prepare' || current.key.startsWith('loop:'))
      && current.key !== `loop:${event.loop_index}`
    ) {
      current = getOrCreateGroup(`loop:${event.loop_index}`, `第 ${event.loop_index} 轮 · 轨迹评估`, 'round')
    }

    let target = current
    if (FINAL_GROUP_STAGES.has(stage)) {
      if (!finalGroup) finalGroup = getOrCreateGroup('final', '收尾 · 验证与入库', 'final')
      target = finalGroup
    }

    const severity = eventSeverity(event)
    if (severity === 'error') target.hasError = true

    // ── 轨迹归属：带 trajectory_id 的事件进入轨迹子卡片，否则是组级事件 ──
    const trajectoryId = stringValue(event.trajectory_id)
    if (trajectoryId && target.kind !== 'final') {
      const existingGroup = groupForTrajectory(trajectoryId)
      if (existingGroup) {
        target = existingGroup
      } else if (target.kind === 'prepare' && typeof event.round_idx === 'number') {
        const idx = numberValue(event.round_idx, 0)
        target = getOrCreateGroup(`round:${idx}`, roundGroupTitle(idx, event.phase), 'round')
        current = target
      }
    }

    if (trajectoryId && target.kind !== 'final' && target.kind !== 'prepare') {
      const map = trajectoryMaps.get(target.key)!
      let trajectory = map.get(trajectoryId)
      if (!trajectory) {
        trajectory = {
          trajectoryId,
          directionId: optionalNumber(event.direction_id),
          hypothesis: '',
          status: 'running',
          reward: null,
          events: [],
        }
        map.set(trajectoryId, trajectory)
        target.trajectories.push(trajectory)
      }
      if (trajectory.directionId === null) trajectory.directionId = optionalNumber(event.direction_id)
      // 假说文本：hypothesis（HypothesisAgent 细化后的最终假说）优先并持续更新；
      // direction（任务规划时的方向假说）作为任务一开始的兜底显示。
      const hypo = stringValue(event.hypothesis)
      if (hypo) trajectory.hypothesis = hypo
      else if (!trajectory.hypothesis && stringValue(event.direction)) trajectory.hypothesis = stringValue(event.direction)
      if (stage === 'evolution_task_finished' || stage === 'round_finished') {
        trajectory.status = numberValue(event.error_count, 0) > 0 ? 'error' : 'finished'
        const reward = optionalNumber(event.reward_score)
        if (reward !== null) trajectory.reward = reward
      } else if (stage === 'bootstrap_quality_gate_decision') {
        trajectory.status = event.rejected === true ? 'rejected' : 'qualified'
      } else if (severity === 'error' && trajectory.status === 'running') {
        trajectory.status = 'error'
      }
      trajectory.events.push(event)
    } else {
      target.events.push(event)
    }
  }

  groups.forEach((group, index) => {
    const hasLaterGroup = groups.slice(index + 1).some((next) => next.kind !== 'prepare')
    if (!hasLaterGroup) return
    for (const trajectory of group.trajectories) {
      if (trajectory.status === 'running') trajectory.status = 'finished'
    }
  })

  return groups.filter((group) => group.events.length > 0 || group.trajectories.length > 0)
}

function trajectoryStatusBadge(trajectory: TimelineTrajectory) {
  switch (trajectory.status) {
    case 'qualified':
      return <Tag color="green">✓ 资格合格</Tag>
    case 'rejected':
      return <Tag color="orange">✕ 资格出局</Tag>
    case 'error':
      return <Tag color="red">失败</Tag>
    case 'finished':
      return trajectory.reward !== null
        ? <Tag color="blue">Reward {formatScore(trajectory.reward, 3)}</Tag>
        : <Tag color="blue">完成</Tag>
    default:
      return <Tag color="processing">进行中</Tag>
  }
}

// ── 因子结果卡片：事件带 factor_summaries 时在时间线内规整展示每个因子及其指标 ──
interface FactorSummaryRow {
  name: string
  expression: string
  description: string
  implementationStatus: string
  reward: number | null
  rewardBreakdown: RewardBreakdownData | null
  metrics: Record<string, unknown>
}

function extractFactorSummaries(event: ProgressEvent): FactorSummaryRow[] {
  if (!Array.isArray(event.factor_summaries)) return []
  return event.factor_summaries
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .map((item) => ({
      name: stringValue(item.factor_name) || stringValue(item.name),
      expression: stringValue(item.expression),
      description: stringValue(item.description),
      implementationStatus: stringValue(item.implementation_status).toLowerCase(),
      reward: optionalNumber(item.reward_score),
      rewardBreakdown: item.reward_breakdown && typeof item.reward_breakdown === 'object'
        ? item.reward_breakdown as RewardBreakdownData
        : null,
      metrics: item.metrics && typeof item.metrics === 'object' ? item.metrics as Record<string, unknown> : {},
    }))
    .filter((row) => row.name || row.expression)
}

// 徽章三级口径：
// ✓ 入库目标 = Sharpe>1.5 且 |IC|>0.03 且 |RankIC|>0.03（target_filter 硬目标，AND）
// 过资格线   = Sharpe>1.0 或 |IC|>0.02（假说资格赛口径，OR）——资格判定与此一致
// 未达标     = 两条线都不过
function factorRowVerdict(row: FactorSummaryRow): { label: string; cls: string } {
  if (row.implementationStatus === 'failed' || row.implementationStatus === 'error') {
    return { label: '实现失败', cls: 'is-failed' }
  }
  if (Object.keys(row.metrics).length === 0) return { label: '无指标', cls: 'is-unknown' }
  const targetPassed = metricValue(row.metrics, 'target_passed')
  if (targetPassed === 1) return { label: '✓ 入库目标', cls: 'is-passed' }
  const sharpe = metricValue(row.metrics, 'sharpe', 'information_ratio')
  const ic = metricValue(row.metrics, 'IC', 'ic')
  const qualifies = (sharpe !== null && sharpe > 1.0) || (ic !== null && Math.abs(ic) > 0.02)
  if (qualifies) return { label: '过资格线', cls: 'is-qualified' }
  if (targetPassed === 0 || sharpe !== null) return { label: '未达标·放弃', cls: 'is-dropped' }
  return { label: '已评估', cls: 'is-unknown' }
}

function FactorMetricChip({ label, value, digits = 4 }: { label: string; value: number | null; digits?: number }) {
  if (value === null) return null
  return (
    <span className="run-factor-chip">
      <span className="run-factor-chip-label">{label}</span>
      <span className="run-factor-chip-value">{value.toFixed(digits)}</span>
    </span>
  )
}

function FactorResultsBlock({ event }: { event: ProgressEvent }) {
  const rows = extractFactorSummaries(event)
  if (!rows.length) return null
  return (
    <div className="run-factor-results">
      {rows.map((row, index) => {
        const verdict = factorRowVerdict(row)
        const mainlineFlag = metricValue(row.metrics, 'mainline_factor')
        const isControlFactor = mainlineFlag !== null && mainlineFlag < 0.5
        return (
          <div key={`${row.name}-${index}`} className={`run-factor-row ${verdict.cls}`}>
            <div className="run-factor-row-head">
              <Tag
                color={verdict.cls === 'is-passed' ? 'green' : verdict.cls === 'is-qualified' ? 'blue' : verdict.cls === 'is-failed' ? 'red' : verdict.cls === 'is-dropped' ? 'orange' : 'default'}
                style={{ marginInlineEnd: 0 }}
              >
                {verdict.label}
              </Tag>
              {isControlFactor && (
                <Tooltip title="对照因子（adjacent/counterfactual）：不代表假说成绩，不参与资格判定">
                  <Tag style={{ marginInlineEnd: 0 }}>对照</Tag>
                </Tooltip>
              )}
              <Tooltip title={row.description || undefined}>
                <Text strong style={{ fontFamily: 'monospace', fontSize: 12 }}>{row.name || '(未命名因子)'}</Text>
              </Tooltip>
              <span className="run-factor-chips">
                <FactorMetricChip label="Sharpe" value={metricValue(row.metrics, 'sharpe', 'information_ratio')} digits={2} />
                <FactorMetricChip label="IC" value={metricValue(row.metrics, 'IC', 'ic')} />
                <FactorMetricChip label="RankIC" value={metricValue(row.metrics, 'Rank IC', 'RankIC', 'rank_ic')} />
                <FactorMetricChip label="Reward" value={row.reward} digits={3} />
              </span>
            </div>
            {row.expression && (
              <div className="run-factor-expression">{row.expression}</div>
            )}
            {row.rewardBreakdown && (
              <details className="run-reward-details">
                <summary>Reward 分解</summary>
                <RewardBreakdown breakdown={row.rewardBreakdown} compact />
              </details>
            )}
          </div>
        )
      })}
    </div>
  )
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
      states.direction_decomposer_agent = { status: 'error', summary: translateErrorText(event.error) }
      current = idx
    } else if (stage.startsWith('factor_coder_')) {
      const eventSummary = buildEventText(event)
      if (stage.endsWith('_failed')) {
        states.factor_coder_agent = { status: 'error', summary: eventSummary }
      } else if (stage === 'factor_coder_finished') {
        states.factor_coder_agent = { status: 'finish', summary: eventSummary }
      } else {
        states.factor_coder_agent = { status: 'process', summary: eventSummary }
      }
      current = AGENT_ORDER.indexOf('factor_coder_agent')
    } else if (event.stage === 'agent_started') {
      states[name] = { status: 'process' }
      current = idx
    } else if (event.stage === 'agent_finished') {
      states[name] = { status: 'finish', summary, elapsed: formatDuration(event.elapsed_seconds) }
      current = Math.min(idx + 1, AGENT_ORDER.length - 1)
    } else if (event.stage === 'agent_failed') {
      states[name] = { status: 'error', summary: translateErrorText(event.error) }
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
      key: 'preprocess',
      label: '数据预处理',
      detail: '校验 / 计算预测标签 fwd_returns',
      icon: <FunctionOutlined />,
      started: has(['panel_data_loaded', 'preprocessing_required']),
      finished: has(['preprocessing_complete', 'panel_data_normalized', 'market_payload_finished']),
      failed: failed(['preprocessing_failed']),
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

function factorCoderPartialProgress(event?: ProgressEvent) {
  if (!event || !event.stage?.startsWith('factor_coder_')) return 0
  const total = Math.max(1, numberValue(event.factor_total, numberValue(event.total_factors, 1)))
  const index = Math.max(1, numberValue(event.factor_index, 1))
  const completedBefore = Math.max(0, Math.min(total - 1, index - 1))
  const weights: Record<string, number> = {
    factor_coder_started: 0.03,
    factor_coder_panel_load_started: 0.05,
    factor_coder_panel_load_finished: 0.08,
    factor_coder_config_ready: 0.1,
    factor_coder_loop_started: 0.12,
    factor_coder_factor_started: 0.2,
    factor_coder_factor_evaluation_started: 0.35,
    factor_coder_value_started: 0.55,
    factor_coder_value_finished: 0.8,
    factor_coder_factor_finished: 1,
    factor_coder_loop_finished: 0.9,
    factor_coder_finished: 0.95,
  }
  const within = weights[event.stage] ?? 0.15
  return Math.max(0, Math.min(0.95, (completedBefore + within) / total))
}

function deriveProgress(events: ProgressEvent[], run?: Run | RunDetail) {
  const roundEvents = events.filter((event) => event.stage === 'evolution_round_planned')
  const plannedTasks = roundEvents.reduce((sum, event) => sum + numberValue(event.task_count), 0)
  const finishedTasks = events.filter((event) => event.stage === 'evolution_task_finished').length
  const finishedAgents = events.filter((event) => event.stage === 'agent_finished').length
  const totalAgents = Math.max(AGENT_ORDER.length, finishedAgents + Number(Boolean(latestEvent(events, (e) => e.stage === 'agent_started'))))
  const latestFactorCoderIndex = events.reduce((latest, event, index) => (
    event.stage?.startsWith('factor_coder_') ? index : latest
  ), -1)
  const latestTaskFinishedIndex = events.reduce((latest, event, index) => (
    event.stage === 'evolution_task_finished' ? index : latest
  ), -1)
  const latestFactorCoder = latestFactorCoderIndex >= 0 && latestFactorCoderIndex > latestTaskFinishedIndex
    ? events[latestFactorCoderIndex]
    : undefined
  const factorCoderPartial = factorCoderPartialProgress(latestFactorCoder)

  if (plannedTasks > 0) return Math.min(99, Math.round(((finishedTasks + factorCoderPartial) / plannedTasks) * 100))
  if (run?.status === 'completed') return 100
  if (run?.status === 'failed') return Math.max(5, Math.min(95, Math.round((finishedAgents / totalAgents) * 100)))
  if (factorCoderPartial > 0) return Math.min(95, Math.round(((finishedAgents + factorCoderPartial) / totalAgents) * 100))
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

function runLoopCount(run?: Run | RunDetail) {
  if (run && 'loops' in run && Array.isArray(run.loops)) return run.loops.length
  return typeof run?.loop_count === 'number' ? run.loop_count : 0
}

function runFactorCount(run?: Run | RunDetail) {
  if (typeof run?.factor_count === 'number') return run.factor_count
  if (run && 'loops' in run && Array.isArray(run.loops)) {
    return run.loops.reduce((sum, loop) => sum + (Array.isArray(loop.factors) ? loop.factors.length : 0), 0)
  }
  return 0
}

function completedTrajectoryCount(events: ProgressEvent[], run?: Run | RunDetail) {
  const finished = events.filter((event) => event.stage === 'evolution_task_finished')
  // 数事件条数会偏小：前端只加载最近 N 条进度事件，更早的"轨迹完成"事件被挤出窗口。
  // loop_index 是每条轨迹的全局递增流水号，因此用"已完成事件里的最大 loop_index"估算
  // 真实完成数，不受事件窗口限制；运行结束后再以数据库的 trajectory_count 为准。
  const fromCount = finished.length
  const fromMaxIndex = finished.reduce((max, event) => (
    typeof event.loop_index === 'number' && event.loop_index > max ? event.loop_index : max
  ), 0)
  return Math.max(fromCount, fromMaxIndex, Number(run?.trajectory_count ?? 0), runLoopCount(run))
}

function producedFactorCount(events: ProgressEvent[], run?: Run | RunDetail) {
  const fromEvents = events
    .filter((event) => event.stage === 'evolution_task_finished')
    .reduce((sum, event) => sum + numberValue(event.factor_count), 0)
  return Math.max(fromEvents, runFactorCount(run))
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

// 时间线文案：首段「结果」加粗、其余「指标明细」淡色，层次更清晰；
// 多文件列表(panel_data_loaded)保留等宽换行，失败事件标红。
function EventText({ event }: { event: ProgressEvent }) {
  const text = buildEventText(event)
  if (event.stage === 'panel_data_loaded') {
    return <Text style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 12 }}>{text}</Text>
  }
  const isError = eventSeverity(event) === 'error'
  const sep = text.indexOf(' · ')
  const lead = sep >= 0 ? text.slice(0, sep) : text
  const detail = sep >= 0 ? text.slice(sep + 3) : ''
  return (
    <div className="run-timeline-text">
      <Text strong className="run-timeline-lead" style={isError ? { color: '#cf1322' } : undefined}>{lead}</Text>
      {detail && <Text type="secondary" className="run-timeline-detail">{detail}</Text>}
    </div>
  )
}

export default function RunProgressView({ run, runId, events, connected = false, compact = false }: RunProgressViewProps) {
  const navigate = useNavigate()
  const [stopping, setStopping] = useState(false)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportDetail, setReportDetail] = useState<ResearchReportDetail | null>(null)
  const [expandedGroups, setExpandedGroups] = useState<string[] | null>(null)
  const currentLoop = currentLoopIndex(events, run)
  const currentRound = currentRoundIndex(events)
  const progress = deriveProgress(events, run)
  const agentStates = deriveAgentStates(events, currentLoop)
  const stageStatuses = deriveStageStatuses(events, run?.status)
  const activeAgent = latestEvent(events, (event) => event.stage === 'agent_started')
  const activeTask = latestEvent(events, (event) => event.stage === 'evolution_task_started' || event.stage === 'evolution_task_planned')
  const latest = latestEvent(events, () => true)
  const errors = events.filter((event) => event.error || event.stage?.includes('failed'))
  const trajectoryCount = completedTrajectoryCount(events, run)
  const factorCount = producedFactorCount(events, run)
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
  // compact 模式：扁平「最近事件」只保留关键事件；完整模式：按阶段分组
  const timeline = compactTimelineEvents(compact ? events.filter(isMilestoneEvent) : events, compact, activeKeys)
  const timelineGroups = compact ? [] : groupTimelineEvents(events)
  const effectiveExpandedGroups = expandedGroups
    ?? (timelineGroups.length ? [timelineGroups[timelineGroups.length - 1].key] : [])
  const phaseEvent = currentPhaseEvent(events)
  const currentPhase = phaseEvent?.phase ?? activeTask?.phase ?? latest?.phase
  const currentEvolutionStage = evolutionStageLabel(phaseEvent?.round_idx ?? currentRound, currentPhase)
  const currentTrajectory = activeTask?.trajectory_id ?? latest?.trajectory_id
  const stopRequested = events.some((event) => event.stage === 'run_stop_requested')
  const isPollingProgress = !connected && (events.length > 0 || run?.status === 'running')
  const connectionBadge = connected
    ? <Badge status="processing" text="实时连接" />
    : isPollingProgress
      ? <Badge status="processing" text="轮询更新" />
      : <Badge status="default" text="离线" />
  const hasReports = reportCountFromRun(run) > 0 || events.some((event) => (
    event.stage === 'research_report_payload_finished'
    || event.stage === 'research_report_payload_started'
    || Boolean(event.research_report_pdf)
  ))

  async function loadReportDetail() {
    if (reportLoading) return
    setReportLoading(true)
    try {
      const res = await fetch(`/api/runs/${runId}/research-report`)
      if (!res.ok) throw new Error(await res.text())
      setReportDetail(await res.json())
    } catch {
      message.error('研报分析读取失败，请检查后端服务和研报文件路径')
    } finally {
      setReportLoading(false)
    }
  }

  async function openReportDetail() {
    setReportOpen(true)
    await loadReportDetail()
  }

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
            {connectionBadge}
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
          {hasReports && (
            <Button icon={<FileSearchOutlined />} onClick={openReportDetail}>
              研报分析
            </Button>
          )}
        </Space>
      </div>

      <Drawer
        title={
          <Space>
            <span>研报分析与假设来源</span>
            <Button size="small" icon={<ReloadOutlined />} loading={reportLoading} onClick={loadReportDetail}>
              刷新
            </Button>
          </Space>
        }
        open={reportOpen}
        onClose={() => setReportOpen(false)}
        width={860}
      >
        {reportLoading ? <Text type="secondary">正在读取研报分析...</Text> : renderReportDetail(reportDetail)}
      </Drawer>

      <div className="run-progress-meter">
        <Progress
          percent={progress}
          status={run?.status === 'failed' ? 'exception' : run?.status === 'completed' ? 'success' : 'active'}
          strokeColor={run?.status === 'failed' ? undefined : { from: '#1677ff', to: '#52c41a' }}
        />
      </div>

      <div className="run-kpi-grid">
        <div className="run-kpi">
          <Text type="secondary">已完成轨迹</Text>
          <strong>{trajectoryCount}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">已产出因子</Text>
          <strong>{factorCount}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">进化阶段</Text>
          <strong>{currentEvolutionStage}</strong>
        </div>
        <div className="run-kpi">
          <Text type="secondary">错误</Text>
          <strong className={errors.length ? 'danger-text' : ''}>{errors.length}</strong>
        </div>
        {!compact && (
          <>
            <div className="run-kpi">
              <Text type="secondary">特征</Text>
              <strong>{latestMarket?.feature_count === undefined ? '-' : String(latestMarket.feature_count)}</strong>
            </div>
            <div className="run-kpi">
              <Text type="secondary">当前执行序号</Text>
              <strong>{currentLoop ?? '-'}</strong>
            </div>
          </>
        )}
      </div>

      <div className="run-kpi-grid" style={{ display: 'none' }}>
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
                              {translateErrorText(e.error ?? buildEventText(e))}
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
          {compact ? (
            <>
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
                      <EventText event={event} />
                      <FactorResultsBlock event={event} />
                    </div>
                  ),
                }))}
              />
              {timeline.length === 0 && <Text type="secondary">等待进度事件</Text>}
            </>
          ) : (
            <>
              <div className="run-timeline-toolbar">
                <Text type="secondary" style={{ fontSize: 11 }}>
                  共 {events.length} 条事件 · {timelineGroups.length} 个阶段
                </Text>
              </div>
              {timelineGroups.length === 0 ? (
                <Text type="secondary">等待进度事件</Text>
              ) : (
                <Collapse
                  className="run-timeline-groups"
                  activeKey={effectiveExpandedGroups}
                  onChange={(keys) => setExpandedGroups(Array.isArray(keys) ? keys.map(String) : [String(keys)])}
                  items={timelineGroups.map((group) => {
                    const groupEvents = group.events
                    const finishedCount = group.trajectories.filter((t) => t.status === 'finished' || t.status === 'qualified').length
                    const rejectedCount = group.trajectories.filter((t) => t.status === 'rejected').length
                    const errorCount = group.trajectories.filter((t) => t.status === 'error').length
                    const runningCount = group.trajectories.filter((t) => t.status === 'running').length
                    return {
                      key: group.key,
                      label: (
                        <Space size={8} wrap>
                          <Text strong>{group.title}</Text>
                          {group.trajectories.length > 0 && (
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {group.trajectories.length} 条轨迹
                              {finishedCount > 0 ? ` · ✓${finishedCount}` : ''}
                              {rejectedCount > 0 ? ` · 出局${rejectedCount}` : ''}
                              {errorCount > 0 ? ` · ✕${errorCount}` : ''}
                              {runningCount > 0 ? ` · 进行中${runningCount}` : ''}
                            </Text>
                          )}
                          {group.hasError && <Tag color="red" style={{ marginInlineEnd: 0 }}>有失败</Tag>}
                        </Space>
                      ),
                      children: (
                        <div className="run-timeline-group-body">
                          {groupEvents.length > 0 && (
                            <Timeline
                              className="run-timeline"
                              items={groupEvents.map((event, index) => ({
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
                                    <EventText event={event} />
                                    <FactorResultsBlock event={event} />
                                  </div>
                                ),
                              }))}
                            />
                          )}
                          {group.trajectories.length > 0 && (
                            <Collapse
                              ghost
                              size="small"
                              className="run-trajectory-collapse"
                              items={[...group.trajectories]
                                .sort((a, b) => (a.directionId ?? 0) - (b.directionId ?? 0))
                                .map((trajectory) => {
                                  const shownEvents = trajectory.events
                                  return {
                                    key: trajectory.trajectoryId,
                                    label: (
                                      <Space size={8} wrap>
                                        <Text strong style={{ fontSize: 13 }}>
                                          方向 #{trajectory.directionId ?? '-'}
                                        </Text>
                                        {trajectoryStatusBadge(trajectory)}
                                        {trajectory.hypothesis && (
                                          <Text type="secondary" style={{ fontSize: 12 }}>
                                            {compactId(trajectory.hypothesis, 72)}
                                          </Text>
                                        )}
                                      </Space>
                                    ),
                                    children: (
                                      <>
                                        {trajectory.hypothesis && (
                                          <div className="run-trajectory-hypothesis">
                                            <Text type="secondary" style={{ fontSize: 11 }}>假说全文</Text>
                                            <Paragraph style={{ fontSize: 12, margin: '2px 0 0', whiteSpace: 'pre-wrap' }}>
                                              {trajectory.hypothesis}
                                            </Paragraph>
                                          </div>
                                        )}
                                        <Timeline
                                        className="run-timeline run-timeline-nested"
                                        items={shownEvents.map((event, index) => ({
                                          key: `${event.timestamp ?? index}-${event.stage}-${index}`,
                                          dot: <TimelineDot event={event} active={activeKeys.has(activeEventKey(event))} />,
                                          children: (
                                            <div className="run-timeline-row">
                                              <div className="run-timeline-head">
                                                <Text type="secondary">{formatTime(event.timestamp)}</Text>
                                                <Tag color={eventSeverity(event) === 'error' ? 'red' : eventSeverity(event) === 'success' ? 'green' : 'default'}>
                                                  {STAGE_LABELS[event.stage] ?? event.stage}
                                                </Tag>
                                              </div>
                                              <EventText event={event} />
                                              <FactorResultsBlock event={event} />
                                            </div>
                                          ),
                                        }))}
                                        />
                                      </>
                                    ),
                                  }
                                })}
                            />
                          )}
                        </div>
                      ),
                    }
                  })}
                />
              )}
            </>
          )}
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
        .run-timeline-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }
        .run-timeline-groups {
          background: transparent;
        }
        .run-timeline-groups > .ant-collapse-item {
          border: 1px solid #eef0f3;
          border-radius: 8px !important;
          margin-bottom: 8px;
          background: #ffffff;
          overflow: hidden;
        }
        .run-timeline-groups > .ant-collapse-item > .ant-collapse-header {
          background: #fafafa;
          align-items: center;
        }
        .run-timeline-group-body {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .run-trajectory-collapse > .ant-collapse-item {
          border: 1px solid #f0f0f0;
          border-radius: 6px !important;
          margin-bottom: 6px;
          background: #fcfcfd;
        }
        .run-trajectory-collapse .ant-collapse-header {
          padding: 6px 10px !important;
        }
        .run-timeline-nested {
          margin-top: 4px;
          margin-left: 4px;
        }
        .run-trajectory-hypothesis {
          background: #f8fafc;
          border: 1px solid #eef0f3;
          border-radius: 6px;
          padding: 8px 12px;
          margin-bottom: 8px;
        }
        .run-factor-results {
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-top: 8px;
        }
        .run-factor-row {
          border: 1px solid #f0f0f0;
          border-radius: 6px;
          padding: 6px 10px;
          background: #ffffff;
        }
        .run-factor-row.is-passed {
          background: #f6ffed;
          border-color: #d9f7be;
        }
        .run-factor-row.is-qualified {
          background: #f0f7ff;
          border-color: #bcd8f7;
        }
        .run-factor-row.is-dropped {
          background: #fffbe6;
          border-color: #ffe58f;
        }
        .run-factor-row.is-failed {
          background: #fff2f0;
          border-color: #ffccc7;
        }
        .run-factor-row-head {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .run-factor-chips {
          display: inline-flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-left: auto;
        }
        .run-factor-chip {
          display: inline-flex;
          gap: 4px;
          align-items: baseline;
          font-size: 12px;
        }
        .run-factor-chip-label {
          color: #8c8c8c;
          font-size: 11px;
        }
        .run-factor-chip-value {
          font-family: monospace;
          font-weight: 600;
          color: #111827;
        }
        .run-factor-expression {
          margin-top: 4px;
          font-family: monospace;
          font-size: 11px;
          color: #555;
          word-break: break-all;
          background: #fafafa;
          border-radius: 4px;
          padding: 3px 8px;
        }
        .run-reward-details {
          margin-top: 6px;
          font-size: 12px;
        }
        .run-reward-details summary {
          cursor: pointer;
          color: #0958d9;
          font-weight: 600;
          user-select: none;
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
          max-height: 70vh;
          min-height: 360px;
          overflow: auto;
          padding: 2px 4px 0 0;
        }
        .run-progress-shell.is-compact .run-timeline {
          max-height: 420px;
          min-height: 0;
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
        .run-timeline-text {
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .run-timeline-lead {
          font-size: 13px;
          line-height: 1.5;
          overflow-wrap: anywhere;
        }
        .run-timeline-detail {
          font-size: 12px;
          line-height: 1.5;
          overflow-wrap: anywhere;
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
