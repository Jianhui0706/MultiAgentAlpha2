export interface Run {
  run_id: string
  status: 'running' | 'completed' | 'failed' | string
  started_at: string
  ended_at: string | null
  factor_count: number
  loop_count: number
  trajectory_count?: number
  evolution_mode: boolean
  loop_count_arg: number
  research_report_pdf: string
  research_report_pdfs?: string[]
  overview: Record<string, unknown>
}

export interface RunDetail extends Run {
  log_dir: string
  args: Record<string, unknown>
  summary: Record<string, unknown>
  loops: Loop[]
}

export interface Loop {
  loop_index: number
  status: string
  phase: string
  round_idx: number
  direction_id: number
  trajectory_id: string
  parent_ids: string[]
  error_count: number
  agent_count: number
  elapsed_seconds: number
  started_at: string
  ended_at: string
  hypothesis: string
  feedback_summary: string
  factors: Factor[]
  metrics: Record<string, number>
}

export interface Factor {
  factor_id: string
  loop_index: number
  trajectory_id: string
  factor_name: string
  expression: string
  description: string
  formulation: string
  implementation_status: string
  phase: string
  round_idx: number
  reward_score: number | null
  // 入库判定（9 项 admission gate，与因子库排行榜同口径）
  admission_passed?: boolean
  admission_failed_checks?: string[]
  // metrics
  sharpe: number | null
  IC: number | null
  rank_ic: number | null
  icir: number | null
  rank_icir: number | null
  coverage: number | null
  turnover: number | null
  ls_return: number | null
  ann_return: number | null
  yearly_returns?: Record<string, number> | null
  quality_gate_passed: number | null
  quality_gate_threshold: number | null
  valid_cross_section_mean: number | null
  cross_section_universe_size: number | null
  ic_win_rate: number | null
  rank_ic_win_rate: number | null
  drawdown: number | null
  long_short_positive_ratio: number | null
  layer_1: number | null
  layer_2: number | null
  layer_3: number | null
  layer_4: number | null
  layer_5: number | null
  layer_monotonicity: number | null
  ls_cumulative_series: [string, number][] | null
  layer_1_cumulative_series: [string, number][] | null
  layer_2_cumulative_series: [string, number][] | null
  layer_3_cumulative_series: [string, number][] | null
  layer_4_cumulative_series: [string, number][] | null
  layer_5_cumulative_series: [string, number][] | null
  benchmark_cumulative_series: [string, number][] | null
  ic_cumulative_series: [string, number][] | null
  rank_ic_cumulative_series: [string, number][] | null
  duplicate_expression_count?: number | null
  duplicate_trajectory_ids?: string[] | null
  factor_reward_score?: number | null
  reward_breakdown?: {
    score?: number | null
    terms?: {
      name: string
      value: number
      weight: number
      contribution: number
    }[]
    target_status?: Record<string, unknown>
  } | null
  // source context
  hypothesis: string
  feedback_summary: string
}

export interface Trajectory {
  trajectory_id: string
  phase: 'original' | 'mutation' | 'crossover' | 'evolved' | string
  round_idx: number
  direction_id: number
  parent_ids: string[]
  hypothesis: string
  reward_score: number | null
  metrics: Record<string, number>
}

export interface ProgressEvent {
  stage: string
  timestamp?: string
  agent_name?: string
  loop_index?: number
  round_idx?: number
  phase?: string
  trajectory_id?: string
  elapsed_seconds?: number
  error?: string
  progress_summary?: { text: string }
  [key: string]: unknown
}
