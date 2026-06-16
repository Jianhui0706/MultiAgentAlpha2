import { useEffect, useState } from 'react'
import useSWR from 'swr'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  AreaChartOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import DataCoverageChart from '../components/DataCoverageChart'

const { Title, Text } = Typography

const fetcher = (url: string) => fetch(url).then((r) => r.json())

interface DataFile {
  path: string
  name: string
  relative: string
  size_mb: string
}

interface PreviewInfo {
  path: string
  rows: number
  symbol_count: number
  datetime_start: string | null
  datetime_end: string | null
  fwd_columns: string[]
  feature_columns: string[]
}

interface ValidationResult {
  report_id: string
  metrics: Record<string, number | string | boolean | null>
  return_diagnostics?: {
    yearly_annualized_returns?: Record<string, number>
    layer_return_table?: Array<{
      group: string
      mean_return: number
      annualized_return: number
      cumulative_return: number
    }>
    compounded_annualized_return?: number
  }
  data_range: {
    start: string
    end: string
    rows: number
    symbols: number
    target_column: string
  }
  chart_urls: Record<string, string>
  download_url: string
  pdf_url: string
}

interface FormValues {
  panel_data_path: string
  factor_name: string
  expression: string
  target_column: string
  rolling_window: number
  rolling_icir_mode: string
}

const metricOrder = [
  'IC',
  'ICIR',
  'Rank IC',
  'Rank ICIR',
  'sharpe',
  'drawdown',
  'coverage',
  'turnover',
  'long_short_mean_return',
  'long_short_positive_ratio',
  'annualized_return',
]

const DEFAULT_TARGET_COLUMNS = ['fwd_returns_4h', 'fwd_returns_24h']

function uniqueTargetColumns(columns?: string[]) {
  return Array.from(new Set([...(columns ?? []), ...DEFAULT_TARGET_COLUMNS]))
}

function preferredTargetColumn(columns: string[], panelPath?: string) {
  const lowerPath = (panelPath ?? '').toLowerCase()
  if (lowerPath.includes('panel_4h') && columns.includes('fwd_returns_4h')) return 'fwd_returns_4h'
  if (lowerPath.includes('panel_24h') && columns.includes('fwd_returns_24h')) return 'fwd_returns_24h'
  if (columns.includes('fwd_returns_4h')) return 'fwd_returns_4h'
  if (columns.includes('fwd_returns_24h')) return 'fwd_returns_24h'
  return columns[0]
}

function fmt(value: unknown, digits = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  return value.toFixed(digits)
}

export default function FactorValidation() {
  const [form] = Form.useForm<FormValues>()
  const [preview, setPreview] = useState<PreviewInfo | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<ValidationResult | null>(null)
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs] | null>(null)
  const selectedPath = Form.useWatch('panel_data_path', form)
  const { data: dataFiles, isLoading: filesLoading, mutate: refreshFiles } =
    useSWR<DataFile[]>('/api/data-files', fetcher, { revalidateOnFocus: false })

  const fileOptions = (dataFiles ?? []).map((f) => ({
    value: f.path,
    label: (
      <Space>
        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{f.relative}</span>
        <Text type="secondary" style={{ fontSize: 11 }}>{f.size_mb}</Text>
      </Space>
    ),
    searchText: `${f.relative} ${f.name}`,
  }))

  useEffect(() => {
    setDateRange(null)
    if (!selectedPath) {
      setPreview(null)
      return
    }
    let cancelled = false
    async function loadPreview() {
      setPreviewLoading(true)
      setResult(null)
      try {
        const res = await fetch('/api/factor-validation/preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ panel_data_path: selectedPath }),
        })
        const data = await res.json()
        if (!res.ok) {
          message.error(data.detail ?? '数据预览失败')
          setPreview(null)
          return
        }
        if (!cancelled) {
          setPreview(data as PreviewInfo)
          const fwd = uniqueTargetColumns((data as PreviewInfo).fwd_columns)
          const currentTarget = form.getFieldValue('target_column')
          if (fwd.length && (!currentTarget || !fwd.includes(currentTarget))) {
            form.setFieldValue('target_column', preferredTargetColumn(fwd, selectedPath))
          }
        }
      } catch {
        if (!cancelled) message.error('数据预览请求失败')
      } finally {
        if (!cancelled) setPreviewLoading(false)
      }
    }
    loadPreview()
    return () => {
      cancelled = true
    }
  }, [selectedPath, form])

  async function onFinish(values: FormValues) {
    setSubmitting(true)
    setResult(null)
    try {
      const res = await fetch('/api/factor-validation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          panel_data_path: values.panel_data_path,
          factor_name: values.factor_name?.trim() || 'manual_factor',
          expression: values.expression,
          target_column: values.target_column,
          rolling_window: values.rolling_window ?? 60,
          rolling_icir_mode: values.rolling_icir_mode ?? 'ewm',
          start_datetime: dateRange?.[0]?.format('YYYY-MM-DD HH:mm:ss') ?? null,
          end_datetime: dateRange?.[1]?.format('YYYY-MM-DD HH:mm:ss') ?? null,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        message.error(data.detail ?? '验证失败')
        return
      }
      setResult(data as ValidationResult)
      message.success('因子验证完成')
    } catch {
      message.error('验证请求失败')
    } finally {
      setSubmitting(false)
    }
  }

  const metricsData = result
    ? metricOrder
        .filter((key) => key in result.metrics)
        .map((key) => ({ key, metric: key, value: fmt(result.metrics[key]) }))
    : []
  const targetColumnOptions = uniqueTargetColumns(preview?.fwd_columns)

  return (
    <div style={{ maxWidth: 1280 }}>
      <Space align="center" style={{ marginBottom: 6 }}>
        <ExperimentOutlined style={{ fontSize: 20, color: '#1677ff' }} />
        <Title level={4} style={{ margin: 0 }}>因子验证</Title>
      </Space>
      <div style={{ marginBottom: 16 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          粘贴已生成的因子表达式，选择固定数据与时间窗口，后端直接计算 IC、Rank IC、Sharpe、覆盖率、换手和图表报告。
        </Text>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(420px, 520px) minmax(0, 1fr)', gap: 16, alignItems: 'start' }}>
        <Card size="small">
          <Form<FormValues>
            form={form}
            layout="vertical"
            initialValues={{
              factor_name: 'manual_factor',
              rolling_window: 60,
              rolling_icir_mode: 'ewm',
            }}
            onFinish={onFinish}
          >
            <Form.Item
              label={
                <Space>
                  <span>回测数据</span>
                  <Button size="small" type="text" icon={<ReloadOutlined />} loading={filesLoading} onClick={() => refreshFiles()} />
                </Space>
              }
              name="panel_data_path"
              rules={[{ required: true, message: '请选择数据文件' }]}
            >
              <Select
                allowClear
                showSearch
                loading={filesLoading}
                placeholder="选择 data/ 下的 parquet 或 csv"
                optionFilterProp="searchText"
                options={fileOptions}
                optionRender={(o) => o.label}
              />
            </Form.Item>

            {previewLoading && <Spin size="small" />}

            {preview && (
              <Descriptions size="small" column={1} bordered style={{ marginBottom: 16 }}>
                <Descriptions.Item label="数据范围">
                  {preview.datetime_start ? dayjs(preview.datetime_start).format('YYYY-MM-DD HH:mm') : '-'} ~{' '}
                  {preview.datetime_end ? dayjs(preview.datetime_end).format('YYYY-MM-DD HH:mm') : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="样本">
                  <Space>
                    <Tag>{preview.rows.toLocaleString()} 行</Tag>
                    <Tag>{preview.symbol_count.toLocaleString()} 标的</Tag>
                    <Tag>{preview.fwd_columns.length} 个已检测目标列</Tag>
                    <Tag color="blue">{targetColumnOptions.length} 个可选目标列</Tag>
                  </Space>
                </Descriptions.Item>
              </Descriptions>
            )}

            <Form.Item
              label="目标收益列"
              name="target_column"
              rules={[{ required: true, message: '请选择目标收益列' }]}
              extra={
                <Text type="secondary" style={{ fontSize: 11 }}>
                  默认提供 fwd_returns_4h / fwd_returns_24h；运行时后端会确认所选列是否真实存在于数据中。
                </Text>
              }
            >
              <Select
                placeholder="选择 fwd_returns_*"
                options={targetColumnOptions.map((col) => ({ value: col, label: col }))}
              />
            </Form.Item>

            <Form.Item
              label="回测时间"
              extra={<Text type="secondary" style={{ fontSize: 11 }}>和新建运行页使用同一套数据覆盖率时间选择：可拖动时间轴，也可点推荐区间、全部数据或最近区间。</Text>}
            >
              <DataCoverageChart
                paths={selectedPath ? [selectedPath] : []}
                selectable
                range={dateRange}
                onRangeChange={setDateRange}
              />
            </Form.Item>

            <Space size={12} align="start">
              <Form.Item label="因子名称" name="factor_name">
                <Input style={{ width: 190 }} />
              </Form.Item>
              <Form.Item label="滚动 ICIR 窗口" name="rolling_window">
                <InputNumber min={2} max={1000} style={{ width: 130 }} />
              </Form.Item>
              <Form.Item
                label="滚动加权方式"
                name="rolling_icir_mode"
                tooltip="EWM=近期 IC 指数加权（默认）；等权=窗口内 IC 平均。IC 已按预测窗口非重叠采样。"
              >
                <Select style={{ width: 150 }}>
                  <Select.Option value="ewm">EWM 近期加权</Select.Option>
                  <Select.Option value="simple">等权</Select.Option>
                </Select>
              </Form.Item>
            </Space>

            <Form.Item
              label="因子表达式"
              name="expression"
              rules={[{ required: true, message: '请粘贴因子表达式' }]}
            >
              <Input.TextArea
                rows={8}
                placeholder="例如：RANK(TS_PCTCHANGE($close, 24))"
                style={{ fontFamily: 'monospace', fontSize: 13 }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={submitting}>
                开始固定验证
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <div>
          {!result && !submitting && (
            <Alert
              type="info"
              showIcon
              message="等待验证"
              description="验证完成后，这里会显示核心指标、实际数据范围、图表预览和报告下载入口。"
            />
          )}

          {submitting && (
            <Card size="small">
              <Spin tip="正在计算因子信号、IC 序列和报告图表..." />
            </Card>
          )}

          {result && (
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Card
                size="small"
                title={<Space><AreaChartOutlined />验证结果</Space>}
                extra={
                  <Space>
                    <Button icon={<DownloadOutlined />} href={result.pdf_url}>下载 PDF</Button>
                    <Button icon={<DownloadOutlined />} href={result.download_url}>下载全部</Button>
                  </Space>
                }
              >
                <Descriptions size="small" column={2} bordered>
                  <Descriptions.Item label="报告 ID">{result.report_id}</Descriptions.Item>
                  <Descriptions.Item label="目标列">{result.data_range.target_column}</Descriptions.Item>
                  <Descriptions.Item label="数据范围">
                    {dayjs(result.data_range.start).format('YYYY-MM-DD HH:mm')} ~ {dayjs(result.data_range.end).format('YYYY-MM-DD HH:mm')}
                  </Descriptions.Item>
                  <Descriptions.Item label="样本">
                    {result.data_range.rows.toLocaleString()} 行 / {result.data_range.symbols.toLocaleString()} 标的
                  </Descriptions.Item>
                </Descriptions>

                <Table
                  size="small"
                  style={{ marginTop: 14 }}
                  pagination={false}
                  dataSource={metricsData}
                  columns={[
                    { title: '指标', dataIndex: 'metric', width: 220 },
                    { title: '值', dataIndex: 'value', align: 'right' },
                  ]}
                />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                <Card size="small" title="分年年化收益率">
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={Object.entries(result.return_diagnostics?.yearly_annualized_returns ?? {}).map(([year, value]) => ({
                      key: year,
                      year,
                      value: fmt(value * 100, 2) + '%',
                    }))}
                    columns={[
                      { title: '年份', dataIndex: 'year' },
                      { title: '年化收益率', dataIndex: 'value', align: 'right' },
                    ]}
                  />
                </Card>
                <Card size="small" title="五分组收益率">
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={(result.return_diagnostics?.layer_return_table ?? []).map((row) => ({
                      key: row.group,
                      group: row.group,
                      mean: fmt(row.mean_return * 100, 4) + '%',
                      annualized: fmt(row.annualized_return * 100, 2) + '%',
                      cumulative: fmt(row.cumulative_return * 100, 2) + '%',
                    }))}
                    columns={[
                      { title: '分组', dataIndex: 'group', width: 64 },
                      { title: '平均单期', dataIndex: 'mean', align: 'right' },
                      { title: '年化', dataIndex: 'annualized', align: 'right' },
                      { title: '累计', dataIndex: 'cumulative', align: 'right' },
                    ]}
                  />
                </Card>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
                {Object.entries({
                  cumulative_ic: '累计 IC',
                  ic_timeseries: 'IC 时序',
                  monthly_ic: '月度 IC',
                  ic_distribution: 'IC 分布',
                  rolling_icir: '滚动 ICIR',
                  yearly_annualized_return: '分年年化收益率',
                  five_group_cumulative_return: '五分组累计收益',
                }).map(([key, title]) => (
                  result.chart_urls[key] ? (
                    <Card key={key} size="small" title={title}>
                      <img
                        src={result.chart_urls[key]}
                        alt={title}
                        style={{ width: '100%', display: 'block' }}
                      />
                    </Card>
                  ) : null
                ))}
              </div>
            </Space>
          )}
        </div>
      </div>
    </div>
  )
}
