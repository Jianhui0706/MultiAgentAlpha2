import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  DatePicker,
  Divider,
  Form,
  Input,
  InputNumber,
  Select,
  Slider,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message,
} from 'antd'
import { CalendarOutlined, FileOutlined, PlayCircleOutlined, ReloadOutlined, UploadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'
import type { UploadFile } from 'antd'

const { Title, Text } = Typography

interface DataFile {
  path: string
  name: string
  relative: string
  size_mb: string
}

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

interface FormValues {
  run_name: string
  loop_count: number
  evolution_mode: boolean
  evolution_iterations: number
  evolution_num_initial_hypotheses: number
  evolution_mutation_tasks_per_round: number
  evolution_crossover_tasks_per_round: number
  initial_hypothesis: string
  train_sharpe_threshold: number | null
  train_coverage_threshold_pct: number
  panel_data_paths: string[]
  prediction_horizon: string
  bar_frequency: string
}

const fetcher = (url: string) => fetch(url).then((r) => r.json())

export default function NewRun() {
  const [form] = Form.useForm<FormValues>()
  const [uploading, setUploading] = useState(false)
  const [pdfPaths, setPdfPaths] = useState<string[]>([])
  const [starting, setStarting] = useState(false)
  const [coverageInfo, setCoverageInfo] = useState<CoverageInfo | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs] | null>(null)
  const navigate = useNavigate()
  const evolutionMode = Form.useWatch('evolution_mode', form)
  const coverageThresholdPct = Form.useWatch('train_coverage_threshold_pct', form) ?? 90

  const CHINESE_DATE_FMT = 'YYYY年MM月DD日 HH:mm'

  const { data: dataFiles, isLoading: filesLoading, mutate: refreshFiles } =
    useSWR<DataFile[]>('/api/data-files', fetcher, { revalidateOnFocus: false })

  const selectedPaths: string[] = Form.useWatch('panel_data_paths', form) ?? []

  const coverageStart = useMemo(
    () => coverageInfo?.union_start_datetime ? dayjs(coverageInfo.union_start_datetime) : null,
    [coverageInfo],
  )
  const coverageEnd = useMemo(
    () => coverageInfo?.union_end_datetime ? dayjs(coverageInfo.union_end_datetime) : null,
    [coverageInfo],
  )
  const coverageRangeMs = coverageStart && coverageEnd
    ? [coverageStart.valueOf(), coverageEnd.valueOf()] as [number, number]
    : null
  const selectedRangeMs = dateRange
    ? [dateRange[0].valueOf(), dateRange[1].valueOf()] as [number, number]
    : coverageRangeMs

  const handleRefresh = useCallback(() => {
    refreshFiles()
    message.info('已刷新数据文件列表')
  }, [refreshFiles])

  useEffect(() => {
    if (selectedPaths.length === 0) {
      setCoverageInfo(null)
      setDateRange(null)
      return
    }
    let cancelled = false
    setCoverageLoading(true)
    fetch('/api/data-files/coverage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: selectedPaths }),
    })
      .then((res) => {
        if (!res.ok) throw new Error('coverage request failed')
        return res.json()
      })
      .then((data: CoverageInfo) => {
        if (cancelled) return
        setCoverageInfo(data)
        setDateRange(null)
      })
      .catch(() => {
        if (!cancelled) {
          setCoverageInfo(null)
          message.error('数据时间范围读取失败')
        }
      })
      .finally(() => {
        if (!cancelled) setCoverageLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedPaths.join('|')])

  const applyPresetRange = useCallback((preset: 'all' | 'recommended' | '30d' | '90d' | '1y') => {
    if (!coverageStart || !coverageEnd) return
    if (preset === 'all') {
      setDateRange(null)
      return
    }
    if (preset === 'recommended') {
      const start = dayjs(coverageInfo?.recommended_start_datetime || coverageInfo?.union_start_datetime)
      const end = dayjs(coverageInfo?.recommended_end_datetime || coverageInfo?.union_end_datetime)
      if (start.isValid() && end.isValid()) setDateRange([start, end])
      return
    }
    const amount = preset === '30d' ? 30 : preset === '90d' ? 90 : 365
    const start = coverageEnd.subtract(amount, 'day')
    setDateRange([start.isBefore(coverageStart) ? coverageStart : start, coverageEnd])
  }, [coverageInfo, coverageStart, coverageEnd])

  async function handleUpload(file: UploadFile) {
    setUploading(true)
    const fd = new FormData()
    fd.append('file', file as unknown as File)
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd })
      const data = await res.json()
      setPdfPaths((prev) => prev.includes(data.path) ? prev : [...prev, data.path])
      message.success(`上传成功：${data.filename}`)
    } catch {
      message.error('上传失败')
    } finally {
      setUploading(false)
    }
    return false
  }

  async function onFinish(values: FormValues) {
    setStarting(true)
    try {
      const { train_coverage_threshold_pct, ...submitValues } = values
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...submitValues,
          train_coverage_threshold: (train_coverage_threshold_pct ?? 90) / 100,
          panel_data_paths: values.panel_data_paths ?? [],
          panel_data_path: (values.panel_data_paths ?? [])[0] ?? '',
          research_report_pdf: pdfPaths[0] ?? '',
          research_report_pdfs: pdfPaths,
          prediction_horizon: values.prediction_horizon ?? '24h',
          bar_frequency: values.bar_frequency ?? 'auto',
          start_date: dateRange ? dateRange[0].format('YYYY-MM-DD HH:mm:ss') : null,
          end_date: dateRange ? dateRange[1].format('YYYY-MM-DD HH:mm:ss') : null,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        message.error(data.detail ?? '启动失败，请检查后端返回信息')
        return
      }
      message.success(`运行已启动：${data.run_id}`)
      navigate(`/runs/${data.run_id}`)
    } catch {
      message.error('启动失败，请检查后端服务')
    } finally {
      setStarting(false)
    }
  }

  const fileOptions = (dataFiles ?? []).map((f) => ({
    value: f.path,
    label: (
      <Space size={6}>
        <FileOutlined style={{ color: '#1677ff' }} />
        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{f.relative}</span>
        <Tag style={{ fontSize: 10, marginLeft: 4 }}>{f.size_mb}</Tag>
      </Space>
    ),
    searchText: `${f.relative} ${f.name}`,
  }))

  const maxCoverageCount = Math.max(1, coverageInfo?.total_symbols ?? 1)

  const thresholdValidInterval = useMemo(() => {
    if (!coverageInfo?.sampled_coverage?.length || !coverageInfo.total_symbols || !coverageEnd) return null
    const requiredCount = Math.ceil((coverageThresholdPct / 100) * coverageInfo.total_symbols)
    const firstValid = coverageInfo.sampled_coverage.find(p => Number(p.count) >= requiredCount)
    if (!firstValid) return null
    const start = dayjs(firstValid.timestamp)
    return start.isValid() ? { start, end: coverageEnd } : null
  }, [coverageInfo, coverageThresholdPct, coverageEnd])

  return (
    <div style={{ maxWidth: 760 }}>
      <Title level={4}>新建运行</Title>
      <Form
        form={form}
        layout="vertical"
        autoComplete="off"
        initialValues={{
          run_name: '',
          loop_count: 3,
          evolution_mode: false,
          evolution_iterations: 5,
          evolution_num_initial_hypotheses: 10,
          evolution_mutation_tasks_per_round: 10,
          evolution_crossover_tasks_per_round: 10,
          initial_hypothesis: '',
          train_coverage_threshold_pct: 90,
          panel_data_paths: [],
          prediction_horizon: '24h',
          bar_frequency: 'auto',
        }}
        onFinish={onFinish}
      >
        <Card title="运行设置" size="small" style={{ marginBottom: 16 }}>
          <Form.Item
            label="运行名称（可选）"
            name="run_name"
            extra={<Text type="secondary" style={{ fontSize: 11 }}>留空则使用启动时间命名，例如 20260604_120000</Text>}
          >
            <Input placeholder="例如：动量反转因子实验" maxLength={60} showCount autoComplete="off" />
          </Form.Item>

          <Form.Item label="进化模式" name="evolution_mode" valuePropName="checked">
            <Switch checkedChildren="进化" unCheckedChildren="单循环" />
          </Form.Item>

          {!evolutionMode && (
            <Form.Item label="循环次数" name="loop_count" rules={[{ required: true }]}>
              <InputNumber min={1} max={100} style={{ width: 140 }} />
            </Form.Item>
          )}

          {evolutionMode && (
            <>
              <Form.Item
                label="进化迭代次数"
                name="evolution_iterations"
                extra={<Text type="secondary">1 次 iteration 会依次执行 1 个 mutation phase 和 1 个 crossover phase；系统会先运行第 0 轮初始探索。</Text>}
              >
                <InputNumber min={1} max={20} style={{ width: 140 }} />
              </Form.Item>

              <Form.Item
                label="初始探索假设数"
                name="evolution_num_initial_hypotheses"
                extra={<Text type="secondary">控制第 0 轮 original trajectories 的数量。默认 10。</Text>}
              >
                <InputNumber min={1} max={50} style={{ width: 140 }} />
              </Form.Item>

              <Space size={16} wrap align="start">
                <Form.Item
                  label="Mutation 每阶段任务数"
                  name="evolution_mutation_tasks_per_round"
                  extra={<Text type="secondary">每个 mutation phase 生成多少条局部修改 trajectory。默认 10。</Text>}
                >
                  <InputNumber min={0} max={50} style={{ width: 160 }} />
                </Form.Item>

                <Form.Item
                  label="Crossover 每阶段任务数"
                  name="evolution_crossover_tasks_per_round"
                  extra={<Text type="secondary">每个 crossover phase 生成多少条交叉重组 trajectory。默认 10。</Text>}
                >
                  <InputNumber min={0} max={50} style={{ width: 160 }} />
                </Form.Item>
              </Space>
            </>
          )}

          <Form.Item label="最低 Sharpe 阈值（可选）" name="train_sharpe_threshold">
            <InputNumber step={0.1} placeholder="不设置则默认 1.5" style={{ width: 190 }} />
          </Form.Item>
        </Card>

        <Card title="输入数据" size="small" style={{ marginBottom: 16 }}>
          <Form.Item
            label={
              <Space>
                <span>面板数据文件</span>
                <Text type="secondary" style={{ fontSize: 12 }}>可多选，留空则使用默认路径</Text>
                <Tooltip title="刷新文件列表">
                  <Button
                    size="small"
                    type="text"
                    icon={<ReloadOutlined />}
                    loading={filesLoading}
                    onClick={handleRefresh}
                  />
                </Tooltip>
              </Space>
            }
            name="panel_data_paths"
          >
            <Select
              mode="multiple"
              allowClear
              showSearch
              placeholder="点击选择数据文件，支持多选"
              loading={filesLoading}
              style={{ width: '100%' }}
              optionFilterProp="searchText"
              notFoundContent={
                filesLoading ? '加载中...' : (
                  <Space orientation="vertical" style={{ padding: '8px 0', width: '100%' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      未找到数据文件，请将 .parquet / .csv 文件放入项目 data/ 目录后刷新。
                    </Text>
                    <Button size="small" icon={<ReloadOutlined />} onClick={handleRefresh}>
                      刷新
                    </Button>
                  </Space>
                )
              }
              options={fileOptions}
              maxTagCount="responsive"
              optionRender={(option) => option.label}
            />
          </Form.Item>

          {(dataFiles ?? []).length === 0 && !filesLoading && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 12, fontSize: 12 }}
              title={
                <>
                  未检测到数据文件。请将 <Text code>.parquet</Text> 或 <Text code>.csv</Text> 文件放入{' '}
                  <Text code>data/</Text> 目录，然后刷新。
                </>
              }
            />
          )}

          <Form.Item
            label="覆盖率门槛（%）"
            name="train_coverage_threshold_pct"
            extra={<Text type="secondary" style={{ fontSize: 11 }}>低于该覆盖率的因子会在回测后、进入保留和入库前直接放弃。默认 90%。</Text>}
          >
            <InputNumber min={0} max={100} step={1} style={{ width: 160 }} />
          </Form.Item>

          {selectedPaths.length > 0 && (
            <Form.Item
              label={
                <Space size={6}>
                  <CalendarOutlined style={{ color: '#1677ff' }} />
                  <span>回测时间窗口</span>
                  {coverageLoading && <Tag color="processing">读取中</Tag>}
                </Space>
              }
              extra={<Text type="secondary" style={{ fontSize: 11 }}>支持精确到分钟。默认使用覆盖厚度达到回测要求后的推荐区间，也可以拖动时间轴或手动输入。</Text>}
              style={{ marginBottom: 12 }}
            >
              <Space orientation="vertical" size={10} style={{ width: '100%' }}>
                {coverageInfo && coverageStart && coverageEnd && (
                  <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 12, background: '#fbfdff' }}>
                    <Space wrap size={8} style={{ marginBottom: 10 }}>
                      <Tag color="blue">
                        数据范围 {coverageStart.format(CHINESE_DATE_FMT)} ~ {coverageEnd.format(CHINESE_DATE_FMT)}
                      </Tag>
                      <Tag>标的 {coverageInfo.total_symbols ?? 0}</Tag>
                      <Tag>最低横截面 {coverageInfo.min_cross_section ?? 5}</Tag>
                    </Space>

                    {thresholdValidInterval && (
                      <div style={{ marginBottom: 8 }}>
                        <Tag color="green" style={{ fontSize: 12 }}>
                          满足 {coverageThresholdPct}% 覆盖率的可用区间：
                          {thresholdValidInterval.start.format(CHINESE_DATE_FMT)} ~ {thresholdValidInterval.end.format(CHINESE_DATE_FMT)}
                        </Tag>
                      </div>
                    )}

                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 46, marginBottom: 8 }}>
                      {(coverageInfo.sampled_coverage ?? []).map((point, idx) => {
                        const ratio = Math.max(0.08, Math.min(1, Number(point.count || 0) / maxCoverageCount))
                        const ok = Number(point.count || 0) >= Number(coverageInfo.min_cross_section ?? 5)
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
                                opacity: ok ? 0.82 : 0.58,
                              }}
                            />
                          </Tooltip>
                        )
                      })}
                    </div>

                    {coverageRangeMs && selectedRangeMs && (
                      <Slider
                        range={{ draggableTrack: true }}
                        min={coverageRangeMs[0]}
                        max={coverageRangeMs[1]}
                        value={selectedRangeMs}
                        tooltip={{ formatter: (value: number | undefined) => value == null ? '' : dayjs(value).format(CHINESE_DATE_FMT) }}
                        onChange={(value: number | number[]) => {
                          const [start, end] = value as [number, number]
                          setDateRange([dayjs(start), dayjs(end)])
                        }}
                      />
                    )}

                    <Space wrap size={8}>
                      <Button size="small" onClick={() => applyPresetRange('recommended')}>推荐区间</Button>
                      <Button size="small" onClick={() => applyPresetRange('all')}>全部数据</Button>
                      <Button size="small" onClick={() => applyPresetRange('30d')}>最近 30 天</Button>
                      <Button size="small" onClick={() => applyPresetRange('90d')}>最近 90 天</Button>
                      <Button size="small" onClick={() => applyPresetRange('1y')}>最近 1 年</Button>
                    </Space>
                  </div>
                )}

                <DatePicker.RangePicker
                  showTime={{ format: 'HH:mm' }}
                  value={dateRange}
                  onChange={(v) => setDateRange(v as [Dayjs, Dayjs] | null)}
                  style={{ width: '100%' }}
                  format="YYYY年MM月DD日 HH:mm"
                  placeholder={['开始时间', '结束时间']}
                />
              </Space>
            </Form.Item>
          )}

          <Space size={16} wrap align="start" style={{ marginBottom: 0 }}>
            <Form.Item
              label="预测时间窗口"
              name="prediction_horizon"
              extra={<Text type="secondary" style={{ fontSize: 11 }}>回测目标列 fwd_returns_Nh；数据须已包含该列，否则运行直接失败。请先在「数据预处理」生成。</Text>}
            >
              <Select style={{ width: 160 }}>
                <Select.Option value="5min">5 分钟</Select.Option>
                <Select.Option value="10min">10 分钟</Select.Option>
                <Select.Option value="15min">15 分钟</Select.Option>
                <Select.Option value="20min">20 分钟</Select.Option>
                <Select.Option value="30min">30 分钟</Select.Option>
                <Select.Option value="1h">1 小时</Select.Option>
                <Select.Option value="4h">4 小时</Select.Option>
                <Select.Option value="12h">12 小时</Select.Option>
                <Select.Option value="24h">24 小时（推荐）</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item
              label="数据 K 线级别"
              name="bar_frequency"
              extra={<Text type="secondary" style={{ fontSize: 11 }}>影响 fwd 偏移根数；默认自动从时间戳间隔推断。</Text>}
            >
              <Select style={{ width: 160 }}>
                <Select.Option value="auto">自动检测（推荐）</Select.Option>
                <Select.Option value="1min">1 分钟</Select.Option>
                <Select.Option value="5min">5 分钟</Select.Option>
                <Select.Option value="15min">15 分钟</Select.Option>
                <Select.Option value="30min">30 分钟</Select.Option>
                <Select.Option value="60min">1 小时</Select.Option>
                <Select.Option value="240min">4 小时</Select.Option>
                <Select.Option value="1d">日线</Select.Option>
              </Select>
            </Form.Item>
          </Space>

          <Form.Item label="研究报告 PDF（可选，可多份）">
            <Space orientation="vertical" style={{ width: '100%' }}>
              <Upload beforeUpload={handleUpload} accept=".pdf" multiple>
                <Button icon={<UploadOutlined />} loading={uploading}>上传研报</Button>
              </Upload>
              {pdfPaths.length > 0 && (
                <Space size={6} wrap>
                  {pdfPaths.map((path) => (
                    <Tag
                      key={path}
                      closable
                      onClose={(event) => {
                        event.preventDefault()
                        setPdfPaths((prev) => prev.filter((item) => item !== path))
                      }}
                    >
                      {path.split(/[\\/]/).pop()}
                    </Tag>
                  ))}
                </Space>
              )}
            </Space>
          </Form.Item>

          <Form.Item label="初始假设文本（可选）" name="initial_hypothesis">
            <Input.TextArea rows={3} placeholder="可直接输入假设文本；未上传 PDF 时会作为初始方向使用。" />
          </Form.Item>
        </Card>

        <Alert
          type="info"
          showIcon
          title="启动后会跳转到实时监控页面，可以观察 Agent 执行进度。"
          style={{ marginBottom: 16 }}
        />

        <Divider />
        <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={starting} size="large">
          启动运行
        </Button>
      </Form>
    </div>
  )
}
