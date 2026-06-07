import { useState, useCallback, useEffect } from 'react'
import {
  Form, Input, InputNumber, Switch, Button, Upload, Card,
  Space, Alert, message, Divider, Typography, Select, Tooltip, Tag, Spin, DatePicker,
} from 'antd'
import { UploadOutlined, PlayCircleOutlined, ReloadOutlined, FileOutlined, CalendarOutlined } from '@ant-design/icons'
import dayjs, { type Dayjs } from 'dayjs'
import { useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import type { UploadFile } from 'antd'

const { Title, Text } = Typography

interface DataFile {
  path: string
  name: string
  relative: string
  size_mb: string
}

interface DatePreview {
  path: string
  start_date: string
  end_date: string
  years: number[]
  row_count: number
}

interface FormValues {
  run_name: string
  loop_count: number
  evolution_mode: boolean
  evolution_iterations: number
  evolution_num_initial_hypotheses: number
  initial_hypothesis: string
  train_sharpe_threshold: number | null
  panel_data_paths: string[]
  year_range: [number, number] | null
}

const fetcher = (url: string) => fetch(url).then((r) => r.json())

export default function NewRun() {
  const [form] = Form.useForm<FormValues>()
  const [uploading, setUploading] = useState(false)
  const [pdfPath, setPdfPath] = useState<string>('')
  const [starting, setStarting] = useState(false)
  const navigate = useNavigate()
  const evolutionMode = Form.useWatch('evolution_mode', form)

  const [datePreview, setDatePreview] = useState<DatePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs] | null>(null)

  const { data: dataFiles, isLoading: filesLoading, mutate: refreshFiles } =
    useSWR<DataFile[]>('/api/data-files', fetcher, { revalidateOnFocus: false })

  // When the first selected file changes, fetch its date range
  const selectedPaths: string[] = Form.useWatch('panel_data_paths', form) ?? []
  useEffect(() => {
    const firstPath = selectedPaths[0]
    if (!firstPath) { setDatePreview(null); setDateRange(null); return }
    setPreviewLoading(true)
    fetch(`/api/data-files/preview?path=${encodeURIComponent(firstPath)}`)
      .then(r => r.json())
      .then((data: DatePreview) => {
        setDatePreview(data)
        // Default: full range
        setDateRange([dayjs(data.start_date), dayjs(data.end_date)])
      })
      .catch(() => { setDatePreview(null); setDateRange(null) })
      .finally(() => setPreviewLoading(false))
  }, [selectedPaths[0]])

  const handleRefresh = useCallback(() => {
    refreshFiles()
    message.info('已刷新数据文件列表')
  }, [refreshFiles])

  async function handleUpload(file: UploadFile) {
    setUploading(true)
    const fd = new FormData()
    fd.append('file', file as unknown as File)
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd })
      const data = await res.json()
      setPdfPath(data.path)
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
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
          panel_data_paths: values.panel_data_paths ?? [],
          panel_data_path: (values.panel_data_paths ?? [])[0] ?? '',
          research_report_pdf: pdfPath,
          start_date: dateRange ? dateRange[0].format('YYYY-MM') : null,
          end_date: dateRange ? dateRange[1].format('YYYY-MM') : null,
        }),
      })
      const data = await res.json()
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
    // plain text for search filtering
    searchText: f.relative + ' ' + f.name,
  }))

  return (
    <div style={{ maxWidth: 700 }}>
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
          initial_hypothesis: '',
          panel_data_paths: [],
        }}
        onFinish={onFinish}
      >
        <Card title="基础设置" size="small" style={{ marginBottom: 16 }}>
          <Form.Item
            label="运行名称（可选）"
            name="run_name"
            extra={<Text type="secondary" style={{ fontSize: 11 }}>留空则自动以时间戳命名，如 20260604_120000</Text>}
          >
            <Input placeholder="例：动量反转策略实验" maxLength={60} showCount autoComplete="off" />
          </Form.Item>

          <Form.Item label="进化模式" name="evolution_mode" valuePropName="checked">
            <Switch checkedChildren="进化" unCheckedChildren="单循环" />
          </Form.Item>

          {!evolutionMode && (
            <Form.Item label="Loop 次数" name="loop_count" rules={[{ required: true }]}>
              <InputNumber min={1} max={100} style={{ width: 120 }} />
            </Form.Item>
          )}

          {evolutionMode && (
            <>
              <Form.Item
                label="进化迭代次数"
                name="evolution_iterations"
                extra={<Text type="secondary">1 次迭代会展开为 1 轮变异 + 1 轮交叉；系统会先跑第 0 轮初始探索。</Text>}
              >
                <InputNumber min={1} max={20} style={{ width: 120 }} />
              </Form.Item>
              <Form.Item
                label="初始探索假说数"
                name="evolution_num_initial_hypotheses"
                extra={<Text type="secondary">只控制第 0 轮初始探索；后续变异/交叉轮按进化任务数继续规划轨迹。</Text>}
              >
                <InputNumber min={1} max={50} style={{ width: 120 }} />
              </Form.Item>
            </>
          )}

          <Form.Item label="最低 Sharpe 阈值（可选）" name="train_sharpe_threshold">
            <InputNumber step={0.1} placeholder="不设置则使用默认" style={{ width: 180 }} />
          </Form.Item>
        </Card>

        <Card title="输入数据" size="small" style={{ marginBottom: 16 }}>
          <Form.Item
            label={
              <Space>
                <span>面板数据文件</span>
                <Text type="secondary" style={{ fontSize: 12 }}>（可多选，留空则用默认路径）</Text>
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
                  <Space direction="vertical" style={{ padding: '8px 0', width: '100%' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      未找到数据文件，请将 .parquet / .csv 文件放入项目 data/ 目录后点击刷新
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
              message={
                <>
                  未检测到数据文件。请将 <Text code>.parquet</Text> 或 <Text code>.csv</Text> 文件放入{' '}
                  <Text code>data/</Text> 目录，然后点击刷新。留空则使用环境变量 / 默认路径。
                </>
              }
            />
          )}

          {/* Year range selector — shows after a file is selected */}
          {selectedPaths.length > 0 && (
            <Form.Item
              label={
                <Space size={6}>
                  <CalendarOutlined style={{ color: '#1677ff' }} />
                  <span>数据年份范围</span>
                  {previewLoading && <Spin size="small" />}
                  {datePreview && !previewLoading && (
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      文件覆盖 {datePreview.start_date} ~ {datePreview.end_date}，共 {datePreview.row_count.toLocaleString()} 行
                    </Text>
                  )}
                </Space>
              }
              style={{ marginBottom: 12 }}
            >
              {previewLoading ? (
                <Spin size="small" style={{ display: 'block', margin: '8px 0' }} />
              ) : datePreview ? (
                <Space direction="vertical" style={{ width: '100%' }} size={6}>
                  <DatePicker.RangePicker
                    picker="month"
                    value={dateRange}
                    onChange={(v) => setDateRange(v as [Dayjs, Dayjs] | null)}
                    disabledDate={(d) =>
                      !d ||
                      d.isBefore(dayjs(datePreview.start_date), 'month') ||
                      d.isAfter(dayjs(datePreview.end_date), 'month')
                    }
                    style={{ width: '100%' }}
                    allowClear={false}
                    format="YYYY-MM"
                    placeholder={['开始月份', '结束月份']}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      可选范围：{datePreview.start_date} ~ {datePreview.end_date}
                    </Text>
                    <Button
                      type="link"
                      size="small"
                      style={{ padding: 0, fontSize: 11 }}
                      onClick={() => setDateRange([dayjs(datePreview.start_date), dayjs(datePreview.end_date)])}
                    >
                      重置为全部数据
                    </Button>
                  </div>
                </Space>
              ) : (
                <Text type="secondary" style={{ fontSize: 12 }}>选择数据文件后自动读取时间范围</Text>
              )}
            </Form.Item>
          )}

          <Form.Item label="研报 PDF（可选）">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Upload beforeUpload={handleUpload} maxCount={1} accept=".pdf">
                <Button icon={<UploadOutlined />} loading={uploading}>上传研报</Button>
              </Upload>
              {pdfPath && <Text type="success" style={{ fontSize: 12 }}>已上传：{pdfPath}</Text>}
            </Space>
          </Form.Item>

          <Form.Item label="初始假说文本（可选）" name="initial_hypothesis">
            <Input.TextArea rows={3} placeholder="可直接输入假说文本，不上传 PDF 时使用" />
          </Form.Item>
        </Card>

        <Alert
          type="info"
          showIcon
          message="启动后会跳转到实时监控页面，可以观察 Agent 执行进度"
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
