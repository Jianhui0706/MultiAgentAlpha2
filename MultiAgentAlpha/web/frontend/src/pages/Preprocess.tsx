import { useState } from 'react'
import useSWR from 'swr'
import {
  Card, Form, Select, Button, Space, Typography, Input, message,
  Alert, Tag, Descriptions, Result, Spin,
} from 'antd'
import { ReloadOutlined, ThunderboltOutlined, FunctionOutlined, CheckCircleFilled, CalendarOutlined } from '@ant-design/icons'
import type { Dayjs } from 'dayjs'
import DataCoverageChart from '../components/DataCoverageChart'

const { Text, Title } = Typography
const fetcher = (url: string) => fetch(url).then((r) => r.json())

interface DataFile {
  path: string
  name: string
  relative: string
  size_mb: string
}

interface PreprocessResult {
  saved_to: string
  filename: string
  relative: string
  prediction_horizon: string
  target_column: string
  target_columns?: string[]
  target_already_present: boolean
  target_already_present_columns?: string[]
  all_fwd_horizons?: boolean
  bar_minutes: number | null
  bars_per_hour: number | null
  bar_frequency_source: string | null
  target_source: string
  rows: number
  symbol_count: number
  datetime_start: string
  datetime_end: string
  fwd_columns: string[]
  feature_columns: string[]
}

export default function Preprocess() {
  const [form] = Form.useForm()
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<PreprocessResult | null>(null)
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs] | null>(null)
  const selectedPaths: string[] = Form.useWatch('panel_data_paths', form) ?? []
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

  async function onFinish(values: {
    panel_data_paths: string[]
    target_horizons: string[]
    bar_frequency: string
    output_name?: string
  }) {
    if (!values.panel_data_paths?.length) {
      message.warning('请至少选择一个数据文件')
      return
    }
    if (!values.target_horizons?.length) {
      message.warning('请至少选择一个预测窗口')
      return
    }
    setSubmitting(true)
    setResult(null)
    try {
      const res = await fetch('/api/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          panel_data_paths: values.panel_data_paths,
          target_horizons: values.target_horizons,
          // 主目标（用于文件名/摘要展示）取所选的第一个窗口。
          prediction_horizon: values.target_horizons[0] ?? '24h',
          bar_frequency: values.bar_frequency ?? 'auto',
          output_name: values.output_name?.trim() || undefined,
          start_date: dateRange ? dateRange[0].format('YYYY-MM-DD HH:mm:ss') : null,
          end_date: dateRange ? dateRange[1].format('YYYY-MM-DD HH:mm:ss') : null,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        message.error(data.detail ?? '预处理失败')
        return
      }
      setResult(data as PreprocessResult)
      message.success('预处理完成，已存入 data/processed/')
      refreshFiles()
    } catch {
      message.error('请求失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: 880 }}>
      <Space align="center" style={{ marginBottom: 4 }}>
        <FunctionOutlined style={{ fontSize: 20, color: '#1677ff' }} />
        <Title level={4} style={{ margin: 0 }}>数据预处理</Title>
      </Space>
      <div style={{ marginBottom: 16 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          在原始行情数据上计算预测标签 fwd_returns_Nh 与派生特征，并以 parquet 存入 <Text code>data/processed/</Text>。
          之后到「新建运行」选择处理后的文件即可回测。主流程不再自动生成标签——若数据缺少 fwd 列，运行会直接失败。
        </Text>
      </div>

      <Card size="small">
        <Form
          form={form}
          layout="vertical"
          initialValues={{ target_horizons: ['24h'], bar_frequency: 'auto', panel_data_paths: [] }}
          onFinish={onFinish}
        >
          <Form.Item
            label={
              <Space>
                <span>原始数据文件</span>
                <Text type="secondary" style={{ fontSize: 12 }}>可多选，将合并后统一处理</Text>
                <Button size="small" type="text" icon={<ReloadOutlined />} loading={filesLoading} onClick={() => refreshFiles()} />
              </Space>
            }
            name="panel_data_paths"
            rules={[{ required: true, message: '请选择数据文件' }]}
          >
            <Select
              mode="multiple"
              allowClear
              showSearch
              placeholder="点击选择 .parquet / .csv 数据文件"
              loading={filesLoading}
              style={{ width: '100%' }}
              optionFilterProp="searchText"
              options={fileOptions}
              maxTagCount="responsive"
              optionRender={(o) => o.label}
            />
          </Form.Item>

          {selectedPaths.length > 0 && (
            <Form.Item
              label={
                <Space size={6}>
                  <CalendarOutlined style={{ color: '#1677ff' }} />
                  <span>数据时间分布</span>
                </Space>
              }
              extra={<Text type="secondary" style={{ fontSize: 11 }}>查看横截面覆盖随时间的分布，并选择要处理的时间区间（留空则处理全部）。fwd 标签在全量数据上计算后再裁剪，区间末端标签仍然有效。</Text>}
            >
              <DataCoverageChart
                paths={selectedPaths}
                selectable
                range={dateRange}
                onRangeChange={setDateRange}
              />
            </Form.Item>
          )}

          <Space size={16} wrap align="start">
            <Form.Item
              label="预测时间窗口（可多选）"
              name="target_horizons"
              rules={[{ required: true, message: '请至少选择一个预测窗口' }]}
              extra={<Text type="secondary" style={{ fontSize: 11 }}>勾选要生成的 forward return 标签，可多选；新建运行时再从中选一个作为回测目标。60min 等于 1h、240min 等于 4h，会自动合并。</Text>}
            >
              <Select mode="multiple" allowClear style={{ minWidth: 320 }} placeholder="选择要生成的预测窗口">
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

            <Form.Item
              label="输出文件名（可选）"
              name="output_name"
              extra={<Text type="secondary" style={{ fontSize: 11 }}>留空则自动命名 panel_{'{'}周期{'}'}_{'{'}时间戳{'}'}。</Text>}
            >
              <Input placeholder="panel_my_data" style={{ width: 200 }} addonAfter=".parquet" />
            </Form.Item>
          </Space>

          <Form.Item style={{ marginBottom: 0, marginTop: 8 }}>
            <Button type="primary" htmlType="submit" icon={<ThunderboltOutlined />} loading={submitting}>
              生成预处理数据
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {submitting && (
        <div style={{ textAlign: 'center', padding: '32px 0' }}>
          <Spin tip="正在计算预测标签与特征…" />
        </div>
      )}

      {result && (
        <Card size="small" style={{ marginTop: 16, borderColor: '#52c41a' }}>
          <Result
            style={{ padding: '8px 0' }}
            icon={<CheckCircleFilled style={{ color: '#52c41a' }} />}
            title="预处理完成"
            subTitle={<Text code>{result.relative}</Text>}
          />
          {result.target_already_present && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 12, fontSize: 12 }}
              message={`原始数据已包含 ${result.target_column}，已直接沿用并完成标准化。`}
            />
          )}
          {result.all_fwd_horizons && (
            <Alert
              type="success"
              showIcon
              style={{ marginBottom: 12, fontSize: 12 }}
              message={`多窗口已生成：${(result.target_columns ?? result.fwd_columns).join('、')}`}
            />
          )}
          <Descriptions size="small" column={2} bordered>
            <Descriptions.Item label="目标列">
              <Tag color="blue">{result.target_column}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="预处理模式">
              <Tag color={result.all_fwd_horizons ? 'green' : 'default'}>
                {result.all_fwd_horizons ? '多窗口' : '单窗口'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="K线级别">
              {result.bar_minutes ? `${result.bar_minutes} 分钟` : '—'}
              {result.bar_frequency_source ? <Text type="secondary" style={{ fontSize: 11 }}>（{result.bar_frequency_source === 'manual' ? '手动' : '自动检测'}）</Text> : null}
            </Descriptions.Item>
            <Descriptions.Item label="行数">{result.rows.toLocaleString()}</Descriptions.Item>
            <Descriptions.Item label="标的数">{result.symbol_count}</Descriptions.Item>
            <Descriptions.Item label="时间范围" span={2}>
              {result.datetime_start} ~ {result.datetime_end}
            </Descriptions.Item>
            <Descriptions.Item label="预测标签列" span={2}>
              <Space wrap>{result.fwd_columns.map((c) => <Tag key={c} color="green">{c}</Tag>)}</Space>
            </Descriptions.Item>
            <Descriptions.Item label="保存路径" span={2}>
              <Text code style={{ fontSize: 11 }}>{result.saved_to}</Text>
            </Descriptions.Item>
          </Descriptions>
          <div style={{ marginTop: 12 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              现在可到「新建运行」选择 <Text code>{result.relative}</Text> 启动回测。
            </Text>
          </div>
        </Card>
      )}
    </div>
  )
}
