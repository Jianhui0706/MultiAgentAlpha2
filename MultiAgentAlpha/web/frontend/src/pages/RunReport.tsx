import { useParams, Link } from 'react-router-dom'
import useSWR from 'swr'
import { Spin, Result, Button, Space, Typography, Alert } from 'antd'
import { ArrowLeftOutlined, ExperimentOutlined, ReloadOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'

const { Title, Text } = Typography

interface ReportResponse {
  content: string
  path: string
  assets_base: string
}

const fetcher = (url: string) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(String(r.status))
  return r.json()
})

// Transform relative image paths in markdown to absolute web paths via assets_base
function patchImageSrc(src: string | undefined, assetsBase: string): string {
  if (!src) return ''
  if (src.startsWith('http') || src.startsWith('/') || src.startsWith('data:')) return src
  const base = assetsBase.replace(/\/$/, '')
  return `${base}/${src}`
}

function buildComponents(assetsBase: string): Components {
  return {
    img({ src, alt, ...props }) {
      const resolvedSrc = patchImageSrc(src, assetsBase)
      return (
        <img
          src={resolvedSrc}
          alt={alt ?? ''}
          {...props}
          style={{ maxWidth: '100%', borderRadius: 6, margin: '12px 0', boxShadow: '0 2px 12px rgba(0,0,0,0.1)' }}
          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
        />
      )
    },
    h1({ children }) {
      return <Title level={2} style={{ marginTop: 24 }}>{children}</Title>
    },
    h2({ children }) {
      return <Title level={3} style={{ marginTop: 20 }}>{children}</Title>
    },
    h3({ children }) {
      return <Title level={4} style={{ marginTop: 16 }}>{children}</Title>
    },
    h4({ children }) {
      return <Title level={5} style={{ marginTop: 14 }}>{children}</Title>
    },
    p({ children }) {
      return <p style={{ lineHeight: 1.8, marginBottom: 10, color: '#1f2937' }}>{children}</p>
    },
    code({ children, className }) {
      const isBlock = className?.startsWith('language-')
      if (isBlock) {
        return (
          <pre style={{ background: '#1a1a2e', borderRadius: 6, padding: '12px 16px', overflowX: 'auto', margin: '12px 0' }}>
            <code style={{ color: '#a8d8ff', fontFamily: 'monospace', fontSize: 12 }}>{children}</code>
          </pre>
        )
      }
      return (
        <code style={{ background: '#f3f4f6', borderRadius: 3, padding: '2px 5px', fontFamily: 'monospace', fontSize: 12, color: '#374151' }}>
          {children}
        </code>
      )
    },
    table({ children }) {
      return (
        <div style={{ overflowX: 'auto', margin: '14px 0' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13 }}>{children}</table>
        </div>
      )
    },
    th({ children }) {
      return (
        <th style={{ border: '1px solid #d1d5db', padding: '6px 12px', background: '#f9fafb', fontWeight: 600, textAlign: 'left' }}>
          {children}
        </th>
      )
    },
    td({ children }) {
      return (
        <td style={{ border: '1px solid #e5e7eb', padding: '5px 12px' }}>{children}</td>
      )
    },
    blockquote({ children }) {
      return (
        <blockquote style={{ borderLeft: '4px solid #1677ff', paddingLeft: 14, margin: '12px 0', color: '#4b5563', background: '#f0f7ff', borderRadius: '0 6px 6px 0', padding: '10px 14px' }}>
          {children}
        </blockquote>
      )
    },
    hr() {
      return <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '20px 0' }} />
    },
    ul({ children }) {
      return <ul style={{ paddingLeft: 24, marginBottom: 10 }}>{children}</ul>
    },
    ol({ children }) {
      return <ol style={{ paddingLeft: 24, marginBottom: 10 }}>{children}</ol>
    },
    li({ children }) {
      return <li style={{ marginBottom: 4, lineHeight: 1.7 }}>{children}</li>
    },
  }
}

export default function RunReport() {
  const { runId } = useParams<{ runId: string }>()
  const { data, error, isLoading, mutate } = useSWR<ReportResponse>(
    runId ? `/api/runs/${runId}/report` : null,
    fetcher,
    { revalidateOnFocus: false }
  )

  const header = (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
      <Space>
        <Link to={`/runs/${runId}/factors`}>
          <Button icon={<ArrowLeftOutlined />}>因子列表</Button>
        </Link>
        <Link to={`/runs/${runId}`}>
          <Button icon={<ExperimentOutlined />}>运行监控</Button>
        </Link>
      </Space>
      <Button icon={<ReloadOutlined />} onClick={() => mutate()}>刷新</Button>
    </div>
  )

  if (isLoading) {
    return (
      <div>
        {header}
        <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div>
        {header}
        <Result
          status="404"
          title="暂无分析报告"
          subTitle={
            <Text type="secondary">
              本次运行尚未生成 Markdown 报告。运行完成后，系统会自动在 <Text code>reports/</Text> 目录下生成报告，刷新后可查看。
            </Text>
          }
          extra={<Button onClick={() => mutate()}>重新检查</Button>}
        />
      </div>
    )
  }

  const components = buildComponents(data.assets_base)

  return (
    <div>
      {header}
      <Alert
        type="info"
        showIcon
        message={<>报告文件路径：<Text code style={{ fontSize: 11 }}>{data.path}</Text></>}
        style={{ marginBottom: 20, borderRadius: 6 }}
        closable
      />
      <div style={{
        background: '#ffffff',
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        padding: '24px 32px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.04)',
        maxWidth: 1100,
      }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {data.content}
        </ReactMarkdown>
      </div>
    </div>
  )
}
