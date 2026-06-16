import { Component, type ReactNode } from 'react'
import { Result, Button, Typography } from 'antd'

const { Paragraph, Text } = Typography

interface Props {
  children: ReactNode
  /** When this value changes, the boundary resets (e.g. route key). */
  resetKey?: string
}

interface State {
  error: Error | null
  info: string | null
}

/**
 * Catches render-time errors in the subtree so a single broken component
 * (e.g. a chart that fails on malformed data) shows a readable message
 * instead of blanking the whole app.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    // Surface to console for full stack while the on-screen panel stays concise.
    console.error('[ErrorBoundary] render error:', error, info?.componentStack)
    this.setState({ info: info?.componentStack ?? null })
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null, info: null })
    }
  }

  render() {
    const { error, info } = this.state
    if (!error) return this.props.children

    return (
      <Result
        status="error"
        title="页面渲染出错"
        subTitle="某个组件在渲染时抛出了异常，已被捕获。下面是错误详情（也已打印到浏览器控制台）。"
        extra={[
          <Button type="primary" key="reset" onClick={() => this.setState({ error: null, info: null })}>
            重试
          </Button>,
          <Button key="reload" onClick={() => window.location.reload()}>刷新页面</Button>,
        ]}
      >
        <div style={{ textAlign: 'left', background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: 6, padding: '12px 16px' }}>
          <Paragraph style={{ marginBottom: 8 }}>
            <Text strong style={{ color: '#cf1322' }}>{error.name}: </Text>
            <Text code style={{ color: '#cf1322' }}>{error.message}</Text>
          </Paragraph>
          {info && (
            <pre style={{ fontSize: 11, color: '#8c8c8c', maxHeight: 240, overflow: 'auto', margin: 0, whiteSpace: 'pre-wrap' }}>
              {info}
            </pre>
          )}
        </div>
      </Result>
    )
  }
}
