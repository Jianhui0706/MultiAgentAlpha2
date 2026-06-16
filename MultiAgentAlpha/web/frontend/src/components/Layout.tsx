import { Layout as AntLayout, Menu } from 'antd'
import {
  AreaChartOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FunctionOutlined,
  PlusCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { Link, useLocation } from 'react-router-dom'

const { Header, Sider, Content } = AntLayout

const menuItems = [
  { key: '/preprocess', icon: <FunctionOutlined />, label: <Link to="/preprocess">数据预处理</Link> },
  { key: '/factor-validation', icon: <AreaChartOutlined />, label: <Link to="/factor-validation">因子验证</Link> },
  { key: '/runs/new', icon: <PlusCircleOutlined />, label: <Link to="/runs/new">新建运行</Link> },
  { key: '/running', icon: <SyncOutlined />, label: <Link to="/running">正在运行</Link> },
  { key: '/history', icon: <DashboardOutlined />, label: <Link to="/history">历史运行</Link> },
  { key: '/factor-library', icon: <ExperimentOutlined />, label: <Link to="/factor-library">因子库</Link> },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  const selectedKey = menuItems
    .filter((m) => pathname === m.key || pathname.startsWith(m.key + '/'))
    .sort((a, b) => b.key.length - a.key.length)[0]?.key ?? '/runs'

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="dark">
        <div style={{ padding: '16px 24px', color: '#fff', fontWeight: 700, fontSize: 15, borderBottom: '1px solid #1f3a5f' }}>
          MultiAgentAlpha
        </div>
        <Menu theme="dark" mode="inline" selectedKeys={[selectedKey]} items={menuItems} />
      </Sider>
      <AntLayout>
        <Header style={{ background: '#fff', padding: '0 24px', borderBottom: '1px solid #f0f0f0', display: 'flex', alignItems: 'center' }}>
          <span style={{ color: '#666', fontSize: 13 }}>因子挖掘与固定验证平台</span>
        </Header>
        <Content style={{ margin: 24, padding: 24, background: '#fff', borderRadius: 8, minHeight: 360 }}>
          {children}
        </Content>
      </AntLayout>
    </AntLayout>
  )
}
